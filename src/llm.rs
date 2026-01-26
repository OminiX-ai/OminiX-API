//! LLM inference engine using qwen3-mlx

use std::collections::HashSet;

use eyre::{Context, Result};
use mlx_rs::ops::indexing::{IndexOp, NewAxis};
use mlx_rs::Stream;
use tokenizers::Tokenizer;

use crate::types::{ChatCompletionRequest, ChatCompletionResponse, ChatChoice, ChatMessage, ChatUsage};

/// Synchronize the given stream
fn synchronize(stream: &Stream) {
    unsafe {
        mlx_sys::mlx_synchronize(stream.as_ptr());
    }
}

/// Set the wired memory limit for better GPU performance
fn set_wired_limit_max() {
    unsafe {
        let info = mlx_sys::mlx_metal_device_info();
        let max_size = info.max_recommended_working_set_size;
        let mut old_limit: usize = 0;
        mlx_sys::mlx_set_wired_limit(&mut old_limit, max_size);
        tracing::info!(
            "Set wired limit to {} MB (was {} MB)",
            max_size / (1024 * 1024),
            old_limit / (1024 * 1024)
        );
    }
}

/// LLM inference engine using Qwen3
pub struct LlmEngine {
    model_id: String,
    model: qwen3_mlx::Model,
    tokenizer: Tokenizer,
    eos_token_id: u32,
}

impl LlmEngine {
    /// Create a new LLM engine by loading a model from HuggingFace Hub
    pub fn new(model_path: &str) -> Result<Self> {
        tracing::info!("Loading LLM model from: {}", model_path);

        // Set wired memory limit
        set_wired_limit_max();

        // Download model from HuggingFace Hub
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(model_path.to_string());

        // Download config and tokenizer
        let config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;

        // Read config to determine model type
        let config_content = std::fs::read_to_string(&config_path)?;
        let config: serde_json::Value = serde_json::from_str(&config_content)?;
        let model_type = config["model_type"].as_str().unwrap_or("");

        tracing::info!("Model type: {}", model_type);

        // Download weights
        if let Ok(index_path) = repo.get("model.safetensors.index.json") {
            let index_content = std::fs::read_to_string(&index_path)?;
            let index: serde_json::Value = serde_json::from_str(&index_content)?;
            if let Some(weight_map) = index["weight_map"].as_object() {
                let weight_files: HashSet<&str> = weight_map.values()
                    .filter_map(|v| v.as_str())
                    .collect();
                tracing::info!("Downloading {} weight files...", weight_files.len());
                for weight_file in &weight_files {
                    let _ = repo.get(weight_file)?;
                }
            }
        } else {
            let _ = repo.get("model.safetensors")?;
        }

        let model_dir = config_path.parent()
            .ok_or_else(|| eyre::eyre!("Could not determine model directory"))?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| eyre::eyre!("Failed to load tokenizer: {}", e))?;

        // Get EOS token ID from tokenizer config
        let tokenizer_config_path = repo.get("tokenizer_config.json").ok();
        let eos_token_id = if let Some(tc_path) = tokenizer_config_path {
            let tc_content = std::fs::read_to_string(&tc_path).unwrap_or_default();
            let tc: serde_json::Value = serde_json::from_str(&tc_content).unwrap_or_default();
            tc["eos_token_id"].as_u64().unwrap_or(151643) as u32
        } else {
            151643 // Qwen3 default
        };

        // Load model
        let model = qwen3_mlx::load_model(model_dir)
            .context("Failed to load Qwen3 model")?;

        tracing::info!("LLM model loaded successfully (eos_token_id={})", eos_token_id);

        Ok(Self {
            model_id: model_path.to_string(),
            model,
            tokenizer,
            eos_token_id,
        })
    }

    /// Get the model ID
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Format chat messages into a prompt string (ChatML format for Qwen3)
    fn format_messages_chatml(&self, messages: &[ChatMessage]) -> String {
        let mut prompt = String::new();

        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    prompt.push_str("<|im_start|>system\n");
                    prompt.push_str(&msg.content);
                    prompt.push_str("<|im_end|>\n");
                }
                "user" => {
                    prompt.push_str("<|im_start|>user\n");
                    prompt.push_str(&msg.content);
                    prompt.push_str("<|im_end|>\n");
                }
                "assistant" => {
                    prompt.push_str("<|im_start|>assistant\n");
                    prompt.push_str(&msg.content);
                    prompt.push_str("<|im_end|>\n");
                }
                role => {
                    prompt.push_str(&format!("<|im_start|>{}\n", role));
                    prompt.push_str(&msg.content);
                    prompt.push_str("<|im_end|>\n");
                }
            }
        }

        // Add assistant prompt to indicate where generation should start
        prompt.push_str("<|im_start|>assistant\n");
        prompt
    }

    /// Generate a chat completion response
    pub fn generate(&self, request: &ChatCompletionRequest) -> Result<ChatCompletionResponse> {
        use qwen3_mlx::{Generate, KVCache};

        let temperature = request.temperature.unwrap_or(0.7);
        let max_tokens = request.max_tokens.unwrap_or(2048);

        // Format messages
        let prompt = self.format_messages_chatml(&request.messages);

        // Tokenize
        let encoding = self.tokenizer.encode(prompt.as_str(), false)
            .map_err(|e| eyre::eyre!("Tokenization failed: {}", e))?;
        let prompt_tokens = encoding.get_ids().len() as u32;
        let prompt_array = mlx_rs::Array::from(encoding.get_ids()).index(NewAxis);

        // Generate (clone model for thread safety)
        let mut model = self.model.clone();
        let mut cache: Vec<Option<KVCache>> = Vec::new();

        let generator = Generate::<KVCache>::new(
            &mut model,
            &mut cache,
            temperature,
            &prompt_array,
        );

        let mut generated_tokens = Vec::new();
        // Qwen3 EOS tokens: 151643 (<|im_end|>), 151645 (<|endoftext|>)
        let eos_tokens: [u32; 2] = [self.eos_token_id, 151645];

        for token in generator.take(max_tokens) {
            let token = token?;
            let token_id = token.item::<u32>();

            if eos_tokens.contains(&token_id) {
                break;
            }

            generated_tokens.push(token);
        }

        // Synchronize before decoding
        synchronize(&Stream::default());

        // Decode
        let token_ids: Vec<u32> = generated_tokens
            .iter()
            .map(|t| t.item::<u32>())
            .collect();
        let completion_tokens = token_ids.len() as u32;

        let content = self.tokenizer.decode(&token_ids, true)
            .map_err(|e| eyre::eyre!("Decoding failed: {}", e))?;

        Ok(ChatCompletionResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: request.model.clone(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content,
                },
                finish_reason: "stop".to_string(),
            }],
            usage: ChatUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        })
    }
}
