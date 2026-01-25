//! LLM inference engine using mistral-mlx

use eyre::{Context, Result};
use mlx_rs::ops::indexing::{IndexOp, NewAxis};
use mlx_rs::transforms::eval;
use mistral_mlx::{load_model, load_tokenizer, Generate, KVCache, Model};
use tokenizers::Tokenizer;

use crate::types::{ChatCompletionRequest, ChatCompletionResponse, ChatChoice, ChatMessage, ChatUsage};

/// LLM inference engine
pub struct LlmEngine {
    model_id: String,
    tokenizer: Tokenizer,
    model: Model,
}

impl LlmEngine {
    /// Create a new LLM engine by loading a model from HuggingFace Hub
    pub fn new(model_path: &str) -> Result<Self> {
        tracing::info!("Loading LLM model from: {}", model_path);

        // Download model from HuggingFace Hub
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(model_path.to_string());

        // Download config and tokenizer
        let config_path = repo.get("config.json")?;
        let _ = repo.get("tokenizer.json")?;

        // Download weights
        if let Ok(index_path) = repo.get("model.safetensors.index.json") {
            let index_content = std::fs::read_to_string(&index_path)?;
            let index: serde_json::Value = serde_json::from_str(&index_content)?;
            if let Some(weight_map) = index["weight_map"].as_object() {
                let weight_files: std::collections::HashSet<&str> = weight_map.values()
                    .filter_map(|v| v.as_str())
                    .collect();
                for weight_file in &weight_files {
                    let _ = repo.get(weight_file)?;
                }
            }
        } else {
            let _ = repo.get("model.safetensors")?;
        }

        let model_dir = config_path.parent()
            .ok_or_else(|| eyre::eyre!("Could not determine model directory"))?;

        // Load tokenizer and model
        let tokenizer = load_tokenizer(model_dir)
            .context("Failed to load tokenizer")?;
        let model = load_model(model_dir)
            .context("Failed to load model")?;

        tracing::info!("LLM model loaded successfully");

        Ok(Self {
            model_id: model_path.to_string(),
            tokenizer,
            model,
        })
    }

    /// Get the model ID
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Format chat messages into a prompt string
    fn format_messages(&self, messages: &[ChatMessage]) -> String {
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

        // For Mistral Instruct format
        if self.model_id.contains("Mistral") {
            // Reformat for Mistral
            let mut mistral_prompt = String::new();
            for msg in messages {
                if msg.role == "user" {
                    mistral_prompt.push_str(&format!("[INST] {} [/INST]", msg.content));
                } else if msg.role == "assistant" {
                    mistral_prompt.push_str(&msg.content);
                } else if msg.role == "system" {
                    mistral_prompt.push_str(&format!("[INST] <<SYS>>\n{}\n<</SYS>>\n\n", msg.content));
                }
            }
            if !mistral_prompt.is_empty() {
                return mistral_prompt;
            }
        }

        // Add assistant prompt to indicate where generation should start
        prompt.push_str("<|im_start|>assistant\n");
        prompt
    }

    /// Generate a chat completion response
    pub fn generate(&self, request: &ChatCompletionRequest) -> Result<ChatCompletionResponse> {
        let temperature = request.temperature.unwrap_or(0.7);
        let max_tokens = request.max_tokens.unwrap_or(2048);

        // Format messages into prompt
        let prompt = self.format_messages(&request.messages);

        // Tokenize prompt
        let encoding = self.tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| eyre::eyre!("Tokenization failed: {}", e))?;
        let prompt_tokens = encoding.get_ids().len() as u32;
        let prompt_array = mlx_rs::Array::from(encoding.get_ids()).index(NewAxis);

        // Create a mutable model reference for generation
        let mut model = self.model.clone();
        let mut cache: Vec<KVCache> = Vec::new();

        // Generate tokens
        let generator = Generate::<KVCache>::new(
            &mut model,
            &mut cache,
            temperature,
            &prompt_array,
        );

        let mut generated_tokens = Vec::new();
        let eos_token: u32 = 2; // </s> for Mistral

        for token in generator.take(max_tokens) {
            let token = token?;
            let token_id = token.item::<u32>();

            if token_id == eos_token {
                break;
            }

            generated_tokens.push(token);

            // Batch eval for efficiency
            if generated_tokens.len() % 10 == 0 {
                eval(&generated_tokens)?;
            }
        }

        // Decode generated tokens
        let token_ids: Vec<u32> = generated_tokens
            .iter()
            .map(|t| t.item::<u32>())
            .collect();
        let completion_tokens = token_ids.len() as u32;

        let content = self.tokenizer
            .decode(&token_ids, true)
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
