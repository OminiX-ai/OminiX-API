//! LLM inference engine supporting multiple model backends
//!
//! Currently supports:
//! - Qwen3 (via qwen3-mlx) — ChatML format
//! - GLM-4.7-Flash (via glm47-flash-mlx) — GLM chat format with MLA attention

use std::path::PathBuf;

use eyre::{Context, Result};
use mlx_rs::ops::indexing::{IndexOp, NewAxis};
use mlx_rs::Stream;
use tokenizers::Tokenizer;

use crate::model_config::{self, ModelAvailability, ModelCategory};
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

/// Which model backend is loaded
enum ModelBackend {
    Qwen3 {
        model: qwen3_mlx::Model,
        eos_tokens: Vec<u32>,
    },
    Glm4Flash {
        model: glm47_flash_mlx::Model,
        eos_tokens: Vec<u32>,
    },
}

/// LLM inference engine supporting multiple model backends
pub struct LlmEngine {
    model_type: String,
    backend: ModelBackend,
    tokenizer: Tokenizer,
}

impl LlmEngine {
    /// Create a new LLM engine by loading a model
    ///
    /// Detects model type from config.json and loads the appropriate backend.
    pub fn new(model_path: &str) -> Result<Self> {
        tracing::info!("Loading LLM model: {}", model_path);

        // Set wired memory limit
        set_wired_limit_max();

        // Resolve model directory
        let model_dir = resolve_model_dir(model_path)?;
        tracing::info!("Using model directory: {:?}", model_dir);

        // Read config.json to determine model type
        let config_path = model_dir.join("config.json");
        let config_content = std::fs::read_to_string(&config_path)
            .context("Failed to read config.json")?;
        let config: serde_json::Value = serde_json::from_str(&config_content)?;
        let model_type = config["model_type"].as_str().unwrap_or("").to_string();
        tracing::info!("Model type: {}", model_type);

        // Load tokenizer
        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(eyre::eyre!("tokenizer.json not found at {:?}", tokenizer_path));
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| eyre::eyre!("Failed to load tokenizer: {}", e))?;

        // Get EOS token IDs from config.json
        let eos_tokens = parse_eos_tokens(&config);

        // Load model based on type
        let backend = match model_type.as_str() {
            "glm4_moe_lite" => {
                tracing::info!("Loading GLM-4.7-Flash backend");
                let model = glm47_flash_mlx::load_model(&model_dir)
                    .map_err(|e| eyre::eyre!("Failed to load GLM model: {}", e))?;
                ModelBackend::Glm4Flash {
                    model,
                    eos_tokens,
                }
            }
            _ => {
                tracing::info!("Loading Qwen3 backend (model_type={})", model_type);
                let model = qwen3_mlx::load_model(&model_dir)
                    .context("Failed to load Qwen3 model")?;
                ModelBackend::Qwen3 {
                    model,
                    eos_tokens,
                }
            }
        };

        tracing::info!("LLM model loaded successfully");

        Ok(Self {
            model_type,
            backend,
            tokenizer,
        })
    }

    /// Format chat messages based on model type
    fn format_messages(&self, messages: &[ChatMessage]) -> String {
        match self.model_type.as_str() {
            "glm4_moe_lite" => self.format_messages_glm(messages),
            _ => self.format_messages_chatml(messages),
        }
    }

    /// Format messages in ChatML format (Qwen3)
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

    /// Format messages in GLM-4 chat format
    ///
    /// GLM-4 uses: <|system|>...\n<|user|>...\n<|assistant|></think>...
    /// Thinking is disabled (</think> immediately after <|assistant|>)
    fn format_messages_glm(&self, messages: &[ChatMessage]) -> String {
        let mut prompt = String::new();

        // GLM requires [gMASK]<sop> prefix
        prompt.push_str("[gMASK]<sop>");

        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    prompt.push_str("<|system|>");
                    prompt.push_str(&msg.content);
                    prompt.push('\n');
                }
                "user" => {
                    prompt.push_str("<|user|>");
                    prompt.push_str(&msg.content);
                    prompt.push('\n');
                }
                "assistant" => {
                    prompt.push_str("<|assistant|></think>");
                    prompt.push_str(&msg.content);
                    prompt.push('\n');
                }
                "observation" | "tool" => {
                    prompt.push_str("<|observation|>");
                    prompt.push_str(&msg.content);
                    prompt.push('\n');
                }
                role => {
                    prompt.push_str(&format!("<|{}|>", role));
                    prompt.push_str(&msg.content);
                    prompt.push('\n');
                }
            }
        }

        // Generation prompt: thinking disabled
        prompt.push_str("<|assistant|></think>");
        prompt
    }

    /// Generate a chat completion response
    pub fn generate(&self, request: &ChatCompletionRequest) -> Result<ChatCompletionResponse> {
        let temperature = request.temperature.unwrap_or(0.7);
        let max_tokens = request.max_tokens.unwrap_or(2048);

        // Format messages using appropriate template
        let prompt = self.format_messages(&request.messages);

        // Tokenize
        let encoding = self.tokenizer.encode(prompt.as_str(), false)
            .map_err(|e| eyre::eyre!("Tokenization failed: {}", e))?;
        let prompt_tokens = encoding.get_ids().len() as u32;
        let prompt_array = mlx_rs::Array::from(encoding.get_ids()).index(NewAxis);

        // Generate based on backend
        let generated_tokens = match &self.backend {
            ModelBackend::Qwen3 { model, eos_tokens } => {
                let mut model = model.clone();
                let mut cache: Vec<Option<qwen3_mlx::KVCache>> = Vec::new();

                let generator = qwen3_mlx::Generate::<qwen3_mlx::KVCache>::new(
                    &mut model,
                    &mut cache,
                    temperature,
                    &prompt_array,
                );

                let mut tokens = Vec::new();
                for token in generator.take(max_tokens) {
                    let token = token?;
                    let token_id = token.item::<u32>();
                    if eos_tokens.contains(&token_id) {
                        break;
                    }
                    tokens.push(token);
                }
                tokens
            }
            ModelBackend::Glm4Flash { model, eos_tokens } => {
                let mut model = model.clone();
                let mut cache: Vec<glm47_flash_mlx::KVCache> = Vec::new();

                let generator = glm47_flash_mlx::Generate::<glm47_flash_mlx::KVCache>::new(
                    &mut model,
                    &mut cache,
                    temperature,
                    &prompt_array,
                );

                let mut tokens = Vec::new();
                for token in generator.take(max_tokens) {
                    let token = token?;
                    let token_id = token.item::<u32>();
                    if eos_tokens.contains(&token_id) {
                        break;
                    }
                    tokens.push(token);
                }
                tokens
            }
        };

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

/// Resolve a model path/ID to a local directory
fn resolve_model_dir(model_path: &str) -> Result<PathBuf> {
    // If it's a direct filesystem path that exists, use it directly
    let direct_path = PathBuf::from(model_path);
    if direct_path.exists() && direct_path.join("config.json").exists() {
        tracing::info!("Using direct model path: {:?}", direct_path);
        return Ok(direct_path);
    }

    // Also try expanding ~ prefix
    if model_path.starts_with("~/") {
        let expanded = PathBuf::from(crate::utils::expand_tilde(model_path));
        if expanded.exists() && expanded.join("config.json").exists() {
            tracing::info!("Using expanded model path: {:?}", expanded);
            return Ok(expanded);
        }
    }

    // Check model configuration for local availability
    match model_config::check_model(model_path, ModelCategory::Llm) {
        ModelAvailability::Ready { local_path, model_name } => {
            tracing::info!("Found locally available model: {} at {:?}", model_name, local_path);
            let path = local_path.ok_or_else(|| eyre::eyre!("Model path not available"))?;
            crate::utils::resolve_hf_snapshot(&path)
        }
        ModelAvailability::NotDownloaded { model_name, model_id } => {
            Err(eyre::eyre!(
                "Model '{}' ({}) is not downloaded.\n\
                 Please download it using OminiX-Studio before starting the API server.",
                model_name, model_id
            ))
        }
        ModelAvailability::WrongCategory { expected, found } => {
            Err(eyre::eyre!(
                "Model '{}' is a {:?} model, not a {:?} model",
                model_path, found, expected
            ))
        }
        ModelAvailability::NotInConfig => {
            Err(eyre::eyre!(
                "Model '{}' not found in local configuration.\n\
                 Please add this model to OminiX-Studio and download it there first.\n\
                 Available LLM models can be viewed at: ~/.moly/local_models_config.json",
                model_path
            ))
        }
    }
}

/// Parse EOS token IDs from config.json
///
/// Handles both single integer and array formats.
fn parse_eos_tokens(config: &serde_json::Value) -> Vec<u32> {
    match &config["eos_token_id"] {
        serde_json::Value::Number(n) => {
            vec![n.as_u64().unwrap_or(151643) as u32]
        }
        serde_json::Value::Array(arr) => {
            arr.iter()
                .filter_map(|v| v.as_u64())
                .map(|v| v as u32)
                .collect()
        }
        _ => {
            // Qwen3 default
            vec![151643, 151645]
        }
    }
}
