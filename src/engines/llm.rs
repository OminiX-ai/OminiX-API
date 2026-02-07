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
use crate::types::{ChatCompletionRequest, ChatCompletionResponse, ChatChoice, ChatMessage, ChatUsage, Tool, ToolCall, FunctionCall};

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
    fn format_messages(&self, messages: &[ChatMessage], tools: Option<&[Tool]>) -> String {
        match self.model_type.as_str() {
            "glm4_moe_lite" => self.format_messages_glm(messages),
            _ => self.format_messages_chatml(messages, tools),
        }
    }

    /// Format messages in ChatML format (Qwen3)
    fn format_messages_chatml(&self, messages: &[ChatMessage], tools: Option<&[Tool]>) -> String {
        let mut prompt = String::new();
        let mut has_system = false;

        for msg in messages {
            let content = msg.content.as_deref().unwrap_or("");
            match msg.role.as_str() {
                "system" => {
                    prompt.push_str("<|im_start|>system\n");
                    // Inject tools into the system message
                    if let Some(tools) = tools {
                        if !tools.is_empty() {
                            prompt.push_str("You are a helpful assistant with access to the following tools. Use them when appropriate by outputting <tool_call> blocks.\n\n");
                            prompt.push_str("<tools>\n");
                            prompt.push_str(&serde_json::to_string_pretty(tools).unwrap_or_default());
                            prompt.push_str("\n</tools>\n\n");
                        }
                    }
                    prompt.push_str(content);
                    prompt.push_str("<|im_end|>\n");
                    has_system = true;
                }
                "user" => {
                    prompt.push_str("<|im_start|>user\n");
                    prompt.push_str(content);
                    prompt.push_str("<|im_end|>\n");
                }
                "assistant" => {
                    prompt.push_str("<|im_start|>assistant\n");
                    // If the message has tool_calls, format them
                    if let Some(tool_calls) = &msg.tool_calls {
                        for tc in tool_calls {
                            let call_json = serde_json::json!({
                                "name": tc.function.name,
                                "arguments": serde_json::from_str::<serde_json::Value>(&tc.function.arguments).unwrap_or(serde_json::Value::Object(Default::default()))
                            });
                            prompt.push_str(&format!("<tool_call>\n{}\n</tool_call>\n", serde_json::to_string(&call_json).unwrap_or_default()));
                        }
                    }
                    prompt.push_str(content);
                    prompt.push_str("<|im_end|>\n");
                }
                "tool" => {
                    prompt.push_str("<|im_start|>user\n");
                    prompt.push_str(&format!("<tool_response>\n{}\n</tool_response>", content));
                    prompt.push_str("<|im_end|>\n");
                }
                role => {
                    prompt.push_str(&format!("<|im_start|>{}\n", role));
                    prompt.push_str(content);
                    prompt.push_str("<|im_end|>\n");
                }
            }
        }

        // If no system message was found but tools are provided, inject one
        if !has_system {
            if let Some(tools) = tools {
                if !tools.is_empty() {
                    let tools_block = format!(
                        "<|im_start|>system\nYou are a helpful assistant with access to the following tools. Use them when appropriate by outputting <tool_call> blocks.\n\n<tools>\n{}\n</tools>\n<|im_end|>\n",
                        serde_json::to_string_pretty(tools).unwrap_or_default()
                    );
                    prompt = format!("{}{}", tools_block, prompt);
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
            let content = msg.content.as_deref().unwrap_or("");
            match msg.role.as_str() {
                "system" => {
                    prompt.push_str("<|system|>");
                    prompt.push_str(content);
                    prompt.push('\n');
                }
                "user" => {
                    prompt.push_str("<|user|>");
                    prompt.push_str(content);
                    prompt.push('\n');
                }
                "assistant" => {
                    prompt.push_str("<|assistant|></think>");
                    prompt.push_str(content);
                    prompt.push('\n');
                }
                "observation" | "tool" => {
                    prompt.push_str("<|observation|>");
                    prompt.push_str(content);
                    prompt.push('\n');
                }
                role => {
                    prompt.push_str(&format!("<|{}|>", role));
                    prompt.push_str(content);
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
        let prompt = self.format_messages(&request.messages, request.tools.as_deref());

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

        let raw_content = self.tokenizer.decode(&token_ids, true)
            .map_err(|e| eyre::eyre!("Decoding failed: {}", e))?;

        // Parse tool calls from output (Qwen3 format: <tool_call>...</tool_call>)
        let has_tools = request.tools.as_ref().map_or(false, |t| !t.is_empty());
        let (content, tool_calls, finish_reason) = if has_tools {
            parse_tool_calls(&raw_content)
        } else {
            (raw_content, vec![], "stop".to_string())
        };

        Ok(ChatCompletionResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: request.model.clone(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: if content.is_empty() { None } else { Some(content) },
                    tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls) },
                    tool_call_id: None,
                },
                finish_reason,
            }],
            usage: ChatUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        })
    }
}

/// Parse Qwen3 <tool_call> blocks from model output
///
/// Handles multiple Qwen3 output variations:
/// - Standard: `<tool_call>{"name": ..., "arguments": ...}</tool_call>`
/// - 8-bit quirk: `<tool_call>{"function": ..., "parameters": ...}<tool_call>` (missing /)
/// - Strips `<think>...</think>` blocks from remaining content
fn parse_tool_calls(content: &str) -> (String, Vec<ToolCall>, String) {
    let mut tool_calls = Vec::new();
    let mut remaining = content.to_string();

    // Strip <think>...</think> blocks first
    while let Some(start) = remaining.find("<think>") {
        if let Some(end) = remaining.find("</think>") {
            let end_abs = end + "</think>".len();
            remaining = format!("{}{}", &remaining[..start], &remaining[end_abs..]);
        } else {
            // Unclosed think block — remove from start to end
            remaining = remaining[..start].to_string();
            break;
        }
    }

    // Find all <tool_call>...</tool_call> or <tool_call>...<tool_call> blocks
    while let Some(start) = remaining.find("<tool_call>") {
        let after_open = start + "<tool_call>".len();

        // Try </tool_call> first, then <tool_call> as closing tag (8-bit model quirk)
        let (end_abs, inner) = if let Some(end) = remaining[after_open..].find("</tool_call>") {
            let end_abs = after_open + end + "</tool_call>".len();
            let inner = remaining[after_open..after_open + end].trim();
            (end_abs, inner.to_string())
        } else if let Some(end) = remaining[after_open..].find("<tool_call>") {
            // Model used <tool_call> as closing tag too
            let end_abs = after_open + end + "<tool_call>".len();
            let inner = remaining[after_open..after_open + end].trim();
            (end_abs, inner.to_string())
        } else {
            // No closing tag — try to extract JSON from remaining content
            let inner = remaining[after_open..].trim();
            (remaining.len(), inner.to_string())
        };

        if let Some((name, arguments)) = parse_tool_call_inner(&inner) {
            if !name.is_empty() {
                let id = format!("call_{}", &uuid::Uuid::new_v4().to_string().replace('-', "")[..24]);
                tool_calls.push(ToolCall {
                    id,
                    call_type: "function".to_string(),
                    function: FunctionCall { name, arguments },
                });
            }
        }

        // Remove the tool_call block from remaining
        remaining = format!("{}{}", &remaining[..start], &remaining[end_abs..]);
    }

    let remaining = remaining.trim().to_string();

    if tool_calls.is_empty() {
        (content.to_string(), vec![], "stop".to_string())
    } else {
        (remaining, tool_calls, "tool_calls".to_string())
    }
}

/// Parse the inner content of a <tool_call> block into (name, arguments_json).
/// Handles multiple formats:
/// 1. JSON: {"name": "fn", "arguments": {...}} (and variants)
/// 2. Python-like: function_name(key="value", key2=123, key3=[...])
fn parse_tool_call_inner(inner: &str) -> Option<(String, String)> {
    let inner = inner.trim();

    // Try JSON first
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(inner) {
        // Standard: {"name": "fn", "arguments": {...}}
        // Variant: {"function": "fn", "parameters": {...}}
        // Variant: {"tool": "fn", "params": {...}}
        // OpenAI-wrapped: {"type": "function", "function": {"name": ..., "arguments": ...}}
        let (name, args) = if let Some(n) = parsed["name"].as_str() {
            let a = parsed.get("arguments").or_else(|| parsed.get("parameters"))
                .map(|v| if v.is_string() { v.as_str().unwrap_or("{}").to_string() } else { serde_json::to_string(v).unwrap_or_default() })
                .unwrap_or_else(|| "{}".to_string());
            (n.to_string(), a)
        } else if let Some(n) = parsed["function"].as_str() {
            let a = parsed.get("parameters").or_else(|| parsed.get("arguments"))
                .map(|v| if v.is_string() { v.as_str().unwrap_or("{}").to_string() } else { serde_json::to_string(v).unwrap_or_default() })
                .unwrap_or_else(|| "{}".to_string());
            (n.to_string(), a)
        } else if let Some(n) = parsed["tool"].as_str() {
            let a = parsed.get("params").or_else(|| parsed.get("parameters")).or_else(|| parsed.get("arguments"))
                .map(|v| if v.is_string() { v.as_str().unwrap_or("{}").to_string() } else { serde_json::to_string(v).unwrap_or_default() })
                .unwrap_or_else(|| "{}".to_string());
            (n.to_string(), a)
        } else if let Some(func_obj) = parsed.get("function").and_then(|f| f.as_object()) {
            let n = func_obj.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
            let a = func_obj.get("arguments")
                .map(|v| if v.is_string() { v.as_str().unwrap_or("{}").to_string() } else { serde_json::to_string(v).unwrap_or_default() })
                .unwrap_or_else(|| "{}".to_string());
            (n, a)
        } else {
            return None;
        };
        if !name.is_empty() { return Some((name, args)); }
    }

    // Fallback: Python-like function call syntax
    // Pattern: function_name(key1="value1", key2=123, key3=["a","b"], key4=true)
    if let Some(paren_pos) = inner.find('(') {
        let name = inner[..paren_pos].trim().to_string();
        if name.chars().all(|c| c.is_alphanumeric() || c == '_') && !name.is_empty() {
            let args_str = inner[paren_pos + 1..].trim_end_matches(')').trim();
            if args_str.is_empty() {
                return Some((name, "{}".to_string()));
            }
            // Parse key=value pairs into a JSON object
            if let Some(args_json) = parse_kwargs_to_json(args_str) {
                return Some((name, args_json));
            }
        }
    }

    None
}

/// Parse Python-like keyword arguments into a JSON string.
/// Input: `id="title", text="My Books", style="h1"`
/// Output: `{"id":"title","text":"My Books","style":"h1"}`
fn parse_kwargs_to_json(input: &str) -> Option<String> {
    let mut result = serde_json::Map::new();
    let mut pos = 0;
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();

    while pos < len {
        // Skip whitespace and commas
        while pos < len && (chars[pos] == ' ' || chars[pos] == ',' || chars[pos] == '\n' || chars[pos] == '\r') {
            pos += 1;
        }
        if pos >= len { break; }

        // Read key
        let key_start = pos;
        while pos < len && chars[pos] != '=' {
            pos += 1;
        }
        if pos >= len { break; }
        let key = input[key_start..pos].trim().to_string();
        pos += 1; // skip '='

        // Skip whitespace
        while pos < len && chars[pos] == ' ' { pos += 1; }
        if pos >= len { break; }

        // Read value
        let value: serde_json::Value = if chars[pos] == '"' || chars[pos] == '\'' {
            // String value
            let quote = chars[pos];
            pos += 1;
            let val_start = pos;
            while pos < len && chars[pos] != quote {
                if chars[pos] == '\\' { pos += 1; } // skip escaped chars
                pos += 1;
            }
            let val = input[val_start..pos].to_string();
            if pos < len { pos += 1; } // skip closing quote
            serde_json::Value::String(val)
        } else if chars[pos] == '[' {
            // Array value - find matching ]
            let val_start = pos;
            let mut depth = 0;
            while pos < len {
                match chars[pos] {
                    '[' => depth += 1,
                    ']' => { depth -= 1; if depth == 0 { pos += 1; break; } },
                    '"' => { pos += 1; while pos < len && chars[pos] != '"' { if chars[pos] == '\\' { pos += 1; } pos += 1; } },
                    _ => {},
                }
                pos += 1;
            }
            let val_str = &input[val_start..pos];
            serde_json::from_str(val_str).unwrap_or(serde_json::Value::String(val_str.to_string()))
        } else if chars[pos] == '{' {
            // Object value - find matching }
            let val_start = pos;
            let mut depth = 0;
            while pos < len {
                match chars[pos] {
                    '{' => depth += 1,
                    '}' => { depth -= 1; if depth == 0 { pos += 1; break; } },
                    '"' => { pos += 1; while pos < len && chars[pos] != '"' { if chars[pos] == '\\' { pos += 1; } pos += 1; } },
                    _ => {},
                }
                pos += 1;
            }
            let val_str = &input[val_start..pos];
            serde_json::from_str(val_str).unwrap_or(serde_json::Value::String(val_str.to_string()))
        } else {
            // Number, boolean, or other literal
            let val_start = pos;
            while pos < len && chars[pos] != ',' && chars[pos] != ')' && chars[pos] != '\n' {
                pos += 1;
            }
            let val_str = input[val_start..pos].trim();
            if val_str == "true" || val_str == "True" {
                serde_json::Value::Bool(true)
            } else if val_str == "false" || val_str == "False" {
                serde_json::Value::Bool(false)
            } else if let Ok(n) = val_str.parse::<i64>() {
                serde_json::Value::Number(n.into())
            } else if let Ok(n) = val_str.parse::<f64>() {
                serde_json::json!(n).as_number().map(|n| serde_json::Value::Number(n.clone())).unwrap_or(serde_json::Value::String(val_str.to_string()))
            } else {
                serde_json::Value::String(val_str.to_string())
            }
        };

        result.insert(key, value);
    }

    if result.is_empty() {
        None
    } else {
        Some(serde_json::to_string(&serde_json::Value::Object(result)).unwrap_or_else(|_| "{}".to_string()))
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
            // Try standard model hub caches (HuggingFace, ModelScope)
            if let Some(hub_path) = crate::utils::resolve_from_hub_cache(model_path) {
                if hub_path.join("config.json").exists() {
                    tracing::info!("Found model in hub cache: {:?}", hub_path);
                    let _ = model_config::register_model(model_path, ModelCategory::Llm, &hub_path);
                    return Ok(hub_path);
                }
            }
            Err(eyre::eyre!(
                "Model '{}' not found in local configuration or hub caches.\n\
                 Please download it via OminiX-Studio or huggingface-cli.\n\
                 Searched: ~/.OminiX/local_models_config.json, ~/.cache/huggingface/hub/, ~/.cache/modelscope/hub/",
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
