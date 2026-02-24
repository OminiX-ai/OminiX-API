//! Default model catalog and download specifications
//!
//! Contains the 6 default models (ported from OminiX-Studio) and functions
//! to build the full model catalog by merging defaults with on-disk state.

use serde::{Deserialize, Serialize};

use crate::model_config::{self, LocalModelsConfig, ModelCategory};

/// Source type for model downloads
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    Huggingface,
    Modelscope,
}

/// Model source for downloads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogSource {
    pub primary_url: String,
    #[serde(default)]
    pub backup_urls: Vec<String>,
    pub source_type: SourceType,
    pub repo_id: Option<String>,
    pub revision: String,
}

/// Storage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogStorage {
    pub local_path: String,
    pub total_size_bytes: u64,
    pub total_size_display: String,
}

/// Runtime requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogRuntime {
    pub memory_required_mb: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_engine: Option<String>,
}

/// Full catalog model definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogModel {
    pub id: String,
    pub name: String,
    pub description: String,
    pub category: ModelCategory,
    pub tags: Vec<String>,
    pub source: CatalogSource,
    pub storage: CatalogStorage,
    pub runtime: CatalogRuntime,
    /// Current status: "ready", "not_downloaded", "downloading", "error"
    pub status: String,
}

/// Get the default models (ported from OminiX-Studio + 8-bit FLUX variant + Qwen3-ASR)
pub fn get_default_models() -> Vec<CatalogModel> {
    vec![
        CatalogModel {
            id: "flux-klein-4b".to_string(),
            name: "FLUX.2-klein-4B".to_string(),
            description: "4B parameter FLUX image generation model. Fast inference with 4-step generation, optimized for Apple Silicon.".to_string(),
            category: ModelCategory::Image,
            tags: vec!["image-generation".into(), "flux".into(), "apple-silicon".into(), "mlx".into()],
            source: CatalogSource {
                primary_url: "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B".to_string(),
                backup_urls: vec![
                    "https://hf-mirror.com/black-forest-labs/FLUX.2-klein-4B".to_string(),
                ],
                source_type: SourceType::Huggingface,
                repo_id: Some("black-forest-labs/FLUX.2-klein-4B".to_string()),
                revision: "main".to_string(),
            },
            storage: CatalogStorage {
                local_path: "~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B".to_string(),
                total_size_bytes: 13_958_643_712,
                total_size_display: "~13 GB".to_string(),
            },
            runtime: CatalogRuntime {
                memory_required_mb: 16384,
                quantization: Some("bf16".into()),
                inference_engine: Some("mlx".into()),
            },
            status: "not_downloaded".to_string(),
        },
        CatalogModel {
            id: "flux-klein-4b-8bit".to_string(),
            name: "FLUX.2-klein-4B 8-bit".to_string(),
            description: "4B parameter FLUX image generation model, 8-bit quantized. Lower memory usage (~12 GB) while maintaining quality.".to_string(),
            category: ModelCategory::Image,
            tags: vec!["image-generation".into(), "flux".into(), "quantized".into(), "8bit".into(), "mlx".into()],
            source: CatalogSource {
                primary_url: "https://huggingface.co/moxin-org/FLUX.2-klein-4B-8bit-mlx".to_string(),
                backup_urls: vec![],
                source_type: SourceType::Huggingface,
                repo_id: Some("moxin-org/FLUX.2-klein-4B-8bit-mlx".to_string()),
                revision: "main".to_string(),
            },
            storage: CatalogStorage {
                local_path: "~/.cache/huggingface/hub/models--moxin-org--FLUX.2-klein-4B-8bit-mlx".to_string(),
                total_size_bytes: 12_584_580_985,
                total_size_display: "~12 GB".to_string(),
            },
            runtime: CatalogRuntime {
                memory_required_mb: 14336,
                quantization: Some("8bit".into()),
                inference_engine: Some("mlx".into()),
            },
            status: "not_downloaded".to_string(),
        },
        CatalogModel {
            id: "zimage-turbo".to_string(),
            name: "Z-Image Turbo".to_string(),
            description: "6B parameter S3-DiT image generation. 9-step turbo inference (~3s/image), 4-bit quantized for efficient memory usage.".to_string(),
            category: ModelCategory::Image,
            tags: vec!["image-generation".into(), "s3-dit".into(), "quantized".into(), "fast".into()],
            source: CatalogSource {
                primary_url: "https://huggingface.co/uqer1244/MLX-z-image".to_string(),
                backup_urls: vec![],
                source_type: SourceType::Huggingface,
                repo_id: Some("uqer1244/MLX-z-image".to_string()),
                revision: "main".to_string(),
            },
            storage: CatalogStorage {
                local_path: "~/.cache/huggingface/hub/models--uqer1244--MLX-z-image".to_string(),
                total_size_bytes: 12_884_901_888,
                total_size_display: "~12 GB".to_string(),
            },
            runtime: CatalogRuntime {
                memory_required_mb: 14336,
                quantization: Some("4bit".into()),
                inference_engine: Some("mlx".into()),
            },
            status: "not_downloaded".to_string(),
        },
        CatalogModel {
            id: "qwen3-8b".to_string(),
            name: "Qwen3 8B".to_string(),
            description: "Powerful 8B parameter language model. Excellent for chat, coding assistance, and general text generation.".to_string(),
            category: ModelCategory::Llm,
            tags: vec!["llm".into(), "chat".into(), "coding".into(), "qwen".into()],
            source: CatalogSource {
                primary_url: "https://huggingface.co/mlx-community/Qwen3-8B-8bit".to_string(),
                backup_urls: vec![],
                source_type: SourceType::Huggingface,
                repo_id: Some("mlx-community/Qwen3-8B-8bit".to_string()),
                revision: "main".to_string(),
            },
            storage: CatalogStorage {
                local_path: "~/.cache/huggingface/hub/models--mlx-community--Qwen3-8B-8bit".to_string(),
                total_size_bytes: 8_589_934_592,
                total_size_display: "~8 GB".to_string(),
            },
            runtime: CatalogRuntime {
                memory_required_mb: 10240,
                quantization: Some("8bit".into()),
                inference_engine: Some("mlx".into()),
            },
            status: "not_downloaded".to_string(),
        },
        CatalogModel {
            id: "funasr-paraformer".to_string(),
            name: "FunASR Paraformer".to_string(),
            description: "High-accuracy Chinese ASR. Downloads PyTorch model and auto-converts to MLX format.".to_string(),
            category: ModelCategory::Asr,
            tags: vec!["asr".into(), "chinese".into(), "speech-recognition".into(), "funasr".into()],
            source: CatalogSource {
                primary_url: "https://modelscope.cn/models/damo/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch".to_string(),
                backup_urls: vec![],
                source_type: SourceType::Modelscope,
                repo_id: Some("damo/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch".to_string()),
                revision: "master".to_string(),
            },
            storage: CatalogStorage {
                local_path: "~/.dora/models/paraformer".to_string(),
                total_size_bytes: 1_073_741_824,
                total_size_display: "~1.0 GB".to_string(),
            },
            runtime: CatalogRuntime {
                memory_required_mb: 2048,
                quantization: None,
                inference_engine: Some("mlx".into()),
            },
            status: "not_downloaded".to_string(),
        },
        CatalogModel {
            id: "funasr-nano".to_string(),
            name: "FunASR Nano".to_string(),
            description: "800M LLM-based ASR supporting 31 languages. Combines Whisper encoder with Qwen LLM for high accuracy.".to_string(),
            category: ModelCategory::Asr,
            tags: vec!["asr".into(), "multilingual".into(), "speech-recognition".into(), "whisper".into(), "qwen".into()],
            source: CatalogSource {
                primary_url: "https://huggingface.co/mlx-community/Fun-ASR-Nano-2512-fp16".to_string(),
                backup_urls: vec![],
                source_type: SourceType::Huggingface,
                repo_id: Some("mlx-community/Fun-ASR-Nano-2512-fp16".to_string()),
                revision: "main".to_string(),
            },
            storage: CatalogStorage {
                local_path: "~/.dora/models/funasr-nano".to_string(),
                total_size_bytes: 2_040_109_465,
                total_size_display: "~1.9 GB".to_string(),
            },
            runtime: CatalogRuntime {
                memory_required_mb: 3072,
                quantization: Some("fp16".into()),
                inference_engine: Some("mlx".into()),
            },
            status: "not_downloaded".to_string(),
        },
        CatalogModel {
            id: "funasr-qwen4b".to_string(),
            name: "FunASR Qwen4B".to_string(),
            description: "SenseVoice encoder + Qwen3-4B LLM ASR. 5.10% CER on AISHELL-1 with 8-bit quantization at 0.56x RTF.".to_string(),
            category: ModelCategory::Asr,
            tags: vec!["asr".into(), "chinese".into(), "speech-recognition".into(), "sensevoice".into(), "qwen".into(), "8bit".into()],
            source: CatalogSource {
                primary_url: "https://huggingface.co/mlx-community/Qwen3-4B-8bit".to_string(),
                backup_urls: vec![
                    "https://huggingface.co/yuechen/funasr-qwen4b-mlx".to_string(),
                ],
                source_type: SourceType::Huggingface,
                repo_id: Some("mlx-community/Qwen3-4B-8bit".to_string()),
                revision: "main".to_string(),
            },
            storage: CatalogStorage {
                local_path: "~/.dora/models/funasr-qwen4b".to_string(),
                total_size_bytes: 6_000_000_000,
                total_size_display: "~5.6 GB".to_string(),
            },
            runtime: CatalogRuntime {
                memory_required_mb: 8192,
                quantization: Some("8bit".into()),
                inference_engine: Some("mlx".into()),
            },
            status: "not_downloaded".to_string(),
        },
        CatalogModel {
            id: "qwen3-asr-1.7b".to_string(),
            name: "Qwen3-ASR 1.7B 8-bit".to_string(),
            description: "Qwen3-ASR 1.7B encoder-decoder ASR. 30+ languages, ~30x realtime on Apple Silicon. Best accuracy among Qwen3-ASR variants.".to_string(),
            category: ModelCategory::Asr,
            tags: vec!["asr".into(), "multilingual".into(), "speech-recognition".into(), "qwen3".into(), "8bit".into()],
            source: CatalogSource {
                primary_url: "https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit".to_string(),
                backup_urls: vec![],
                source_type: SourceType::Huggingface,
                repo_id: Some("mlx-community/Qwen3-ASR-1.7B-8bit".to_string()),
                revision: "main".to_string(),
            },
            storage: CatalogStorage {
                local_path: "~/.OminiX/models/qwen3-asr-1.7b".to_string(),
                total_size_bytes: 2_641_100_800,
                total_size_display: "~2.5 GB".to_string(),
            },
            runtime: CatalogRuntime {
                memory_required_mb: 4096,
                quantization: Some("8bit".into()),
                inference_engine: Some("mlx".into()),
            },
            status: "not_downloaded".to_string(),
        },
        CatalogModel {
            id: "qwen3-asr-0.6b".to_string(),
            name: "Qwen3-ASR 0.6B 8-bit".to_string(),
            description: "Qwen3-ASR 0.6B encoder-decoder ASR. 30+ languages, compact model (~1 GB). Faster download, slightly lower accuracy.".to_string(),
            category: ModelCategory::Asr,
            tags: vec!["asr".into(), "multilingual".into(), "speech-recognition".into(), "qwen3".into(), "8bit".into(), "compact".into()],
            source: CatalogSource {
                primary_url: "https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-8bit".to_string(),
                backup_urls: vec![],
                source_type: SourceType::Huggingface,
                repo_id: Some("mlx-community/Qwen3-ASR-0.6B-8bit".to_string()),
                revision: "main".to_string(),
            },
            storage: CatalogStorage {
                local_path: "~/.OminiX/models/qwen3-asr-0.6b".to_string(),
                total_size_bytes: 1_085_276_160,
                total_size_display: "~1.0 GB".to_string(),
            },
            runtime: CatalogRuntime {
                memory_required_mb: 2048,
                quantization: Some("8bit".into()),
                inference_engine: Some("mlx".into()),
            },
            status: "not_downloaded".to_string(),
        },
    ]
}

/// Build the full model catalog by merging defaults with on-disk state.
pub fn get_model_catalog() -> Vec<CatalogModel> {
    // Load config once to avoid repeated disk reads
    let config = LocalModelsConfig::load()
        .unwrap_or_else(LocalModelsConfig::default_empty);
    let mut catalog = get_default_models();

    // Update status based on actual disk state (uses pre-loaded config)
    for model in &mut catalog {
        let availability = config.check_model_availability(&model.id, model.category.clone());
        match availability {
            model_config::ModelAvailability::Ready { .. } => {
                model.status = "ready".to_string();
            }
            _ => {
                // Also check by repo_id if available
                if let Some(ref repo_id) = model.source.repo_id {
                    let avail2 = config.check_model_availability(repo_id, model.category.clone());
                    if matches!(avail2, model_config::ModelAvailability::Ready { .. }) {
                        model.status = "ready".to_string();
                    }
                }
            }
        }
    }

    // Add any models from config that aren't in the defaults
    for config_model in &config.models {
        let already_in_catalog = catalog.iter().any(|c| {
            c.id == config_model.id
                || (c.source.repo_id.is_some()
                    && c.source.repo_id.as_deref() == config_model.source.repo_id.as_deref())
        });
        if already_in_catalog {
            continue;
        }

            catalog.push(CatalogModel {
                id: config_model.id.clone(),
                name: config_model.name.clone(),
                description: config_model
                    .description
                    .clone()
                    .unwrap_or_default(),
                category: config_model.category.clone(),
                tags: vec![],
                source: CatalogSource {
                    primary_url: config_model
                        .source
                        .primary_url
                        .clone()
                        .unwrap_or_default(),
                    backup_urls: vec![],
                    source_type: if config_model.source.source_type.as_deref()
                        == Some("modelscope")
                    {
                        SourceType::Modelscope
                    } else {
                        SourceType::Huggingface
                    },
                    repo_id: config_model.source.repo_id.clone(),
                    revision: "main".to_string(),
                },
                storage: CatalogStorage {
                    local_path: config_model.storage.local_path.clone(),
                    total_size_bytes: config_model.storage.total_size_bytes.unwrap_or(0),
                    total_size_display: config_model
                        .storage
                        .total_size_display
                        .clone()
                        .unwrap_or_default(),
                },
                runtime: CatalogRuntime {
                    memory_required_mb: 0,
                    quantization: None,
                    inference_engine: None,
                },
                status: if config_model.is_ready() {
                    "ready".to_string()
                } else {
                    "not_downloaded".to_string()
                },
            });
        }

    catalog
}

/// Look up a model spec by ID for downloading.
pub fn get_download_spec(model_id: &str) -> Option<CatalogModel> {
    get_default_models()
        .into_iter()
        .find(|m| m.id == model_id)
}
