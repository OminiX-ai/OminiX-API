use salvo::cors::*;
use salvo::prelude::*;

use crate::handlers;
use crate::state::AppState;

pub fn build_router(state: AppState) -> Router {
    let cors = Cors::new()
        .allow_origin(AllowOrigin::any())
        .allow_methods(AllowMethods::any())
        .allow_headers(AllowHeaders::any())
        .into_handler();

    Router::new()
        .hoop(affix_state::inject(state))
        .hoop(cors)
        .push(Router::with_path("health").get(handlers::health::health))
        .push(Router::with_path("version").get(handlers::version::get_version))
        .push(
            Router::with_path("v1")
                // Version
                .push(Router::with_path("version").get(handlers::version::get_version))
                // Models — no handler on "models" itself, use empty-path child for exact match
                .push(Router::with_path("models")
                    .push(Router::new().get(handlers::health::list_models))
                    .push(Router::with_path("status").get(handlers::health::model_status))
                    .push(Router::with_path("report").get(handlers::health::models_report))
                    .push(Router::with_path("load").post(handlers::models::load_model))
                    .push(Router::with_path("unload").post(handlers::models::unload_model))
                    .push(Router::with_path("quantize").post(handlers::image::quantize_model))
                    .push(Router::with_path("catalog").get(handlers::download::model_catalog))
                    .push(Router::with_path("download")
                        .push(Router::new().post(handlers::download::download_model))
                        .push(Router::with_path("progress").get(handlers::download::download_progress_sse))
                        .push(Router::with_path("cancel").post(handlers::download::cancel_download))
                    )
                    .push(Router::with_path("remove").post(handlers::download::remove_model))
                    .push(Router::with_path("scan").post(handlers::download::scan_models))
                )
                // Chat
                .push(Router::with_path("chat/completions").post(handlers::chat::chat_completions))
                // Audio — model-specific endpoints (preferred)
                .push(Router::with_path("audio")
                    // TTS model-specific
                    .push(Router::with_path("tts")
                        .push(Router::with_path("qwen3").post(handlers::audio::tts_qwen3))
                        .push(Router::with_path("clone").post(handlers::audio::tts_clone))
                        .push(Router::with_path("sovits").post(handlers::audio::tts_sovits))
                    )
                    // ASR model-specific
                    .push(Router::with_path("asr")
                        .push(Router::with_path("qwen3").post(handlers::audio::asr_qwen3))
                        .push(Router::with_path("paraformer").post(handlers::audio::asr_paraformer))
                    )
                    // Legacy endpoints (backward-compatible, auto-routes)
                    .push(Router::with_path("transcriptions").post(handlers::audio::audio_transcriptions))
                    .push(Router::with_path("speech")
                        .push(Router::new().post(handlers::audio::audio_speech))
                        .push(Router::with_path("clone").post(handlers::audio::audio_speech_clone))
                    )
                )
                // Images
                .push(Router::with_path("images/generations").post(handlers::image::images_generations))
                // VLM
                .push(Router::with_path("vlm/completions").post(handlers::vlm::vlm_completions))
                // Voices
                .push(Router::with_path("voices")
                    .push(Router::new().get(handlers::training::list_voices))
                    .push(Router::with_path("train")
                        .push(Router::new().post(handlers::training::start_voice_training))
                        .push(Router::with_path("status").get(handlers::training::get_training_status))
                        .push(Router::with_path("progress").get(handlers::training::training_progress_sse))
                        .push(Router::with_path("cancel").post(handlers::training::cancel_training))
                    )
                )
        )
        .push(Router::with_path("ws/v1/tts").get(handlers::ws_tts::ws_tts))
}
