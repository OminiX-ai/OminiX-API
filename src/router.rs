use salvo::cors::*;
use salvo::prelude::*;

use crate::handlers;
use crate::state::AppState;

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .hoop(affix_state::inject(state))
        .hoop(
            Cors::new()
                .allow_origin(AllowOrigin::any())
                .allow_methods(AllowMethods::any())
                .allow_headers(AllowHeaders::any())
                .into_handler(),
        )
        // Health & Models
        .push(Router::with_path("health").get(handlers::health::health))
        .push(Router::with_path("v1/models").get(handlers::health::list_models))
        .push(Router::with_path("v1/models/status").get(handlers::health::model_status))
        .push(Router::with_path("v1/models/report").get(handlers::health::models_report))
        .push(Router::with_path("v1/models/load").post(handlers::models::load_model))
        .push(Router::with_path("v1/models/unload").post(handlers::models::unload_model))
        .push(Router::with_path("v1/models/quantize").post(handlers::image::quantize_model))
        // Chat completions
        .push(Router::with_path("v1/chat/completions").post(handlers::chat::chat_completions))
        // Audio endpoints
        .push(Router::with_path("v1/audio/transcriptions").post(handlers::audio::audio_transcriptions))
        .push(Router::with_path("v1/audio/speech").post(handlers::audio::audio_speech))
        // Image generation
        .push(Router::with_path("v1/images/generations").post(handlers::image::images_generations))
        // WebSocket TTS
        .push(Router::with_path("ws/v1/tts").get(handlers::ws_tts::ws_tts))
        // Voice cloning training
        .push(Router::with_path("v1/voices").get(handlers::training::list_voices))
        .push(Router::with_path("v1/voices/train").post(handlers::training::start_voice_training))
        .push(Router::with_path("v1/voices/train/status").get(handlers::training::get_training_status))
        .push(Router::with_path("v1/voices/train/progress").get(handlers::training::training_progress_sse))
        .push(Router::with_path("v1/voices/train/cancel").post(handlers::training::cancel_training))
}
