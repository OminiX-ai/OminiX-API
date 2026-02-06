use std::time::Duration;

use salvo::prelude::*;

use crate::inference::InferenceRequest;
use crate::types::ChatCompletionRequest;

use super::helpers::{get_state, send_and_wait};

/// Timeout for LLM chat completions (can be long for large responses)
const CHAT_TIMEOUT: Duration = Duration::from_secs(300); // 5 minutes

/// POST /v1/chat/completions - OpenAI-compatible chat completions
#[handler]
pub async fn chat_completions(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    let request: ChatCompletionRequest = req.parse_json().await.map_err(|e| {
        tracing::error!("Failed to parse request: {}", e);
        StatusError::bad_request()
    })?;

    let is_streaming = request.stream.unwrap_or(false);

    let response = send_and_wait(
        &state.inference_tx,
        |tx| InferenceRequest::Chat { request, response_tx: tx },
        CHAT_TIMEOUT,
    )
    .await?;

    if is_streaming {
        // Convert non-streaming response to SSE format for clients that expect streaming
        let content = response.choices.first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();
        let finish_reason = response.choices.first()
            .map(|c| c.finish_reason.clone())
            .unwrap_or_else(|| "stop".to_string());

        let chunk1 = serde_json::json!({
            "id": response.id,
            "object": "chat.completion.chunk",
            "created": response.created,
            "model": response.model,
            "choices": [{
                "index": 0,
                "delta": { "role": "assistant", "content": content },
                "finish_reason": serde_json::Value::Null
            }]
        });
        let chunk2 = serde_json::json!({
            "id": response.id,
            "object": "chat.completion.chunk",
            "created": response.created,
            "model": response.model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        });

        let chunk1_str = serde_json::to_string(&chunk1).unwrap_or_default();
        let chunk2_str = serde_json::to_string(&chunk2).unwrap_or_default();
        let body = format!("data: {}\n\ndata: {}\n\ndata: [DONE]\n\n", chunk1_str, chunk2_str);

        res.headers_mut().insert(
            salvo::http::header::CONTENT_TYPE,
            "text/event-stream; charset=utf-8".parse().unwrap(),
        );
        res.headers_mut().insert(
            salvo::http::header::CACHE_CONTROL,
            "no-cache".parse().unwrap(),
        );
        res.write_body(body).ok();
    } else {
        res.render(Json(response));
    }
    Ok(())
}
