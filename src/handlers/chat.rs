use std::time::Duration;

use salvo::prelude::*;

use crate::engines::ascend;
use crate::inference::InferenceRequest;
use crate::types::{ChatCompletionRequest, MessageContent, VlmCompletionRequest};

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

    let request: ChatCompletionRequest = req
        .parse_json_with_max_size(20 * 1024 * 1024)
        .await
        .map_err(|e| {
            tracing::error!("Failed to parse request: {}", e);
            StatusError::bad_request()
        })?;

    let is_streaming = request.stream.unwrap_or(false);

    // Detect multimodal (image) content → route to VLM engine
    let image_b64 = request.messages.iter().rev().find_map(|m| {
        m.content.as_ref().and_then(|c| c.image_base64())
    });
    if let Some(image) = image_b64 {
        let prompt = request.messages.iter().rev().find_map(|m| {
            m.content.as_ref().map(|c| c.as_text().to_string()).filter(|t| !t.is_empty())
        }).unwrap_or_default();

        let vlm_req = VlmCompletionRequest {
            model: request.model.clone(),
            image,
            prompt,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
        };

        let vlm_response = send_and_wait(
            &state.inference_tx,
            |tx| InferenceRequest::VlmCompletion { request: vlm_req, response_tx: tx },
            CHAT_TIMEOUT,
        )
        .await?;

        // Wrap VLM response in OpenAI chat completions format
        let chat_response = serde_json::json!({
            "id": vlm_response.id,
            "object": "chat.completion",
            "created": vlm_response.created,
            "model": vlm_response.model,
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": vlm_response.content },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": vlm_response.usage.prompt_tokens,
                "completion_tokens": vlm_response.usage.completion_tokens,
                "total_tokens": vlm_response.usage.total_tokens
            }
        });

        if is_streaming {
            let content = vlm_response.content;
            let sse = format!(
                "data: {}\n\ndata: [DONE]\n\n",
                serde_json::json!({
                    "id": vlm_response.id,
                    "object": "chat.completion.chunk",
                    "created": vlm_response.created,
                    "model": vlm_response.model,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": "stop"}]
                })
            );
            res.headers_mut().insert("Content-Type", "text/event-stream".parse().unwrap());
            res.write_body(sse).ok();
        } else {
            res.render(Json(chat_response));
        }
        return Ok(());
    }

    let response = send_and_wait(
        &state.inference_tx,
        |tx| InferenceRequest::Chat { request, response_tx: tx },
        CHAT_TIMEOUT,
    )
    .await?;

    if is_streaming {
        // Convert non-streaming response to SSE format for clients that expect streaming
        let content = response.choices.first()
            .and_then(|c| c.message.content.as_ref().map(|mc| mc.as_text().to_string()))
            .unwrap_or_default();
        let tool_calls = response.choices.first()
            .and_then(|c| c.message.tool_calls.clone());
        let finish_reason = response.choices.first()
            .map(|c| c.finish_reason.clone())
            .unwrap_or_else(|| "stop".to_string());

        // Build delta with role, optional content, and optional tool_calls
        let mut delta = serde_json::json!({ "role": "assistant" });
        if !content.is_empty() {
            delta["content"] = serde_json::json!(content);
        }
        if let Some(tc) = &tool_calls {
            // Add index to each tool call for OpenAI streaming compatibility
            let indexed_tc: Vec<serde_json::Value> = tc.iter().enumerate().map(|(i, call)| {
                let mut v = serde_json::to_value(call).unwrap_or_default();
                if let Some(obj) = v.as_object_mut() {
                    obj.insert("index".to_string(), serde_json::json!(i));
                }
                v
            }).collect();
            delta["tool_calls"] = serde_json::json!(indexed_tc);
        }

        let chunk1 = serde_json::json!({
            "id": response.id,
            "object": "chat.completion.chunk",
            "created": response.created,
            "model": response.model,
            "choices": [{
                "index": 0,
                "delta": delta,
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

/// POST /v1/chat/completions/ascend — LLM chat on Ascend NPU
///
/// Proxies to a llama-server instance running on the Ascend NPU.
/// Supports the full OpenAI chat completions API (100+ model architectures).
#[handler]
pub async fn chat_completions_ascend(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = get_state(depot)?;

    let ascend_cfg = state.ascend_config.as_ref().ok_or_else(|| {
        tracing::error!("Ascend backend not configured");
        StatusError::internal_server_error()
    })?;

    if !ascend_cfg.has_llm() {
        crate::error::render_error(
            res,
            salvo::http::StatusCode::SERVICE_UNAVAILABLE,
            "Ascend LLM not available. Set ASCEND_LLM_MODEL.",
            "unavailable",
        );
        return Ok(());
    }

    let request: ChatCompletionRequest = req.parse_json().await.map_err(|e| {
        tracing::error!("Failed to parse request: {}", e);
        StatusError::bad_request()
    })?;

    let cfg = ascend_cfg.clone();
    let response = tokio::task::spawn_blocking(move || {
        let server = ascend::AscendLlmServer::new((*cfg).clone())?;
        server.chat_completions(&request)
    })
    .await
    .map_err(|e| {
        tracing::error!("Ascend LLM task failed: {}", e);
        StatusError::internal_server_error()
    })?
    .map_err(|e| {
        tracing::error!("Ascend LLM error: {}", e);
        StatusError::internal_server_error()
    })?;

    res.render(Json(response));
    Ok(())
}
