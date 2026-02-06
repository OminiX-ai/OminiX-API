use std::time::Duration;

use salvo::prelude::*;
use salvo::websocket::{Message, WebSocket, WebSocketUpgrade};
use tokio::sync::oneshot;
use tokio::time::timeout;

use crate::inference::InferenceRequest;
use crate::state::AppState;
use crate::types::SpeechRequest;

use super::helpers::get_state;

/// Timeout for text-to-speech
const TTS_TIMEOUT: Duration = Duration::from_secs(60); // 1 minute

/// GET /ws/v1/tts - WebSocket streaming TTS
///
/// Protocol (MiniMax T2A compatible):
/// 1. Connect -> Server sends {"event": "connected_success"}
/// 2. Client sends {"event": "task_start", "voice_setting": {...}, "audio_setting": {...}}
///    -> Server sends {"event": "task_started"}
/// 3. Client sends {"event": "task_continue", "text": "..."}
///    -> Server streams {"event": "task_progress", "data": {"audio": "<hex>"}, "is_final": bool}
/// 4. Client sends {"event": "task_finish"} -> connection closes
#[handler]
pub async fn ws_tts(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), StatusError> {
    let state = get_state(depot)?.clone();

    WebSocketUpgrade::new()
        .upgrade(req, res, |ws| handle_tts_websocket(ws, state))
        .await
}

async fn handle_tts_websocket(mut ws: WebSocket, state: AppState) {
    // Send connection success
    let msg = serde_json::json!({"event": "connected_success"});
    if ws.send(Message::text(msg.to_string())).await.is_err() {
        return;
    }

    // Session state from task_start
    let mut voice: Option<String> = None;
    let mut speed: f32 = 1.0;
    let mut audio_format = "wav".to_string();
    const CHUNK_SIZE: usize = 8192;

    while let Some(msg) = ws.recv().await {
        let msg = match msg {
            Ok(m) => m,
            Err(_) => break,
        };

        if msg.is_close() {
            break;
        }

        let text = match msg.as_str() {
            Ok(t) => t.to_string(),
            Err(_) => continue, // skip non-text (ping/pong/binary)
        };

        let event: serde_json::Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(_) => {
                let err = serde_json::json!({"event": "error", "message": "Invalid JSON"});
                let _ = ws.send(Message::text(err.to_string())).await;
                continue;
            }
        };

        match event.get("event").and_then(|e| e.as_str()) {
            Some("task_start") => {
                if let Some(vs) = event.get("voice_setting") {
                    voice = vs
                        .get("voice_id")
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    speed = vs
                        .get("speed")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(1.0) as f32;
                }
                if let Some(audio_s) = event.get("audio_setting") {
                    audio_format = audio_s
                        .get("format")
                        .and_then(|v| v.as_str())
                        .unwrap_or("wav")
                        .to_string();
                }

                let resp = serde_json::json!({"event": "task_started"});
                if ws.send(Message::text(resp.to_string())).await.is_err() {
                    break;
                }
            }

            Some("task_continue") => {
                let input_text = match event.get("text").and_then(|t| t.as_str()) {
                    Some(t) => t.to_string(),
                    None => {
                        let err = serde_json::json!({"event": "error", "message": "Missing 'text' field"});
                        let _ = ws.send(Message::text(err.to_string())).await;
                        continue;
                    }
                };

                // Per-message voice_id override, falls back to task_start voice
                let msg_voice = event
                    .get("voice_id")
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .or_else(|| voice.clone());

                let speech_req = SpeechRequest {
                    model: None,
                    input: input_text,
                    voice: msg_voice,
                    response_format: audio_format.clone(),
                    speed,
                };

                let (response_tx, response_rx) = oneshot::channel();
                if state
                    .inference_tx
                    .send(InferenceRequest::Speech {
                        request: speech_req,
                        response_tx,
                    })
                    .await
                    .is_err()
                {
                    let err = serde_json::json!({"event": "error", "message": "Inference unavailable"});
                    let _ = ws.send(Message::text(err.to_string())).await;
                    break;
                }

                match timeout(TTS_TIMEOUT, response_rx).await {
                    Ok(Ok(Ok(audio_data))) => {
                        if audio_data.is_empty() {
                            let resp = serde_json::json!({
                                "event": "task_progress",
                                "data": {"audio": ""},
                                "is_final": true
                            });
                            let _ = ws.send(Message::text(resp.to_string())).await;
                        } else {
                            let total = audio_data.len();
                            let mut sent = 0;

                            for chunk in audio_data.chunks(CHUNK_SIZE) {
                                sent += chunk.len();
                                let is_final = sent >= total;
                                let hex_audio = hex::encode(chunk);

                                let resp = serde_json::json!({
                                    "event": "task_progress",
                                    "data": {"audio": hex_audio},
                                    "is_final": is_final
                                });

                                if ws.send(Message::text(resp.to_string())).await.is_err() {
                                    return;
                                }
                            }
                        }
                    }
                    Ok(Ok(Err(e))) => {
                        let err = serde_json::json!({
                            "event": "error",
                            "message": format!("TTS error: {}", e)
                        });
                        let _ = ws.send(Message::text(err.to_string())).await;
                    }
                    Ok(Err(_)) => {
                        let err = serde_json::json!({
                            "event": "error",
                            "message": "Inference channel dropped"
                        });
                        let _ = ws.send(Message::text(err.to_string())).await;
                    }
                    Err(_) => {
                        let err = serde_json::json!({
                            "event": "error",
                            "message": format!("TTS timed out after {:?}", TTS_TIMEOUT)
                        });
                        let _ = ws.send(Message::text(err.to_string())).await;
                    }
                }
            }

            Some("task_finish") => {
                break;
            }

            Some(unknown) => {
                let err = serde_json::json!({
                    "event": "error",
                    "message": format!("Unknown event: {}", unknown)
                });
                let _ = ws.send(Message::text(err.to_string())).await;
            }

            None => {
                let err = serde_json::json!({
                    "event": "error",
                    "message": "Missing 'event' field"
                });
                let _ = ws.send(Message::text(err.to_string())).await;
            }
        }
    }

    let _ = ws.close().await;
}
