//! Explicit model routing to an authenticated OminiX-SGLang worker-v0 shim.
//!
//! This is deliberately configuration-driven. CUDA availability is not a
//! sufficient routing signal: the same model name can exist in local MLX and
//! remote CUDA formats, and silently changing providers would make failures and
//! output drift very hard to diagnose.

use std::collections::BTreeSet;
use std::io;
use std::sync::Arc;
use std::time::Duration;

use reqwest::header::{HeaderValue, ACCEPT, AUTHORIZATION, CONTENT_TYPE};
use reqwest::redirect::Policy;
use reqwest::{Client, Response as UpstreamResponse, Url};
use salvo::hyper::body::Bytes;
use salvo::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use tokio_stream::StreamExt;
use uuid::Uuid;

use crate::types::{ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage};
use crate::types::{ChatUsage, MessageContent};

const PROTOCOL_VERSION: &str = "ominix.worker.v0";
const GENERATE_MESSAGE_TYPE: &str = "GenerateRequest";
const EVENT_MESSAGE_TYPE: &str = "WorkerEvent";
const SHIM_BACKEND: &str = "sglang-ominix-v0-http-shim";
const DEFAULT_TIMEOUT_SECS: u64 = 1_800;
const DEFAULT_MAX_TOKENS: usize = 2_048;
const PROBE_TIMEOUT: Duration = Duration::from_secs(3);
const MAX_SSE_FRAME_BYTES: usize = 2 * 1024 * 1024;
const MAX_CONTROL_BODY_BYTES: usize = 64 * 1024;
const MAX_NON_STREAM_BODY_BYTES: usize = 64 * 1024 * 1024;

const URL_ENV: &str = "OMINIX_V0_SCHEDULER_URL";
const TOKEN_ENV: &str = "OMINIX_V0_SCHEDULER_TOKEN";
const MODELS_ENV: &str = "OMINIX_V0_SCHEDULER_MODELS";
const SERVED_MODEL_ENV: &str = "OMINIX_V0_SERVED_MODEL";
const TIMEOUT_ENV: &str = "OMINIX_V0_SCHEDULER_TIMEOUT_SECS";
const CHAT_TEMPLATE_KWARGS_ENV: &str = "OMINIX_V0_CHAT_TEMPLATE_KWARGS_JSON";

#[derive(Clone)]
pub struct SglangRouter {
    client: Client,
    generate_url: Url,
    server_info_url: Url,
    model_info_url: Url,
    abort_url: Url,
    models: Arc<BTreeSet<String>>,
    served_model: Arc<str>,
    authorization: HeaderValue,
    chat_template_kwargs: Arc<Map<String, Value>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SglangBackendStatus {
    pub configured: bool,
    pub models: Vec<String>,
    pub served_model: String,
    pub shim_reachable: bool,
    pub model_ready: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum SglangError {
    #[error("invalid OminiX-SGLang configuration: {0}")]
    Config(String),
    #[error("the mapped OminiX-SGLang route does not support {0}")]
    UnsupportedRequest(&'static str),
    #[error("OminiX-SGLang transport failed: {0}")]
    Transport(#[from] reqwest::Error),
    #[error("OminiX-SGLang returned HTTP {status}: {message}")]
    UpstreamStatus { status: u16, message: String },
    #[error("invalid OminiX worker-v0 stream: {0}")]
    Protocol(String),
    #[error("OminiX-SGLang generation failed: {0}")]
    RemoteEvent(String),
}

impl SglangError {
    pub fn is_client_error(&self) -> bool {
        matches!(self, Self::UnsupportedRequest(_))
            || matches!(
                self,
                Self::UpstreamStatus {
                    status: 400 | 422,
                    ..
                }
            )
    }

    pub fn is_timeout(&self) -> bool {
        matches!(self, Self::Transport(error) if error.is_timeout())
            || matches!(
                self,
                Self::UpstreamStatus {
                    status: 408 | 504,
                    ..
                }
            )
    }

    pub fn is_unavailable(&self) -> bool {
        matches!(
            self,
            Self::UpstreamStatus {
                status: 401 | 403 | 429 | 503,
                ..
            }
        )
    }

    pub fn log_class(&self) -> &'static str {
        match self {
            Self::Config(_) => "configuration",
            Self::UnsupportedRequest(_) => "unsupported_request",
            Self::Transport(error) if error.is_timeout() => "timeout",
            Self::Transport(_) => "transport",
            Self::UpstreamStatus { .. } => "upstream_status",
            Self::Protocol(_) => "protocol",
            Self::RemoteEvent(_) => "remote_event",
        }
    }

    pub fn public_message(&self) -> &'static str {
        if self.is_client_error() {
            "The selected CUDA model does not support this request"
        } else if self.is_timeout() {
            "The CUDA inference backend timed out"
        } else {
            "The CUDA inference backend failed"
        }
    }
}

impl SglangRouter {
    /// Build the optional router from environment variables.
    ///
    /// The route is disabled only when all route variables are absent. Partial
    /// configuration is a startup error so a mapped model can never silently
    /// fall back to a different local backend.
    pub fn from_env() -> Result<Option<Self>, SglangError> {
        let names = [
            URL_ENV,
            TOKEN_ENV,
            MODELS_ENV,
            SERVED_MODEL_ENV,
            TIMEOUT_ENV,
            CHAT_TEMPLATE_KWARGS_ENV,
        ];
        if !names.iter().any(|name| std::env::var_os(name).is_some()) {
            return Ok(None);
        }

        let url = strict_env(URL_ENV)?;
        let token = strict_env(TOKEN_ENV)?;
        let models = strict_env(MODELS_ENV)?;
        let served_model = strict_env(SERVED_MODEL_ENV)?;
        let timeout = strict_env(TIMEOUT_ENV)?;
        let chat_template_kwargs = strict_env(CHAT_TEMPLATE_KWARGS_ENV)?;

        Self::from_values(
            url.as_deref(),
            models.as_deref(),
            served_model.as_deref(),
            token.as_deref(),
            timeout.as_deref(),
            chat_template_kwargs.as_deref(),
        )
        .map(Some)
    }

    fn from_values(
        base_url: Option<&str>,
        models: Option<&str>,
        served_model: Option<&str>,
        token: Option<&str>,
        timeout_secs: Option<&str>,
        chat_template_kwargs: Option<&str>,
    ) -> Result<Self, SglangError> {
        let base_url = required(base_url, URL_ENV)?;
        let token = required(token, TOKEN_ENV)?;
        let models = parse_models(required(models, MODELS_ENV)?)?;
        let served_model = required(served_model, SERVED_MODEL_ENV)?;
        if served_model.contains(',') {
            return Err(SglangError::Config(format!(
                "{SERVED_MODEL_ENV} must contain one exact served model id"
            )));
        }
        let timeout = parse_timeout(timeout_secs)?;
        let chat_template_kwargs = parse_chat_template_kwargs(chat_template_kwargs)?;
        let base_url = parse_base_url(base_url)?;

        let client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(timeout)
            .no_proxy()
            .redirect(Policy::none())
            .build()
            .map_err(SglangError::Transport)?;
        let mut authorization = HeaderValue::from_str(&format!("Bearer {token}"))
            .map_err(|_| SglangError::Config(format!("{TOKEN_ENV} is not a valid bearer token")))?;
        authorization.set_sensitive(true);

        Ok(Self {
            client,
            generate_url: base_url
                .join("generate")
                .map_err(|error| SglangError::Config(error.to_string()))?,
            server_info_url: base_url
                .join("server_info")
                .map_err(|error| SglangError::Config(error.to_string()))?,
            model_info_url: base_url
                .join("get_model_info")
                .map_err(|error| SglangError::Config(error.to_string()))?,
            abort_url: base_url
                .join("abort_request")
                .map_err(|error| SglangError::Config(error.to_string()))?,
            models: Arc::new(models),
            served_model: Arc::from(served_model),
            authorization,
            chat_template_kwargs: Arc::new(chat_template_kwargs),
        })
    }

    pub fn routes_model(&self, model: &str) -> bool {
        self.models.contains(model)
    }

    pub fn model_ids(&self) -> Vec<String> {
        self.models.iter().cloned().collect()
    }

    pub fn served_model(&self) -> &str {
        &self.served_model
    }

    /// Probe shim liveness and model readiness separately. `/server_info`
    /// proves that this is a worker-v0 gRPC shim; `/get_model_info` must then
    /// traverse the scheduler and return the exact configured served model.
    /// Neither endpoint is used for service discovery.
    pub async fn status(&self) -> SglangBackendStatus {
        let (shim_reachable, production_shim) =
            tokio::time::timeout(PROBE_TIMEOUT, self.probe_server_info())
                .await
                .unwrap_or((false, false));

        let model_ready = if production_shim {
            tokio::time::timeout(PROBE_TIMEOUT, self.probe_model_identity())
                .await
                .unwrap_or(false)
        } else {
            false
        };

        SglangBackendStatus {
            configured: true,
            models: self.model_ids(),
            served_model: self.served_model.to_string(),
            shim_reachable,
            model_ready,
        }
    }

    pub async fn render_chat_completion(
        &self,
        request: &ChatCompletionRequest,
        res: &mut salvo::Response,
    ) -> Result<(), SglangError> {
        self.validate_chat_request(request)?;
        let backend_ready = self.probe_backend_identity().await;
        if !backend_ready {
            return Err(SglangError::UpstreamStatus {
                status: 503,
                message: "configured worker identity is not ready".to_string(),
            });
        }
        let request_id = format!("ominix-{}", Uuid::new_v4().simple());
        let mut abort_guard = AbortGuard::new(self, request_id.clone());
        let upstream = self.send_generate(request, &request_id).await?;

        if request.stream.unwrap_or(false) {
            self.render_stream(
                upstream,
                request.model.clone(),
                request_id,
                abort_guard,
                res,
            );
        } else {
            let body = read_bounded_response(upstream, MAX_NON_STREAM_BODY_BYTES).await?;
            let accumulator = parse_worker_sse(&body, &request_id)?;
            abort_guard.disarm();
            res.render(Json(
                accumulator.into_chat_response(request.model.clone(), request_id),
            ));
        }
        Ok(())
    }

    fn validate_chat_request(&self, request: &ChatCompletionRequest) -> Result<(), SglangError> {
        if request.max_tokens == Some(0) {
            return Err(SglangError::UnsupportedRequest(
                "max_tokens must be greater than zero",
            ));
        }
        if request
            .tools
            .as_ref()
            .is_some_and(|tools| !tools.is_empty())
            || request
                .tool_choice
                .as_ref()
                .is_some_and(|choice| choice.as_str() != Some("none"))
            || request.messages.iter().any(|message| {
                message.role == "tool"
                    || message
                        .tool_calls
                        .as_ref()
                        .is_some_and(|calls| !calls.is_empty())
                    || message.tool_call_id.is_some()
            })
        {
            return Err(SglangError::UnsupportedRequest("tool calling"));
        }
        if request
            .messages
            .iter()
            .any(|message| matches!(message.content.as_ref(), Some(MessageContent::Parts(_))))
        {
            return Err(SglangError::UnsupportedRequest(
                "multipart or multimodal chat content",
            ));
        }
        Ok(())
    }

    async fn send_generate(
        &self,
        request: &ChatCompletionRequest,
        request_id: &str,
    ) -> Result<UpstreamResponse, SglangError> {
        let payload = self.worker_request(request, request_id);
        let response = self
            .client
            .post(self.generate_url.clone())
            .header(AUTHORIZATION, self.authorization.clone())
            .header(ACCEPT, "text/event-stream")
            .json(&payload)
            .send()
            .await
            .map_err(SglangError::Transport)?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let raw = read_bounded_response(response, MAX_CONTROL_BODY_BYTES)
                .await
                .unwrap_or_default();
            return Err(SglangError::UpstreamStatus {
                status,
                message: upstream_error_message(&String::from_utf8_lossy(&raw)),
            });
        }

        let content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default();
        if !content_type
            .to_ascii_lowercase()
            .starts_with("text/event-stream")
        {
            return Err(SglangError::Protocol(
                "worker returned a non-SSE success response".to_string(),
            ));
        }
        Ok(response)
    }

    fn worker_request(&self, request: &ChatCompletionRequest, request_id: &str) -> Value {
        let mut sampling = Map::new();
        sampling.insert(
            "max_new_tokens".to_string(),
            json!(request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS)),
        );
        if let Some(temperature) = request.temperature {
            sampling.insert("temperature".to_string(), json!(temperature));
        }
        if let Some(top_p) = request.top_p {
            sampling.insert("top_p".to_string(), json!(top_p));
        }

        json!({
            "protocol_version": PROTOCOL_VERSION,
            "message_type": GENERATE_MESSAGE_TYPE,
            "request_id": request_id,
            "model": &*self.served_model,
            "input": {
                "kind": "chat",
                "messages": request.messages,
                "chat_template_kwargs": &*self.chat_template_kwargs,
            },
            "sampling": sampling,
            "response_options": {
                "skip_special_tokens": true,
                "spaces_between_special_tokens": true,
            },
            "stream": request.stream.unwrap_or(false),
        })
    }

    async fn authorized_get(&self, url: Url) -> Result<UpstreamResponse, reqwest::Error> {
        self.client
            .get(url)
            .header(AUTHORIZATION, self.authorization.clone())
            .send()
            .await
    }

    async fn probe_server_info(&self) -> (bool, bool) {
        let Ok(response) = self.authorized_get(self.server_info_url.clone()).await else {
            return (false, false);
        };
        if !response.status().is_success() {
            return (true, false);
        }
        let Ok(body) = read_bounded_response(response, MAX_CONTROL_BODY_BYTES).await else {
            return (true, false);
        };
        let Ok(value) = serde_json::from_slice::<Value>(&body) else {
            return (true, false);
        };
        let valid = value.get("scheduler_backend").and_then(Value::as_str) == Some(SHIM_BACKEND)
            && value.get("protocol_version").and_then(Value::as_str) == Some(PROTOCOL_VERSION)
            && value.get("mode").and_then(Value::as_str) == Some("grpc")
            && value.get("public_openai_api").and_then(Value::as_bool) == Some(false)
            && value.get("auth_required").and_then(Value::as_bool) == Some(true)
            && value.pointer("/routes/generate").and_then(Value::as_str) == Some("/generate");
        (true, valid)
    }

    async fn probe_model_identity(&self) -> bool {
        let Ok(response) = self.authorized_get(self.model_info_url.clone()).await else {
            return false;
        };
        if !response.status().is_success() {
            return false;
        }
        let Ok(body) = read_bounded_response(response, MAX_CONTROL_BODY_BYTES).await else {
            return false;
        };
        let Ok(value) = serde_json::from_slice::<Value>(&body) else {
            return false;
        };
        value.get("served_model_name").and_then(Value::as_str) == Some(&*self.served_model)
            && value.get("is_generation").and_then(Value::as_bool) == Some(true)
    }

    async fn probe_backend_identity(&self) -> bool {
        let (_, production_shim) = tokio::time::timeout(PROBE_TIMEOUT, self.probe_server_info())
            .await
            .unwrap_or((false, false));
        production_shim
            && tokio::time::timeout(PROBE_TIMEOUT, self.probe_model_identity())
                .await
                .unwrap_or(false)
    }

    fn render_stream(
        &self,
        upstream: UpstreamResponse,
        model: String,
        request_id: String,
        mut abort_guard: AbortGuard,
        res: &mut salvo::Response,
    ) {
        let mut upstream = upstream.bytes_stream();
        let stream = async_stream::stream! {
            let mut decoder = SseDecoder::default();
            let mut adapter = OpenAiStreamAdapter::new(model, request_id);

            yield Ok::<Bytes, io::Error>(Bytes::from(adapter.initial_chunk()));

            while let Some(chunk) = upstream.next().await {
                let chunk = match chunk {
                    Ok(chunk) => chunk,
                    Err(_) => {
                        tracing::error!(error_class = "transport", "OminiX-SGLang stream failed");
                        yield Ok(Bytes::from(openai_stream_error("CUDA inference stream failed")));
                        yield Ok(Bytes::from_static(b"data: [DONE]\n\n"));
                        return;
                    }
                };

                let frames = match decoder.push(&chunk) {
                    Ok(frames) => frames,
                    Err(_) => {
                        tracing::error!(error_class = "sse_decode", "OminiX-SGLang stream failed");
                        yield Ok(Bytes::from(openai_stream_error("CUDA inference stream was invalid")));
                        yield Ok(Bytes::from_static(b"data: [DONE]\n\n"));
                        return;
                    }
                };

                for frame in frames {
                    match adapter.consume(&frame) {
                        Ok(chunks) => {
                            let transport_done = adapter.transport_done;
                            if transport_done {
                                abort_guard.disarm();
                            }
                            for chunk in chunks {
                                yield Ok(Bytes::from(chunk));
                            }
                            if transport_done {
                                return;
                            }
                        }
                        Err(error) => {
                            tracing::error!(
                                error_class = error.log_class(),
                                "OminiX-SGLang worker event failed"
                            );
                            yield Ok(Bytes::from(openai_stream_error(error.public_message())));
                            yield Ok(Bytes::from_static(b"data: [DONE]\n\n"));
                            return;
                        }
                    }
                }
            }

            if decoder.finish().is_err() {
                tracing::error!(
                    error_class = "incomplete_sse",
                    "OminiX-SGLang stream ended with an incomplete frame"
                );
            }
            if !adapter.transport_done {
                tracing::error!("OminiX-SGLang stream ended before data: [DONE]");
                yield Ok(Bytes::from(openai_stream_error(
                    "CUDA inference stream ended prematurely",
                )));
                yield Ok(Bytes::from_static(b"data: [DONE]\n\n"));
            }
        };

        res.headers_mut().insert(
            salvo::http::header::CONTENT_TYPE,
            "text/event-stream; charset=utf-8".parse().unwrap(),
        );
        res.headers_mut().insert(
            salvo::http::header::CACHE_CONTROL,
            "no-cache".parse().unwrap(),
        );
        res.stream(stream);
    }
}

/// Best-effort cancellation for work already admitted by the scheduler. The
/// guard is moved into the public SSE body, so dropping a disconnected client
/// also sends exactly one bounded abort request. A clean worker `[DONE]`
/// disarms it before the public terminal event is yielded.
struct AbortGuard {
    client: Client,
    url: Url,
    authorization: HeaderValue,
    request_id: Option<String>,
}

impl AbortGuard {
    fn new(router: &SglangRouter, request_id: String) -> Self {
        Self {
            client: router.client.clone(),
            url: router.abort_url.clone(),
            authorization: router.authorization.clone(),
            request_id: Some(request_id),
        }
    }

    fn disarm(&mut self) {
        self.request_id = None;
    }
}

impl Drop for AbortGuard {
    fn drop(&mut self) {
        let Some(request_id) = self.request_id.take() else {
            return;
        };
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let client = self.client.clone();
        let url = self.url.clone();
        let authorization = self.authorization.clone();
        runtime.spawn(async move {
            let request = client
                .post(url)
                .header(AUTHORIZATION, authorization)
                .json(&json!({
                    "protocol_version": PROTOCOL_VERSION,
                    "message_type": "AbortRequest",
                    "request_id": request_id,
                    "reason": "client_disconnect_or_upstream_failure",
                }))
                .send();
            if tokio::time::timeout(Duration::from_secs(2), request)
                .await
                .is_err()
            {
                tracing::warn!("Timed out aborting disconnected OminiX-SGLang request");
            }
        });
    }
}

async fn read_bounded_response(
    response: UpstreamResponse,
    limit: usize,
) -> Result<Vec<u8>, SglangError> {
    let mut stream = response.bytes_stream();
    let mut body = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(SglangError::Transport)?;
        if body.len().saturating_add(chunk.len()) > limit {
            return Err(SglangError::Protocol(
                "worker response exceeded the configured safety limit".to_string(),
            ));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn strict_env(name: &str) -> Result<Option<String>, SglangError> {
    match std::env::var(name) {
        Ok(value) => {
            let value = value.trim().to_string();
            if value.is_empty() {
                Err(SglangError::Config(format!(
                    "{name} must not be empty when present"
                )))
            } else {
                Ok(Some(value))
            }
        }
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(SglangError::Config(format!(
            "{name} must contain valid UTF-8"
        ))),
    }
}

fn required<'a>(value: Option<&'a str>, name: &str) -> Result<&'a str, SglangError> {
    value.ok_or_else(|| SglangError::Config(format!("{name} is required when routing is enabled")))
}

fn parse_models(raw: &str) -> Result<BTreeSet<String>, SglangError> {
    let models: BTreeSet<String> = raw
        .split(',')
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .map(ToOwned::to_owned)
        .collect();
    if models.is_empty() {
        return Err(SglangError::Config(format!(
            "{MODELS_ENV} must contain at least one exact model id"
        )));
    }
    if models.contains("*") {
        return Err(SglangError::Config(format!(
            "{MODELS_ENV} does not accept wildcards; list exact model ids"
        )));
    }
    Ok(models)
}

fn parse_timeout(raw: Option<&str>) -> Result<Duration, SglangError> {
    let seconds = match raw {
        Some(raw) => raw.parse::<u64>().map_err(|_| {
            SglangError::Config(format!("{TIMEOUT_ENV} must be a positive integer"))
        })?,
        None => DEFAULT_TIMEOUT_SECS,
    };
    if seconds == 0 || seconds > 86_400 {
        return Err(SglangError::Config(format!(
            "{TIMEOUT_ENV} must be between 1 and 86400 seconds"
        )));
    }
    Ok(Duration::from_secs(seconds))
}

fn parse_chat_template_kwargs(raw: Option<&str>) -> Result<Map<String, Value>, SglangError> {
    let Some(raw) = raw else {
        return Ok(Map::new());
    };
    serde_json::from_str::<Map<String, Value>>(raw).map_err(|error| {
        SglangError::Config(format!(
            "{CHAT_TEMPLATE_KWARGS_ENV} must be a JSON object: {error}"
        ))
    })
}

fn parse_base_url(raw: &str) -> Result<Url, SglangError> {
    let mut url = Url::parse(raw)
        .map_err(|error| SglangError::Config(format!("{URL_ENV} is invalid: {error}")))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(SglangError::Config(format!(
            "{URL_ENV} must use http or https"
        )));
    }
    if url.scheme() == "http" && !matches!(url.host_str(), Some("127.0.0.1" | "localhost" | "::1"))
    {
        return Err(SglangError::Config(format!(
            "{URL_ENV} may use plaintext HTTP only on loopback; use HTTPS for a remote worker"
        )));
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(SglangError::Config(format!(
            "{URL_ENV} must not embed credentials"
        )));
    }
    if url.query().is_some() || url.fragment().is_some() {
        return Err(SglangError::Config(format!(
            "{URL_ENV} must not contain a query or fragment"
        )));
    }
    if !matches!(url.path(), "" | "/") {
        return Err(SglangError::Config(format!(
            "{URL_ENV} must be the shim base URL without a path"
        )));
    }
    url.set_path("/");
    Ok(url)
}

fn upstream_error_message(raw: &str) -> String {
    let parsed = serde_json::from_str::<Value>(raw).ok();
    parsed
        .as_ref()
        .and_then(|value| value.pointer("/error/message"))
        .and_then(Value::as_str)
        .unwrap_or(raw)
        .chars()
        .take(512)
        .collect()
}

#[derive(Debug, Clone, Deserialize)]
struct WorkerEvent {
    protocol_version: String,
    message_type: String,
    request_id: String,
    kind: String,
    #[serde(rename = "created_at_ms")]
    _created_at_ms: u64,
    #[serde(default)]
    sequence_index: Option<usize>,
    #[serde(default)]
    text_delta: Option<String>,
    #[serde(default)]
    usage: Option<WorkerUsage>,
    #[serde(default)]
    finish_reason: Option<String>,
    #[serde(default)]
    error: Option<WorkerEventError>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct WorkerUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    #[serde(default)]
    total_tokens: u32,
}

impl From<WorkerUsage> for ChatUsage {
    fn from(usage: WorkerUsage) -> Self {
        Self {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
struct WorkerEventError {
    #[serde(default)]
    message: String,
}

fn parse_worker_event(data: &str, expected_request_id: &str) -> Result<WorkerEvent, SglangError> {
    let event: WorkerEvent = serde_json::from_str(data)
        .map_err(|error| SglangError::Protocol(format!("invalid WorkerEvent JSON: {error}")))?;
    if event.protocol_version != PROTOCOL_VERSION {
        return Err(SglangError::Protocol(format!(
            "unsupported protocol version {:?}",
            event.protocol_version
        )));
    }
    if event.message_type != EVENT_MESSAGE_TYPE {
        return Err(SglangError::Protocol(format!(
            "unexpected message type {:?}",
            event.message_type
        )));
    }
    if event.request_id != expected_request_id {
        return Err(SglangError::Protocol(
            "worker event request_id did not match the request".to_string(),
        ));
    }
    if event.sequence_index.unwrap_or(0) != 0 {
        return Err(SglangError::Protocol(
            "multiple output sequences are not supported by this endpoint".to_string(),
        ));
    }
    Ok(event)
}

fn validate_finish_reason(reason: Option<String>) -> Result<String, SglangError> {
    let reason = reason.ok_or_else(|| {
        SglangError::Protocol("worker done event omitted finish_reason".to_string())
    })?;
    match reason.as_str() {
        "stop" | "length" | "content_filter" => Ok(reason),
        "abort" => Err(SglangError::RemoteEvent(
            "worker aborted generation".to_string(),
        )),
        "error" => Err(SglangError::RemoteEvent(
            "worker terminated generation with an error".to_string(),
        )),
        "tool_calls" => Err(SglangError::Protocol(
            "worker returned tool_calls without structured tool-call deltas".to_string(),
        )),
        _ => Err(SglangError::Protocol(
            "worker done event contained an invalid finish_reason".to_string(),
        )),
    }
}

#[derive(Default)]
struct WorkerAccumulator {
    text: String,
    usage: WorkerUsage,
    finish_reason: Option<String>,
    saw_done: bool,
    transport_done: bool,
}

impl WorkerAccumulator {
    fn consume(&mut self, data: &str, request_id: &str) -> Result<(), SglangError> {
        if self.transport_done {
            return Err(SglangError::Protocol(
                "worker emitted data after transport completion".to_string(),
            ));
        }
        if data == "[DONE]" {
            if !self.saw_done {
                return Err(SglangError::Protocol(
                    "transport ended before the worker emitted done".to_string(),
                ));
            }
            self.transport_done = true;
            return Ok(());
        }
        if self.saw_done {
            return Err(SglangError::Protocol(
                "worker emitted an event after done".to_string(),
            ));
        }

        let event = parse_worker_event(data, request_id)?;
        match event.kind.as_str() {
            "prefill_done" | "usage" => {
                if let Some(usage) = event.usage {
                    self.usage = usage;
                }
            }
            "token" => {
                if let Some(text) = event.text_delta {
                    self.text.push_str(&text);
                }
            }
            "done" => {
                if let Some(text) = event.text_delta {
                    self.text.push_str(&text);
                }
                self.finish_reason = Some(validate_finish_reason(event.finish_reason)?);
                self.saw_done = true;
            }
            "error" => {
                return Err(SglangError::RemoteEvent(
                    event
                        .error
                        .map(|error| error.message)
                        .filter(|message| !message.is_empty())
                        .unwrap_or_else(|| "worker returned an error event".to_string()),
                ));
            }
            other => {
                return Err(SglangError::Protocol(format!(
                    "unknown WorkerEvent kind {other:?}"
                )));
            }
        }
        Ok(())
    }

    fn into_chat_response(self, model: String, request_id: String) -> ChatCompletionResponse {
        ChatCompletionResponse {
            id: format!("chatcmpl-{request_id}"),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: Some(MessageContent::Text(self.text)),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: self.finish_reason.unwrap_or_else(|| "stop".to_string()),
            }],
            usage: self.usage.into(),
        }
    }
}

fn parse_worker_sse(body: &[u8], request_id: &str) -> Result<WorkerAccumulator, SglangError> {
    let mut decoder = SseDecoder::default();
    let mut accumulator = WorkerAccumulator::default();
    for frame in decoder.push(body)? {
        accumulator.consume(&frame, request_id)?;
    }
    decoder.finish()?;
    if !accumulator.transport_done {
        return Err(SglangError::Protocol(
            "worker stream ended before data: [DONE]".to_string(),
        ));
    }
    Ok(accumulator)
}

struct OpenAiStreamAdapter {
    id: String,
    model: String,
    request_id: String,
    created: i64,
    usage: WorkerUsage,
    pending_finish_reason: Option<String>,
    saw_done: bool,
    transport_done: bool,
}

impl OpenAiStreamAdapter {
    fn new(model: String, request_id: String) -> Self {
        Self {
            id: format!("chatcmpl-{request_id}"),
            model,
            request_id,
            created: chrono::Utc::now().timestamp(),
            usage: WorkerUsage::default(),
            pending_finish_reason: None,
            saw_done: false,
            transport_done: false,
        }
    }

    fn initial_chunk(&self) -> String {
        openai_sse(&json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": Value::Null,
            }],
        }))
    }

    fn consume(&mut self, data: &str) -> Result<Vec<String>, SglangError> {
        if self.transport_done {
            return Err(SglangError::Protocol(
                "worker emitted data after transport completion".to_string(),
            ));
        }
        if data == "[DONE]" {
            if !self.saw_done {
                return Err(SglangError::Protocol(
                    "transport ended before the worker emitted done".to_string(),
                ));
            }
            self.transport_done = true;
            let finish_reason = self.pending_finish_reason.take().ok_or_else(|| {
                SglangError::Protocol("worker done event omitted finish_reason".to_string())
            })?;
            return Ok(vec![
                self.finish_chunk(finish_reason),
                "data: [DONE]\n\n".to_string(),
            ]);
        }
        if self.saw_done {
            return Err(SglangError::Protocol(
                "worker emitted an event after done".to_string(),
            ));
        }

        let event = parse_worker_event(data, &self.request_id)?;
        let mut output = Vec::new();
        match event.kind.as_str() {
            "prefill_done" | "usage" => {
                if let Some(usage) = event.usage {
                    self.usage = usage;
                }
            }
            "token" => {
                if let Some(text) = event.text_delta.filter(|text| !text.is_empty()) {
                    output.push(self.content_chunk(text));
                }
            }
            "done" => {
                if let Some(text) = event.text_delta.filter(|text| !text.is_empty()) {
                    output.push(self.content_chunk(text));
                }
                self.pending_finish_reason = Some(validate_finish_reason(event.finish_reason)?);
                self.saw_done = true;
            }
            "error" => {
                return Err(SglangError::RemoteEvent(
                    event
                        .error
                        .map(|error| error.message)
                        .filter(|message| !message.is_empty())
                        .unwrap_or_else(|| "worker returned an error event".to_string()),
                ));
            }
            other => {
                return Err(SglangError::Protocol(format!(
                    "unknown WorkerEvent kind {other:?}"
                )));
            }
        }
        Ok(output)
    }

    fn content_chunk(&self, text: String) -> String {
        openai_sse(&json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": {"content": text},
                "finish_reason": Value::Null,
            }],
        }))
    }

    fn finish_chunk(&self, finish_reason: String) -> String {
        openai_sse(&json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            },
        }))
    }
}

fn openai_sse(payload: &Value) -> String {
    format!(
        "data: {}\n\n",
        serde_json::to_string(payload).unwrap_or_default()
    )
}

fn openai_stream_error(message: &str) -> String {
    openai_sse(&json!({
        "error": {
            "message": message,
            "type": "upstream_error",
            "code": Value::Null,
        }
    }))
}

#[derive(Default)]
struct SseDecoder {
    buffer: Vec<u8>,
}

impl SseDecoder {
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<String>, SglangError> {
        self.buffer.extend_from_slice(bytes);
        let mut events = Vec::new();

        while let Some((index, delimiter_len)) = next_sse_boundary(&self.buffer) {
            if index > MAX_SSE_FRAME_BYTES {
                return Err(SglangError::Protocol(
                    "SSE frame exceeded the configured safety limit".to_string(),
                ));
            }
            let frame: Vec<u8> = self.buffer.drain(..index).collect();
            self.buffer.drain(..delimiter_len);
            if let Some(data) = parse_sse_frame(&frame)? {
                events.push(data);
            }
        }

        if self.buffer.len() > MAX_SSE_FRAME_BYTES {
            return Err(SglangError::Protocol(
                "SSE frame exceeded the configured safety limit".to_string(),
            ));
        }
        Ok(events)
    }

    fn finish(&self) -> Result<(), SglangError> {
        if self.buffer.iter().all(u8::is_ascii_whitespace) {
            Ok(())
        } else {
            Err(SglangError::Protocol(
                "incomplete SSE frame at end of response".to_string(),
            ))
        }
    }
}

fn next_sse_boundary(buffer: &[u8]) -> Option<(usize, usize)> {
    let lf = buffer.windows(2).position(|window| window == b"\n\n");
    let crlf = buffer.windows(4).position(|window| window == b"\r\n\r\n");
    match (lf, crlf) {
        (Some(left), Some(right)) if left <= right => Some((left, 2)),
        (Some(_), Some(right)) => Some((right, 4)),
        (Some(left), None) => Some((left, 2)),
        (None, Some(right)) => Some((right, 4)),
        (None, None) => None,
    }
}

fn parse_sse_frame(frame: &[u8]) -> Result<Option<String>, SglangError> {
    let frame = std::str::from_utf8(frame)
        .map_err(|_| SglangError::Protocol("SSE frame was not valid UTF-8".to_string()))?;
    let normalized = frame.replace("\r\n", "\n");
    let mut data = Vec::new();
    for line in normalized.lines() {
        if line.starts_with(':') {
            continue;
        }
        if let Some(value) = line.strip_prefix("data:") {
            data.push(value.strip_prefix(' ').unwrap_or(value));
        }
    }
    if data.is_empty() {
        Ok(None)
    } else {
        Ok(Some(data.join("\n")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChatMessage;
    use std::sync::Mutex;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;

    #[derive(Default)]
    struct MockCalls {
        generate_payloads: Vec<Value>,
        abort_request_ids: Vec<String>,
    }

    async fn spawn_mock_worker(
        mode: &'static str,
        served_model: &'static str,
        is_generation: bool,
    ) -> (
        String,
        Arc<Mutex<MockCalls>>,
        oneshot::Sender<()>,
        tokio::task::JoinHandle<()>,
    ) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let calls = Arc::new(Mutex::new(MockCalls::default()));
        let calls_for_server = calls.clone();
        let (shutdown_tx, mut shutdown_rx) = oneshot::channel();
        let task = tokio::spawn(async move {
            loop {
                let accepted = tokio::select! {
                    _ = &mut shutdown_rx => break,
                    accepted = listener.accept() => accepted,
                };
                let Ok((mut socket, _)) = accepted else {
                    break;
                };

                let mut request = Vec::new();
                let header_end = loop {
                    if let Some(index) = request.windows(4).position(|part| part == b"\r\n\r\n") {
                        break index + 4;
                    }
                    let mut chunk = [0_u8; 4096];
                    let read = socket.read(&mut chunk).await.unwrap();
                    if read == 0 {
                        break request.len();
                    }
                    request.extend_from_slice(&chunk[..read]);
                };
                let headers = String::from_utf8_lossy(&request[..header_end]).into_owned();
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        line.strip_prefix("Content-Length: ")
                            .or_else(|| line.strip_prefix("content-length: "))
                    })
                    .and_then(|value| value.trim().parse::<usize>().ok())
                    .unwrap_or(0);
                while request.len() < header_end + content_length {
                    let mut chunk = [0_u8; 4096];
                    let read = socket.read(&mut chunk).await.unwrap();
                    if read == 0 {
                        break;
                    }
                    request.extend_from_slice(&chunk[..read]);
                }
                let path = headers
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                    .unwrap_or("/");
                let authorized = headers
                    .lines()
                    .any(|line| line.eq_ignore_ascii_case("Authorization: Bearer test-token"));

                let (status, content_type, body) = if !authorized {
                    (
                        "401 Unauthorized",
                        "application/json",
                        json!({"error":{"message":"unauthorized"}}).to_string(),
                    )
                } else {
                    match path {
                        "/server_info" => (
                            "200 OK",
                            "application/json",
                            json!({
                                "scheduler_backend": SHIM_BACKEND,
                                "mode": mode,
                                "protocol_version": PROTOCOL_VERSION,
                                "routes": {"generate":"/generate"},
                                "public_openai_api": false,
                                "auth_required": true,
                            })
                            .to_string(),
                        ),
                        "/get_model_info" => (
                            "200 OK",
                            "application/json",
                            json!({
                                "served_model_name": served_model,
                                "is_generation": is_generation,
                            })
                            .to_string(),
                        ),
                        "/generate" => {
                            let payload: Value =
                                serde_json::from_slice(&request[header_end..]).unwrap();
                            let request_id = payload["request_id"].as_str().unwrap();
                            calls_for_server
                                .lock()
                                .unwrap()
                                .generate_payloads
                                .push(payload.clone());
                            let body = [
                                format!(
                                    "data: {}\n\n",
                                    event(request_id, "token", json!({"text_delta":"po"}))
                                ),
                                format!(
                                    "data: {}\n\n",
                                    event(
                                        request_id,
                                        "done",
                                        json!({"text_delta":"ng","finish_reason":"stop"})
                                    )
                                ),
                                "data: [DONE]\n\n".to_string(),
                            ]
                            .concat();
                            ("200 OK", "text/event-stream", body)
                        }
                        "/abort_request" => {
                            let payload: Value =
                                serde_json::from_slice(&request[header_end..]).unwrap();
                            calls_for_server
                                .lock()
                                .unwrap()
                                .abort_request_ids
                                .push(payload["request_id"].as_str().unwrap().to_string());
                            (
                                "200 OK",
                                "application/json",
                                json!({"success":true}).to_string(),
                            )
                        }
                        _ => (
                            "404 Not Found",
                            "application/json",
                            json!({"error":{"message":"not found"}}).to_string(),
                        ),
                    }
                };
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                socket.write_all(response.as_bytes()).await.unwrap();
            }
        });
        (format!("http://{address}"), calls, shutdown_tx, task)
    }

    fn router() -> SglangRouter {
        SglangRouter::from_values(
            Some("http://127.0.0.1:19091"),
            Some("C2Rust-FP8-DFlash,second-model"),
            Some("C2Rust-FP8-DFlash"),
            Some("not-a-real-token"),
            Some("30"),
            Some(r#"{"enable_thinking":false}"#),
        )
        .unwrap()
    }

    fn request(stream: bool) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "C2Rust-FP8-DFlash".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: Some(MessageContent::Text("translate this".to_string())),
                tool_calls: None,
                tool_call_id: None,
            }],
            temperature: Some(0.0),
            max_tokens: Some(64),
            stream: Some(stream),
            top_p: Some(0.9),
            tools: None,
            tool_choice: None,
        }
    }

    fn event(request_id: &str, kind: &str, extra: Value) -> String {
        let mut value = json!({
            "protocol_version": PROTOCOL_VERSION,
            "message_type": EVENT_MESSAGE_TYPE,
            "request_id": request_id,
            "kind": kind,
            "created_at_ms": 1,
            "sequence_index": 0,
        });
        value
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        serde_json::to_string(&value).unwrap()
    }

    #[test]
    fn route_requires_exact_model_ids() {
        let router = router();
        assert!(router.routes_model("C2Rust-FP8-DFlash"));
        assert!(router.routes_model("second-model"));
        assert!(!router.routes_model("c2rust-fp8-dflash"));
        assert!(!router.routes_model("C2Rust-FP8"));
    }

    #[test]
    fn partial_or_unsafe_configuration_is_rejected() {
        assert!(SglangRouter::from_values(
            None,
            Some("model"),
            Some("model"),
            Some("token"),
            None,
            None,
        )
        .is_err());
        assert!(SglangRouter::from_values(
            Some("http://127.0.0.1:19091"),
            Some("*"),
            Some("model"),
            Some("token"),
            None,
            None,
        )
        .is_err());
        assert!(SglangRouter::from_values(
            Some("http://user:secret@localhost:19091"),
            Some("model"),
            Some("model"),
            Some("token"),
            None,
            None,
        )
        .is_err());
        assert!(SglangRouter::from_values(
            Some("http://127.0.0.1:19091/generate"),
            Some("model"),
            Some("model"),
            Some("token"),
            None,
            None,
        )
        .is_err());

        let empty_env = "OMINIX_ROUTING_TEST_EMPTY_VALUE";
        std::env::set_var(empty_env, "   ");
        assert!(strict_env(empty_env).is_err());
        std::env::remove_var(empty_env);
    }

    #[test]
    fn unsupported_chat_features_are_rejected_explicitly() {
        let router = router();
        let mut invalid = request(false);
        invalid.max_tokens = Some(0);
        assert!(router.validate_chat_request(&invalid).is_err());

        let mut no_tools = request(false);
        no_tools.tool_choice = Some(Value::String("none".to_string()));
        assert!(router.validate_chat_request(&no_tools).is_ok());

        let mut tool_history = request(false);
        tool_history.messages[0].role = "tool".to_string();
        tool_history.messages[0].tool_call_id = Some("call-1".to_string());
        assert!(router.validate_chat_request(&tool_history).is_err());

        let mut content_parts = request(false);
        content_parts.messages[0].content =
            Some(serde_json::from_value(json!([{"type":"text","text":"translate this"}])).unwrap());
        assert!(router.validate_chat_request(&content_parts).is_err());
    }

    #[test]
    fn worker_request_uses_chat_envelope_and_sampling() {
        let router = router();
        let payload = router.worker_request(&request(false), "request-1");
        assert_eq!(payload["protocol_version"], PROTOCOL_VERSION);
        assert_eq!(payload["message_type"], GENERATE_MESSAGE_TYPE);
        assert_eq!(payload["request_id"], "request-1");
        assert_eq!(payload["model"], "C2Rust-FP8-DFlash");
        assert_eq!(payload["input"]["kind"], "chat");
        assert_eq!(payload["input"]["messages"][0]["content"], "translate this");
        assert_eq!(
            payload["input"]["chat_template_kwargs"]["enable_thinking"],
            false
        );
        assert_eq!(payload["sampling"]["max_new_tokens"], 64);
        assert_eq!(payload["sampling"]["temperature"], 0.0);
        let top_p = payload["sampling"]["top_p"].as_f64().unwrap();
        assert!((top_p - 0.9).abs() < f32::EPSILON as f64);

        let mut defaulted = request(false);
        defaulted.model = "second-model".to_string();
        defaulted.max_tokens = None;
        let payload = router.worker_request(&defaulted, "request-default");
        assert_eq!(payload["model"], "C2Rust-FP8-DFlash");
        assert_eq!(payload["sampling"]["max_new_tokens"], DEFAULT_MAX_TOKENS);
    }

    #[test]
    fn sse_decoder_handles_every_chunk_boundary() {
        let body = b"data: {\"one\":1}\r\n\r\ndata: {\"two\":2}\n\ndata: [DONE]\n\n";
        for split in 0..=body.len() {
            let mut decoder = SseDecoder::default();
            let mut frames = decoder.push(&body[..split]).unwrap();
            frames.extend(decoder.push(&body[split..]).unwrap());
            decoder.finish().unwrap();
            assert_eq!(frames, vec![r#"{"one":1}"#, r#"{"two":2}"#, "[DONE]"]);
        }
    }

    #[test]
    fn nonstream_worker_events_become_openai_response() {
        let request_id = "request-2";
        let body = [
            format!(
                "data: {}\n\n",
                event(request_id, "token", json!({"text_delta":"safe "}))
            ),
            format!(
                "data: {}\n\n",
                event(request_id, "token", json!({"text_delta":"Rus"}))
            ),
            format!(
                "data: {}\n\n",
                event(
                    request_id,
                    "usage",
                    json!({"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}})
                )
            ),
            format!(
                "data: {}\n\n",
                event(
                    request_id,
                    "done",
                    json!({"text_delta":"t","finish_reason":"stop"})
                )
            ),
            "data: [DONE]\n\n".to_string(),
        ]
        .concat();
        let response = parse_worker_sse(body.as_bytes(), request_id)
            .unwrap()
            .into_chat_response("C2Rust-FP8-DFlash".to_string(), request_id.to_string());
        assert_eq!(
            response.choices[0]
                .message
                .content
                .as_ref()
                .unwrap()
                .as_text(),
            "safe Rust"
        );
        assert_eq!(response.usage.total_tokens, 12);
        assert_eq!(response.choices[0].finish_reason, "stop");
    }

    #[test]
    fn incomplete_or_mismatched_stream_is_rejected() {
        let request_id = "request-3";
        let incomplete = format!(
            "data: {}\n\n",
            event(request_id, "done", json!({"finish_reason":"stop"}))
        );
        assert!(parse_worker_sse(incomplete.as_bytes(), request_id).is_err());

        let mismatch = format!(
            "data: {}\n\ndata: [DONE]\n\n",
            event("other-request", "done", json!({"finish_reason":"stop"}))
        );
        assert!(parse_worker_sse(mismatch.as_bytes(), request_id).is_err());
    }

    #[test]
    fn unsupported_or_failed_worker_terminals_are_not_successful_completions() {
        let request_id = "request-terminal-failure";
        for reason in ["abort", "error", "tool_calls"] {
            let body = format!(
                "data: {}\n\ndata: [DONE]\n\n",
                event(request_id, "done", json!({"finish_reason":reason}))
            );
            assert!(parse_worker_sse(body.as_bytes(), request_id).is_err());

            let mut adapter =
                OpenAiStreamAdapter::new("C2Rust-FP8-DFlash".to_string(), request_id.to_string());
            assert!(adapter
                .consume(&event(request_id, "done", json!({"finish_reason":reason}),))
                .is_err());
            assert!(!adapter.transport_done);
        }
    }

    #[test]
    fn stream_adapter_emits_openai_chunks_and_done() {
        let request_id = "request-4";
        let mut adapter =
            OpenAiStreamAdapter::new("C2Rust-FP8-DFlash".to_string(), request_id.to_string());
        let token = adapter
            .consume(&event(request_id, "token", json!({"text_delta":"pong"})))
            .unwrap();
        assert!(token[0].contains("pong"));
        let finish = adapter
            .consume(&event(
                request_id,
                "done",
                json!({"text_delta":"!","finish_reason":"stop"}),
            ))
            .unwrap();
        assert_eq!(finish.len(), 1);
        assert!(finish[0].contains("!"));
        assert!(finish[0].contains(r#""finish_reason":null"#));
        let terminal = adapter.consume("[DONE]").unwrap();
        assert_eq!(terminal.len(), 2);
        assert!(terminal[0].contains("finish_reason"));
        assert_eq!(terminal[1], "data: [DONE]\n\n");
        assert!(adapter.transport_done);
        assert!(adapter.consume("[DONE]").is_err());
    }

    #[tokio::test]
    async fn production_worker_probe_generation_and_abort_are_authenticated() {
        let (base_url, calls, shutdown, task) =
            spawn_mock_worker("grpc", "C2Rust-FP8-DFlash", true).await;
        let router = SglangRouter::from_values(
            Some(&base_url),
            Some("public-c2rust"),
            Some("C2Rust-FP8-DFlash"),
            Some("test-token"),
            Some("30"),
            None,
        )
        .unwrap();

        let status = router.status().await;
        assert!(status.shim_reachable);
        assert!(status.model_ready);
        assert!(router.probe_backend_identity().await);

        let mut public_request = request(false);
        public_request.model = "public-c2rust".to_string();
        let request_id = "integration-request";
        let response = router
            .send_generate(&public_request, request_id)
            .await
            .unwrap();
        let body = read_bounded_response(response, MAX_NON_STREAM_BODY_BYTES)
            .await
            .unwrap();
        let completed = parse_worker_sse(&body, request_id).unwrap();
        assert_eq!(completed.text, "pong");

        let guard = AbortGuard::new(&router, "cancel-me".to_string());
        drop(guard);
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if calls.lock().unwrap().abort_request_ids.len() == 1 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();

        let calls = calls.lock().unwrap();
        assert_eq!(calls.generate_payloads.len(), 1);
        assert_eq!(calls.generate_payloads[0]["model"], "C2Rust-FP8-DFlash");
        assert_eq!(calls.abort_request_ids, vec!["cancel-me"]);
        drop(calls);
        let _ = shutdown.send(());
        task.await.unwrap();
    }

    #[tokio::test]
    async fn fake_or_wrong_model_worker_never_becomes_ready() {
        for (mode, served_model, is_generation) in [
            ("fake", "C2Rust-FP8-DFlash", true),
            ("grpc", "different-model", true),
            ("grpc", "C2Rust-FP8-DFlash", false),
        ] {
            let (base_url, _, shutdown, task) =
                spawn_mock_worker(mode, served_model, is_generation).await;
            let router = SglangRouter::from_values(
                Some(&base_url),
                Some("C2Rust-FP8-DFlash"),
                Some("C2Rust-FP8-DFlash"),
                Some("test-token"),
                Some("30"),
                None,
            )
            .unwrap();
            let status = router.status().await;
            assert!(status.shim_reachable);
            assert!(!status.model_ready);
            assert!(!router.probe_backend_identity().await);
            let _ = shutdown.send(());
            task.await.unwrap();
        }
    }
}
