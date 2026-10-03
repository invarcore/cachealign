//! HTTP Proxy routes and forwarding logic for Anthropic and OpenAI compatible APIs.

use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use reqwest::Client;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tracing::{error, info};

use crate::canonicalizer::canonicalize_tool_schemas;
use crate::partitioner::partition_system_and_messages;

#[derive(Clone)]
pub struct AppState {
    pub http_client: Client,
    pub upstream_anthropic: String,
    pub upstream_openai: String,
    pub total_requests: Arc<AtomicU64>,
    pub optimized_requests: Arc<AtomicU64>,
}

pub async fn health_check() -> impl IntoResponse {
    Json(json!({
        "status": "healthy",
        "service": "cachealign-proxy",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

pub async fn metrics(State(state): State<AppState>) -> impl IntoResponse {
    Json(json!({
        "total_requests": state.total_requests.load(Ordering::Relaxed),
        "optimized_requests": state.optimized_requests.load(Ordering::Relaxed),
    }))
}

pub async fn forward_messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut payload): Json<Value>,
) -> Result<Response, (StatusCode, String)> {
    state.total_requests.fetch_add(1, Ordering::Relaxed);

    // 1. Canonicalize tools if present
    if let Some(tools) = payload.get_mut("tools").and_then(|t| t.as_array_mut()) {
        if canonicalize_tool_schemas(tools).is_ok() {
            // Inject cache breakpoint on last tool
            if let Some(last_tool) = tools.last_mut().and_then(|t| t.as_object_mut()) {
                last_tool.insert("cache_control".to_string(), json!({"type": "ephemeral"}));
            }
        }
    }

    // 2. Partition system prompt
    let system_str = payload.get("system").and_then(|s| s.as_str()).map(|s| s.to_string());
    if let Some(sys) = system_str {
        if let Some(messages) = payload.get_mut("messages").and_then(|m| m.as_array_mut()) {
            let (clean_sys, _extracted) = partition_system_and_messages(Some(&sys), messages);
            if let Some(cleaned) = clean_sys {
                // Convert system prompt to structured block with cache_control
                payload["system"] = json!([
                    {
                        "type": "text",
                        "text": cleaned,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]);
            }
        }
    }

    state.optimized_requests.fetch_add(1, Ordering::Relaxed);

    // 3. Forward request upstream
    let target_url = format!("{}/v1/messages", state.upstream_anthropic);
    let mut req_builder = state.http_client.post(&target_url);

    // Forward upstream authorization and custom headers
    for (name, val) in headers.iter() {
        if name != "host" && name != "content-length" {
            req_builder = req_builder.header(name.as_str(), val.as_bytes());
        }
    }

    let upstream_res = req_builder
        .json(&payload)
        .send()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Upstream gateway error: {}", e)))?;

    let status = StatusCode::from_u16(upstream_res.status().as_u16())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

    let mut response_headers = HeaderMap::new();
    for (name, val) in upstream_res.headers() {
        response_headers.insert(name.clone(), val.clone());
    }

    let stream = upstream_res.bytes_stream();
    let body = Body::from_stream(stream);

    let mut response = Response::new(body);
    *response.status_mut() = status;
    *response.headers_mut() = response_headers;

    Ok(response)
}

pub async fn forward_chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut payload): Json<Value>,
) -> Result<Response, (StatusCode, String)> {
    state.total_requests.fetch_add(1, Ordering::Relaxed);

    // Canonicalize tools if present
    if let Some(tools) = payload.get_mut("tools").and_then(|t| t.as_array_mut()) {
        let _ = canonicalize_tool_schemas(tools);
    }

    state.optimized_requests.fetch_add(1, Ordering::Relaxed);

    let target_url = format!("{}/v1/chat/completions", state.upstream_openai);
    let mut req_builder = state.http_client.post(&target_url);

    for (name, val) in headers.iter() {
        if name != "host" && name != "content-length" {
            req_builder = req_builder.header(name.as_str(), val.as_bytes());
        }
    }

    let upstream_res = req_builder
        .json(&payload)
        .send()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Upstream gateway error: {}", e)))?;

    let status = StatusCode::from_u16(upstream_res.status().as_u16())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

    let mut response_headers = HeaderMap::new();
    for (name, val) in upstream_res.headers() {
        response_headers.insert(name.clone(), val.clone());
    }

    let stream = upstream_res.bytes_stream();
    let body = Body::from_stream(stream);

    let mut response = Response::new(body);
    *response.status_mut() = status;
    *response.headers_mut() = response_headers;

    Ok(response)
}