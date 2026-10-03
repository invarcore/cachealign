//! CacheAlign Sidecar Proxy Main Entrypoint.

mod canonicalizer;
mod partitioner;
mod proxy;

use axum::{
    routing::{get, post},
    Router,
};
use reqwest::Client;
use std::net::SocketAddr;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::proxy::{forward_chat_completions, forward_messages, health_check, metrics, AppState};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let port: u16 = std::env::var("PORT")
        .unwrap_or_else(|_| "8080".to_string())
        .parse()
        .unwrap_or(8080);

    let upstream_anthropic = std::env::var("UPSTREAM_ANTHROPIC")
        .unwrap_or_else(|_| "https://api.anthropic.com".to_string());
    let upstream_openai = std::env::var("UPSTREAM_OPENAI")
        .unwrap_or_else(|_| "https://api.openai.com".to_string());

    let state = AppState {
        http_client: Client::builder().build()?,
        upstream_anthropic,
        upstream_openai,
        total_requests: Arc::new(AtomicU64::new(0)),
        optimized_requests: Arc::new(AtomicU64::new(0)),
    };

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/metrics", get(metrics))
        .route("/v1/messages", post(forward_messages))
        .route("/v1/chat/completions", post(forward_chat_completions))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!("CacheAlign Proxy listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install CTRL+C signal handler");
    info!("Shutting down CacheAlign Proxy gracefully...");
}