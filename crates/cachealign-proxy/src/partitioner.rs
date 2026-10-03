//! Static/Dynamic Prompt Partitioner in Rust.
//! Extracts volatile ephemeral variables and relocates them to the dynamic tail
//! enclosed in an isolated XML boundary.

use regex::Regex;
use serde_json::Value;
use std::sync::LazyLock;

static VOLATILE_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        Regex::new(r"(?im)^\s*(?:current\s+time|timestamp|current\s+date(?:\s+and\s+time)?|system\s+time|date)\s*[:=]\s*(.+)$").unwrap(),
        Regex::new(r"(?m)^\s*\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\s*$").unwrap(),
        Regex::new(r"(?im)^\s*(?:session|request|conversation|trace)[_-]id\s*[:=]\s*([a-zA-Z0-9_\-\.]+)\s*$").unwrap(),
        Regex::new(r"(?im)^\s*(?:turn|step)\s*[:=]\s*(\d+(?:\s*(?:of|/)\s*\d+)?)\s*$").unwrap(),
    ]
});

/// Extracts volatile header lines from prompt text, returning (clean_text, extracted_lines).
pub fn extract_volatile_elements(text: &str) -> (String, Vec<String>) {
    let mut kept = Vec::new();
    let mut extracted = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim();
        let mut matched = false;
        if !trimmed.is_empty() {
            for pattern in VOLATILE_PATTERNS.iter() {
                if pattern.is_match(trimmed) {
                    extracted.push(trimmed.to_string());
                    matched = true;
                    break;
                }
            }
        }
        if !matched {
            kept.push(line);
        }
    }

    (kept.join("\n").trim().to_string(), extracted)
}

/// Builds an XML container for migrated volatile context.
pub fn build_ephemeral_context_block(extracted: &[String]) -> String {
    let inner = extracted.join("; ");
    format!("\n<cachealign_ephemeral_context>\n[Context: {}]\n</cachealign_ephemeral_context>", inner)
}

/// Partitions system prompt and migrates extracted headers to the tail message.
pub fn partition_system_and_messages(
    system: Option<&str>,
    messages: &mut [Value],
) -> (Option<String>, Vec<String>) {
    let Some(sys_text) = system else {
        return (None, Vec::new());
    };

    let (clean_sys, extracted) = extract_volatile_elements(sys_text);
    if extracted.is_empty() || messages.is_empty() {
        return (Some(clean_sys), extracted);
    }

    let context_block = build_ephemeral_context_block(&extracted);

    // Safely append to the final message
    if let Some(last_msg) = messages.last_mut() {
        if let Some(content) = last_msg.get_mut("content") {
            if let Some(text) = content.as_str() {
                *content = Value::String(format!("{}{}", text, context_block));
            }
        }
    }

    (Some(clean_sys), extracted)
}