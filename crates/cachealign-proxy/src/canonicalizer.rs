//! RFC 8785 JSON Canonicalization Scheme (JCS) Normalizer in Rust.
//! Ensures tool definitions and schemas have byte-identical serialized output.

use serde_json::{Map, Value};
use std::collections::BTreeMap;

pub const MAX_SCHEMA_DEPTH: usize = 20;

/// Recursively sorts all JSON object keys lexicographically according to RFC 8785.
pub fn canonicalize_value(val: &Value, depth: usize) -> Result<Value, &'static str> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err("Schema nesting depth exceeded maximum recursion limit (20)");
    }

    match val {
        Value::Object(map) => {
            let mut btree = BTreeMap::new();
            for (k, v) in map {
                let canonical_v = canonicalize_value(v, depth + 1)?;
                btree.insert(k.clone(), canonical_v);
            }
            let mut sorted_map = Map::new();
            for (k, v) in btree {
                sorted_map.insert(k, v);
            }
            Ok(Value::Object(sorted_map))
        }
        Value::Array(arr) => {
            let mut canonical_arr = Vec::with_capacity(arr.len());
            for item in arr {
                canonical_arr.push(canonicalize_value(item, depth + 1)?);
            }
            Ok(Value::Array(canonical_arr))
        }
        _ => Ok(val.clone()),
    }
}

/// Extracts a deterministic tool name identifier from Anthropic or OpenAI tool definitions.
fn get_tool_identifier(tool: &Value) -> String {
    if let Some(name) = tool.get("name").and_then(|n| n.as_str()) {
        return name.to_string();
    }
    if let Some(func) = tool.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str()) {
        return func.to_string();
    }
    tool.to_string()
}

/// Canonicalizes an array of tool definitions.
pub fn canonicalize_tool_schemas(tools: &mut [Value]) -> Result<(), &'static str> {
    // 1. Sort tools deterministically by their identifier
    tools.sort_by_key(|t| get_tool_identifier(t));

    // 2. Canonicalize nested schema keys inside each tool
    for tool in tools.iter_mut() {
        *tool = canonicalize_value(tool, 0)?;
    }
    Ok(())
}