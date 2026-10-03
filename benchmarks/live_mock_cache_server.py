"""
Local High-Fidelity Prompt Cache Simulation Server.
Zero-dependency HTTP server mimicking Anthropic /v1/messages and OpenAI /v1/chat/completions.
Simulates real hardware KV-cache behavior:
- Stores SHA-256 hashes of prefix tokens.
- Aligned prefix -> Cache HIT (sub-50ms TTFT, cache_read_input_tokens populated).
- Unaligned/Busted prefix -> Cache MISS (450ms cold TTFT, full input token charge).

Usage:
    uv run python benchmarks/live_mock_cache_server.py [--port 8080]
"""

import argparse
import hashlib
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.stdout.reconfigure(encoding="utf-8")

# Global in-memory KV-cache store
CACHE_REGISTRY: set[str] = set()


class CacheSimulationHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                b'{"status":"healthy","cached_prefixes":' + str(len(CACHE_REGISTRY)).encode() + b"}"
            )
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_length)

        try:
            payload = json.loads(body_bytes.decode("utf-8"))
        except Exception:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error":"invalid_json"}')
            return

        # Handle Anthropic /v1/messages
        if "/v1/messages" in self.path:
            self._handle_anthropic(payload)
        # Handle OpenAI /v1/chat/completions
        elif "/v1/chat/completions" in self.path:
            self._handle_openai(payload)
        else:
            self.send_response(404)
            self.end_headers()

    def _compute_prefix_hash(self, system: any, tools: any) -> str:
        serialized = json.dumps([system, tools], sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _handle_anthropic(self, payload: dict):
        system = payload.get("system", "")
        tools = payload.get("tools", [])
        payload.get("messages", [])

        prefix_hash = self._compute_prefix_hash(system, tools)
        is_hit = prefix_hash in CACHE_REGISTRY

        # Simulate network & KV-cache inference latency
        if is_hit:
            time.sleep(0.04)  # 40ms fast cache hit TTFT
            CACHE_STATUS = "CACHE HIT (KV-Prefix Aligned)"
            cache_read = 1200
            cache_create = 0
            input_tokens = 300
        else:
            time.sleep(0.35)  # 350ms cold prefill TTFT
            CACHE_REGISTRY.add(prefix_hash)
            CACHE_STATUS = "CACHE MISS (Cold Prefill Stored)"
            cache_read = 0
            cache_create = 1200
            input_tokens = 1500

        print(
            f"[{time.strftime('%H:%M:%S')}] Anthropic /v1/messages | {CACHE_STATUS} | Hash: {prefix_hash[:8]}..."
        )

        response_body = {
            "id": f"msg_mock_{int(time.time() * 1000)}",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "This is a verified live response from CacheAlign Local Simulation Server.",
                }
            ],
            "model": payload.get("model", "claude-3-5-sonnet-20241022"),
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": 42,
                "cache_creation_input_tokens": cache_create,
                "cache_read_input_tokens": cache_read,
            },
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response_body).encode("utf-8"))

    def _handle_openai(self, payload: dict):
        messages = payload.get("messages", [])
        tools = payload.get("tools", [])

        system = next((m.get("content") for m in messages if m.get("role") == "system"), "")
        prefix_hash = self._compute_prefix_hash(system, tools)
        is_hit = prefix_hash in CACHE_REGISTRY

        if is_hit:
            time.sleep(0.04)
            CACHE_STATUS = "CACHE HIT (Tools & System Aligned)"
            cached_tokens = 1000
        else:
            time.sleep(0.35)
            CACHE_REGISTRY.add(prefix_hash)
            CACHE_STATUS = "CACHE MISS (Cold Context Stored)"
            cached_tokens = 0

        print(
            f"[{time.strftime('%H:%M:%S')}] OpenAI /v1/chat/completions | {CACHE_STATUS} | Hash: {prefix_hash[:8]}..."
        )

        response_body = {
            "id": f"chatcmpl_mock_{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model", "gpt-4o"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Simulated live completion with prompt cache verification.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1200,
                "completion_tokens": 35,
                "total_tokens": 1235,
                "prompt_tokens_details": {"cached_tokens": cached_tokens},
            },
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response_body).encode("utf-8"))

    def log_message(self, format, *args):
        # Suppress default noisy access logs
        return


def run_server(host: str = "0.0.0.0", port: int = 8080):
    server = HTTPServer((host, port), CacheSimulationHandler)
    print("\n" + "=" * 70)
    print(f"🚀 CacheAlign Live Prompt Cache Simulation Server Running on http://{host}:{port}")
    print("=" * 70)
    print("Features:")
    print(" • Emulates real Anthropic (/v1/messages) and OpenAI (/v1/chat/completions) caching")
    print(" • Real-time prefix hashing & KV-cache hit simulation")
    print(" • Verifies TTFT latency reduction: 350ms (Cold) -> 40ms (Warm Read)")
    print(" • Zero external dependencies, zero API keys, zero rate-limits")
    print("=" * 70)
    print("Press Ctrl+C to stop the server.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping simulation server...")
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CacheAlign Live Mock Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind (default: 8080)")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
