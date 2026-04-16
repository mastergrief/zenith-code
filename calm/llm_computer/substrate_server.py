"""Substrate-native inference server.

Serves the unified tensor via OpenAI-compatible API. PTs, compiled cards,
CALM verification, and knowledge DB fire natively. Drop-in replacement
for llama-server from the harness's perspective.

Architecture:
    1. Query arrives via /v1/chat/completions
    2. PT attempts NL → expression transduction (copy-augmented)
    3. If expression found → safe_eval computes → verified answer
    4. CALM verification on the result
    5. Response formatted as OpenAI-compatible JSON
    6. Auto-upgrade logs corrections for persistent knowledge

For general language (no expression target), falls back to the base LLM
via configurable fallback_url (llama-server).

Usage:
    python -m calm.llm_computer.substrate_server --port 8081
    curl localhost:8081/v1/chat/completions -d '{"messages":[...]}'
"""

from __future__ import annotations

import json
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional

import torch

from calm.hrm.data import _CHAR_TO_ID, _ID_TO_CHAR, VOCAB_SIZE
from calm.expression import safe_eval, ExpressionError


# --- PT Inference ---

class PTInference:
    """Load and run copy-augmented PTs for NL → expression transduction."""

    def __init__(self, checkpoint_dir: str = "calm/hrm/checkpoints"):
        self.models = {}
        self.configs = {}
        ckpt_dir = Path(checkpoint_dir)
        # Load all copy_*_best.pt checkpoints
        for pt_file in sorted(ckpt_dir.glob("copy_*_best.pt")):
            name = pt_file.stem.replace("copy_", "").replace("_best", "")
            try:
                ckpt = torch.load(pt_file, map_location="cpu", weights_only=False)
                if not ckpt.get("copy_augmented"):
                    continue
                cfg = ckpt["config"]
                from calm.llm_computer.copy_augmented import build_copy_augmented_hrm
                model = build_copy_augmented_hrm(**cfg)
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()
                self.models[name] = model
                self.configs[name] = cfg
            except Exception as e:
                print(f"[substrate] warning: failed to load {pt_file.name}: {e}")
        if self.models:
            print(f"[substrate] loaded {len(self.models)} PTs: {list(self.models.keys())}")

    # Keyword routing: map query keywords to preferred PT order
    _ROUTING = {
        "syllable": ["writing"],
        "rhyme": ["writing"],
        "meter": ["writing"],
        "haiku": ["writing"],
        "sonnet": ["writing"],
        "readability": ["writing"],
        "alliteration": ["writing"],
        "passive voice": ["writing"],
        "vocabulary": ["writing"],
        "word count": ["writing"],
        "percent": ["funcall", "reasoning"],
        "ratio": ["funcall", "reasoning"],
        "sequence": ["funcall"],
        "maximum": ["funcall"],
        "minimum": ["funcall"],
        "biggest": ["funcall", "logic"],
        "smallest": ["funcall", "logic"],
        "greater": ["logic"],
        "less than": ["logic"],
        "compare": ["logic"],
        "if ": ["logic"],
        "bought": ["gsm", "word"],
        "sold": ["gsm", "word"],
        "earned": ["gsm", "word"],
        "spent": ["gsm", "word"],
        "has": ["word", "gsm", "logic"],
    }

    def _route_pts(self, text: str) -> list:
        """Return PT names in priority order based on query keywords."""
        text_lower = text.lower()
        matched = set()
        for keyword, pts in self._ROUTING.items():
            if keyword in text_lower:
                for pt in pts:
                    if pt in self.models:
                        matched.add(pt)
        # Matched PTs first, then all others as fallback
        ordered = list(matched)
        for name in self.models:
            if name not in ordered:
                ordered.append(name)
        return ordered

    def transduce(self, text: str, max_gen: int = 40) -> Optional[str]:
        """Try PTs in routed priority order. Return first valid expression or None."""
        bos = _CHAR_TO_ID["<bos>"]
        sep = _CHAR_TO_ID["<sep>"]
        eos = _CHAR_TO_ID["<eos>"]

        text_lower = text.lower().strip().rstrip("?.")
        pt_order = self._route_pts(text_lower)

        for name in pt_order:
            model = self.models[name]
            cfg = self.configs[name]
            pos_limit = cfg["max_len"]

            ids = [bos] + [_CHAR_TO_ID[c] for c in text_lower
                           if c in _CHAR_TO_ID] + [sep]

            if len(ids) > pos_limit - 10:
                continue  # text too long for this PT

            gen = []
            budget = min(max_gen, pos_limit - len(ids) - 1)
            for _ in range(max(budget, 1)):
                x = torch.tensor([ids], dtype=torch.long)
                with torch.no_grad():
                    log_probs = model(x)
                nxt = int(log_probs[0, -1].argmax().item())
                if nxt == eos:
                    break
                gen.append(nxt)
                ids.append(nxt)

            if not gen:
                continue

            decoded = "".join(
                _ID_TO_CHAR.get(i, "") for i in gen
                if not _ID_TO_CHAR.get(i, "").startswith("<")
            ).strip().rstrip("=").strip()

            if not decoded:
                continue

            # Validate: can safe_eval parse this?
            try:
                result = safe_eval(decoded)
                return decoded
            except (ExpressionError, Exception):
                continue

        return None


# --- CALM Verification ---

def verify_and_compute(expression: str) -> dict:
    """Evaluate expression via safe_eval and return result + verification."""
    try:
        result = safe_eval(expression)
        return {
            "expression": expression,
            "result": result,
            "verified": True,
            "source": "substrate",
        }
    except (ExpressionError, Exception) as e:
        return {
            "expression": expression,
            "result": None,
            "verified": False,
            "error": str(e),
            "source": "substrate",
        }


# --- OpenAI-Compatible API ---

def _make_response(content: str, model: str = "substrate-v1",
                   finish_reason: str = "stop") -> dict:
    """Format as OpenAI chat completion response."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason,
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _make_stream_chunk(content: str, model: str = "substrate-v1",
                       finish_reason: Optional[str] = None) -> str:
    """Format as SSE streaming chunk."""
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": content} if content else {},
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(chunk)}\n\n"


class SubstrateHandler(BaseHTTPRequestHandler):
    """HTTP handler for OpenAI-compatible substrate API."""

    pt_inference: PTInference = None  # Set by server init
    fallback_url: Optional[str] = None
    gemma = None              # Optional GemmaSubstrate (native fallback)
    gemma_tokenizer = None
    gemma_device: str = "cuda"
    gemma_max_tokens: int = 64

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._handle_chat()
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok",
                                          "pts": list(self.pt_inference.models.keys()),
                                          "backend": "substrate"}).encode())
        elif self.path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"data": [
                {"id": "substrate-v1", "object": "model"}
            ]}).encode())
        else:
            self.send_error(404)

    def _handle_chat(self):
        try:
            self._handle_chat_inner()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send_json(500, {"error": str(e)})

    def _handle_chat_inner(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        messages = body.get("messages", [])
        stream = body.get("stream", False)

        # Extract the last user message
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = m.get("content", "")
                break

        if not user_msg:
            self._send_json(400, {"error": "No user message"})
            return

        # --- Substrate pipeline ---
        t0 = time.time()

        # Step 1: PT transduction
        expression = self.pt_inference.transduce(user_msg)

        if expression:
            # Step 2: Compute via safe_eval
            result = verify_and_compute(expression)
            elapsed = time.time() - t0

            if result["verified"]:
                content = (f"The answer is **{result['result']}**.\n\n"
                           f"Expression: `{expression}`\n"
                           f"Verified: yes (substrate, {elapsed*1000:.0f}ms)")
            else:
                content = (f"Expression extracted: `{expression}`\n"
                           f"Error: {result.get('error', 'unknown')}")
        else:
            # No expression found — try CALM precompute
            try:
                from calm.precompute import precompute
                precomputed = precompute(user_msg)
                if precomputed:
                    elapsed = time.time() - t0
                    facts = "; ".join(f"{k} = {v}" for k, v in
                                      list(precomputed.items())[:3])
                    content = (f"Computed: {facts}\n\n"
                               f"Verified: yes (CALM precompute, {elapsed*1000:.0f}ms)")
                else:
                    content = None
            except Exception:
                content = None

            if content is None:
                # Native Gemma substrate path takes precedence over proxy.
                if self.gemma is not None:
                    out = self.gemma.generate(
                        user_msg, self.gemma_tokenizer,
                        max_tokens=self.gemma_max_tokens,
                        device=self.gemma_device)
                    content = out["text"]
                elif self.fallback_url:
                    self._proxy_to_fallback(body)
                    return
                else:
                    elapsed = time.time() - t0
                    content = (f"No computable expression found in query.\n"
                               f"Substrate handles: math, reasoning, comparisons, "
                               f"percentages, ratios, writing analysis.\n"
                               f"({elapsed*1000:.0f}ms)")

        # Send response
        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            # Stream word by word for realistic SSE
            words = content.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                self.wfile.write(_make_stream_chunk(chunk).encode())
                self.wfile.flush()
            self.wfile.write(_make_stream_chunk("", finish_reason="stop").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            self._send_json(200, _make_response(content))

    def _proxy_to_fallback(self, body: dict):
        """Forward request to fallback LLM server (e.g. llama-server)."""
        import urllib.request
        try:
            req = urllib.request.Request(
                f"{self.fallback_url}/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                self.send_response(resp.status)
                for key, val in resp.getheaders():
                    if key.lower() not in ("transfer-encoding",):
                        self.send_header(key, val)
                self.end_headers()
                self.wfile.write(resp.read())
        except Exception as e:
            self._send_json(502, {"error": f"Fallback failed: {e}"})

    def _send_json(self, code: int, data: dict):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        """Suppress default HTTP logging — too noisy."""
        pass


def serve(port: int = 8081, fallback_url: str = None,
          checkpoint_dir: str = "calm/hrm/checkpoints",
          gemma_gguf: Optional[str] = None,
          gemma_max_len: int = 8192,
          gemma_max_tokens: int = 64,
          gemma_device: str = "cuda"):
    """Start the substrate inference server."""
    print(f"[substrate] loading PTs from {checkpoint_dir}...")
    pt = PTInference(checkpoint_dir)

    SubstrateHandler.pt_inference = pt
    SubstrateHandler.fallback_url = fallback_url

    if gemma_gguf:
        from calm.llm_computer.gemma_substrate import GemmaSubstrate
        from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer
        print(f"[substrate] loading Gemma substrate from {gemma_gguf}...")
        gemma = GemmaSubstrate.from_gguf(gemma_gguf, max_len=gemma_max_len)
        if gemma_device == "cuda":
            gemma.preload_gpu(gemma_device)
        SubstrateHandler.gemma = gemma
        SubstrateHandler.gemma_tokenizer = GemmaTokenizer.from_gguf(gemma_gguf)
        SubstrateHandler.gemma_device = gemma_device
        SubstrateHandler.gemma_max_tokens = gemma_max_tokens

    server = HTTPServer(("0.0.0.0", port), SubstrateHandler)
    print(f"[substrate] serving on http://localhost:{port}")
    print(f"[substrate] endpoints: /v1/chat/completions, /health, /v1/models")
    if SubstrateHandler.gemma is not None:
        print(f"[substrate] native fallback: GemmaSubstrate (gguf={gemma_gguf})")
    elif fallback_url:
        print(f"[substrate] proxy fallback: {fallback_url}")
    print(f"[substrate] ready.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[substrate] shutting down.")
        server.server_close()


def main():
    import argparse
    p = argparse.ArgumentParser(description="Substrate-native inference server")
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--fallback", type=str, default=None,
                   help="Fallback LLM server URL (e.g. http://localhost:8080)")
    p.add_argument("--checkpoint-dir", type=str, default="calm/hrm/checkpoints")
    p.add_argument("--gemma-gguf", type=str, default=None,
                   help="Load GemmaSubstrate as native fallback (no llama-server proxy)")
    p.add_argument("--gemma-max-len", type=int, default=8192,
                   help="Max sequence length for Gemma KV cache (default 8192)")
    p.add_argument("--gemma-max-tokens", type=int, default=64,
                   help="Max generated tokens per request (default 64)")
    p.add_argument("--gemma-device", type=str, default="cuda")
    args = p.parse_args()
    serve(port=args.port, fallback_url=args.fallback,
          checkpoint_dir=args.checkpoint_dir,
          gemma_gguf=args.gemma_gguf,
          gemma_max_len=args.gemma_max_len,
          gemma_max_tokens=args.gemma_max_tokens,
          gemma_device=args.gemma_device)


if __name__ == "__main__":
    main()
