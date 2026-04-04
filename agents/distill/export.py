"""Export trained specialist models to GGUF and register with Ollama.

Usage:
    python -m agents.distill.export --domain python
    python -m agents.distill.export --domain all
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

from agents.distill.config import DOMAINS, MERGED_DIR, MODELS_DIR, OLLAMA_URL


LLAMA_CPP_PATH = os.environ.get("LLAMA_CPP_PATH", os.path.expanduser("~/llama.cpp"))


def find_llama_cpp() -> Path:
    """Find the llama.cpp installation."""
    path = Path(LLAMA_CPP_PATH)
    convert_script = path / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        print(f"Error: llama.cpp not found at {path}")
        print(f"\nInstall it with:")
        print(f"  git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp")
        print(f"  pip install -r ~/llama.cpp/requirements.txt")
        print(f"\nOr set LLAMA_CPP_PATH environment variable.")
        sys.exit(1)
    return path


def convert_to_gguf(domain: str, merged_dir: Path, output_path: Path, quant: str = "q4_k_m"):
    """Convert a merged HuggingFace model to GGUF format."""
    llama_cpp = find_llama_cpp()
    convert_script = llama_cpp / "convert_hf_to_gguf.py"

    print(f"  Converting to GGUF ({quant})...")
    cmd = [
        sys.executable, str(convert_script),
        str(merged_dir),
        "--outfile", str(output_path),
        "--outtype", quant,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"  Error converting to GGUF:")
        print(f"  {result.stderr[:500]}")
        return False

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  GGUF saved: {output_path} ({size_mb:.0f} MB)")
    return True


def create_modelfile(domain: str, gguf_path: Path) -> Path:
    """Generate an Ollama Modelfile for a specialist."""
    domain_config = DOMAINS[domain]
    modelfile_path = MODELS_DIR / f"Modelfile.{domain_config['ollama_name']}"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    system_prompt = domain_config["system_prompt"].replace('"', '\\"')

    content = f"""FROM {gguf_path}

PARAMETER num_ctx 32768
PARAMETER temperature 0.7
PARAMETER top_k 20
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1
PARAMETER num_gpu 999

SYSTEM \"\"\"{domain_config['system_prompt']}\"\"\"
"""
    modelfile_path.write_text(content, encoding="utf-8")
    print(f"  Modelfile: {modelfile_path}")
    return modelfile_path


def register_with_ollama(domain: str, modelfile_path: Path) -> bool:
    """Register the model with Ollama."""
    ollama_name = DOMAINS[domain]["ollama_name"]
    print(f"  Registering as '{ollama_name}'...")

    result = subprocess.run(
        ["ollama", "create", ollama_name, "-f", str(modelfile_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        print(f"  Error registering with Ollama:")
        print(f"  {result.stderr[:500]}")
        return False

    print(f"  Registered: {ollama_name}")
    return True


def smoke_test(domain: str) -> bool:
    """Send a test prompt to verify the specialist works."""
    ollama_name = DOMAINS[domain]["ollama_name"]
    print(f"  Smoke testing '{ollama_name}'...")

    try:
        payload = json.dumps({
            "model": ollama_name,
            "messages": [{"role": "user", "content": "Say hello in one sentence."}],
            "stream": False,
        }).encode()

        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())

        reply = result["message"]["content"]
        print(f"  Response: {reply[:100]}")
        return bool(reply.strip())

    except Exception as e:
        print(f"  Smoke test failed: {e}")
        return False


def export_specialist(domain: str):
    """Full export pipeline: GGUF conversion → Modelfile → Ollama registration → smoke test."""
    if domain not in DOMAINS:
        print(f"Error: Unknown domain '{domain}'. Available: {list(DOMAINS.keys())}")
        return

    merged_dir = MERGED_DIR / domain
    if not merged_dir.exists():
        print(f"Error: Merged model not found at {merged_dir}")
        print(f"Run: python -m agents.distill.train --domain {domain}")
        return

    gguf_path = MERGED_DIR / f"{domain}.gguf"

    print(f"\n{'='*60}")
    print(f"Exporting specialist: {domain}")
    print(f"{'='*60}\n")

    # Step 1: Convert to GGUF
    if not gguf_path.exists():
        if not convert_to_gguf(domain, merged_dir, gguf_path):
            return
    else:
        size_mb = gguf_path.stat().st_size / (1024 * 1024)
        print(f"  GGUF already exists: {gguf_path} ({size_mb:.0f} MB)")

    # Step 2: Create Modelfile
    modelfile_path = create_modelfile(domain, gguf_path)

    # Step 3: Register with Ollama
    if not register_with_ollama(domain, modelfile_path):
        return

    # Step 4: Smoke test
    smoke_test(domain)

    print(f"\nExport complete: {DOMAINS[domain]['ollama_name']}")


def main():
    parser = argparse.ArgumentParser(description="Export specialist models to Ollama")
    parser.add_argument(
        "--domain", "-d",
        required=True,
        help=f"Domain to export. Options: {', '.join(DOMAINS.keys())}, all",
    )
    args = parser.parse_args()

    if args.domain == "all":
        for domain in DOMAINS:
            export_specialist(domain)
    else:
        export_specialist(args.domain)


if __name__ == "__main__":
    main()
