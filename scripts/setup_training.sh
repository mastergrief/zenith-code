#!/bin/bash
# Setup training environment in WSL2
set -e

export PATH="$HOME/.local/bin:$PATH"

echo "=== Installing PyTorch with CUDA ==="
pip3 install torch --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -5

echo "=== Installing Unsloth + training deps ==="
pip3 install unsloth transformers trl datasets peft accelerate bitsandbytes 2>&1 | tail -5

echo "=== Verifying GPU access ==="
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"

echo "=== Setup complete ==="
