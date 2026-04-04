#!/bin/bash
# Stage 1: Train a reasoning base model from qwen3:0.6b + Claude reasoning data
# Run in WSL2: bash /mnt/c/Users/gabes/Projects/claw-code/agents/distill/train_reasoning_base.sh
export PATH="$HOME/.local/bin:$PATH"
cd /mnt/c/Users/gabes/Projects/claw-code
PYTHONPATH=. python3 -c "
from agents.distill.train_base import train_reasoning_base
train_reasoning_base()
"
