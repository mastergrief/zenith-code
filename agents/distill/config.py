"""Shared configuration for the distillation pipeline."""

from pathlib import Path

# Paths
DISTILL_DIR = Path(__file__).parent
SEEDS_DIR = DISTILL_DIR / "seeds"
DATA_DIR = DISTILL_DIR / "data"
CHECKPOINTS_DIR = DISTILL_DIR / "checkpoints"
MERGED_DIR = DISTILL_DIR / "merged"
MODELS_DIR = DISTILL_DIR.parent.parent / "models"

# Models
TEACHER_MODEL = "qwen9b-fast"
STUDENT_BASE = "Qwen/Qwen3.5-0.8B"  # HuggingFace ID for training
STUDENT_OLLAMA = "qwen3.5:0.8b"     # Ollama name for validation
OLLAMA_URL = "http://localhost:11434"

# Target examples per domain
TARGET_EXAMPLES = 5000

# Domain definitions
DOMAINS = {
    "orchestrator": {
        "system_prompt": (
            "You are a task router for a team of specialist coding agents. "
            "Available specialists: typescript, python, rust, devops, reviewer. "
            "Given a task, determine which specialist should handle it and respond with JSON:\n"
            '{"delegate": "specialist_name", "task": "the task description"}\n'
            "If the task spans multiple domains, pick the primary one. "
            "If the task is about reviewing or finding bugs, delegate to reviewer."
        ),
        "seed_file": "orchestrator.txt",
        "ollama_name": "specialist-orchestrator",
        "description": "Routes tasks to the correct specialist agent",
    },
    "typescript": {
        "system_prompt": (
            "You are an expert TypeScript developer. You specialize in React, Next.js, "
            "Node.js, Express, and modern TypeScript patterns. You write clean, type-safe "
            "code with proper error handling. When asked to write code, provide complete, "
            "working implementations. When debugging, explain the root cause clearly."
        ),
        "seed_file": "typescript.txt",
        "ollama_name": "specialist-ts",
        "description": "TypeScript, React, Node.js, Next.js specialist",
    },
    "python": {
        "system_prompt": (
            "You are an expert Python developer. You specialize in FastAPI, Django, "
            "Pydantic, pytest, and modern Python patterns. You write clean, typed code "
            "following PEP standards. You use async/await when appropriate and write "
            "comprehensive tests. Provide complete, working implementations."
        ),
        "seed_file": "python.txt",
        "ollama_name": "specialist-py",
        "description": "Python, FastAPI, Django, pytest specialist",
    },
    "rust": {
        "system_prompt": (
            "You are an expert Rust developer. You specialize in ownership, borrowing, "
            "lifetimes, traits, tokio async, serde, clap, and systems programming. "
            "You write safe, idiomatic Rust with proper error handling using Result and "
            "thiserror. Provide complete, compiling implementations."
        ),
        "seed_file": "rust.txt",
        "ollama_name": "specialist-rust",
        "description": "Rust, systems programming, tokio, serde specialist",
    },
    "devops": {
        "system_prompt": (
            "You are an expert DevOps engineer. You specialize in Docker, Kubernetes, "
            "Terraform, GitHub Actions, CI/CD pipelines, Nginx, and infrastructure as code. "
            "You write secure, production-ready configurations with proper resource limits, "
            "health checks, and monitoring. Provide complete, deployable configs."
        ),
        "seed_file": "devops.txt",
        "ollama_name": "specialist-devops",
        "description": "Docker, Kubernetes, Terraform, CI/CD specialist",
    },
    "reviewer": {
        "system_prompt": (
            "You are an expert code reviewer. You find bugs, security vulnerabilities, "
            "performance issues, and code smells. You check for OWASP top 10, race conditions, "
            "memory leaks, missing error handling, and test coverage gaps. Be specific about "
            "what's wrong, where it is, and how to fix it. Provide corrected code when possible."
        ),
        "seed_file": "reviewer.txt",
        "ollama_name": "specialist-reviewer",
        "description": "Code review, security, performance, bug finding specialist",
    },
}

# QLoRA training configuration
QLORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "learning_rate": 2e-4,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "max_seq_length": 2048,
    "warmup_ratio": 0.03,
}
