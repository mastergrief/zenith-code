"""Specialist Swarm Distillation Pipeline.

Distills a large teacher model (9B) into a swarm of specialized 0.6B models,
each fine-tuned for a specific domain (TypeScript, Python, Rust, DevOps, etc.).
"""

from agents.distill.config import DOMAINS, TEACHER_MODEL, STUDENT_BASE
