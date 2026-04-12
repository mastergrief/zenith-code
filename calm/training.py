"""
CALM v0.1 training signal collector.

Captures every prediction vs actual pair from the interceptor's
training_log and persists them to JSONL on disk. Each line is a
labeled training example generated for free during normal CALM usage.

Format per line:
{
  "timestamp": "2026-04-12T...",
  "prompt": "original user prompt",
  "instruction": "mul",
  "claimed": [401],       // what the model predicted (null if <pending>)
  "actual": [391],        // what the VM computed
  "correct": false,       // did the model predict correctly?
  "context": "push 17\npush 23\nmul -> [401]"  // surrounding CALM block
}

Usage:
    from calm.training import TrainingCollector
    collector = TrainingCollector()
    collector.save(engine_result, prompt="What is 17*23?")
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


DEFAULT_LOG_DIR = Path(".calm_training")


class TrainingCollector:
    """Persists training signal to JSONL files."""

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or DEFAULT_LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "signal.jsonl"

    def save(self, engine_result, prompt: str = "") -> int:
        """
        Append training signal entries from an EngineResult.
        Returns the number of entries written.
        """
        now = datetime.now(timezone.utc).isoformat()
        written = 0

        with open(self.log_file, "a") as f:
            for entry in engine_result.training_log:
                record = {
                    "timestamp": now,
                    "prompt": prompt,
                    "instruction": entry["instruction"],
                    "claimed": entry["claimed"],
                    "actual": entry["actual"],
                    "correct": entry["correct"],
                }
                f.write(json.dumps(record) + "\n")
                written += 1

        return written

    def load(self) -> list:
        """Load all training signal entries."""
        if not self.log_file.exists():
            return []
        entries = []
        with open(self.log_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def stats(self) -> dict:
        """Return summary statistics of the training log."""
        entries = self.load()
        if not entries:
            return {"total": 0}
        correct = sum(1 for e in entries if e["correct"])
        wrong = sum(1 for e in entries if not e["correct"])
        pending = sum(1 for e in entries if e["claimed"] is None)
        return {
            "total": len(entries),
            "correct": correct,
            "wrong": wrong,
            "pending": pending,
            "accuracy": correct / (correct + wrong) * 100 if (correct + wrong) > 0 else 0,
        }
