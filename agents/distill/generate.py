"""Dataset generator: uses the 9B teacher model to create training data for specialists."""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

from agents.distill.config import (
    DATA_DIR,
    DOMAINS,
    OLLAMA_URL,
    SEEDS_DIR,
    TARGET_EXAMPLES,
    TEACHER_MODEL,
)


class DatasetGenerator:
    """Generate training JSONL from a teacher model for each specialist domain."""

    def __init__(self, teacher_model: str = TEACHER_MODEL):
        self.teacher_model = teacher_model
        self.api_url = f"{OLLAMA_URL}/api/chat"

    def _call_teacher(self, messages: list[dict], temperature: float = 0.8) -> str:
        """Send messages to the teacher model and get a response."""
        payload = json.dumps({
            "model": self.teacher_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }).encode()

        req = urllib.request.Request(
            self.api_url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read())

        return result["message"]["content"]

    def _load_seeds(self, domain: str) -> list[str]:
        """Load seed prompts from a domain's seed file."""
        seed_file = SEEDS_DIR / DOMAINS[domain]["seed_file"]
        lines = seed_file.read_text(encoding="utf-8").strip().split("\n")
        return [line.strip() for line in lines if line.strip()]

    def _count_existing(self, output_path: Path) -> int:
        """Count existing examples in a JSONL file (for resuming)."""
        if not output_path.exists():
            return 0
        count = 0
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def _save_example(self, example: dict, output_path: Path):
        """Append a single example to the JSONL file."""
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    def _generate_coding_example(self, seed: str, domain_config: dict) -> list[dict]:
        """Generate training examples from a coding seed prompt."""
        system_prompt = domain_config["system_prompt"]
        examples = []

        # 1. Direct: teacher answers the seed prompt as the specialist
        try:
            response = self._call_teacher([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": seed},
            ])
            examples.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": seed},
                    {"role": "assistant", "content": response},
                ]
            })
        except Exception as e:
            print(f"    Error on direct: {e}")

        # 2. Variations: ask teacher to generate related prompts, then answer them
        try:
            variation_prompt = (
                f"Generate 5 different coding tasks related to this topic but with different specifics:\n"
                f'"{seed}"\n\n'
                f"Return each task on its own line, numbered 1-5. Just the task descriptions, nothing else."
            )
            variations_text = self._call_teacher([
                {"role": "system", "content": "You generate coding task descriptions. Be specific and varied."},
                {"role": "user", "content": variation_prompt},
            ], temperature=1.0)

            # Parse numbered lines
            variations = []
            for line in variations_text.strip().split("\n"):
                line = line.strip()
                if line and line[0].isdigit():
                    # Strip number prefix like "1. " or "1) "
                    task = line.lstrip("0123456789.)- ").strip()
                    if task:
                        variations.append(task)

            # Answer each variation as the specialist
            for task in variations[:5]:
                try:
                    response = self._call_teacher([
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": task},
                    ])
                    examples.append({
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": task},
                            {"role": "assistant", "content": response},
                        ]
                    })
                except Exception as e:
                    print(f"    Error on variation '{task[:40]}': {e}")

        except Exception as e:
            print(f"    Error generating variations: {e}")

        return examples

    def _generate_orchestrator_example(self, seed: str, domain_config: dict) -> list[dict]:
        """Generate routing/classification examples for the orchestrator."""
        system_prompt = domain_config["system_prompt"]
        examples = []

        # Ask the teacher to classify the task
        try:
            response = self._call_teacher([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": seed},
            ], temperature=0.3)  # Low temp for consistent classification

            examples.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": seed},
                    {"role": "assistant", "content": response},
                ]
            })
        except Exception as e:
            print(f"    Error on orchestrator: {e}")

        # Generate variation tasks and classify them too
        try:
            variation_prompt = (
                "Generate 5 diverse coding/devops tasks that each clearly belong to one of these categories: "
                "typescript, python, rust, devops, reviewer.\n"
                "Return each on its own line, numbered 1-5. Just task descriptions."
            )
            variations_text = self._call_teacher([
                {"role": "system", "content": "You generate diverse coding task descriptions."},
                {"role": "user", "content": variation_prompt},
            ], temperature=1.0)

            for line in variations_text.strip().split("\n"):
                line = line.strip()
                if line and line[0].isdigit():
                    task = line.lstrip("0123456789.)- ").strip()
                    if task:
                        try:
                            response = self._call_teacher([
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": task},
                            ], temperature=0.3)
                            examples.append({
                                "messages": [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": task},
                                    {"role": "assistant", "content": response},
                                ]
                            })
                        except Exception as e:
                            print(f"    Error on orchestrator variation: {e}")
        except Exception as e:
            print(f"    Error generating orchestrator variations: {e}")

        return examples

    def generate_domain(self, domain: str, target_count: int = TARGET_EXAMPLES):
        """Generate training data for a specific domain."""
        if domain not in DOMAINS:
            print(f"Error: Unknown domain '{domain}'. Available: {list(DOMAINS.keys())}")
            return

        domain_config = DOMAINS[domain]
        output_path = DATA_DIR / f"{domain}.jsonl"
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        existing = self._count_existing(output_path)
        seeds = self._load_seeds(domain)

        print(f"\n{'='*60}")
        print(f"Domain: {domain}")
        print(f"Model: {self.teacher_model}")
        print(f"Seeds: {len(seeds)}")
        print(f"Existing examples: {existing}")
        print(f"Target: {target_count}")
        print(f"Output: {output_path}")
        print(f"{'='*60}\n")

        if existing >= target_count:
            print(f"Already have {existing} examples (target: {target_count}). Skipping.")
            return

        is_orchestrator = domain == "orchestrator"
        total_generated = existing
        seed_idx = existing // 6  # Approximate which seed to resume from

        for i, seed in enumerate(seeds[seed_idx:], start=seed_idx):
            if total_generated >= target_count:
                break

            print(f"  [{total_generated}/{target_count}] Seed {i+1}/{len(seeds)}: {seed[:60]}...")
            start = time.time()

            if is_orchestrator:
                examples = self._generate_orchestrator_example(seed, domain_config)
            else:
                examples = self._generate_coding_example(seed, domain_config)

            for ex in examples:
                self._save_example(ex, output_path)
                total_generated += 1

            elapsed = time.time() - start
            rate = len(examples) / elapsed if elapsed > 0 else 0
            print(f"    +{len(examples)} examples ({elapsed:.1f}s, {rate:.1f} ex/s)")

        print(f"\nDone. Total examples: {total_generated}")
        print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate training data for specialist models")
    parser.add_argument(
        "--domain", "-d",
        default="all",
        help=f"Domain to generate for (default: all). Options: {', '.join(DOMAINS.keys())}, all",
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=TARGET_EXAMPLES,
        help=f"Target number of examples (default: {TARGET_EXAMPLES})",
    )
    parser.add_argument(
        "--model", "-m",
        default=TEACHER_MODEL,
        help=f"Teacher model (default: {TEACHER_MODEL})",
    )
    args = parser.parse_args()

    generator = DatasetGenerator(teacher_model=args.model)

    if args.domain == "all":
        for domain in DOMAINS:
            generator.generate_domain(domain, args.count)
    else:
        generator.generate_domain(args.domain, args.count)


if __name__ == "__main__":
    main()
