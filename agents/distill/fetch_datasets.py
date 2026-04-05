"""Fetch external distillation datasets from HuggingFace and convert to our JSONL format.

Usage:
    python -m agents.distill.fetch_datasets
"""

import json
import urllib.request
from pathlib import Path

from agents.distill.config import DATA_DIR

# HuggingFace datasets to fetch (parquet URLs)
DATASETS = [
    {
        "name": "nohurry/Opus-4.6-Reasoning-3000x-filtered",
        "dataset_id": "nohurry%2FOpus-4.6-Reasoning-3000x-filtered",
        "total": 3000,
    },
    {
        "name": "TeichAI/Claude-Opus-4.6-Reasoning-887x",
        "dataset_id": "TeichAI%2FClaude-Opus-4.6-Reasoning-887x",
        "total": 887,
    },
    {
        "name": "Crownelius/Opus-4.6-Reasoning-2100x-formatted",
        "dataset_id": "Crownelius%2FOpus-4.6-Reasoning-2100x-formatted",
        "total": 2160,
    },
]

BATCH_SIZE = 100

OUTPUT_FILE = DATA_DIR / "claude_reasoning.jsonl"


def fetch_dataset(dataset_id: str, name: str, total: int) -> list[dict]:
    """Fetch all rows from HuggingFace datasets API in batches."""
    print(f"  Fetching {name} ({total} rows)...")
    all_rows = []

    for offset in range(0, total, BATCH_SIZE):
        url = (
            f"https://datasets-server.huggingface.co/rows?"
            f"dataset={dataset_id}&config=default&split=train"
            f"&offset={offset}&length={BATCH_SIZE}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        rows = data.get("rows", [])
        all_rows.extend(rows)
        print(f"    {len(all_rows)}/{total}...")

    print(f"  Got {len(all_rows)} rows total")
    return all_rows


def convert_row(row: dict) -> dict | None:
    """Convert a HuggingFace row to our JSONL format."""
    row_data = row.get("row", row)
    messages = row_data.get("messages", [])

    if not messages:
        return None

    # Convert to our format: simple role/content messages
    converted = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Some datasets have thinking/reasoning in separate fields
        thinking = msg.get("thinking", "") or msg.get("reasoning", "")

        if thinking and role == "assistant":
            # Prepend thinking to content for chain-of-thought training
            content = f"<think>\n{thinking}\n</think>\n\n{content}"

        if role and content:
            converted.append({"role": role, "content": content})

    if len(converted) < 2:
        return None

    return {"messages": converted}


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    total = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ds in DATASETS:
            try:
                rows = fetch_dataset(ds["dataset_id"], ds["name"], ds["total"])
                count = 0
                for row in rows:
                    example = convert_row(row)
                    if example:
                        f.write(json.dumps(example, ensure_ascii=False) + "\n")
                        count += 1
                        total += 1
                print(f"  Converted {count} examples from {ds['name']}")
            except Exception as e:
                print(f"  Error fetching {ds['name']}: {e}")

    print(f"\nTotal: {total} examples saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
