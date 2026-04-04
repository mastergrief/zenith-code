"""Validate specialist models against the base 0.6B using the 9B teacher as judge.

Usage:
    python -m agents.distill.validate --domain python
    python -m agents.distill.validate --domain all
"""

import argparse
import json
import urllib.request
from pathlib import Path

from agents.distill.config import (
    DOMAINS,
    OLLAMA_URL,
    SEEDS_DIR,
    STUDENT_OLLAMA,
    TEACHER_MODEL,
)


def call_model(model: str, messages: list[dict], temperature: float = 0.7) -> str:
    """Call an Ollama model and return the response."""
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())

    return result["message"]["content"]


def judge_responses(prompt: str, response_a: str, response_b: str, domain: str) -> dict:
    """Use the 9B teacher to judge which response is better."""
    judge_prompt = (
        f"You are judging two AI responses to a {domain} coding task.\n\n"
        f"TASK: {prompt}\n\n"
        f"RESPONSE A:\n{response_a[:2000]}\n\n"
        f"RESPONSE B:\n{response_b[:2000]}\n\n"
        f"Which response is better for a {domain} specialist? Consider:\n"
        f"- Correctness and completeness\n"
        f"- Code quality and best practices\n"
        f"- Clarity of explanation\n\n"
        f'Respond with JSON: {{"winner": "A" or "B", "reason": "brief explanation"}}'
    )

    try:
        result = call_model(TEACHER_MODEL, [
            {"role": "user", "content": judge_prompt},
        ], temperature=0.2)

        # Try to parse JSON from response
        import re
        match = re.search(r'\{[^{}]+\}', result)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"    Judge error: {e}")

    return {"winner": "tie", "reason": "judge failed to respond"}


def validate_specialist(domain: str, num_tests: int = 20):
    """Compare a specialist model against the base 0.6B."""
    if domain not in DOMAINS:
        print(f"Error: Unknown domain '{domain}'. Available: {list(DOMAINS.keys())}")
        return

    domain_config = DOMAINS[domain]
    specialist_model = domain_config["ollama_name"]
    system_prompt = domain_config["system_prompt"]

    # Load seed prompts (use last N as holdout for validation)
    seeds = []
    seed_file = SEEDS_DIR / domain_config["seed_file"]
    lines = seed_file.read_text(encoding="utf-8").strip().split("\n")
    seeds = [l.strip() for l in lines if l.strip()]

    # Use the last num_tests seeds as holdout
    test_prompts = seeds[-num_tests:]

    print(f"\n{'='*60}")
    print(f"Validating: {specialist_model} vs {STUDENT_OLLAMA}")
    print(f"Domain: {domain}")
    print(f"Test prompts: {len(test_prompts)}")
    print(f"Judge: {TEACHER_MODEL}")
    print(f"{'='*60}\n")

    specialist_wins = 0
    base_wins = 0
    ties = 0
    results = []

    for i, prompt in enumerate(test_prompts):
        print(f"  [{i+1}/{len(test_prompts)}] {prompt[:60]}...")

        # Get responses from both models
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            specialist_response = call_model(specialist_model, messages)
        except Exception as e:
            print(f"    Specialist error: {e}")
            specialist_response = f"Error: {e}"

        try:
            base_response = call_model(STUDENT_OLLAMA, messages)
        except Exception as e:
            print(f"    Base error: {e}")
            base_response = f"Error: {e}"

        # Judge (randomly assign A/B to avoid position bias)
        import random
        if random.random() < 0.5:
            judgment = judge_responses(prompt, specialist_response, base_response, domain)
            specialist_is = "A"
        else:
            judgment = judge_responses(prompt, base_response, specialist_response, domain)
            specialist_is = "B"

        winner = judgment.get("winner", "tie")
        if winner == specialist_is:
            specialist_wins += 1
            outcome = "SPECIALIST"
        elif winner == "tie":
            ties += 1
            outcome = "TIE"
        else:
            base_wins += 1
            outcome = "BASE"

        print(f"    Winner: {outcome} — {judgment.get('reason', 'N/A')[:80]}")

        results.append({
            "prompt": prompt,
            "outcome": outcome,
            "reason": judgment.get("reason", ""),
        })

    # Summary
    total = specialist_wins + base_wins + ties
    spec_pct = (specialist_wins / total * 100) if total else 0

    print(f"\n{'='*60}")
    print(f"Results: {specialist_model}")
    print(f"{'='*60}")
    print(f"  Specialist wins: {specialist_wins}/{total} ({spec_pct:.0f}%)")
    print(f"  Base wins:       {base_wins}/{total} ({base_wins/total*100:.0f}%)")
    print(f"  Ties:            {ties}/{total}")
    print(f"  {'PASS' if spec_pct >= 60 else 'FAIL'} (threshold: 60%)")
    print(f"{'='*60}")

    return {
        "domain": domain,
        "specialist_model": specialist_model,
        "specialist_wins": specialist_wins,
        "base_wins": base_wins,
        "ties": ties,
        "win_rate": spec_pct,
        "passed": spec_pct >= 60,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate specialist models")
    parser.add_argument(
        "--domain", "-d",
        required=True,
        help=f"Domain to validate. Options: {', '.join(DOMAINS.keys())}, all",
    )
    parser.add_argument(
        "--tests", "-n",
        type=int,
        default=20,
        help="Number of test prompts (default: 20)",
    )
    args = parser.parse_args()

    if args.domain == "all":
        all_results = {}
        for domain in DOMAINS:
            result = validate_specialist(domain, args.tests)
            if result:
                all_results[domain] = result

        print(f"\n\n{'='*60}")
        print(f"OVERALL RESULTS")
        print(f"{'='*60}")
        for domain, r in all_results.items():
            status = "PASS" if r["passed"] else "FAIL"
            print(f"  {domain:20s} {r['win_rate']:5.0f}% [{status}]")
    else:
        validate_specialist(args.domain, args.tests)


if __name__ == "__main__":
    main()
