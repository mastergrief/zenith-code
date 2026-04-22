"""Quick diagnostic: what does Gemma natively emit after 'Answer: '?
Compare number_theory vs base_conversion prompts."""
from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
assert "m" in globals() and "tok" in globals()  # type: ignore[name-defined]
sys.path.insert(0, str(ROOT))
from calm.llm_computer.gemma_substrate import KVCache
from calm.llm_computer.facades.retrieval import _monkey_patch_fast_encode
_monkey_patch_fast_encode(tok)  # type: ignore[name-defined]

def probe(prompt, n=5):
    ids = tok.encode(prompt)  # type: ignore[name-defined]
    cache = KVCache(m.config.n_layers, device="cuda")  # type: ignore[name-defined]
    with torch.no_grad():
        logits = m.forward(torch.tensor([ids]), device="cuda",  # type: ignore[name-defined]
                            kv_cache=cache, start_pos=0)
    # Top n tokens with logit
    top = logits[0, -1].float().topk(n)
    print(f'  prompt={prompt!r}')
    print(f'  last-token logit top-{n}:')
    for v, i in zip(top.values.tolist(), top.indices.tolist()):
        t = tok.id_to_token.get(i, "?")  # type: ignore[name-defined]
        print(f'    {i:>7} {t!r:<20} logit={v:.3f}')

print("=== number theory prompts ===")
probe("What is 25 mod 7? Answer: ")
probe("What is the LCM of 12 and 18? Answer: ")
probe("What is the GCD of 48 and 180? Answer: ")
print("\n=== base conversion prompts (working) ===")
probe("What is 0xFF in decimal? Answer: ")
probe("What is 0x100 in decimal? Answer: ")
print("DONE")
