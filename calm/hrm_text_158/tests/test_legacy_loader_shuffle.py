"""Focused test for the diagnostic --legacy-loader-shuffle flag (codex msg
1779652915624 constraint 2).

Proves the flag changes the DataLoader construction/order path as intended:
- default (False): explicit Generator seeded by --seed → order is seed-pinned
  and DECOUPLED from global RNG state (the post-1656ead path).
- diagnostic (True): NO explicit generator → order FOLLOWS global RNG
  (pre-1656ead path).

No GPU / no model load. Tests `_build_train_loader` directly.
"""
import importlib.util
import os
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_spec = importlib.util.spec_from_file_location(
    "_train_hrm_text_158", os.path.join(_REPO, "scripts", "train_hrm_text_158.py")
)
_thr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_thr)

_build = _thr._build_train_loader


class _DummyDS(torch.utils.data.Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


def _order(loader):
    return [int(x) for batch in loader for x in batch]


def _loader(legacy, seed=17, n=40, bs=4):
    return _build(_DummyDS(n), bs, seed, legacy, collate_fn=list)


# --------------------------------------------------------------------------- #
# Construction path: generator present (default) vs absent (legacy)
# --------------------------------------------------------------------------- #

def test_default_attaches_explicit_seeded_generator():
    loader = _loader(legacy=False, seed=17)
    assert loader.generator is not None, "default must use an explicit generator"
    assert loader.generator.initial_seed() == 17, "generator must be seeded by --seed"


def test_legacy_uses_no_explicit_generator():
    loader = _loader(legacy=True)
    assert loader.generator is None, "legacy path must NOT pass an explicit generator"


# --------------------------------------------------------------------------- #
# Order semantics: default is seed-pinned/global-RNG-decoupled; legacy follows global RNG
# --------------------------------------------------------------------------- #

def test_default_order_is_seed_pinned_and_global_rng_decoupled():
    # Same seed under DIFFERENT global RNG states must yield the SAME order.
    torch.manual_seed(999)
    o1 = _order(_loader(legacy=False, seed=17))
    torch.manual_seed(123456)
    o2 = _order(_loader(legacy=False, seed=17))
    assert o1 == o2, "explicit-generator order must be seed-pinned, not global-RNG dependent"
    # Different seed → different order (the generator actually drives the shuffle).
    o3 = _order(_loader(legacy=False, seed=18))
    assert o3 != o1, "different --seed must change the explicit-generator order"
    # And it is a real shuffle (not identity).
    assert o1 != list(range(40)), "default order should be shuffled, not identity"


def test_legacy_order_follows_global_rng():
    # Legacy (no explicit generator) order depends on global RNG state.
    torch.manual_seed(111)
    a = _order(_loader(legacy=True))
    torch.manual_seed(222)
    b = _order(_loader(legacy=True))
    assert a != b, "legacy order should follow global RNG (different global seed → different order)"
    # Re-seeding global RNG identically reproduces the legacy order.
    torch.manual_seed(111)
    a2 = _order(_loader(legacy=True))
    assert a == a2, "legacy order must be reproducible under identical global RNG state"


def test_legacy_ignores_seed_param_while_default_uses_it():
    # The behavioral crux: legacy follows GLOBAL RNG (ignores the `seed` param),
    # default follows `seed`. With global RNG perturbed away from `seed`, the
    # two paths produce different orders — proving the flag swaps the RNG source.
    torch.manual_seed(999)            # global state != seed (17)
    leg = _order(_loader(legacy=True, seed=17))    # uses global (999-state)
    exp = _order(_loader(legacy=False, seed=17))   # uses explicit gen(17)
    assert leg != exp, "legacy (global-RNG) and explicit-seeded orders should differ"
    # Legacy ignores the seed param: same global state, different `seed` arg → same order.
    torch.manual_seed(555)
    l_a = _order(_loader(legacy=True, seed=1))
    torch.manual_seed(555)
    l_b = _order(_loader(legacy=True, seed=2))
    assert l_a == l_b, "legacy order must be independent of the --seed param (global-RNG driven)"


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"  {_name}: PASS")
    print("legacy-loader-shuffle tests: PASS")
