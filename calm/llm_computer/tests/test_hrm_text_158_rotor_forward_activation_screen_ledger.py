"""CPU-static regression for the phase4a bits-ledger key contract.

Gate-2 audit finding: kv_ternary must emit ternary-named ledger keys, NOT
turbo2/turbo3 (which would mislabel ternary values as 2.125-bpw turbo2).
Pure helper test — no GPU / probe loop.
"""
import importlib.util
import os


def _load_screen():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                        "scripts", "hrm_text_158_rotor_forward_activation_screen.py")
    spec = importlib.util.spec_from_file_location("_rotor_fwd_screen", os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_kv_ternary_emits_ternary_keys_not_turbo():
    m = _load_screen()
    specs = m.ledger_key_specs("kv_ternary")
    assert set(specs) == {"ternary_fp16_scale", "ternary_int8_scale"}
    assert "turbo2" not in specs and "turbo3" not in specs
    assert all(s["family"] == "ternary" for s in specs.values())
    assert specs["ternary_int8_scale"]["kwargs"] == {"scale_dtype": "int8"}
    assert specs["ternary_fp16_scale"]["kwargs"] == {}


def test_non_ternary_surfaces_keep_turbo_keys():
    m = _load_screen()
    for surface in ("kv_turbo", "residual", "activations"):
        specs = m.ledger_key_specs(surface)
        assert set(specs) == {"turbo2", "turbo3"}
        assert specs["turbo2"]["family"] == 2 and specs["turbo3"]["family"] == 3
        assert "ternary_fp16_scale" not in specs
