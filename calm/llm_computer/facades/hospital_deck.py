"""HospitalDeck — verticalized medical substrate composition.

Per `.claude/rules/augmentation_thesis.md` §"Customer verticals = card decks",
each customer substrate = Gemma + their own deck of Tier-2/3 cards. This
module is the canonical hospital example.

Shipped facades in HospitalDeck:
  - Icd10RecallFacade (72,748-code CMS DB) — tier-3 text recall
  - DaysBetween (auto-generated via MetaFacade L2) — date arithmetic for
    treatment / admission / follow-up intervals
  - NumberTheoryFacade — GCD/LCM/mod (dosage divisibility, cycle length)
  - PlannerFacade — auto-dispatcher for all above

Not yet shipped (pending backends or DBs):
  - DrugInteractionFacade — needs drug-interaction DB
  - DosageCalculator — needs weight-adjusted dosing rules
  - ChiefComplaintToIcdFacade — needs symptom→ICD mapping DB
  - LabValueClassifier — needs lab range tables

Commercial context: each HospitalDeck install is ~a day's work (curate DB
+ test cases + A/B vs Gemma baseline). Per §"Factorial scaling per domain"
the marginal cost of the 100th hospital card ≈ the 1st. With MetaFacade L2
any function in safe_eval's registry becomes a shipping facade in minutes.

Usage:
    deck = HospitalDeck(icd10_db_path=Path('.cache/icd10/icd10cm_codes_2022.json'))
    deck.install(gemma, tokenizer)
    r = deck.solve("What is the diagnosis for ICD-10 code E11.9?")
    r = deck.solve("Days between 2024-01-01 and 2024-06-15?")
    # r.facade contains which card dispatched
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from calm.llm_computer.facades.planner import PlannerFacade, PlannerResult


@dataclass
class HospitalDeck:
    """Medical vertical composition of decoded-path facades.

    Wraps PlannerFacade with pre-registered medical-relevant cards and
    an ICD-10 DB. Exposes the PlannerResult shape unchanged.
    """
    device: str = "cuda"
    icd10_db_path: Optional[Path] = None
    _planner: Optional[PlannerFacade] = field(default=None, init=False)

    # Facades most useful in medical workflows. Subset of Planner's full
    # auto-facade catalog — the others (factorial, fibonacci, etc.) don't
    # typically show up in hospital dialogue but remain active via Planner
    # as a superset.
    MEDICAL_RELEVANT_TAGS = {
        "icd10",           # diagnosis lookup
        "days_between",    # admission duration, follow-up intervals
        "number_theory",   # dosage divisibility (GCD / LCM / mod)
        "multi_step",      # generic infix math on lab values
        "base_conv",       # occasionally for encoded identifiers
        "numeric_encode",  # same
        "is_prime",        # (not clinical but kept for audit completeness)
    }

    def build(self) -> PlannerFacade:
        """Instantiate the underlying Planner with auto-facades. Loads
        the ICD-10 DB if a path was supplied."""
        p = PlannerFacade(device=self.device, register_auto=True)
        if self.icd10_db_path is not None:
            p.load_icd10_db(self.icd10_db_path)
        self._planner = p
        return p

    def install(self, gemma, tokenizer) -> None:
        if self._planner is None:
            self.build()
        assert self._planner is not None
        self._planner.install(gemma, tokenizer)

    def detach(self) -> None:
        if self._planner is not None:
            self._planner.detach()

    def solve(self, prompt: str, *, use_bias: bool = True) -> PlannerResult:
        """Route a prompt through the medical substrate deck. Returns a
        PlannerResult whose `.facade` field documents which card fired.
        """
        if self._planner is None:
            raise RuntimeError("HospitalDeck.install(...) must be called first")
        return self._planner.solve(prompt, use_bias=use_bias)

    def classify(self, prompt: str) -> Optional[str]:
        """Pure classification, no Gemma inference."""
        if self._planner is None:
            self.build()
        assert self._planner is not None
        return self._planner.classify(prompt)

    def shipped_cards(self) -> list[str]:
        """Return the list of active card tags. Useful for audit reports."""
        if self._planner is None:
            self.build()
        assert self._planner is not None
        base = ["icd10", "base_conv", "number_theory", "multi_step", "numeric_encode"]
        auto = [t for t, _ in self._planner.auto_facades]
        return base + auto
