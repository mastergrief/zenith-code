"""Substrate Card abstraction — the v3 CHRLM composition primitive.

A card is a substrate-compliant unit of capability. Compiled programs,
trained models (SubstrateLM, SubstrateHRM, SubstrateHRLM), and external
LLM clients (Gemma) all implement the `Card` protocol so the orchestrator
can route uniformly.

This is the composition mechanism for CHRLM builds: a build is just a
`CardOrchestrator` holding an ordered list of cards. At query time the
orchestrator tries cards in priority order and returns the first one
that can produce a useful answer.

Naming convention (see `.claude/CLAUDE.md`):
  substrate     = architectural standard (Small2DTransformer + protocols)
  card          = individual .pt file compliant with the spec
  build         = curated set of cards orchestrated together (CHRLM is one)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Tuple


# ----- Card protocol -----

class Card(Protocol):
    """Uniform interface every substrate card exposes to the orchestrator."""
    name: str
    priority: int   # lower = tried first (compiled exact > HRM > trained > external)

    def applies_to(self, query: str, context: dict) -> bool:
        """Return True if this card might handle `query`. Fast check (regex,
        mode token, trigger predicate) — don't do heavy work here."""
        ...

    def invoke(self, query: str, context: dict) -> Optional[str]:
        """Produce an answer, or None if the card decides it can't handle
        this query after all (fall through to next card)."""
        ...


# ----- CompiledProgramCard -----

@dataclass
class CompiledProgramCard:
    """Wraps a compiled gate-graph program triggered by a regex pattern.

    The `evaluator` callable takes the regex match object and returns the
    exact answer (via LLM-Computer interpreter or a direct safe_eval call).
    Compiled programs run via Python execution of their gate-graph IR, not
    by instantiating the Small2DTransformer weights at orchestration time —
    the weights exist as a separate .pt file that compile_program would
    build on demand.
    """
    name: str
    pattern: re.Pattern
    evaluator: Callable[[re.Match], str]
    priority: int = 100

    def applies_to(self, query: str, context: dict) -> bool:
        return bool(self.pattern.search(query))

    def invoke(self, query: str, context: dict) -> Optional[str]:
        match = self.pattern.search(query)
        if match is None:
            return None
        try:
            return str(self.evaluator(match))
        except Exception:
            return None


# ----- SubstrateLMCard -----

@dataclass
class SubstrateLMCard:
    """Wraps a trained SubstrateLM / SubstrateHRLM checkpoint.

    Lazy-loads the checkpoint + tokenizer on first invocation. Once loaded,
    kept in memory for subsequent calls. The model responds to any query
    (so `applies_to` returns True) — use priority to ensure the trained
    card is a fallback AFTER compiled programs and HRM specialists.
    """
    name: str
    checkpoint_path: str
    tokenizer_path: str
    priority: int = 500
    max_new_tokens: int = 200
    temperature: float = 0.7
    top_k: int = 40
    device: str = "cpu"
    # Optional mode token prefix ("<|lm|>", "<|hrm|>") — relevant for
    # hybrid SubstrateHRLM cards that respond differently per mode.
    mode_prefix: str = ""

    def __post_init__(self):
        self._model = None
        self._tok = None

    def _lazy_load(self):
        if self._model is not None:
            return
        import torch
        from tokenizers import Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        # Import locally to avoid a hard dep on substrate_lm at card import.
        from calm.llm_computer.substrate_lm import (
            SubstrateLMConfig, build_substrate_lm,
        )

        ckpt = torch.load(self.checkpoint_path, map_location=self.device,
                          weights_only=False)
        cfg = SubstrateLMConfig(**ckpt["config"])
        model = build_substrate_lm(cfg).to(self.device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        tok = Tokenizer.from_file(self.tokenizer_path)
        tok.decoder = ByteLevelDecoder()
        self._model = model
        self._tok = tok

    def applies_to(self, query: str, context: dict) -> bool:
        # Trained brain is the universal fallback; applies to anything.
        return True

    def invoke(self, query: str, context: dict) -> Optional[str]:
        from calm.llm_computer.substrate_lm import generate
        self._lazy_load()
        prompt = f"{self.mode_prefix}{query}" if self.mode_prefix else query
        try:
            return generate(
                self._model, self._tok, prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
            )
        except Exception:
            return None


# ----- ExternalLLMCard -----

@dataclass
class ExternalLLMCard:
    """Wraps an external chat-completions HTTP endpoint (llama-server /
    OpenAI-compatible). Applies when no substrate-native card handles
    the query. Useful while SubstrateLM is too weak to be a brain.

    Checks endpoint health on construction; `applies_to` short-circuits
    to False if the endpoint is offline at query time.
    """
    name: str
    endpoint: str = "http://localhost:8080/v1/chat/completions"
    health_endpoint: str = "http://localhost:8080/health"
    priority: int = 900
    system_prompt: str = "You are a helpful assistant."
    timeout: float = 120.0

    def _is_up(self) -> bool:
        import urllib.error
        import urllib.request
        try:
            req = urllib.request.Request(self.health_endpoint)
            with urllib.request.urlopen(req, timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    def applies_to(self, query: str, context: dict) -> bool:
        return self._is_up()

    def invoke(self, query: str, context: dict) -> Optional[str]:
        import json
        import urllib.request
        body = json.dumps({
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query},
            ],
            "temperature": 0.7,
            "max_tokens": 512,
            "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint, data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                resp = json.loads(r.read().decode("utf-8"))
            return resp["choices"][0]["message"]["content"]
        except Exception:
            return None


# ----- Orchestrator -----

@dataclass
class RouteResult:
    """What a routing attempt produced."""
    answer: Optional[str]
    card: Optional[str]           # which card responded, None if none matched
    tried: list[str] = field(default_factory=list)  # card names attempted


@dataclass
class CardOrchestrator:
    """Holds an ordered list of cards and routes queries through them.

    Cards are tried in ascending priority order. First card where
    `applies_to` returns True AND `invoke` returns non-None wins.
    """
    cards: list  # list[Card] but Protocol in dataclass is awkward
    name: str = "CHRLM"

    def __post_init__(self):
        # Sort once by priority so routing is O(n) not O(n log n) per call.
        self.cards = sorted(self.cards, key=lambda c: c.priority)

    def route(self, query: str, context: Optional[dict] = None) -> RouteResult:
        context = context or {}
        tried = []
        for card in self.cards:
            if not card.applies_to(query, context):
                continue
            tried.append(card.name)
            answer = card.invoke(query, context)
            if answer is not None:
                return RouteResult(answer=answer, card=card.name, tried=tried)
        return RouteResult(answer=None, card=None, tried=tried)
