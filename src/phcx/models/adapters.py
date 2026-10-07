"""Pluggable LLM adapters (spec section 3).

Two clearly separated capability classes:

* BLACK-BOX adapters (OpenAI-compatible, Ollama chat, ...):
  generation only.  They NEVER expose hidden states -- PHC-X never pretends
  internal signals exist where they do not.

* WHITE-BOX adapters (local transformers): hidden states, per-layer norms,
  token entropy, logit margins, representation statistics.

No model name is hardcoded as *required*; adapters are registered lazily and
the pipeline degrades gracefully when a backend is absent.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence


class ModelAdapter(ABC):
    name: str = "base"
    whitebox: bool = False

    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def generate(self, prompt: str, history: Optional[Sequence[Dict[str, str]]] = None,
                 max_tokens: int = 256) -> str: ...

    def whitebox_signals(self, prompt: str, history: Optional[Sequence[Dict[str, str]]] = None) -> Dict[str, float]:
        """White-box only; black-box adapters must return {} (never fake it)."""
        return {}


class BlackBoxAdapter(ModelAdapter):
    whitebox = False


class WhiteBoxAdapter(ModelAdapter):
    whitebox = True


# --------------------------------------------------------------------- black box
class OllamaAdapter(BlackBoxAdapter):
    name = "ollama"

    def __init__(self, model: Optional[str] = None, url: Optional[str] = None, timeout: float = 60.0):
        self.model = model or os.environ.get("PHC_OLLAMA_MODEL", "")
        self.url = (url or os.environ.get("PHC_OLLAMA_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.timeout = timeout

    def available(self) -> bool:
        if not self.model:
            return False
        try:
            import httpx

            r = httpx.get(f"{self.url}/api/tags", timeout=3.0)
            if r.status_code != 200:
                return False
            names = [m.get("name", "") for m in r.json().get("models", [])]
            return any(self.model in n or n.startswith(self.model) for n in names)
        except Exception:
            return False

    def generate(self, prompt: str, history: Optional[Sequence[Dict[str, str]]] = None,
                 max_tokens: int = 256) -> str:
        import httpx

        msgs: List[Dict[str, str]] = []
        for h in history or []:
            msgs.append({"role": h.get("role", "user"), "content": h.get("content", "")})
        msgs.append({"role": "user", "content": prompt})
        r = httpx.post(
            f"{self.url}/api/chat",
            json={"model": self.model, "messages": msgs, "stream": False,
                  "options": {"num_predict": max_tokens}},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return str(r.json().get("message", {}).get("content", ""))


class OpenAICompatAdapter(BlackBoxAdapter):
    name = "openai_compat"

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model or os.environ.get("PHC_LLM_MODEL", "")
        self.base_url = (base_url or os.environ.get("PHC_LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def available(self) -> bool:
        return bool(self.base_url and self.model and self.api_key)

    def generate(self, prompt: str, history: Optional[Sequence[Dict[str, str]]] = None,
                 max_tokens: int = 256) -> str:
        import httpx

        msgs = [{"role": h.get("role", "user"), "content": h.get("content", "")} for h in history or []]
        msgs.append({"role": "user", "content": prompt})
        r = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": msgs, "max_tokens": max_tokens},
            timeout=60.0,
        )
        r.raise_for_status()
        return str(r.json()["choices"][0]["message"]["content"])


class ScriptedAdapter(BlackBoxAdapter):
    """Deterministic offline adapter used by tests and demos."""

    name = "scripted"

    def __init__(self, responses: Optional[Dict[str, str]] = None, default: str = ""):
        self.responses = responses or {}
        self.default = default

    def available(self) -> bool:
        return True

    def generate(self, prompt: str, history: Optional[Sequence[Dict[str, str]]] = None,
                 max_tokens: int = 256) -> str:
        for key, val in self.responses.items():
            if key.lower() in (prompt or "").lower():
                return val
        return self.default or "I do not have verified information to answer that."


# ------------------------------------------------------------------ white box
class TransformersLocalAdapter(WhiteBoxAdapter):
    """Local HF model: exposes hidden states / entropy / margins.

    Loaded lazily; requires `transformers` + a locally available model id.
    """

    name = "transformers_local"

    def __init__(self, model_id: Optional[str] = None, device: str = "cpu"):
        self.model_id = model_id or os.environ.get("PHC_LOCAL_MODEL", "")
        self.device = device
        self._tok = None
        self._model = None

    def _load(self) -> bool:
        if self._model is not None:
            return True
        if not self.model_id:
            return False
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.float32)
            self._model.to(self.device).eval()
            return True
        except Exception:
            self._model = None
            return False

    def available(self) -> bool:
        return self._load()

    def generate(self, prompt: str, history: Optional[Sequence[Dict[str, str]]] = None,
                 max_tokens: int = 256) -> str:
        if not self._load():
            return ""
        import torch

        text = self._format(prompt, history)
        ids = self._tok(text, return_tensors="pt")
        with torch.no_grad():
            out = self._model.generate(**ids, max_new_tokens=max_tokens, do_sample=False,
                                       pad_token_id=self._tok.eos_token_id)
        return self._tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

    def _format(self, prompt: str, history: Optional[Sequence[Dict[str, str]]]) -> str:
        parts = []
        for h in history or []:
            parts.append(f"{h.get('role', 'user').upper()}: {h.get('content', '')}")
        parts.append(f"USER: {prompt}")
        parts.append("ASSISTANT:")
        return "\n".join(parts)

    def whitebox_signals(self, prompt: str, history: Optional[Sequence[Dict[str, str]]] = None) -> Dict[str, float]:
        if not self._load():
            return {}
        import math

        import torch

        text = self._format(prompt, history)
        enc = self._tok(text, return_tensors="pt")
        with torch.no_grad():
            out = self._model(**enc, output_hidden_states=True)
        logits = out.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        entropy = float(-(probs * torch.log(probs + 1e-12)).sum())
        top2 = torch.topk(probs, k=2).values
        margin = float(top2[0] - top2[1])
        states = out.hidden_states
        final = states[-1][0, -1, :]
        norms = [float(torch.norm(states[l][0, -1, :])) for l in range(len(states))]
        return {
            "wb_entropy": entropy,
            "wb_entropy_first_token": entropy,
            "wb_logit_margin": margin,
            "wb_max_prob": float(top2[0]),
            "wb_rep_norm": float(torch.norm(final)),
            "wb_hidden_norm_mean": sum(norms) / max(1, len(norms)),
        }


_REGISTRY = {
    "ollama": OllamaAdapter,
    "openai_compat": OpenAICompatAdapter,
    "scripted": ScriptedAdapter,
    "transformers_local": TransformersLocalAdapter,
}


def get_adapter(kind: str = "auto", **kwargs: Any) -> ModelAdapter:
    """Resolve an adapter. ``auto`` prefers white-box local, then ollama, then scripted."""
    if kind != "auto":
        if kind not in _REGISTRY:
            raise ValueError(f"unknown adapter '{kind}'; available: {sorted(_REGISTRY)}")
        return _REGISTRY[kind](**kwargs)
    candidates: List[ModelAdapter] = [TransformersLocalAdapter(), OllamaAdapter(), OpenAICompatAdapter()]
    for c in candidates:
        try:
            if c.available():
                return c
        except Exception:
            continue
    return ScriptedAdapter()


def register_adapter(name: str, cls: type) -> None:
    _REGISTRY[name] = cls
