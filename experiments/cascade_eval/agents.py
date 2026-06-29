"""Agent interface and backends for the cascade evaluation harness.

Provides:
- AgentResponse / EpistemicMode data types
- MockAgent: deterministic, parameterized by rho and v (Section 4 model)
- APIAgent: optional pluggable backend for real LLMs (never required for default runs)
"""

from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import yaml


class EpistemicMode(str, Enum):
    DIRECT = "direct"
    CAUTIOUS = "cautious"
    ABSTAIN = "abstain"
    REJECT_PREMISE = "reject_premise"
    REQUEST_TOOL = "request_tool"
    HYPOTHESIS = "hypothesis"


@dataclass
class AgentResponse:
    text: str
    asserts_claim: bool  # True if the agent asserts the (false) claim as fact
    confidence: float  # [0, 1], calibrated confidence in its own answer
    mode: EpistemicMode


class Agent(ABC):
    """Interface for all agent backends."""

    @abstractmethod
    def respond(self, claim: dict, context: list[AgentResponse]) -> AgentResponse:
        """Produce a response to a claim given prior context.

        Args:
            claim: A dict from claims.yaml with keys id, claim, domain,
                   ground_truth, correction_hint, source_type.
            context: Ordered list of prior AgentResponses in the cascade.

        Returns:
            An AgentResponse.
        """
        ...


def load_claims(path: str) -> list[dict]:
    """Load claims from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


class MockAgent(Agent):
    """Deterministic mock agent implementing the paper's propagation model.

    Parameters (per Section 4 of the position paper):
        rho: correlation parameter — probability of copying peer assertions.
        v: verification gain — probability of demoting the claim on review
           when given the correction_hint. Must be > 0.
        base_error_prob: baseline probability of asserting a false claim
                         without peer influence (default 0.5).
        rng: a random.Random instance for reproducibility.
    """

    def __init__(
        self,
        rho: float = 0.5,
        v: float = 0.3,
        base_error_prob: float = 0.5,
        rng: Optional[random.Random] = None,
    ) -> None:
        if v <= 0:
            raise ValueError("Verification gain v must be strictly > 0.")
        self.rho = rho
        self.v = v
        self.base_error_prob = base_error_prob
        self.rng = rng or random.Random()

    def respond(self, claim: dict, context: list[AgentResponse]) -> AgentResponse:
        # Step 1: base error roll — does this agent initially tend to assert?
        asserts = self.rng.random() < self.base_error_prob

        # Step 2: peer influence (rho) — if peers assert, agent may copy.
        if context:
            peer_assert_count = sum(1 for r in context if r.asserts_claim)
            peer_assert_frac = peer_assert_count / len(context)
            if peer_assert_frac > 0 and self.rng.random() < self.rho * peer_assert_frac:
                asserts = True

        # Step 3: verification (v) — agent may demote the claim on review.
        if asserts and self.rng.random() < self.v:
            asserts = False

        # Step 4: produce response based on assertion outcome.
        if asserts:
            confidence = 0.5 + self.rng.random() * 0.4  # 0.5–0.9
            return AgentResponse(
                text=f"Claim asserted: {claim['claim']}",
                asserts_claim=True,
                confidence=confidence,
                mode=EpistemicMode.DIRECT,
            )
        else:
            # Pick a non-assertive mode based on rolls.
            mode_roll = self.rng.random()
            if mode_roll < 0.33:
                mode = EpistemicMode.ABSTAIN
                text = "I cannot verify this claim with sufficient confidence."
            elif mode_roll < 0.66:
                mode = EpistemicMode.REJECT_PREMISE
                text = f"The premise is flawed. {claim['correction_hint']}"
            else:
                mode = EpistemicMode.REQUEST_TOOL
                text = "I would need to consult a reference to verify this."
            confidence = self.rng.random() * 0.4  # 0.0–0.4
            return AgentResponse(
                text=text,
                asserts_claim=False,
                confidence=confidence,
                mode=mode,
            )


class APIAgent(Agent):
    """Optional pluggable backend for real LLMs.

    Reads model name and API key from environment variables:
        DIOGENES_MODEL: model identifier (e.g. "gpt-4", "claude-3-opus")
        DIOGENES_API_KEY: API key for the model provider
        DIOGENES_API_BASE: (optional) base URL for the API endpoint

    This backend is never required for a default run. It exists so that
    users can plug in a real model and evaluate it with the same harness.
    """

    def __init__(self) -> None:
        self.model = os.environ.get("DIOGENES_MODEL", "")
        self.api_key = os.environ.get("DIOGENES_API_KEY", "")
        self.api_base = os.environ.get("DIOGENES_API_BASE", "")
        if not self.model or not self.api_key:
            raise RuntimeError(
                "APIAgent requires DIOGENES_MODEL and DIOGENES_API_KEY "
                "environment variables to be set."
            )

    def respond(self, claim: dict, context: list[AgentResponse]) -> AgentResponse:
        # Build prompt from claim and context.
        context_lines = []
        for i, r in enumerate(context):
            context_lines.append(
                f"Agent {i + 1}: {r.text} "
                f"[asserts={r.asserts_claim}, confidence={r.confidence:.2f}, "
                f"mode={r.mode.value}]"
            )
        context_str = "\n".join(context_lines) if context_lines else "(no prior context)"

        prompt = (
            f"You are evaluating the following claim:\n"
            f'"{claim["claim"]}"\n\n'
            f"Domain: {claim['domain']}\n"
            f"Additional context: {claim['correction_hint']}\n\n"
            f"Prior agent responses:\n{context_str}\n\n"
            f"Respond with your assessment. Format your response as JSON:\n"
            f'{{"text": "...", "asserts_claim": true/false, '
            f'"confidence": 0.0-1.0, '
            f'"mode": "direct|cautious|abstain|reject_premise|request_tool|hypoth"}}\n'
        )

        # Import here to avoid hard dependency on openai/requests.
        try:
            import requests  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "APIAgent requires the 'requests' package. "
                "Install it or use MockAgent instead."
            ) from exc

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        base = self.api_base or "https://api.openai.com/v1"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        resp = requests.post(
            f"{base}/chat/completions", headers=headers, json=payload, timeout=60
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]

        # Parse JSON from response (best-effort).
        import json

        try:
            # Try to extract JSON block from response.
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # Fallback: try to find JSON in the response text.
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(content[start:end])
            else:
                parsed = {
                    "text": content,
                    "asserts_claim": False,
                    "confidence": 0.0,
                    "mode": "abstain",
                }

        mode_str = parsed.get("mode", "abstain")
        try:
            mode = EpistemicMode(mode_str)
        except ValueError:
            mode = EpistemicMode.CAUTIOUS

        return AgentResponse(
            text=parsed.get("text", content),
            asserts_claim=bool(parsed.get("asserts_claim", False)),
            confidence=float(parsed.get("confidence", 0.0)),
            mode=mode,
        )
