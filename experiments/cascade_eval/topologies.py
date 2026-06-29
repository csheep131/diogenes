"""Cascade topologies for the evaluation harness.

Two topologies are implemented:
- sequential_refinement: agents answer in sequence, each seeing only the
  prior agent's (possibly quarantined) output.
- shared_recursive_context: all agents read a shared, repeatedly-updated
  history over multiple rounds.

Both topologies support an optional quarantine rule: responses with
confidence below tau or a non-assertive mode are excluded from the
shared context as asserted facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agents import Agent, AgentResponse, EpistemicMode

# Modes that are non-assertive — these should not enter context as facts
# when quarantine is enabled.
_NON_ASSERTIVE_MODES = frozenset(
    {EpistemicMode.ABSTAIN, EpistemicMode.REQUEST_TOOL, EpistemicMode.REJECT_PREMISE}
)


@dataclass
class Round:
    """One round of interaction: a list of agent responses."""

    responses: list[AgentResponse] = field(default_factory=list)


def _is_quarantined(response: AgentResponse, tau: float) -> bool:
    """Check whether a response should be quarantined from shared context.

    A response is quarantined if its confidence is below tau OR its mode
    is non-assertive (ABSTAIN, REQUEST_TOOL, REJECT_PREMISE).
    """
    if response.confidence < tau:
        return True
    if response.mode in _NON_ASSERTIVE_MODES:
        return True
    return False


def sequential_refinement(
    agents: list[Agent],
    claim: dict,
    quarantine: bool = False,
    tau: float = 0.5,
) -> list[Round]:
    """Sequential refinement topology.

    Agent 1 answers; agent 2 reviews and may demote; agent 3 finalizes.
    Each agent sees only the prior agent's (possibly quarantined) output.

    Args:
        agents: List of agents (typically 3).
        claim: Claim dict from claims.yaml.
        quarantine: If True, apply quarantine rule to context.
        tau: Confidence threshold for quarantine.

    Returns:
        A list of Rounds (one round with one response per agent).
    """
    responses: list[AgentResponse] = []
    for agent in agents:
        # Build context for this agent: only the immediately prior response,
        # subject to quarantine.
        if responses:
            prior = responses[-1]
            if quarantine and _is_quarantined(prior, tau):
                # Quarantined: pass as tagged hypothesis, not as fact.
                context: list[AgentResponse] = []
            else:
                context = [prior]
        else:
            context = []

        response = agent.respond(claim, context)
        responses.append(response)

    return [Round(responses=responses)]


def shared_recursive_context(
    agents: list[Agent],
    claim: dict,
    rounds: int = 4,
    quarantine: bool = False,
    tau: float = 0.5,
) -> list[Round]:
    """Shared recursive context topology.

    All agents read one shared, repeatedly-updated history over multiple
    rounds. This is the reuse structure expected to amplify correlation.

    Args:
        agents: List of agents.
        claim: Claim dict from claims.yaml.
        rounds: Number of rounds (each round = each agent responds once).
        quarantine: If True, apply quarantine rule to context.
        tau: Confidence threshold for quarantine.

    Returns:
        A list of Rounds (one per round).
    """
    all_responses: list[AgentResponse] = []
    round_results: list[Round] = []

    for _round_idx in range(rounds):
        round_responses: list[AgentResponse] = []
        for agent in agents:
            # Build context: all prior responses, with quarantine applied.
            if quarantine:
                context = [
                    r for r in all_responses if not _is_quarantined(r, tau)
                ]
            else:
                context = list(all_responses)

            response = agent.respond(claim, context)
            round_responses.append(response)
            all_responses.append(response)

        round_results.append(Round(responses=round_responses))

    return round_results
