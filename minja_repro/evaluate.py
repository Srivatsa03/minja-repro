"""Rates, with intervals.

The paper reports ISR and ASR as a mean and a standard deviation, for example
68.9 +/- 19.1 for ASR on MMLU, over 10 attack queries per victim-target pair. A
standard deviation across pairs is not an interval on a proportion, and at ten
trials per pair the spread is wide enough that the headline figure does not
distinguish between fairly different underlying rates.

So every rate here carries a Wilson score interval, and
:func:`interval_for_published_rate` exists to say what the published numbers
themselves support. That is a re-analysis of the paper's own arithmetic and needs
no rerun, which is worth separating from any replication claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .agent import MemoryAgent
from .attack import AttackSpec, probe_victim, run_attack
from .memory import MemoryBank, MemoryRecord
from .models import Model


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float, float]:
    """Point estimate with a Wilson score interval.

    Wilson rather than the normal approximation because these proportions sit
    near 1 (ISR is reported at 95 to 100) where the normal interval misbehaves
    and can run past 1.
    """
    if total == 0:
        return (float("nan"), 0.0, 1.0)
    p = successes / total
    d = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def interval_for_published_rate(rate_percent: float, trials: int) -> tuple[float, float, float]:
    """The Wilson interval implied by a published rate and its trial count.

    No rerun involved. It converts a reported percentage back to a success count
    and reports the interval that count supports, which is what lets a reader see
    how much of the gap between two published numbers is real.
    """
    successes = round(rate_percent / 100.0 * trials)
    return wilson(successes, trials)


def pooled_interval_for_published_rate(
    rate_percent: float, trials_per_pair: int, pairs: int
) -> tuple[float, float, float]:
    """The interval on a rate pooled over every pair.

    Kept separate from :func:`interval_for_published_rate` because the two answer
    different questions and conflating them is the easiest way to overstate this
    analysis.

    A per-pair interval says how little ten trials pin down one pair's rate. A
    pooled interval over ``trials_per_pair * pairs`` trials is much narrower, and
    is the right object if the claim is about the average across pairs. Neither
    is the paper's reported ``+/-``, which is a standard deviation *between*
    pairs: a measure of how much pairs differ from each other, not of sampling
    error in either quantity.

    So the honest statement is that each pair's rate is imprecise, and that the
    reported spread is consistent with that imprecision. Saying the headline is
    uninformative would need the pair count, which has to come from the paper.
    """
    total = trials_per_pair * pairs
    return wilson(round(rate_percent / 100.0 * total), total)


@dataclass
class PairOutcome:
    spec: AttackSpec
    injected: bool
    victim_hits: int
    victim_trials: int
    baseline_hits: int
    poisoned_records: int

    @property
    def attack_succeeded(self) -> bool:
        return self.victim_hits > 0


@dataclass
class Experiment:
    model_name: str
    retriever_name: str
    rounds: int
    outcomes: list[PairOutcome] = field(default_factory=list)

    @property
    def injection_rate(self) -> tuple[float, float, float]:
        return wilson(sum(1 for o in self.outcomes if o.injected), len(self.outcomes))

    @property
    def attack_rate(self) -> tuple[float, float, float]:
        """ASR over every victim probe, not over pairs.

        Pair-level averaging is what produces a standard deviation across pairs
        and hides how few probes sit under each one.
        """
        hits = sum(o.victim_hits for o in self.outcomes)
        trials = sum(o.victim_trials for o in self.outcomes)
        return wilson(hits, trials)

    @property
    def baseline_rate(self) -> tuple[float, float, float]:
        """How often the target answer appears with no attack at all.

        Without this control an ASR is uninterpretable: a target the agent would
        have said anyway is not an attack.
        """
        hits = sum(o.baseline_hits for o in self.outcomes)
        trials = sum(o.victim_trials for o in self.outcomes)
        return wilson(hits, trials)


def run_pair(
    model: Model,
    spec: AttackSpec,
    victim_queries: Sequence[str],
    *,
    seed_memory: Sequence[MemoryRecord] = (),
    rounds: int = 5,
    retriever=None,
    top_k: int = 3,
) -> PairOutcome:
    """One victim-target pair: baseline probes, attack, then victim probes.

    A fresh bank per pair, so one pair's poison cannot help the next and inflate
    the rate.
    """
    bank = MemoryBank(retriever=retriever, top_k=top_k)
    bank.seed(list(seed_memory))
    agent = MemoryAgent(model, bank)

    baseline_hits = sum(1 for q in victim_queries if probe_victim(agent, q, spec)[0])
    result = run_attack(agent, spec, rounds=rounds)
    victim_hits = sum(1 for q in victim_queries if probe_victim(agent, q, spec)[0])

    return PairOutcome(
        spec=spec,
        injected=result.injection_succeeded,
        victim_hits=victim_hits,
        victim_trials=len(victim_queries),
        baseline_hits=baseline_hits,
        poisoned_records=result.poisoned_records,
    )


def format_experiment(exp: Experiment) -> str:
    def pct(t: tuple[float, float, float]) -> str:
        p, lo, hi = t
        if math.isnan(p):
            return "n/a"
        return f"{p * 100:.1f}% (95% CI {lo * 100:.1f} to {hi * 100:.1f})"

    probes = sum(o.victim_trials for o in exp.outcomes)
    lines = [
        f"model      {exp.model_name}",
        f"retriever  {exp.retriever_name}",
        f"pairs      {len(exp.outcomes)}   attack rounds per pair {exp.rounds}   victim probes {probes}",
        "",
        f"  injection success rate  {pct(exp.injection_rate)}",
        f"  attack success rate     {pct(exp.attack_rate)}",
        f"  baseline (no attack)    {pct(exp.baseline_rate)}",
    ]
    if any(o.baseline_hits for o in exp.outcomes):
        lines.append(
            "  NOTE: the target appears without any attack, so the attack rate above is"
            " not attributable to the attack alone"
        )
    return "\n".join(lines)
