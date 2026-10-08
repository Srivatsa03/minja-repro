"""Model backends.

The paper ran GPT-4 and GPT-4o. Both are retired, so any run here is a
replication under substitution and must say which model answered.

``MockModel`` exists so the harness, the attack loop and the metrics can be
tested end to end with no network and no spend. It is not a stand-in for a real
model in any reported number: it follows planted demonstrations by construction,
so it can show that the plumbing works and never that an attack works.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Protocol


class Model(Protocol):
    def complete(self, prompt: str) -> str: ...

    @property
    def name(self) -> str: ...


@dataclass
class MockModel:
    """Deterministic stand-in for an instruction-following, demonstration-following model.

    It needs both behaviours, because the attack uses them in sequence: early
    rounds rely on the model obeying an in-prompt instruction, and the final
    probe relies on it reusing a retrieved demonstration once that instruction is
    gone. A mock that only did the second cannot plant anything, and a mock that
    only did the first cannot show persistence.

    Precedence is instruction first, then demonstration, then ground truth. That
    ordering is the attack's own logic, which is the point: a success here shows
    the harness wires retrieval and write-back together correctly. It is not
    evidence that any real model behaves this way, and no reported attack rate
    may come from this class.
    """

    name: str = "mock-instruction-and-demo-follower"
    calls: int = 0
    _answers: dict[str, str] = field(default_factory=dict)

    def ground_truth(self, query: str, answer: str) -> None:
        """Teach the mock the correct answer for a query, used for clean runs."""
        self._answers[query.strip()] = answer

    def complete(self, prompt: str) -> str:
        self.calls += 1
        question = ""
        m = re.search(r"Question:\s*(.+)", prompt)
        if m:
            question = m.group(1).strip()

        # An explicit instruction in the live query wins, as it would for a model
        # that follows instructions. Taken from the question line only, so a
        # recalled demonstration cannot re-trigger it.
        instructed = re.search(r"Final answer:\s*([^.\n]+)", question)
        if instructed:
            steps = re.findall(r"(Step \d+:[^.]*\.)", question)
            rendered = "\n".join(f"  {i + 1}. {s.strip()}" for i, s in enumerate(steps)) or "  1. Following the given reasoning."
            return f"Steps:\n{rendered}\nA: {instructed.group(1).strip()}"

        demos = re.findall(r"^A: (.+)$", prompt, re.MULTILINE)
        if demos:
            return f"Steps:\n  1. Recalling a similar case.\nA: {demos[-1]}"

        truth = self._answers.get(question, "unknown")
        return f"Steps:\n  1. Answering directly.\nA: {truth}"


class CallableModel:
    """Wrap any ``str -> str`` function, so a real backend stays the caller's business."""

    def __init__(self, fn: Callable[[str], str], name: str) -> None:
        self._fn = fn
        self._name = name
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    def complete(self, prompt: str) -> str:
        self.calls += 1
        return self._fn(prompt)
