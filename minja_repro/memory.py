"""The agent memory bank.

A retrieval-augmented agent keeps past interactions and feeds the most similar
ones back as demonstrations for the next query. That write-back is the whole
attack surface: the agent is the thing that stores the record, so an attacker who
can only send queries can still get content into the bank.

Records are plain data and the bank has no notion of trust, which is faithful to
the systems MINJA targets. Nothing here validates what it stores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .retriever import LexicalRetriever, Retriever


@dataclass
class MemoryRecord:
    """One stored interaction.

    ``reasoning`` is the step sequence the agent produced. It is replayed to later
    queries as a demonstration, which is why planting steps here is more powerful
    than planting an answer.
    """

    query: str
    reasoning: tuple[str, ...]
    answer: str
    origin: str = "agent"

    def as_demonstration(self) -> str:
        steps = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(self.reasoning))
        return f"Q: {self.query}\nSteps:\n{steps}\nA: {self.answer}"

    @property
    def retrieval_text(self) -> str:
        """What the retriever matches against.

        The query is weighted by repetition rather than by a tuned coefficient,
        because the systems MINJA targets index the question far more heavily than
        the stored reasoning, and a reproduction that indexed the whole record
        equally would make the poisoned record easier to retrieve than it is in
        practice.
        """
        return f"{self.query} {self.query} {' '.join(self.reasoning)}"


class MemoryBank:
    def __init__(self, retriever: Retriever | None = None, top_k: int = 3) -> None:
        self.retriever: Retriever = retriever or LexicalRetriever()
        self.top_k = top_k
        self.records: list[MemoryRecord] = []

    def __len__(self) -> int:
        return len(self.records)

    def seed(self, records: Iterable[MemoryRecord]) -> None:
        self.records.extend(records)

    def write(self, record: MemoryRecord) -> None:
        self.records.append(record)

    def retrieve(self, query: str, top_k: int | None = None) -> list[tuple[float, MemoryRecord]]:
        k = self.top_k if top_k is None else top_k
        scored = [
            (self.retriever.score(query, r.retrieval_text), r)
            for r in self.records
        ]
        scored = [(s, r) for s, r in scored if s > 0.0]
        # Ties break toward the earlier record so a later poisoned copy cannot win
        # a tie purely by being appended last.
        scored.sort(key=lambda pair: -pair[0])
        return scored[:k]

    def poisoned(self) -> list[MemoryRecord]:
        return [r for r in self.records if r.origin == "attack"]
