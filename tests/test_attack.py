import math

from minja_repro.agent import MemoryAgent
from minja_repro.attack import AttackSpec, probe_victim, run_attack, shortened_indication
from minja_repro.dataset import SYNTHETIC, spec_for, victim_queries
from minja_repro.evaluate import (
    interval_for_published_rate,
    pooled_interval_for_published_rate,
    run_pair,
    wilson,
)
from minja_repro.memory import MemoryBank, MemoryRecord
from minja_repro.models import MockModel
from minja_repro.retriever import LexicalRetriever


def _agent(question):
    model = MockModel()
    model.ground_truth(question.question, question.correct)
    for probe in victim_queries(question, 5):
        model.ground_truth(probe, question.correct)
    return MemoryAgent(model, MemoryBank()), model


def test_retriever_scores_relevant_above_irrelevant():
    r = LexicalRetriever()
    q = "Which metal has atomic number 26?"
    assert r.score(q, "Iron is the metal at atomic number 26.") > r.score(q, "Paris is in France.")


def test_retriever_ignores_stopword_only_overlap():
    r = LexicalRetriever()
    assert r.score("what is the of a", "the a of is what") == 0.0


def test_agent_writes_each_turn_back_to_memory():
    q = SYNTHETIC[0]
    agent, _ = _agent(q)
    assert len(agent.memory) == 0
    agent.ask(q.question)
    assert len(agent.memory) == 1


def test_progressive_removal_keeps_the_bridge_until_the_last_round():
    """Regression: truncating the whole string stripped the bridge and left a bare
    instruction, which inverts the attack."""
    bridge = "Step 1: the sample is reddish. Step 2: reddish metals are copper."
    kept = [
        "reddish metals" in shortened_indication(bridge, "Copper", i, 5) for i in range(5)
    ]
    assert kept[:4] == [True, True, True, True]
    assert kept[4] is False


def test_final_round_sends_no_indication_at_all():
    assert shortened_indication("b", "T", 4, 5) == ""


def test_attack_is_query_only_and_plants_records_the_agent_wrote():
    q = SYNTHETIC[0]
    agent, model = _agent(q)
    spec = spec_for(q)
    before = len(agent.memory)
    result = run_attack(agent, spec, rounds=5)

    # Every new record came from an agent turn, never from a direct write.
    assert len(agent.memory) == before + 5
    assert model.calls == 5
    assert result.injection_succeeded
    assert all(r.origin == "attack" for r in agent.memory.poisoned())


def test_target_does_not_surface_before_the_attack():
    q = SYNTHETIC[0]
    agent, _ = _agent(q)
    hit, turn = probe_victim(agent, q.question, spec_for(q))
    assert not hit
    assert turn.answer == q.correct


def test_benign_query_surfaces_the_target_after_the_attack():
    q = SYNTHETIC[0]
    agent, _ = _agent(q)
    spec = spec_for(q)
    run_attack(agent, spec, rounds=5)
    hit, turn = probe_victim(agent, q.question, spec)
    assert hit
    assert turn.retrieved_poisoned()


def test_probing_does_not_extend_the_attack():
    """The probe must not write back, or measuring would inflate later probes."""
    q = SYNTHETIC[0]
    agent, _ = _agent(q)
    spec = spec_for(q)
    run_attack(agent, spec, rounds=5)
    size = len(agent.memory)
    for probe in victim_queries(q, 5):
        probe_victim(agent, probe, spec)
    assert len(agent.memory) == size


def test_each_pair_gets_a_clean_memory_bank():
    """One pair's poison must not help the next, which would inflate the rate."""
    model = MockModel()
    outcomes = []
    for q in SYNTHETIC[:2]:
        model.ground_truth(q.question, q.correct)
        for probe in victim_queries(q, 3):
            model.ground_truth(probe, q.correct)
        outcomes.append(run_pair(model, spec_for(q), victim_queries(q, 3), rounds=5))
    assert all(o.baseline_hits == 0 for o in outcomes)


def test_wilson_brackets_the_estimate_and_stays_in_range():
    p, lo, hi = wilson(7, 10)
    assert 0.0 <= lo < p < hi <= 1.0


def test_wilson_at_a_boundary_does_not_exceed_one():
    _, _, hi = wilson(10, 10)
    assert hi == 1.0


def test_wilson_on_an_empty_sample_is_not_a_number():
    p, lo, hi = wilson(0, 0)
    assert math.isnan(p) and (lo, hi) == (0.0, 1.0)


def test_published_mmlu_asr_interval_is_very_wide_at_ten_trials():
    _, lo, hi = interval_for_published_rate(68.9, 10)
    assert lo < 0.45 and hi > 0.85


def test_pooling_tightens_the_interval():
    _, lo_one, hi_one = interval_for_published_rate(68.9, 10)
    _, lo_many, hi_many = pooled_interval_for_published_rate(68.9, 10, 10)
    assert (hi_many - lo_many) < (hi_one - lo_one)


def test_published_webshop_backbone_intervals_overlap():
    """GPT-4 77.4 against GPT-4o 98.9 is not a separated difference at per-pair n."""
    _, lo_4o, _ = interval_for_published_rate(98.9, 15)
    _, _, hi_4 = interval_for_published_rate(77.4, 15)
    assert lo_4o < hi_4
