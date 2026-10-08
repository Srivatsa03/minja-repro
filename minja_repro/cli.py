"""Command line interface."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .backends import OPENAI_COMPATIBLE, BackendError, list_models, resolve
from .dataset import SYNTHETIC, load_mmlu_csv, spec_for, victim_queries
from .guard import format_guard_test, run_guard_test
from .evaluate import (
    Experiment,
    format_experiment,
    interval_for_published_rate,
    pooled_interval_for_published_rate,
    run_pair,
)
from .models import MockModel
from .retriever import LexicalRetriever

#: Table 1 of arXiv:2503.03704. Attack queries per victim-target pair: 10 on MMLU, 15 elsewhere.
PUBLISHED = (
    ("EHRAgent MIMIC-III", "GPT-4", 95.6, 57.0, 15),
    ("EHRAgent eICU", "GPT-4", 98.5, 90.0, 15),
    ("RAP Webshop", "GPT-4", 96.3, 77.4, 15),
    ("RAP Webshop", "GPT-4o", 99.3, 98.9, 15),
    ("QA agent MMLU", "GPT-4 / GPT-4o", 100.0, 68.9, 10),
)


def cmd_reanalyze(args: argparse.Namespace) -> int:
    print("Re-analysis of the published rates in arXiv:2503.03704, Table 1.")
    print("No rerun. Each reported percentage is converted back to a success count and")
    print("the Wilson interval that count supports is shown.\n")
    print(f"{'row':22} {'backbone':16} {'n':>3}  {'ISR 95% CI':>20}  {'ASR 95% CI':>20}")
    for row, backbone, isr, asr, n in PUBLISHED:
        _, ilo, ihi = interval_for_published_rate(isr, n)
        _, alo, ahi = interval_for_published_rate(asr, n)
        print(
            f"{row:22} {backbone:16} {n:>3}  "
            f"{isr:5.1f} [{ilo * 100:4.1f},{ihi * 100:5.1f}]  "
            f"{asr:5.1f} [{alo * 100:4.1f},{ahi * 100:5.1f}]"
        )
    print("\nThose are per-pair intervals: what one victim-target pair's rate rests on.")
    if args.pairs:
        print(f"\nPooled over {args.pairs} pairs, if that is the pair count:")
        for row, backbone, isr, asr, n in PUBLISHED:
            _, alo, ahi = pooled_interval_for_published_rate(asr, n, args.pairs)
            print(f"  {row:22} ASR {asr:5.1f} -> [{alo * 100:4.1f},{ahi * 100:5.1f}]  (n={n * args.pairs})")
    print("\nThe reported +/- in the paper is a standard deviation between pairs, which is")
    print("neither of these. A pooled claim needs the pair count read from the paper.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    questions = (
        load_mmlu_csv(Path(args.mmlu), limit=args.limit)
        if args.mmlu
        else list(SYNTHETIC)[: args.limit]
    )
    if not questions:
        print("no questions loaded")
        return 2

    mock = args.backend == "mock"
    if mock:
        model = MockModel()
        for q in questions:
            model.ground_truth(q.question, q.correct)
            for probe in victim_queries(q, args.probes):
                model.ground_truth(probe, q.correct)
    else:
        try:
            model = resolve(args.backend)
        except BackendError as exc:
            print(f"backend unavailable: {exc}")
            return 2

    per_pair = args.probes * 2 + args.rounds
    calls = per_pair * len(questions)
    if not mock:
        print(f"about to make ~{calls} calls to {model.name} "
              f"({len(questions)} pairs x {per_pair} per pair)")
        if not args.yes:
            print("re-run with --yes to proceed; nothing has been sent")
            return 0

    retriever = LexicalRetriever()
    exp = Experiment(model_name=model.name, retriever_name=retriever.name, rounds=args.rounds)
    for q in questions:
        exp.outcomes.append(
            run_pair(
                model,
                spec_for(q),
                victim_queries(q, args.probes),
                rounds=args.rounds,
                retriever=retriever,
            )
        )

    print(format_experiment(exp))
    print(f"\nrun date  {datetime.now(timezone.utc).date().isoformat()}")
    if mock:
        print("\nThe model above is a deterministic mock. It follows instructions and")
        print("demonstrations by construction, so this run shows the harness is wired")
        print("correctly and says nothing about whether a real model is vulnerable.")
    else:
        print("\nA hosted model is not a fixed artifact, so quote the model name and the")
        print("run date with any rate from this run. The retriever is lexical, not the")
        print("embedding similarity the paper used.")
        if args.backend.startswith("ollama"):
            print("A small local model is a weaker substitution than the paper's GPT-4-class")
            print("backbones. This answers whether the attack transfers to it, nothing wider.")
    if questions[0].source.startswith("synthetic"):
        print("The questions are synthetic, not MMLU.")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    """Ask the provider what it serves, so the run names a model that exists."""
    try:
        names = list_models(args.provider)
    except BackendError as exc:
        print(f"could not list models: {exc}")
        return 2
    print(f"{args.provider} serves {len(names)} models:")
    for name in names:
        print(f"  {name}")
    return 0


def cmd_guard(args: argparse.Namespace) -> int:
    """Test the paper's claim that MINJA bypasses detection-based moderation."""
    try:
        guard = resolve(args.guard)
    except BackendError as exc:
        print(f"guard unavailable: {exc}")
        return 2
    questions = list(SYNTHETIC)[: args.limit]
    specs = [spec_for(q) for q in questions]
    calls = len(specs) * args.rounds + 4
    print(f"about to make {calls} calls to {guard.name}")
    if not args.yes:
        print("re-run with --yes to proceed; nothing has been sent")
        return 0
    scores = run_guard_test(guard, specs, rounds=args.rounds)
    print()
    print(format_guard_test(scores, guard.name))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="minja-repro", description="Reproduction of MINJA, arXiv:2503.03704.")
    p.add_argument("--version", action="version", version=f"minja-repro {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("reanalyze", help="intervals implied by the paper's published rates")
    r.add_argument("--pairs", type=int, default=0, help="pair count, to also show pooled intervals")
    r.set_defaults(func=cmd_reanalyze)

    x = sub.add_parser("run", help="run the attack against a model")
    x.add_argument("--mmlu", help="path to an MMLU subject CSV; omit to use the synthetic set")
    x.add_argument("--limit", type=int, default=4, help="questions to use")
    x.add_argument("--probes", type=int, default=10, help="victim probes per pair, 10 matches the paper")
    x.add_argument("--rounds", type=int, default=5, help="attack queries per pair")
    x.add_argument("--backend", default="mock",
                   help="mock, anthropic:<model>, or ollama:<model>")
    x.add_argument("--yes", action="store_true",
                   help="confirm a real-backend run after seeing the call estimate")
    x.set_defaults(func=cmd_run)

    g = sub.add_parser("guard", help="score MINJA's own queries with an injection detector")
    g.add_argument("--guard", default="groq:meta-llama/llama-prompt-guard-2-86m",
                   help="detector backend")
    g.add_argument("--limit", type=int, default=4, help="victim-target pairs to score")
    g.add_argument("--rounds", type=int, default=5, help="attack rounds per pair")
    g.add_argument("--yes", action="store_true", help="confirm after seeing the call estimate")
    g.set_defaults(func=cmd_guard)

    m = sub.add_parser("models", help="list the models a provider serves")
    m.add_argument("provider", choices=sorted(OPENAI_COMPATIBLE), help="provider to query")
    m.set_defaults(func=cmd_models)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
