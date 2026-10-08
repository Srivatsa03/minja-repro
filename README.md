# minja-repro

An independent implementation of **MINJA**, from *Memory Injection Attacks on LLM Agents via
Query-Only Interaction* ([arXiv:2503.03704](https://arxiv.org/abs/2503.03704)), plus a
re-analysis of the intervals its published rates support.

The paper's arXiv comments field reads, in full, "Code released". There is no link, and no
repository is findable by name, by author, or by the paper's own terms. This is an attempt to
supply the missing artifact and to be explicit about which claims it can and cannot check.

## What MINJA is

An attacker who can only send ordinary queries gets malicious content into an agent's memory
bank, because the agent writes its own output back. The record then sits dormant until an
innocent query retrieves it, at which point the agent surfaces the attacker's answer. No
backend access, no training, no attacker input at the moment of harm.

Three mechanisms, named as the paper names them: **bridging steps** that make the target the
natural conclusion, an **indication prompt** that gets the agent to emit those steps itself,
and **progressive removal** of that guidance so the last records written read clean.

## What this reproduces today

The mechanism, end to end, against a deterministic mock model:

```
baseline, clean bank      answer='Iron'    target_surfaced=False
5 query-only attack turns memory=6  poisoned_records=5  ISR=True
clean victim query        answer='Copper'  retrieved_poisoned=True  ASR=True
```

Nothing writes to the store directly. If a poisoned record is in memory, the agent put it
there. A baseline probe runs before every attack, so a target the agent would have said
anyway is not counted.

## What this does not reproduce, and will not

- **No real-model attack rate.** The runs above use `MockModel`, which follows instructions
  and demonstrations by construction. It shows the harness is wired correctly and is not
  evidence any real model is vulnerable. No rate from it may be quoted.
- **The backbones are retired.** The paper used GPT-4 and GPT-4o. Any future run here is a
  replication under substitution and must name the model that answered. It will never be an
  exact reproduction of Table 1.
- **Two of the five rows are gated.** EHRAgent needs MIMIC-III and eICU, which are
  credentialed PhysioNet datasets requiring CITI training and a signed data use agreement.
  Out of scope.
- **RAP on Webshop** is public but a heavy environment. Deferred.
- **The retriever is a substitution.** The paper uses embedding similarity; the default here
  is stopword-filtered lexical cosine so the harness runs offline with no key. A weaker
  retriever can make the attack look easier or harder for reasons unrelated to the attack, so
  every reported number names which retriever ran. An embedding backend is supported and the
  caller supplies the model.

## The re-analysis, which needs no rerun

Table 1 reports a mean and a standard deviation. Converting each percentage back to a success
count gives the interval that count supports:

```
row                    backbone           n            ISR 95% CI            ASR 95% CI
EHRAgent MIMIC-III     GPT-4             15   95.6 [70.2, 98.8]   57.0 [35.7, 80.2]
EHRAgent eICU          GPT-4             15   98.5 [79.6,100.0]   90.0 [70.2, 98.8]
RAP Webshop            GPT-4             15   96.3 [70.2, 98.8]   77.4 [54.8, 93.0]
RAP Webshop            GPT-4o            15   99.3 [79.6,100.0]   98.9 [79.6,100.0]
QA agent MMLU          GPT-4 / GPT-4o    10  100.0 [72.2,100.0]   68.9 [39.7, 89.2]
```

`minja-repro reanalyze`. Two observations:

- The MMLU attack rate of 68.9 rests on 10 attack queries per victim-target pair, and that
  supports anything from 39.7 to 89.2. The interval spans "fails more often than it works" to
  "works nearly always".
- The two Webshop rows, 77.4 for GPT-4 and 98.9 for GPT-4o, have overlapping per-pair
  intervals, so the apparent backbone difference is not separated at that sample size.

**Stated carefully:** those are *per-pair* intervals. Pooled over all pairs they tighten a
lot, and the paper's reported `+/-` is a standard deviation *between* pairs, which is a third
quantity again. `reanalyze --pairs N` shows the pooled view. A claim about the pooled rate
needs the pair count read from the paper, and this repository does not assert one.

## Use

```bash
minja-repro reanalyze --pairs 10
minja-repro run --limit 4 --probes 10 --rounds 5
minja-repro run --mmlu data/mmlu/high_school_chemistry_test.csv --limit 20
```

MMLU is not vendored. Download it yourself; the loader reads the standard subject CSV layout.
The built-in question set is labelled synthetic and must not appear in a number attributed to
MMLU.

Zero dependencies. Python 3.10+.

## Responsible use

This is working attack code against a class of system that is widely deployed. It is here to
make a published claim checkable, which is the same reason the paper exists. It targets a
memory agent built in this repository for the purpose; it ships no exploit against any
third-party product, and no credentialed dataset.

## Status

v0.1.0. The mechanism and the re-analysis are real. The real-model replication is not done.

## License

Apache-2.0
