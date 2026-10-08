# Reproduction project: MINJA

Researched 2026-10-08. Status: CANDIDATE CHOSEN, SCOPE NEEDS HIS SIGN-OFF.

## The paper

**Memory Injection Attacks on LLM Agents via Query-Only Interaction**
arXiv:2503.03704 (cs.LG), March 2025.

An attacker poisons an LLM agent's **memory bank** using nothing but ordinary queries. No
backend access, no training, no privileged position. The agent itself is induced to write
the malicious record, which lies dormant until a victim query retrieves it. Mechanism is
three parts: *bridging steps* that connect a victim query to harmful reasoning, an
*indication prompt* that makes the agent generate those steps itself, and *progressive
removal* of the guidance so the stored record looks clean.

## Why this one, out of everything checked

**Its arXiv comments field says, verbatim, "Code released". Two words. No URL.** There is
no repository anywhere: searched by name, by author, and by the paper's distinctive terms
("bridging step", "indication prompt", "memory injection query-only"). The only `minja`
repos on GitHub are an unrelated C++ Jinja implementation (google/minja) and two Keyboard
Maestro galleries. So the paper claims released code and provides no way to find it.

That is a better story than a paper that never promised code. The project writes itself:
the claim is public, the artifact is not, here is a working implementation and here is
exactly which numbers survive.

Everything else in the lane already has code and was ruled out: PoisonedRAG
(sleeepeer/PoisonedRAG, 304 stars), BIPIA (microsoft/BIPIA, 160), Agent Security Bench
(agiresearch/ASB, 310), AgentPoison (AI-secure/AgentPoison, 244), LongPIBench
(runchu-tian/LongPiBench, 14), MPIB (jhlee0619/mpib-eval). Second-best no-code candidate
was "Defending Against Prompt Injection With a Few Defensive Tokens" (ACM,
10.1145/3733799.3762982), dropped because reproducing a *defense* needs model internals or
fine-tuning, which the 8 GB M1 cannot host.

It also sits exactly between his two existing assets: rag-redteam tests retrieval poisoning,
mcp-snitch watches cross-call agent sessions, and MINJA is memory poisoning that only works
across a session. And it is already cited as a threat-model reference in NVIDIA garak issue
#1950, so it is recognised in the serious tooling conversation.

## The numbers to reproduce

Table 1. ISR is injection success, ASR is end-to-end attack success.

| agent / dataset | backbone | ISR | ASR |
|---|---|---|---|
| EHRAgent / MIMIC-III | GPT-4 | 95.6 ± 7.0 | 57.0 ± 10.3 |
| EHRAgent / eICU | GPT-4 | 98.5 ± 2.8 | 90.0 ± 3.5 |
| RAP / Webshop | GPT-4 | 96.3 ± 4.6 | 77.4 ± 14.5 |
| RAP / Webshop | GPT-4o | 99.3 ± 2.1 | 98.9 ± 2.2 |
| QA agent / MMLU | GPT-4 and GPT-4o | 100.0 ± 0.0 | 68.9 ± 19.1 |

Attack budget: 10 attack queries per victim-target pair on MMLU, 15 on the other three.
Progressive shortening applied 4, 5, 5 and 5 times by pair type.

## Three honest obstacles, found before starting rather than after

1. **The backbones are retired.** GPT-4 and GPT-4o are both gone by late 2026. This cannot
   be an exact reproduction, it is a replication under substitution. That is arguably the
   more interesting question anyway: does a query-only memory attack still land on 2026
   models? But the write-up must never claim to have reproduced the original figures.
2. **Two of the five rows are gated.** EHRAgent needs MIMIC-III and eICU, which are
   credentialed PhysioNet datasets requiring CITI training and a signed data use agreement.
   That is a weeks-long approval, not an afternoon. Do not scope those into v1.
3. **RAP on Webshop** is public but a heavy environment to stand up.

## Where the real contribution is

Not in re-running the attack. In the error bars.

**ASR on MMLU is 68.9 ± 19.1, from 10 attack queries per victim-target pair.** That is a
very small sample carrying a very wide spread, and the paper reports it as a point estimate
with a standard deviation rather than an interval on a proportion. Ten trials cannot
separate 68.9 from roughly 40 or from roughly 90. The same Wilson-interval treatment he
applied to rag-redteam v0.7, where he retracted three of his own claims after Fisher's
exact test, applied here would say precisely how much the headline supports.

So the deliverable is two things: the missing implementation, and an honest re-statement of
what the sample size licenses. A replication that lands *and* shows the published interval
is wider than advertised is a stronger artifact than either half alone.

## Proposed scope for v1

Start with the **MMLU QA agent** row only. Fully public, cheapest to stand up, no
credentialing, and it is the row whose statistics are weakest and therefore most worth
redoing. Defer Webshop to v2 and EHRAgent to never unless PhysioNet access arrives.

Deliverable: a working query-only memory-injection implementation, a replication of the
MMLU row against current models, Wilson intervals on every rate, and a plain statement of
which original claims survive substitution and which cannot be checked.

## Risks

- A clean replication is the *less* interesting outcome and still publishable; a fragile one
  is better. Either way the implementation is the artifact
- API cost is the real budget line, not compute. Query-only means many calls
- Reproducing an attack paper means publishing working attack code. Needs a responsible
  release decision up front: which agent frameworks, what gets withheld, and a SECURITY.md
  before anything goes public
