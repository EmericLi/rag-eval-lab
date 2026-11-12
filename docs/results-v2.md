# Results v2a — Generation and judge calibration on AlloprofRetrieval

## Setup

| | |
|---|---|
| Dataset | `lyon-nlp/alloprof`, the first 30 queries of the v1 sample (seed 42) |
| Contexts | 5: `none` (no passage), `bm25`, `e5-small`, `bge-m3` (the v1 best), `oracle` (labelled documents) |
| Generator | `llama3.2:latest` (3B), temperature 0, `num_ctx` 8192, max 300 tokens |
| Judge | `llama3.1:8B`, temperature 0, `num_ctx` 8192, max 200 tokens |
| Passages | 5 per answer; oracle chunks with `fixed_1000_200`, never ranked against the question |
| Answers | 150 (30 × 5), 0 errors |
| Human labels | 60, balanced across contexts, collected blind to the judge's verdicts |
| Hardware | AMD Ryzen 7 8845HS, CPU only |
| Reproduce | `rel generate configs/v2.yaml` → `rel judge results/v2 configs/v2.yaml` → `rel annotate results/v2 --n 60` → `rel report-gen results/v2` |

## Finding 1 — the free local judge is not usable, and that is measurable

| Criterion | n | Agreement | 95% CI | Cohen's κ | Judge says yes | Human says yes | tp/fp/fn/tn |
|---|---|---|---|---|---|---|---|
| faithfulness | 48 | 0.396 | [0.270, 0.537] | **−0.308** | 72.9% | 58.3% | 17/18/11/2 |
| correctness | 51 | 0.588 | [0.452, 0.712] | 0.249 | 60.8% | 27.5% | 12/19/2/18 |

A negative κ means the judge agrees with a human **less often than chance would**. The confusion
matrix shows why: of the 20 answers a human called unfaithful, `llama3.1:8B` recognised 2. It
says "correct" more than twice as often as a human does (60.8% vs 27.5%). The bias is
systematic and one-directional: this judge rewards fluent answers regardless of grounding.

**Consequence: every number this judge produced is unusable, including the ones in
`results/v2/report-generation.md`.** They are kept in the repository as the evidence for this
finding, not as results.

## Finding 2 — the uncalibrated judge inverts the conclusion

The same 150 answers, scored by the judge and by a human, give opposite rankings.

| Context | Correctness — human | Correctness — judge | Faithfulness — human |
|---|---|---|---|
| **oracle** | **0.58** [0.32, 0.81] | 0.67 | **0.83** [0.55, 0.95] |
| bm25 | 0.30 [0.11, 0.60] | 0.83 | 0.42 [0.19, 0.68] |
| e5-small | 0.20 [0.06, 0.51] | 0.67 | 0.50 [0.25, 0.75] |
| none | 0.11 [0.02, 0.44] | 0.75 | n/a |
| bge-m3 | 0.10 [0.02, 0.40] | 0.42 | 0.58 [0.32, 0.81] |

Read by the judge, the benchmark says *retrieval is useless*: the closed-book context (0.75)
beats every retrieval setup, and perfect retrieval (0.67) beats none of them convincingly.
That is the report a project without human calibration would have published.

Read by a human, the benchmark says the opposite: closed-book is the **worst** setting (0.11),
and perfect retrieval is **five times better** (0.58). Retrieval does help — when it is right.

Significance, Fisher's exact test on the human labels:

- oracle vs `bge-m3` correctness (7/12 vs 1/10): **p = 0.031**
- oracle vs closed-book correctness (7/12 vs 1/9): p = 0.067
- oracle vs `bm25` faithfulness (10/12 vs 5/12): p = 0.089

So the "perfect retrieval beats the alternatives" direction is solid, and the fine ranking
between the three real retrievers is not: with 9–12 labels per context, those differences are
noise. The headline of this version is the judge, not the retriever ranking.

## Finding 3 — a 3B generator fails to use passages it has been given

Reading the 60 annotated answers shows three recurring failure modes, all independent of
retrieval quality:

| Failure | Example | What happened |
|---|---|---|
| **Refusal despite relevant passages** | "Qui étaient Jean Talon, Frontenac et Mgr de Laval ?" | Passages [1] and [2] were the Jean Talon fact sheets; the answer states that "the extracts contain no information on Jean Talon" |
| **Invention despite a correct formula** | "Comment calculer la température avant la réaction d'un mélange ?" | Passage [2] contains `Q = m·c·ΔT`; the answer produced `ΔH = ΔU + pΔV` and `ΔT = ΔH/(nCp)`, neither of which appears anywhere in the context |
| **Contradicting a worked example** | "Convertir 0,15 g/L en % m/V" | Two passages solve the identical exercise by dividing by 10; the answer multiplied by 10 and returned 1.5% instead of 0.015% |

This explains why better retrieval does not translate linearly into better answers here: with
`oracle` passages the model still gets 42% of answers wrong. The bottleneck at this model size
is the generator's ability to exploit its context, not the retriever.

## Limitations

- **9 to 12 human labels per context.** Only the strongest contrast survives a significance
  test; no fine ranking between retrievers should be read from this table.
- **One annotator, no second pass.** Inter-annotator agreement is unknown, so the "human"
  column is one person's judgement, not a gold standard.
- **Reference answers are conversational.** Many Alloprof answers are "here is the fact sheet"
  with a link that the dataset does not carry, so correctness was unjudgeable for 9 of 60
  answers (recorded as null, excluded).
- **One generator, one judge, one prompt per task.** A larger judge (or an API judge) might
  behave differently; that is the natural next experiment and the interface already allows it.
- **`num_ctx` = 8192**: 10 of 200 prompts in the first run still exceeded it. See the incident
  below.
- **Latency was measured on a laptop CPU** and includes one run interrupted by system sleep.

## Incident: a client default that silently changed the measurement

The first complete run (200 answers, archived locally in `results/v2-ctx2048/`) was discarded.
Ollama applies a default context window of 2048 tokens when `num_ctx` is not set, and the
whole-document contexts exceeded it (`bm25` ≈ 6100 tokens, `e5-small` ≈ 3600), so the model
silently received a truncated prompt. The benchmark was measuring a client default rather than
retrieval quality.

The fix sets `num_ctx` explicitly for the generator **and** the judge, records it in every
result record, and includes it in the resume key so a window change can never reuse answers
produced under another window.

## What v2b should test

1. A stronger judge (`llama3.1:70B` through a free API tier, or a calibrated prompt with
   few-shot examples) measured against the same 60 labels — the harness already supports it.
2. A larger generator (8B) on the oracle context, to separate "the model cannot use context"
   from "3B cannot use context".
3. More human labels on the two extreme contexts only (oracle and closed-book), which is where
   the signal is, rather than spreading labels across five contexts.
