# Results v2a — Generation and judge calibration on AlloprofRetrieval

## Setup

| | |
|---|---|
| Dataset | `lyon-nlp/alloprof`, the first 30 queries of the v1 sample (seed 42) |
| Contexts | 5: `none` (no passage), `bm25` (the raw v1 tokenizer, not the tuned `bm25-fr` added since), `e5-small`, `bge-m3`, `oracle` (labelled documents) |
| Generator | `llama3.2:latest` (3B), temperature 0, `num_ctx` 8192, max 300 tokens |
| Judge | `llama3.1:8B`, temperature 0, `num_ctx` 8192, max 200 tokens |
| Passages | 5 per answer; `oracle` takes the first `fixed_1000_200` chunks of the labelled documents, never ranked against the question |
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

**Is this an artefact of how refusals were labelled? Partly, and not enough to change the
verdict.** 29 of the 150 answers are refusals ("the extracts do not allow me to answer"). The
annotation convention counted a refusal as faithful — it asserts nothing — and that convention
was not given to the judge. Splitting the 48 faithfulness pairs:

| Faithfulness pairs | n | Agreement | Cohen's κ | tp/fp/fn/tn |
|---|---|---|---|---|
| all | 48 | 0.396 | −0.308 | 17/18/11/2 |
| refusals only | 9 | 0.222 | −0.235 | 2/1/6/0 |
| substantive answers only | 39 | 0.436 | **−0.147** | 15/17/5/2 |

Six of the eleven "judge too severe" disagreements are refusals, so part of the headline κ is
a guideline mismatch rather than a judging error. On substantive answers alone the judge still
agrees with a human less often than chance (κ = −0.147) and still waves through 17 of 19
unfaithful answers. The honest summary is the pair (−0.308, −0.147), not the first number alone.

**Consequence: every number this judge produced is unusable, including the ones in
`results/v2/report-generation.md`.** They are kept in the repository as the evidence for this
finding, not as results.

## Finding 2 — the uncalibrated judge inverts the conclusion

The same 150 answers, scored by the judge and by a human, give opposite rankings.

The human column is the annotated subset (9–12 labels per context, `?` answers excluded); the
judge column is its verdict over all 30 answers of that context, as published in
`results/v2/report-generation.md`.

| Context | Correctness — human (n) | Correctness — judge (n=30) | Faithfulness — human (n) |
|---|---|---|---|
| **oracle** | **0.58** [0.32, 0.81] (12) | 0.700 | **0.83** [0.55, 0.95] (12) |
| bm25 | 0.30 [0.11, 0.60] (10) | 0.633 | 0.42 [0.19, 0.68] (12) |
| e5-small | 0.20 [0.06, 0.51] (10) | 0.733 | 0.50 [0.25, 0.75] (12) |
| none | 0.11 [0.02, 0.44] (9) | 0.833 | n/a |
| bge-m3 | 0.10 [0.02, 0.40] (10) | 0.567 | 0.58 [0.32, 0.81] (12) |

Read by the judge, the benchmark says *retrieval is useless*: the closed-book context scores
highest (0.833) and the labelled documents (0.700) beat none of the retrievers convincingly.
That is the report a project without human calibration would have published.

Read by a human, the benchmark says the opposite: closed-book sits at the bottom (0.11, one
correct answer in nine) and the labelled documents at the top (0.58, seven in twelve).
Retrieval does help — when it brings the right document.

One row deserves to be read plainly rather than skipped: `bge-m3`, the best retriever of v1,
scores 0.10 — as low as no context at all — while `bm25` reaches 0.30. With ten labels per
context none of these gaps is significant, but the data gives no support to "a better
retriever produced better answers" below the oracle. Only the oracle separates.

Significance, Fisher's exact test on the human labels:

- oracle vs `bge-m3` correctness (7/12 vs 1/10): **p = 0.031**
- oracle vs closed-book correctness (7/12 vs 1/9): p = 0.067
- oracle vs `bm25` faithfulness (10/12 vs 5/12): p = 0.089

These p-values are **unadjusted**, and the reported pair was chosen after seeing the data: the
oracle–`bge-m3` gap is the largest of the ten available pairwise comparisons, so p = 0.031
would not survive a Bonferroni correction (0.005 across ten pairs). The argument rests on the
effect size of the oracle-vs-closed-book contrast (0.58 vs 0.11), not on that one test.

The comparison is also **unpaired**: the annotation sample draws independently per context, so
the 12 oracle answers and the 12 `bge-m3` answers share only half their queries, and question
difficulty is confounded with context. Sampling the same queries across contexts would be the
cheapest way to gain power at this sample size.

So the "labelled documents beat the alternatives" direction is solid, and the fine ranking
between the three real retrievers is not: with 9–12 labels per context, those differences are
noise. The headline of this version is the judge, not the retriever ranking.

## Finding 3 — a 3B generator fails to use passages it has been given

Reading the 60 annotated answers shows three recurring failure modes, all independent of
retrieval quality:

| Failure | Example | What happened |
|---|---|---|
| **Refusal despite relevant passages** (`e5-small`, answer `490896f38aa2`) | "Qui étaient Jean Talon, Frontenac et Mgr de Laval ?" | Passages [1] and [2] were the Jean Talon fact sheets; the answer states that "the extracts contain no information on Jean Talon" |
| **Invention despite a correct formula** (`e5-small`, answer `bfc5763900b4`) | "Comment calculer la température avant la réaction d'un mélange ?" | Passage [2] contains `Q = m·c·ΔT`; the answer produced `ΔH = ΔU + pΔV` and `ΔT = ΔH/(nCp)`, neither of which appears anywhere in the context. The same question under `oracle` (`94241e012ac2`) carries the calorimetry sheet in three of five passages, and the model refuses outright |
| **Contradicting a worked example** (`bge-m3`, answer `29aad9c3db05`) | "Convertir 0,15 g/L en % m/V" | Two passages solve the identical exercise by dividing by 10; the answer multiplied by 10 and returned 1.5% instead of 0.015% |

This explains why better retrieval does not translate linearly into better answers here: with
the labelled documents in its prompt the model still gets 42% of answers wrong. Retrieval is
necessary — closed-book is at the bottom — but at this model size it is not sufficient: the
second bottleneck is the generator's ability to exploit what it is given.

## Limitations

- **9 to 12 human labels per context.** Only the strongest contrast survives a significance
  test; no fine ranking between retrievers should be read from this table.
- **One annotator, no second pass.** Inter-annotator agreement is unknown, so the "human"
  column is one person's judgement, not a gold standard.
- **The annotator was not blind to the condition.** The session displayed the context name
  (`oracle`, `bm25`, …) above each item, and the annotator is the author, who expected oracle
  to win. The labels are blind to the *judge's* verdicts, which is what the calibration
  requires, but not to the experimental condition, which Finding 2 rests on. The display has
  since been changed; a future round will be blind on both counts.
- **The answer prompt licenses a refusal only when passages are given.** With context the
  model is told to say so "if the extracts do not allow an answer"; the closed-book prompt has
  no such exit. Refusals are counted as incorrect, and they are not evenly spread: 9 of 30
  `oracle` answers, 7 `bge-m3`, 6 `bm25`, 5 `e5-small`, against 2 closed-book. Part of
  Finding 3 is therefore the prompt doing what it was asked; a variant without the refusal
  clause would be needed to separate the two.
- **`oracle` is the right document, not the right passage.** It takes the first five
  `fixed_1000_200` chunks of the labelled documents. For 11 of the 30 queries those chunks
  cover only part of the labelled text, so the answer may sit outside what the model saw.
  "Upper bound" overstates it; "labelled documents, beginning first" is exact.
- **Some labels were discussed with an AI assistant.** While annotating, the author asked an
  assistant how to apply the criteria on roughly eight of the sixty items, and it suggested
  labels for several. The assistant read the answers and passages, never the judge's verdicts,
  so the calibration stays blind to the judge; but those labels are not an independent human
  judgement, and the refusal convention above was settled during that exchange.
- **Contexts differ by chunking as well as by retriever.** `bm25` and `e5-small` index whole
  documents (`chunker: none`, the v1 best for each), `bge-m3` and `oracle` use
  `fixed_1000_200`. Their prompts therefore differ by a factor of five in size (median 23,048
  and 13,508 passage characters against 4,846 and 4,113), so a row-to-row comparison is not a
  clean retriever comparison.
- **Reference answers are conversational.** Many Alloprof answers are "here is the fact sheet"
  with a link that the dataset does not carry, so correctness was unjudgeable for 9 of 60
  answers (recorded as null, excluded).
- **One generator, one judge, one prompt per task.** A larger judge (or an API judge) might
  behave differently; that is the natural next experiment and the interface already allows it.
- **The published run still truncates the two whole-document contexts.** With `num_ctx` set to
  8192, an estimated 10 of 30 `bm25` prompts and 3 of 30 `e5-small` prompts still exceed the
  window (median ≈ 6,980 and ≈ 4,090 tokens, largest ≈ 21,800, estimated at 3.3 characters per
  token); `none`, `oracle` and `bge-m3` stay below 1,600 tokens and are unaffected. The two
  truncated rows are therefore measured on a prompt the model saw only in part, which can only
  understate them. The headline results — the judge inversion, and the oracle-vs-closed-book
  and oracle-vs-`bge-m3` contrasts — involve no truncated context. The run was not repeated:
  a correct measurement needs the token counts, which the harness now records
  (`prompt_eval_count`), rather than this character-based estimate.
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

## Related work

Two papers bear directly on these findings; both were read before being cited here.

**[Pandey, *Can Small Language Models Use What They Retrieve?*](https://arxiv.org/abs/2603.11513)
(arXiv 2603.11513, March 2026 — single-author preprint, not peer-reviewed)** runs the same
grid shape on Qwen2.5 1.5B/3B/7B over NQ Open and HotpotQA, scored by exact match, with
no-context / BM25 / dense / oracle conditions. It corroborates Finding 3 closely: with oracle
retrieval, models of 7B or less "fail to extract the correct answer 85 to 100 percent of the
time on questions they cannot answer alone", and the dominant failure is irrelevant
generation — the model ignoring the context entirely, which is this benchmark's Jean Talon
case.

Its headline — that RAG is a net negative below 7B — looks at first like a contradiction of
Finding 2, and is not. The paper separates questions the model already answers closed-book
from those it does not. For Qwen2.5-3B, exact match on **unknown** questions goes from 0.0%
with no context to 12.8% with oracle (retrieval helps, this benchmark's direction), while on
**known** questions it falls from 100% to 54.4% (context destroys parametric answers). The net
negative comes entirely from the second group. Here the closed-book correctness is 0.11, so
that second group is almost empty on this corpus and only the gain remains. The two results
describe different regimes rather than disagreeing.

**[Thakur et al., *Judging the Judges*](https://arxiv.org/abs/2406.12624) (GEM² workshop at
ACL 2025, UMass Amherst and Meta)** evaluates thirteen judges on TriviaQA answer correctness
against human majority votes. It supports the mechanism behind Finding 1: judges "tend to
judge positively when in doubt", an effect the paper reports as "more pronounced for small
models than for larger ones", and they "struggle with under-specified answers and tend to be
lenient".

It does not support the magnitude found here, and the difference is informative. Their
Llama-3 8B reaches a Scott's π of 0.59–0.78 against a human-human π of 96.2, far above the
κ = −0.308 measured here. Their task is deliberately easy — answer correctness against a
reference string — while faithfulness requires checking every claim against five passages.
Their paper evaluates no RAG setting and no groundedness criterion at all. A note on metrics:
they deliberately dropped Cohen's κ for Scott's π, judging it more discriminative; this
report uses κ, which is the more common choice but not the only defensible one.

**The gap this fills.** Neither paper, nor anything else found in a search, publishes
judge-human agreement for *faithfulness in a RAG setting, broken down by judge size*. The
κ = −0.308 for `llama3.1:8B` on 48 labelled pairs is a small, single-annotator data point in
that gap — enough to justify not trusting a free local judge on this criterion, not enough to
generalise.

## What v2b should test

1. A stronger judge (`llama3.1:70B` through a free API tier, or a calibrated prompt with
   few-shot examples) measured against the same 60 labels — the harness already supports it.
2. A larger generator (8B) on the oracle context, to separate "the model cannot use context"
   from "3B cannot use context".
3. More human labels on the two extreme contexts only (oracle and closed-book), which is where
   the signal is, rather than spreading labels across five contexts.
