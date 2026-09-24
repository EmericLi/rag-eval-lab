# Results v1 — Retrieval on AlloprofRetrieval

## Setup

| | |
|---|---|
| Dataset | [`lyon-nlp/alloprof`](https://huggingface.co/datasets/lyon-nlp/alloprof) (Apache-2.0), `documents` + `queries` test split |
| Corpus | 2,556 French school-help documents, indexed as `title + "\n\n" + text` |
| Queries | 300 sampled student questions (seed 42), out of 2,316 |
| Relevance | document-level, as published with the dataset |
| Grid | 3 chunkers × 4 retrievers × 2 rerankers, minus 4 excluded = 20 configurations |
| Chunk sizes | characters: `fixed_1000_200` (1,000 chars, 200 overlap), `recursive_1000` |
| Candidates reranked | 30 chunks |
| Encoders | `intfloat/multilingual-e5-small`, `BAAI/bge-m3` |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| Hardware | AMD Ryzen 7 8845HS, 28 GB RAM, CPU only (no GPU) |
| Reproduce | `uv run rel run configs/v1.yaml && uv run rel report results/v1` |

Per-run provenance (library versions, git commit, hardware) is stored in every record of
`results/v1/results.jsonl`.

## Main results

Ranked by nDCG@10 (full table: [`results/v1/report.md`](../results/v1/report.md)).

| # | Chunker | Retriever | Reranker | nDCG@10 | MRR@10 | R@5 | R@10 | p50 (ms) | p95 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | fixed_1000_200 | dense:bge-m3 | none | **0.554** | 0.522 | 0.635 | 0.709 | 221 | 336 |
| 2 | none | dense:bge-m3 | none | 0.545 | 0.516 | 0.616 | 0.693 | 286 | 446 |
| 3 | none | hybrid:bm25+bge-m3 | none | 0.542 | 0.509 | 0.608 | **0.713** | 309 | 510 |
| 4 | recursive_1000 | dense:bge-m3 | none | 0.538 | 0.508 | 0.618 | 0.688 | 218 | 333 |
| 5 | fixed_1000_200 | hybrid:bm25+bge-m3 | none | 0.527 | 0.490 | 0.609 | 0.702 | 309 | 679 |
| 7 | fixed_1000_200 | hybrid:bm25+bge-m3 | minilm | 0.521 | 0.479 | 0.592 | 0.710 | 2,276 | 3,029 |
| 9 | fixed_1000_200 | dense:bge-m3 | minilm | 0.504 | 0.461 | 0.574 | 0.695 | 2,348 | 2,778 |
| 11 | fixed_1000_200 | bm25 | minilm | 0.433 | 0.417 | 0.478 | 0.551 | 2,310 | 3,243 |
| 14 | none | bm25 | none | 0.351 | 0.323 | 0.410 | 0.502 | **25** | 98 |
| 18 | none | dense:e5-small | none | 0.328 | 0.303 | 0.389 | 0.455 | **18** | 33 |
| 20 | fixed_1000_200 | dense:e5-small | none | 0.311 | 0.276 | 0.362 | 0.454 | 20 | 35 |

![Quality vs latency](../results/v1/pareto.png)

Only 4 of the 20 configurations are on the Pareto front: `e5-small` (18 ms), `bm25` (25 ms),
`recursive_1000 + bge-m3` (218 ms) and `fixed_1000_200 + bge-m3` (221 ms). Every reranked
configuration is dominated.

## Findings

### 1. The embedding model decides, not the lexical/dense dichotomy

bge-m3 beats BM25 by **+58%** nDCG@10 (0.554 vs 0.351), but e5-small — also a dense
bi-encoder — scores **below** BM25 (0.328 vs 0.351). "Switching to dense retrieval" is not a
decision by itself: the same architecture spans the best and the worst results in this grid.
Model capacity matters here (bge-m3: 568M parameters, 1024-dim; e5-small: 118M, 384-dim), and
so does multilingual training quality on French.

### 2. A reranker weaker than the retriever destroys the ranking

| First stage | nDCG@10 without reranker | with `minilm` | Δ | latency p50 |
|---|---|---|---|---|
| bm25 (fixed_1000_200) | 0.349 | 0.433 | **+24%** | 105 → 2,310 ms |
| hybrid bm25+bge-m3 (fixed_1000_200) | 0.527 | 0.521 | −1% | 309 → 2,276 ms |
| dense bge-m3 (fixed_1000_200) | 0.554 | 0.504 | **−9%** | 221 → 2,348 ms |

Reranking is usually presented as a free quality gain. It is not: `mmarco-mMiniLMv2` rescues a
weak lexical first stage (+24%) but degrades a strong dense one (−9%), while multiplying
latency by 11. A cross-encoder only helps when it is more accurate than the ranking it
rewrites — and reranking the top 30 also caps recall at what the first stage already found.

### 3. Hybrid search pays only when both sides are comparable

With bge-m3, hybrid RRF matches the dense retriever (0.542 vs 0.545) and adds ~90 ms. Yet the
two systems are genuinely complementary — measured on hit@10 over the 300 queries:

| | queries solved (hit@10) |
|---|---|
| dense bge-m3 | 224 / 300 |
| bm25 | 164 / 300 |
| hybrid (RRF) | 226 / 300 |
| **union of dense and bm25 (oracle)** | **241 / 300** |

17 queries are solved by BM25 alone — typically those with an exact lexical cue ("estimation
d'une addition de nombres décimaux", "apprendre des verbes par cœur"). Hybrid RRF recovers
only 2 of them: with equal weights, 164 correct BM25 rankings are fused with 136 incorrect
ones, and the noise cancels the gain. **5 points of hit@10 are left on the table**, which
suggests weighted fusion (or a learned gate) rather than plain RRF as the next step.

### 4. Chunking has a small effect on this corpus

Fixed-size chunking gives bge-m3 +1.7% (0.545 → 0.554) and costs BM25 nothing to −2%
(0.351 → 0.349). The expected "long documents get truncated at 512 tokens" effect is real but
modest here, because Alloprof documents are mostly short, single-topic fact sheets: chunking
mainly splits documents that already had one topic. On a corpus of long, multi-topic documents
the effect should be larger — an assumption this benchmark does not test.

### 5. Practical recommendation

Pick by latency budget, not by reputation:

- **< 30 ms**: `bm25` on whole documents (0.351). Free to index, no model to host.
- **~220 ms**: `fixed_1000_200 + dense:bge-m3` (0.554). +58% quality for +200 ms — the best
  trade in this grid.
- **Reranking**: not worth it here (−9% quality, +2 s) unless the first stage is weak.

## Failure analysis

76 of 300 queries have no relevant document in the top 10 for the best configuration. Reading
them shows three distinct causes.

| Query | Expected document | Top-3 retrieved | Cause |
|---|---|---|---|
| "estimation d'une addition de nombres décimaux" | *Le calcul mental* | *L'addition de nombres décimaux*, *La multiplication de nombres décimaux*, *Les systèmes de numération* | **Label sparsity**: the retrieved sheets are arguably valid answers; a single labelled document penalises a reasonable ranking |
| "j'ai une évaluation demain, 4 opérations sur les entiers relatifs" (4 relevant docs) | *La division / soustraction / addition / multiplication de nombres entiers* | primary-school revision repertoires | **Level mismatch**: the query states "secondaire 1" but nothing in the corpus text carries school level with enough weight |
| "je dois écrire *tel* ou *telle* dans cette phrase…" | *Tel et tel que* | *La modalisation*, *Either*, *Le subjonctif* | **Lost lexical cue**: the decisive token is a grammar keyword; the dense encoder paraphrases it away — BM25 finds it |

The first cause inflates the apparent error rate: part of the 25% "failures" are ranking
choices a human would accept. Any absolute number here should be read as a lower bound on real
quality, which is exactly why the benchmark compares configurations against each other rather
than reporting a single score.

## Limitations

- **One dataset, one domain.** French school-help sheets, short and single-topic. Conclusions
  about chunking in particular will not transfer to long technical or legal documents.
- **300 of 2,316 queries.** Sampled with a fixed seed for a CPU budget; differences below
  ~2 points of nDCG@10 should not be treated as significant.
- **Document-level relevance, single labelled answer for most queries** — see the failure
  analysis: the metric under-counts acceptable answers.
- **Latency is CPU-only, single process, batch size 1**, measured on a laptop. On a GPU the
  reranker's 2 s collapses and conclusion 2 would need re-measuring — the trade-off is
  hardware-dependent, the quality ranking is not.
- **One index-time measurement is invalid**: `fixed_1000_200 + dense:bge-m3` reports
  77,157 s because the machine slept mid-encoding. The real value, extrapolated from the
  encoding rate, is ≈ 7,500 s (~2 h); comparable configurations took 1,996 s (2,556 chunks)
  and 6,904 s (11,022 chunks). Wall-clock index times are not sleep-aware.
- **Embedding cache**: index times are cold-cache only when a model/chunker pair is computed
  for the first time; cached runs report ~0 s.
- **No generation stage.** This benchmark stops at retrieval; answer faithfulness is v2.

## What v2 should test

1. Weighted or learned fusion instead of plain RRF — the oracle union suggests +5 points
   of hit@10 are reachable.
2. A stronger reranker (`bge-reranker-v2-m3`) on a subset, to separate "reranking is useless"
   from "this reranker is too weak".
3. The full generation stage: faithfulness and answer relevance judged by an LLM, with the
   judge calibrated against human labels.
