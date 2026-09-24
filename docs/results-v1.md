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
| Encoders | `intfloat/multilingual-e5-small`, `BAAI/bge-m3`, both capped at `max_seq_length = 512` tokens |
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
two systems are genuinely complementary. Measured on hit@10 over the 300 queries, all at
`chunker: none` so the three systems are compared like for like:

| | queries solved (hit@10) |
|---|---|
| dense bge-m3 | 221 / 300 |
| bm25 | 164 / 300 |
| hybrid (RRF) | 226 / 300 |
| **union of dense and bm25 (oracle)** | **244 / 300** |

The net gain of fusion (+5 queries) hides a much larger churn. 23 queries are solved by BM25
alone — typically those carrying an exact lexical cue. Equal-weight RRF **recovers 16 of those
23, but forfeits 21 queries that the dense retriever alone got right**: a rank-3 hit in the
strong system is dragged out of the top 10 by a weak system's confident-but-wrong ranking,
because RRF weighs both sides identically while one is 35% more accurate.

The same churn appears with the other chunkers (`fixed_1000_200`: 12 recovered / 21 forfeited;
`recursive_1000`: 14 / 17), so it is a property of the fusion, not of one configuration.

**6 points of hit@10 separate RRF from the oracle union** (226 → 244 of 300). That gap is the
argument for weighted fusion or a learned gate rather than plain RRF — and it is invisible to
the aggregate metrics, which show hybrid and dense as equivalent.

### 4. Chunking's effect is entangled with the 512-token cap — and this grid cannot separate them

Fixed-size chunking gives bge-m3 +1.7% (0.545 → 0.554) and costs BM25 nothing to −2%
(0.351 → 0.349). That looks like "chunking barely matters here", but the comparison is
confounded, and the corpus statistics say why.

Alloprof documents are **not** short: mean 3,541 characters, median 2,568, p90 7,550, max
47,973 — about **60% exceed ~2,000 characters**, roughly 512 tokens of French. Both encoders
run with `max_seq_length = 512` (a project setting, not a model limit: bge-m3 supports 8,192),
so at `chunker: none` the dense retriever never sees the second half of the median document.
The chunked configurations are therefore not "the same content, cut up" — they are the only
configurations that index the whole corpus.

Two consequences for reading this grid:

- The +1.7% chunking gain is measured against a **truncated** baseline. Chunking's real effect
  on this corpus is unresolved, not small.
- At `chunker: none`, BM25 indexes the full document text while the dense retriever sees a
  prefix, so "e5-small scores below BM25" (0.328 vs 0.351) is not a like-for-like comparison
  at that row. Finding 1's headline is unaffected: it rests on the best dense configuration,
  which is chunked and therefore untruncated.

Separating the two effects needs one extra run with `max_seq_length: 8192` at
`chunker: none` — the first thing v2 should measure.

### 5. Practical recommendation

Pick by latency budget, not by reputation:

- **< 30 ms**: `bm25` on whole documents (0.351). Free to index, no model to host.
- **~220 ms**: `fixed_1000_200 + dense:bge-m3` (0.554). +58% quality for +200 ms — the best
  trade in this grid.
- **Reranking**: not worth it here (−9% quality, +2 s) unless the first stage is weak.

## Failure analysis

76 of 300 queries have no relevant document in the top 10 for the best configuration (79 for
the same retriever at `chunker: none`, the setting used in the table below so that dense and
BM25 stay comparable). Reading them shows four distinct causes; ranks come from the committed
per-query runs (`results/v1/runs/`).

| Query | Expected document | What the dense retriever returned | BM25 | Cause |
|---|---|---|---|---|
| "estimation d'une addition de nombres décimaux" | *Le calcul mental* | *L'addition de nombres décimaux*, *La multiplication de nombres décimaux*, *L'addition* | rank 5 | **Label sparsity**: the retrieved sheets are arguably valid answers; one labelled document penalises a reasonable ranking |
| "calculer la température avant la réaction d'un mélange de deux substances" | *La calorimétrie (Q = m c ΔT)* | *La détermination du point de fusion*, *La mesure de la température*, *La température* | **rank 1** | **Lost technical term**: the query paraphrases a formula the sheet names explicitly; the dense encoder drifts to the generic topic, exact term matching wins |
| "évaluation demain, les 4 opérations sur les entiers relatifs" (4 relevant docs) | *La division / soustraction / addition / multiplication de nombres entiers* | *Répertoire de révision – Mathématiques – Primaire*, *Révision et examen en mathématiques*, … | **rank 1** | **Meta-intent capture**: the dense encoder anchors on "révision / examen" and on the wrong school level instead of the notions being revised |
| "je dois écrire *tel* ou *telle* dans cette phrase…" | *Tel et tel que* | *La modalisation*, *Either*, *Le subjonctif* | not in top 10 | **Vocabulary mismatch**: neither system retrieves the sheet — its title is the bare grammar form, which matches neither the paraphrase nor the surrounding medical sentence |

Two of these four are cases where BM25 ranks the answer first or fifth while the best
configuration misses it entirely — the churn quantified in Finding 3, seen one query at a time.

The first cause also inflates the apparent error rate: part of the 25% "failures" are ranking
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
- **Dense retrieval is truncated at 512 tokens** while BM25 indexes full documents, so rows at
  `chunker: none` are not like-for-like across retriever families (see Finding 4).
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

1. `max_seq_length: 8192` for bge-m3 at `chunker: none`, to separate the effect of chunking
   from the effect of the 512-token cap (Finding 4).
2. Weighted or learned fusion instead of plain RRF — the oracle union is 6 points of hit@10
   above RRF, and the recovered/forfeited split (16 vs 21) says where the loss comes from.
3. A stronger reranker (`bge-reranker-v2-m3`) on a subset, to separate "reranking is useless"
   from "this reranker is too weak".
4. The full generation stage: faithfulness and answer relevance judged by an LLM, with the
   judge calibrated against human labels.
