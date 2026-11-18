# rag-eval-lab

[![CI](https://github.com/EmericLi/rag-eval-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/EmericLi/rag-eval-lab/actions/workflows/ci.yml)

**What this is for.** When a RAG assistant answers badly, is the retriever to blame, or the
model — and can the tool that scores the answers be trusted? Teams rarely measure any of the
three; they tune prompts instead. This project measures all three on a real French corpus:
what each retrieval component is worth in quality and latency, what a small generator does
with the passages it is handed, and how far an LLM judge agrees with a human.

Everything runs locally on a laptop CPU — no GPU, no paid API.

## Retrieval (v1)

Benchmark: [AlloprofRetrieval](https://huggingface.co/datasets/lyon-nlp/alloprof) — 2,556
French school-help documents, 300 sampled student questions (seed 42), 30 configurations
(chunking × lexical / dense / hybrid retrieval × cross-encoder reranking), AMD Ryzen 7 8845HS,
CPU only.

| Chunking | Retriever | Reranker | nDCG@10 | Recall@10 | Latency p50 |
|---|---|---|---|---|---|
| none | hybrid:bm25-fr+bge-m3 | none | **0.571** | **0.762** | 152 ms |
| fixed_1000_200 | dense:bge-m3 | none | 0.554 | 0.709 | 188 ms |
| fixed_1000_200 | dense:bge-m3 | minilm | 0.504 | 0.695 | 2,348 ms |
| none | bm25-fr (tuned) | none | 0.452 | 0.643 | 7 ms |
| none | bm25 (raw tokens) | none | 0.351 | 0.502 | 20 ms |
| none | dense:e5-small | none | 0.328 | 0.455 | 26 ms |

![Quality vs latency](results/v1/pareto.png)

**Four findings**

1. **An untuned baseline had inflated the headline.** The first BM25 here only lowercased
   text, so "divisions" did not match "division", and bge-m3 beat it by +58%. Folding accents,
   dropping function words and stripping plurals (`bm25-fr`) lifts BM25 from 0.351 to 0.452,
   and the real gap is **+22%** (0.554 vs 0.453 at equal chunking). An earlier version of this
   page reported the +58%; both baselines are kept in the grid so the difference stays visible.
2. **The embedding model decides, not "lexical vs dense".** bge-m3 scores 22% above a tuned
   BM25; e5-small, also a dense bi-encoder, scores 27% *below* it (0.328 vs 0.452).
3. **Hybrid search pays once the lexical side is competent.** Fused with the raw BM25, RRF only
   matched bge-m3 alone (0.542 vs 0.545), recovering 16 queries and forfeiting 21. Fused with
   `bm25-fr` it is the best configuration of the grid: 0.571 nDCG@10 and 0.762 recall@10
   against 0.545 and 0.693, recovering 24 and forfeiting 13.
4. **A reranker only helps a first stage weaker than itself.** `mmarco-mMiniLMv2` adds +24% to
   the raw BM25, +9% to the tuned one, and costs the hybrid −6% and bge-m3 −9% — for about
   two more seconds per query. Every reranked configuration is off the Pareto front.

Full analysis, failure cases and limitations: [docs/results-v1.md](docs/results-v1.md).

## Generation (v2a) — is the LLM judge even right?

v2a extends the benchmark past retrieval: it generates an answer per query under five
contexts (no passage, three real retrievers, and the labelled documents as an upper bound),
scores all 150 answers with a local judge (`llama3.1:8B`), and then **calibrates that judge
against 60 answers labelled by hand**, blind to its verdicts.

| | Judge (150 answers) | Human (60 labels) |
|---|---|---|
| Best context by correctness | `none` — no passage at all (0.83) | **`oracle` — the labelled documents (0.58)** |
| Worst context by correctness | `bge-m3` (0.57) | `bge-m3` (0.10) and `none` (0.11) |

**The judge inverts the conclusion.** Read through `llama3.1:8B`, this benchmark says
retrieval is useless and a closed-book model answers best. Read through human labels, the
closed-book setting sits at the bottom (1 correct answer in 9) and the labelled documents at
the top (7 in 12).

Three findings:

1. **A free local judge cannot be trusted here, and the number says so**: agreement 0.40 with
   a Cohen's κ of **−0.308** on faithfulness — worse than chance. Of 20 answers a human called
   unfaithful, it recognised 2. It is systematically indulgent (says "correct" 61% of the
   time against a human's 27%). Refusals excluded, κ is still negative (−0.147).
2. **Retrieval helps when it brings the right document — and only that is shown**: oracle vs
   `bge-m3` correctness, 7/12 vs 1/10, Fisher p = 0.031 (unadjusted, pair chosen post hoc).
   Below the oracle nothing separates: the best v1 retriever scores no higher than no context
   at all, and no ranking between the three real retrievers is claimed at 9–12 labels each.
3. **A 3B generator fails to use passages it was handed**: it refuses while the answer sits in
   passage [1], invents formulas while the right one is in context, and contradicts a worked
   example by a factor of 100. Retrieval is necessary but not sufficient: with the labelled
   documents in its prompt the model is still wrong 42% of the time. Part of this is the
   prompt itself, which invites a refusal whenever passages are given (9 of 30 oracle answers
   are refusals).

An incident worth reading: the first full run measured an Ollama client default (a 2048-token
window silently truncating whole-document contexts) rather than retrieval quality. It was
caught, fixed, and documented.

Full analysis, failure cases and limitations: [docs/results-v2.md](docs/results-v2.md).

## Quickstart

```bash
uv sync --extra models
uv run rel run configs/v1.yaml     # retrieval grid (resumable)
uv run rel report results/v1       # report.md + charts
```

The first run downloads the dataset and the models (~3 GB) into the Hugging Face cache.
Embeddings are cached on disk, so re-running a grid only recomputes what changed.

The generation benchmark also needs a local [Ollama](https://ollama.com) server with the two
models it uses:

```bash
ollama pull llama3.2
ollama pull llama3.1:8b
uv run rel generate configs/v2.yaml             # one answer per (context, query)
uv run rel judge results/v2 configs/v2.yaml     # three verdicts per answer
uv run rel annotate results/v2 --n 60           # label a sample by hand; judge verdicts stay hidden
uv run rel report-gen results/v2                # quality per context + judge calibration
```

Every stage appends to a JSON-lines file and can be interrupted and resumed. Expect about
eight hours for `generate` and `judge` on a laptop CPU with `configs/v2.yaml`.

## How it works

```
retrieval   dataset -> chunker -> retriever -> reranker (optional) -> ranked documents -> metrics
generation  dataset -> context source -> generator -> answers -> judge -> verdicts --+
                                              +-> human annotation ------------------> calibration
```

| Module | Role |
|---|---|
| `data.py` | Loads AlloprofRetrieval or a local JSONL corpus; deterministic sampling |
| `chunking.py` | `none`, fixed-size with overlap, recursive (paragraph → line → sentence → word) |
| `text.py` | Tokenizer, and the French analyzer behind `bm25-fr` |
| `retrieval.py` | BM25, dense (bi-encoder), hybrid via Reciprocal Rank Fusion |
| `reranking.py` | Cross-encoder reranking of the top candidates |
| `pipeline.py` | Wires the stages and maps chunk hits back to documents (max-score per document) |
| `experiment.py`, `report.py` | Retrieval grid runner; ranking table, Pareto front, charts |
| `context.py` | What the generator is given: nothing, retrieved passages, or the labelled documents |
| `ollama_client.py`, `generation.py`, `judging.py` | Local model calls, answer generation, three binary verdicts per answer |
| `annotation.py`, `calibration.py` | Blind human labelling session; agreement, Cohen's κ, Wilson intervals, Fisher test |
| `generation_experiment.py`, `report_generation.py` | Generation and judging stages; quality per context and judge calibration |
| `jsonl.py` | Append-only result files that survive a crash mid-write |
| `cli.py` | `rel run`, `report`, `generate`, `judge`, `annotate`, `report-gen` |

Each stage is a small interchangeable component, so a grid of configurations is just a YAML
file. A configuration or an item that fails is recorded as an error and does not abort the
run, and every result carries what produced it: metrics, latency, library versions, git
commit, model digest, prompt version and context window.

## Configuration

```yaml
dataset: { name: alloprof, n_queries: 300, seed: 42 }
k: 10
ks: [5, 10]
candidates: 30            # chunks passed to the reranker
grid:
  chunker:   [none, fixed_1000_200, recursive_1000]   # sizes in characters
  retriever: [bm25, bm25-fr, dense:e5-small, dense:bge-m3, hybrid:bm25+bge-m3, hybrid:bm25-fr+bge-m3]
  reranker:  [none, minilm]
exclude:
  - { chunker: none, reranker: minilm }
```

Available specs:

- **chunker**: `none`, `fixed_<size>_<overlap>`, `recursive_<size>` (characters)
- **retriever**: `bm25` (raw lowercase tokens), `bm25-fr` (French analysis), `dense:<encoder>`,
  `hybrid:<a>+<b>`
- **encoder**: `e5-small`, `e5-base`, `bge-m3`, `hashing` (model-free baseline)
- **reranker**: `none`, `minilm`, `bge-v2-m3`

Any corpus can be benchmarked by pointing the config at a JSONL dataset
(`{"id", "text"}` documents and `{"id", "text", "relevant": [...], "answer": "..."}` queries;
`answer` is only needed for the generation benchmark). `configs/v2.yaml` shows the generation
side: contexts, generator, judge and their context window.

## Development

```bash
uv sync
uv run pytest      # unit tests, no network, no model download
uv run ruff check .
```

Tests use a model-free hashing encoder and injected fake models, so CI stays fast and
offline.

## Roadmap

- **v2b** — a stronger judge (free-tier API) measured against the same 60 human labels; an 8B
  generator on the oracle context; more labels on the two extreme contexts.
- **v3** — synthetic question generation for any corpus, and a small demo app on top of the
  best configuration.

## License

MIT — see [LICENSE](LICENSE). Dataset: AlloprofRetrieval (`lyon-nlp/alloprof`, Apache-2.0).
