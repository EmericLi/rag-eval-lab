# rag-eval-lab v2a — Generation evaluation: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mesurer si une meilleure recherche produit de meilleures réponses, en générant des réponses sur 5 contextes (dont deux bornes), en les jugeant avec un LLM local, et en calibrant ce juge contre 60 annotations humaines.

**Architecture :** quatre étapes découplées, chacune reprenable et écrivant son propre `jsonl` : `generate` (contexte → réponse), `judge` (réponse → trois verdicts binaires), `annotate` (session humaine), `report-gen` (tableaux, graphique, calibration). Les briques suivent les conventions de la v1 : protocoles, construction validée par pydantic avant tout calcul, isolation des erreurs, reprise par identifiant stable.

**Tech Stack :** Python ≥ 3.11, uv, pydantic v2, typer, matplotlib, Ollama (HTTP local, client en bibliothèque standard), pytest, ruff.

**Spec :** `docs/design/specs/2026-10-06-rag-eval-lab-v2a-generation-design.md`

## Global Constraints

- Code, docstrings, README et rapport en **anglais**. Les **prompts envoyés aux modèles sont en français**, car le corpus et les questions le sont.
- Plans et specs en français.
- **Aucun appel réseau dans les tests** : le client Ollama est injecté sous forme de double de test. Aucun modèle n'est téléchargé en CI.
- Aucune dépendance nouvelle : le client HTTP utilise `urllib` de la bibliothèque standard.
- Tout tourne en local sur CPU (Ryzen 7 8845HS, ~28 Go RAM). Ollama écoute sur `http://127.0.0.1:11434` et dispose de `llama3.2:latest`, `llama3.2:1B`, `llama3.1:8B`.
- Modèles : générateur `llama3.2:latest`, juge `llama3.1:8B`. Température 0, graine 0, pour la reproductibilité.
- Trois critères binaires : `faithfulness`, `relevance`, `correctness`. **Pour le contexte `empty` (aucun passage), `faithfulness` et `relevance` valent `null`** : sans passage, la fidélité n'a pas de sens ; seule la justesse est jugée. Les valeurs `null` sont exclues des moyennes et des calculs d'accord.
- `answer_id = make_key(context_name, query_id, generator_model, PROMPT_VERSION)[:12]` : clé de reprise **et** clé de jointure entre `answers.jsonl`, `judgments.jsonl` et `annotations.jsonl`.
- `answers.jsonl` stocke le **texte** des passages, pas seulement leurs identifiants : `judge`, `annotate` et `report-gen` n'ont alors besoin d'aucune pipeline de recherche.
- Commits : messages conventionnels (`feat:`, `fix:`, `test:`, `docs:`, `chore:`), **sans ligne `Co-Authored-By`** (l'utilisateur est seul auteur). Identité git du dépôt déjà configurée, ne pas y toucher.
- `uv` n'est pas dans le PATH : préfixer chaque commande Bash par
  `export PATH="/c/Users/Emeric/AppData/Local/Microsoft/WinGet/Packages/astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe:$PATH";`
- Toutes les commandes se lancent depuis `C:\Users\Emeric\travail\portofolio\rag-eval-lab`.
- Avant chaque commit : `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .` (si le formateur veut modifier, lancer `uv run ruff format .` et garder le résultat).

## File Structure

```
src/rag_eval_lab/
├── types.py                   # MODIFIÉ : Query gagne un champ `answer`
├── data.py                    # MODIFIÉ : alloprof_from_rows et load_jsonl remplissent `answer`
├── pipeline.py                # MODIFIÉ : search_passages(), search() réécrite au-dessus
├── ollama_client.py           # NOUVEAU : client HTTP stdlib (generate, model_digest)
├── prompts.py                 # NOUVEAU : gabarits français + PROMPT_VERSION
├── context.py                 # NOUVEAU : EmptyContext, RetrievedContext, OracleContext
├── generation.py              # NOUVEAU : OllamaGenerator
├── judging.py                 # NOUVEAU : OllamaJudge, Verdict, parsing JSON tolérant
├── generation_config.py       # NOUVEAU : GenerationConfig et sous-modèles pydantic
├── generation_experiment.py   # NOUVEAU : build_dataset, run_generation, run_judging
├── annotation.py              # NOUVEAU : tirage équilibré, session interactive
├── calibration.py             # NOUVEAU : accord, kappa, Wilson, confusion
├── report_generation.py       # NOUVEAU : rapport v2 + graphique + calibration
└── cli.py                     # MODIFIÉ : commandes generate, judge, annotate, report-gen
tests/
├── test_ollama_client.py      ├── test_prompts.py        ├── test_context.py
├── test_generation.py         ├── test_judging.py        ├── test_generation_config.py
├── test_generation_experiment.py ├── test_annotation.py  ├── test_calibration.py
├── test_report_generation.py  └── test_cli_generation.py
configs/
├── v2-smoke.yaml              # 4 questions, mesure de vitesse
└── v2.yaml                    # grille complète
```

---

### Task 1: Plomberie v1 — réponse du prof et accès aux passages

**Files:**
- Modify: `src/rag_eval_lab/types.py`, `src/rag_eval_lab/data.py`, `src/rag_eval_lab/pipeline.py`
- Test: `tests/test_data.py`, `tests/test_pipeline.py` (ajouts)

**Interfaces:**
- Consumes: `Query(id, text)`, `Dataset`, `RetrievalPipeline` de la v1.
- Produces :
  - `Query(id: str, text: str, answer: str = "")` — `answer` est la réponse rédigée par le prof (champ `answer` d'Alloprof), chaîne vide quand elle est absente.
  - `RetrievalPipeline.search_passages(query: str, k: int) -> list[Hit]` — les `k` meilleurs passages après rerank éventuel, score décroissant.
  - `RetrievalPipeline.search(query, k)` inchangée dans son comportement.

- [ ] **Step 1: Écrire les tests**

Ajouter à `tests/test_data.py` :

```python
def test_alloprof_rows_keep_the_teacher_answer():
    docs = [{"uuid": "u1", "title": "Le volcan", "text": "Un volcan est...", "topic": "x"}]
    queries = [{"id": 7, "text": "c'est quoi un volcan", "relevant": ["u1"], "answer": "Bonjour, un volcan..."}]
    dataset = alloprof_from_rows(docs, queries)
    assert dataset.queries[0].answer == "Bonjour, un volcan..."


def test_alloprof_rows_tolerate_a_missing_answer():
    docs = [{"uuid": "u1", "title": "T", "text": "texte"}]
    dataset = alloprof_from_rows(docs, [{"id": 1, "text": "q", "relevant": ["u1"]}])
    assert dataset.queries[0].answer == ""


def test_load_jsonl_reads_an_optional_answer(tmp_path):
    directory = tmp_path / "mini"
    directory.mkdir()
    (directory / "corpus.jsonl").write_text('{"id": "d1", "text": "texte"}\n', encoding="utf-8")
    (directory / "queries.jsonl").write_text(
        '{"id": "q1", "text": "question", "relevant": ["d1"], "answer": "ref"}\n', encoding="utf-8"
    )
    dataset = load_jsonl(directory)
    assert dataset.queries[0].answer == "ref"
```

Ajouter à `tests/test_pipeline.py` :

```python
def test_search_passages_returns_chunk_hits(mini_dataset):
    pipeline = RetrievalPipeline(FixedChunker(size=200, overlap=50), BM25Retriever())
    pipeline.index(mini_dataset.documents)
    hits = pipeline.search_passages("Comment expliquer le volcan ?", k=3)
    assert len(hits) == 3
    assert hits[0].chunk.doc_id == "d0"
    assert "volcan" in hits[0].chunk.text
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_search_and_search_passages_agree(mini_dataset):
    pipeline = RetrievalPipeline(FixedChunker(size=200, overlap=50), BM25Retriever())
    pipeline.index(mini_dataset.documents)
    docs = pipeline.search("Comment expliquer le volcan ?", k=3)
    passages = pipeline.search_passages("Comment expliquer le volcan ?", k=50)
    assert docs[0] == passages[0].chunk.doc_id
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/test_pipeline.py -q -k search_passages`
Expected: FAIL avec `AttributeError: 'RetrievalPipeline' object has no attribute 'search_passages'`

- [ ] **Step 3: Implémenter**

Dans `src/rag_eval_lab/types.py`, remplacer la classe `Query` :

```python
@dataclass(frozen=True)
class Query:
    id: str
    text: str
    answer: str = ""
    """Reference answer written by a human, when the dataset provides one."""
```

Dans `src/rag_eval_lab/data.py`, dans `load_jsonl`, remplacer la création de la requête :

```python
        queries.append(Query(qid, row["text"], row.get("answer", "")))
```

et dans `alloprof_from_rows` :

```python
        queries.append(Query(qid, row["text"], row.get("answer", "") or ""))
```

Dans `src/rag_eval_lab/pipeline.py`, remplacer la méthode `search` par :

```python
    def search_passages(self, query: str, k: int) -> list[Hit]:
        """Top-k chunk hits, after reranking when a reranker is configured."""
        hits = self.retriever.search(query, max(self.candidates, k))
        if self.reranker is not None:
            hits = self.reranker.rerank(query, hits, len(hits))
        return hits[:k]

    def search(self, query: str, k: int) -> list[str]:
        hits = self.search_passages(query, max(self.candidates, k))
        return aggregate_to_docs(hits, k)
```

- [ ] **Step 4: Donner une réponse de référence aux fixtures partagées**

Dans `tests/conftest.py`, remplacer la construction des requêtes de `mini_dataset` :

```python
    queries = [
        Query(
            id=f"q{i}",
            text=f"Comment expliquer le {TOPICS[i]} ?",
            answer=f"Le {TOPICS[i]} est un sujet important.",
        )
        for i in range(5)
    ]
```

et, dans `mini_jsonl`, la ligne qui écrit une requête :

```python
            f.write(
                json.dumps(
                    {"id": q.id, "text": q.text, "relevant": relevant, "answer": q.answer},
                    ensure_ascii=False,
                )
                + "
"
            )
```

Ajouter à `tests/test_data.py` :

```python
def test_mini_jsonl_round_trips_the_reference_answer(mini_jsonl):
    dataset = load_jsonl(mini_jsonl)
    assert dataset.queries[0].answer == "Le volcan est un sujet important."
```

- [ ] **Step 5: Vérifier le succès**

Run: `uv run pytest -q`
Expected: `80 passed` (74 + 6 nouveaux)

- [ ] **Step 6: Commit**

```bash
git add src/rag_eval_lab/types.py src/rag_eval_lab/data.py src/rag_eval_lab/pipeline.py tests/conftest.py tests/test_data.py tests/test_pipeline.py
git commit -m "feat: carry reference answers and expose passage-level search"
```

---

### Task 2: Client Ollama

**Files:**
- Create: `src/rag_eval_lab/ollama_client.py`
- Test: `tests/test_ollama_client.py`

**Interfaces:**
- Produces :
  - `OllamaError(RuntimeError)`
  - `Completion(text: str, latency_ms: float, eval_count: int, prompt_eval_count: int)` (dataclass frozen)
  - `OllamaClient(host: str = "http://127.0.0.1:11434", timeout_s: float = 180.0, retries: int = 2, transport: Callable[[str, str, dict | None], dict] | None = None)`
    - `generate(model: str, prompt: str, *, temperature: float = 0.0, max_tokens: int = 300, json_mode: bool = False, seed: int = 0) -> Completion`
    - `model_digest(model: str) -> str` — 12 caractères ; lève `OllamaError` nommant `ollama pull <model>` si le modèle est absent.
  - `transport(method, path, payload) -> dict` est le point d'injection des tests : `method` vaut `"GET"` ou `"POST"`.

- [ ] **Step 1: Écrire les tests**

```python
import pytest

from rag_eval_lab.ollama_client import Completion, OllamaClient, OllamaError


class FakeTransport:
    def __init__(self, responses, fail_times=0, error=TimeoutError("timeout")):
        self.responses = responses
        self.fail_times = fail_times
        self.error = error
        self.calls = []

    def __call__(self, method, path, payload):
        self.calls.append((method, path, payload))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise self.error
        return self.responses[path]


def test_generate_builds_payload_and_parses_response():
    transport = FakeTransport({"/api/generate": {"response": "  une réponse  ", "eval_count": 12,
                                                 "prompt_eval_count": 345}})
    client = OllamaClient(transport=transport)
    completion = client.generate("llama3.2:latest", "prompt", max_tokens=50)
    assert isinstance(completion, Completion)
    assert completion.text == "une réponse"
    assert (completion.eval_count, completion.prompt_eval_count) == (12, 345)
    assert completion.latency_ms >= 0
    method, path, payload = transport.calls[0]
    assert (method, path) == ("POST", "/api/generate")
    assert payload["model"] == "llama3.2:latest"
    assert payload["stream"] is False
    assert payload["options"] == {"temperature": 0.0, "num_predict": 50, "seed": 0}
    assert "format" not in payload


def test_generate_json_mode_sets_format():
    transport = FakeTransport({"/api/generate": {"response": "{}"}})
    OllamaClient(transport=transport).generate("m", "p", json_mode=True)
    assert transport.calls[0][2]["format"] == "json"


def test_generate_retries_then_succeeds():
    transport = FakeTransport({"/api/generate": {"response": "ok"}}, fail_times=2)
    client = OllamaClient(transport=transport, retries=2)
    assert client.generate("m", "p").text == "ok"
    assert len(transport.calls) == 3


def test_generate_raises_after_exhausting_retries():
    transport = FakeTransport({"/api/generate": {"response": "ok"}}, fail_times=5)
    with pytest.raises(OllamaError, match="ollama serve"):
        OllamaClient(transport=transport, retries=1).generate("m", "p")


def test_generate_rejects_a_response_without_text():
    transport = FakeTransport({"/api/generate": {"error": "boom"}})
    with pytest.raises(OllamaError, match="unexpected response"):
        OllamaClient(transport=transport).generate("m", "p")


def test_model_digest_finds_the_model():
    transport = FakeTransport({"/api/tags": {"models": [
        {"name": "llama3.1:8B", "digest": "abcdef0123456789"},
        {"name": "llama3.2:latest", "digest": "0123456789abcdef"},
    ]}})
    assert OllamaClient(transport=transport).model_digest("llama3.2:latest") == "0123456789ab"


def test_model_digest_names_the_pull_command():
    transport = FakeTransport({"/api/tags": {"models": []}})
    with pytest.raises(OllamaError, match="ollama pull mistral"):
        OllamaClient(transport=transport).model_digest("mistral")
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/test_ollama_client.py -q`
Expected: FAIL avec `ModuleNotFoundError: No module named 'rag_eval_lab.ollama_client'`

- [ ] **Step 3: Implémenter**

```python
"""Minimal HTTP client for a local Ollama server (standard library only)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

DEFAULT_HOST = "http://127.0.0.1:11434"

Transport = Callable[[str, str, dict | None], dict]


class OllamaError(RuntimeError):
    """Ollama is unreachable, the model is missing, or the response is unusable."""


@dataclass(frozen=True)
class Completion:
    text: str
    latency_ms: float
    eval_count: int
    prompt_eval_count: int


def _http_transport(host: str, timeout_s: float) -> Transport:
    def transport(method: str, path: str, payload: dict | None) -> dict:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{host}{path}", data=data, method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    return transport


class OllamaClient:
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        timeout_s: float = 180.0,
        retries: int = 2,
        transport: Transport | None = None,
    ) -> None:
        self.host = host
        self.timeout_s = timeout_s
        self.retries = retries
        self._transport = transport or _http_transport(host, timeout_s)

    def _call(self, method: str, path: str, payload: dict | None = None) -> dict:
        last: Exception | None = None
        for _ in range(self.retries + 1):
            try:
                return self._transport(method, path, payload)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
        raise OllamaError(
            f"cannot reach Ollama at {self.host} ({last}). Start it with `ollama serve`."
        ) from last

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 300,
        json_mode: bool = False,
        seed: int = 0,
    ) -> Completion:
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "seed": seed},
        }
        if json_mode:
            payload["format"] = "json"
        start = time.perf_counter()
        data = self._call("POST", "/api/generate", payload)
        latency_ms = (time.perf_counter() - start) * 1000
        if "response" not in data:
            raise OllamaError(f"unexpected response from Ollama for model '{model}': {data}")
        return Completion(
            text=str(data["response"]).strip(),
            latency_ms=latency_ms,
            eval_count=int(data.get("eval_count", 0)),
            prompt_eval_count=int(data.get("prompt_eval_count", 0)),
        )

    def model_digest(self, model: str) -> str:
        data = self._call("GET", "/api/tags")
        for entry in data.get("models", []):
            if model in (entry.get("name"), entry.get("model")):
                return str(entry.get("digest", "unknown"))[:12]
        raise OllamaError(f"model '{model}' is not installed. Run: ollama pull {model}")
```

- [ ] **Step 4: Vérifier le succès**

Run: `uv run pytest tests/test_ollama_client.py -q`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add src/rag_eval_lab/ollama_client.py tests/test_ollama_client.py
git commit -m "feat: add a standard-library Ollama client"
```

---

### Task 3: Prompts

**Files:**
- Create: `src/rag_eval_lab/prompts.py`
- Test: `tests/test_prompts.py`

**Interfaces:**
- Consumes: `make_key` de `rag_eval_lab.cache`.
- Produces :
  - `render_answer_prompt(question: str, passages: list[str]) -> str` — sans passage, demande explicitement de répondre de mémoire.
  - `render_grounding_prompt(question: str, passages: list[str], answer: str) -> str` — attend `{"faithful": bool, "relevant": bool, "reason": str}`.
  - `render_correctness_prompt(question: str, answer: str, reference: str) -> str` — attend `{"correct": bool, "reason": str}`.
  - `PROMPT_VERSION: str` — 8 caractères, haché sur le contenu des trois gabarits.

- [ ] **Step 1: Écrire les tests**

```python
from rag_eval_lab.prompts import (
    PROMPT_VERSION,
    render_answer_prompt,
    render_correctness_prompt,
    render_grounding_prompt,
)


def test_answer_prompt_numbers_passages_and_carries_the_question():
    prompt = render_answer_prompt("Qu'est-ce qu'un volcan ?", ["premier extrait", "second extrait"])
    assert "Qu'est-ce qu'un volcan ?" in prompt
    assert "[1] premier extrait" in prompt
    assert "[2] second extrait" in prompt
    assert prompt.index("[1]") < prompt.index("[2]")


def test_answer_prompt_without_passages_asks_for_memory():
    prompt = render_answer_prompt("Question ?", [])
    assert "connaissances" in prompt
    assert "[1]" not in prompt


def test_grounding_prompt_contains_answer_passages_and_json_keys():
    prompt = render_grounding_prompt("Q ?", ["extrait"], "la réponse")
    assert "extrait" in prompt and "la réponse" in prompt
    for key in ("faithful", "relevant", "reason"):
        assert key in prompt


def test_correctness_prompt_contains_the_reference():
    prompt = render_correctness_prompt("Q ?", "réponse générée", "réponse du prof")
    assert "réponse du prof" in prompt
    assert "correct" in prompt
    assert "réponse générée" in prompt


def test_prompt_version_is_short_and_stable():
    assert len(PROMPT_VERSION) == 8
    from rag_eval_lab import prompts

    assert PROMPT_VERSION == prompts.PROMPT_VERSION
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/test_prompts.py -q`
Expected: FAIL avec `ModuleNotFoundError: No module named 'rag_eval_lab.prompts'`

- [ ] **Step 3: Implémenter**

```python
"""Prompt templates. Prompts are in French because the corpus and questions are."""

from __future__ import annotations

from rag_eval_lab.cache import make_key

ANSWER_WITH_CONTEXT = """Tu es un assistant qui répond à des questions d'élèves.

Réponds à la question en t'appuyant UNIQUEMENT sur les extraits numérotés ci-dessous.
Si les extraits ne permettent pas de répondre, dis-le explicitement.
Réponds en français, en 5 phrases maximum.

Extraits :
{passages}

Question : {question}

Réponse :"""

ANSWER_WITHOUT_CONTEXT = """Tu es un assistant qui répond à des questions d'élèves.

Aucun document ne t'est fourni : réponds à partir de tes propres connaissances.
Réponds en français, en 5 phrases maximum.

Question : {question}

Réponse :"""

GROUNDING = """Tu évalues la réponse d'un assistant.

Extraits fournis à l'assistant :
{passages}

Question : {question}

Réponse de l'assistant : {answer}

Réponds en JSON strict avec exactement ces clés :
- "faithful" : true si CHAQUE affirmation de la réponse est appuyée par les extraits, false sinon.
- "relevant" : true si la réponse répond bien à la question posée, false sinon.
- "reason" : une phrase courte justifiant ton verdict.

JSON :"""

CORRECTNESS = """Tu compares deux réponses à la même question.

Question : {question}

Réponse de référence (rédigée par un enseignant) : {reference}

Réponse à évaluer : {answer}

Réponds en JSON strict avec exactement ces clés :
- "correct" : true si la réponse à évaluer est d'accord sur le fond avec la référence, false si
  elle la contredit, passe à côté de la question ou n'apporte pas l'information attendue.
  Une formulation différente n'est pas une erreur.
- "reason" : une phrase courte justifiant ton verdict.

JSON :"""

PROMPT_VERSION = make_key(ANSWER_WITH_CONTEXT, ANSWER_WITHOUT_CONTEXT, GROUNDING, CORRECTNESS)[:8]


def _numbered(passages: list[str]) -> str:
    return "\n\n".join(f"[{i}] {text}" for i, text in enumerate(passages, start=1))


def render_answer_prompt(question: str, passages: list[str]) -> str:
    if not passages:
        return ANSWER_WITHOUT_CONTEXT.format(question=question)
    return ANSWER_WITH_CONTEXT.format(passages=_numbered(passages), question=question)


def render_grounding_prompt(question: str, passages: list[str], answer: str) -> str:
    return GROUNDING.format(passages=_numbered(passages), question=question, answer=answer)


def render_correctness_prompt(question: str, answer: str, reference: str) -> str:
    return CORRECTNESS.format(question=question, answer=answer, reference=reference)
```

- [ ] **Step 4: Vérifier le succès**

Run: `uv run pytest tests/test_prompts.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/rag_eval_lab/prompts.py tests/test_prompts.py
git commit -m "feat: add French prompt templates with a version hash"
```

---

### Task 4: Sources de contexte

**Files:**
- Create: `src/rag_eval_lab/context.py`
- Test: `tests/test_context.py`

**Interfaces:**
- Consumes: `Chunk`, `Query`, `Qrels`, `Document` (types) ; `Chunker` (chunking) ; `RetrievalPipeline.search_passages` (Task 1).
- Produces :
  - `ContextSource` (Protocol) : attribut `name: str`, méthode `passages(query: Query) -> list[Chunk]`
  - `EmptyContext(name: str = "none")`
  - `RetrievedContext(name: str, pipeline: RetrievalPipeline, n_passages: int)`
  - `OracleContext(name: str, qrels: Qrels, documents: list[Document], chunker: Chunker, n_passages: int)` — prend les premiers passages de chaque document pertinent, en alternance (ordre : identifiants triés), sans utiliser la question.

- [ ] **Step 1: Écrire les tests**

```python
from rag_eval_lab.chunking import FixedChunker
from rag_eval_lab.context import EmptyContext, OracleContext, RetrievedContext
from rag_eval_lab.pipeline import RetrievalPipeline
from rag_eval_lab.retrieval import BM25Retriever
from rag_eval_lab.types import Document, Query


def test_empty_context_returns_nothing():
    source = EmptyContext()
    assert source.name == "none"
    assert source.passages(Query("q1", "une question")) == []


def test_retrieved_context_returns_n_passages(mini_dataset):
    pipeline = RetrievalPipeline(FixedChunker(size=200, overlap=50), BM25Retriever())
    pipeline.index(mini_dataset.documents)
    source = RetrievedContext("bm25", pipeline, n_passages=3)
    passages = source.passages(mini_dataset.queries[0])
    assert len(passages) == 3
    assert passages[0].doc_id == "d0"


def test_oracle_context_uses_gold_documents_only(mini_dataset):
    source = OracleContext(
        "oracle", mini_dataset.qrels, mini_dataset.documents,
        FixedChunker(size=100, overlap=0), n_passages=2,
    )
    passages = source.passages(mini_dataset.queries[1])
    assert len(passages) == 2
    assert {p.doc_id for p in passages} == {"d1"}


def test_oracle_context_alternates_between_gold_documents():
    documents = [Document("d1", "a" * 300), Document("d2", "b" * 300)]
    qrels = {"q1": {"d1": 1, "d2": 1}}
    source = OracleContext("oracle", qrels, documents, FixedChunker(size=100, overlap=0), 3)
    passages = source.passages(Query("q1", "peu importe"))
    assert [p.doc_id for p in passages] == ["d1", "d2", "d1"]


def test_oracle_context_ignores_the_question_text():
    documents = [Document("d1", "x" * 250)]
    source = OracleContext("oracle", {"q1": {"d1": 1}}, documents,
                           FixedChunker(size=100, overlap=0), 2)
    first = source.passages(Query("q1", "question A"))
    second = source.passages(Query("q1", "question B totalement différente"))
    assert [p.id for p in first] == [p.id for p in second]


def test_oracle_context_without_gold_documents_returns_nothing():
    source = OracleContext("oracle", {}, [Document("d1", "texte")],
                           FixedChunker(size=100, overlap=0), 2)
    assert source.passages(Query("q1", "q")) == []
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/test_context.py -q`
Expected: FAIL avec `ModuleNotFoundError: No module named 'rag_eval_lab.context'`

- [ ] **Step 3: Implémenter**

```python
"""Context sources: what passages a generator is given for a query."""

from __future__ import annotations

from typing import Protocol

from rag_eval_lab.chunking import Chunker
from rag_eval_lab.pipeline import RetrievalPipeline
from rag_eval_lab.types import Chunk, Document, Qrels, Query


class ContextSource(Protocol):
    name: str

    def passages(self, query: Query) -> list[Chunk]: ...


class EmptyContext:
    """No passage at all: measures what the generator answers from memory."""

    def __init__(self, name: str = "none") -> None:
        self.name = name

    def passages(self, query: Query) -> list[Chunk]:
        return []


class RetrievedContext:
    """Top passages from a v1 retrieval pipeline."""

    def __init__(self, name: str, pipeline: RetrievalPipeline, n_passages: int) -> None:
        self.name = name
        self.pipeline = pipeline
        self.n_passages = n_passages

    def passages(self, query: Query) -> list[Chunk]:
        return [hit.chunk for hit in self.pipeline.search_passages(query.text, self.n_passages)]


class OracleContext:
    """Upper bound: the first chunks of the labelled documents, round-robin.

    Passages are never selected by similarity to the question — that would build a
    perfect context out of the question itself and inflate the bound.
    """

    def __init__(
        self,
        name: str,
        qrels: Qrels,
        documents: list[Document],
        chunker: Chunker,
        n_passages: int,
    ) -> None:
        self.name = name
        self.qrels = qrels
        self.chunker = chunker
        self.n_passages = n_passages
        self._chunks = {doc.id: chunker.chunk(doc) for doc in documents}

    def passages(self, query: Query) -> list[Chunk]:
        gold = sorted(d for d, gain in self.qrels.get(query.id, {}).items() if gain > 0)
        selected: list[Chunk] = []
        position = 0
        while len(selected) < self.n_passages:
            available = [d for d in gold if len(self._chunks.get(d, [])) > position]
            if not available:
                break
            for doc_id in available:
                selected.append(self._chunks[doc_id][position])
                if len(selected) == self.n_passages:
                    break
            position += 1
        return selected
```

- [ ] **Step 4: Vérifier le succès**

Run: `uv run pytest tests/test_context.py -q`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/rag_eval_lab/context.py tests/test_context.py
git commit -m "feat: add empty, retrieved and oracle context sources"
```

---

### Task 5: Générateur

**Files:**
- Create: `src/rag_eval_lab/generation.py`
- Test: `tests/test_generation.py`

**Interfaces:**
- Consumes: `OllamaClient`, `Completion` (Task 2) ; `render_answer_prompt` (Task 3) ; `Chunk` (types).
- Produces :
  - `Generator` (Protocol) : `model: str`, `answer(question: str, passages: list[Chunk]) -> Completion`
  - `OllamaGenerator(client: OllamaClient, model: str, max_tokens: int = 300, temperature: float = 0.0)`

- [ ] **Step 1: Écrire les tests**

```python
from rag_eval_lab.generation import OllamaGenerator
from rag_eval_lab.ollama_client import Completion, OllamaClient
from rag_eval_lab.types import Chunk


class RecordingTransport:
    def __init__(self):
        self.payloads = []

    def __call__(self, method, path, payload):
        self.payloads.append(payload)
        return {"response": "réponse générée", "eval_count": 7, "prompt_eval_count": 100}


def _generator(transport, **kwargs):
    return OllamaGenerator(OllamaClient(transport=transport), "llama3.2:latest", **kwargs)


def test_generator_sends_the_passages_and_returns_a_completion():
    transport = RecordingTransport()
    completion = _generator(transport, max_tokens=120).answer(
        "Qu'est-ce qu'un volcan ?", [Chunk("d1#0", "d1", "Un volcan est une ouverture.")]
    )
    assert isinstance(completion, Completion)
    assert completion.text == "réponse générée"
    prompt = transport.payloads[0]["prompt"]
    assert "Un volcan est une ouverture." in prompt
    assert "Qu'est-ce qu'un volcan ?" in prompt
    assert transport.payloads[0]["options"]["num_predict"] == 120
    assert transport.payloads[0]["options"]["temperature"] == 0.0


def test_generator_without_passages_uses_the_memory_prompt():
    transport = RecordingTransport()
    _generator(transport).answer("Question ?", [])
    assert "connaissances" in transport.payloads[0]["prompt"]


def test_generator_exposes_its_model():
    assert _generator(RecordingTransport()).model == "llama3.2:latest"
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/test_generation.py -q`
Expected: FAIL avec `ModuleNotFoundError: No module named 'rag_eval_lab.generation'`

- [ ] **Step 3: Implémenter**

```python
"""Answer generation on top of a context source."""

from __future__ import annotations

from typing import Protocol

from rag_eval_lab.ollama_client import Completion, OllamaClient
from rag_eval_lab.prompts import render_answer_prompt
from rag_eval_lab.types import Chunk


class Generator(Protocol):
    model: str

    def answer(self, question: str, passages: list[Chunk]) -> Completion: ...


class OllamaGenerator:
    def __init__(
        self,
        client: OllamaClient,
        model: str,
        max_tokens: int = 300,
        temperature: float = 0.0,
    ) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def answer(self, question: str, passages: list[Chunk]) -> Completion:
        prompt = render_answer_prompt(question, [p.text for p in passages])
        return self.client.generate(
            self.model, prompt, temperature=self.temperature, max_tokens=self.max_tokens
        )
```

- [ ] **Step 4: Vérifier le succès**

Run: `uv run pytest tests/test_generation.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/rag_eval_lab/generation.py tests/test_generation.py
git commit -m "feat: add the Ollama answer generator"
```

---

### Task 6: Juge

**Files:**
- Create: `src/rag_eval_lab/judging.py`
- Test: `tests/test_judging.py`

**Interfaces:**
- Consumes: `OllamaClient` (Task 2) ; `render_grounding_prompt`, `render_correctness_prompt` (Task 3) ; `Chunk` (types).
- Produces :
  - `JudgeParseError(RuntimeError)` — porte `raw_output: str`.
  - `Verdict(faithfulness: bool | None, relevance: bool | None, correctness: bool | None, rationales: dict[str, str], latency_ms: float)` (dataclass frozen)
  - `Judge` (Protocol) : `model: str`, `verdict(question, passages, answer, reference) -> Verdict`
  - `OllamaJudge(client: OllamaClient, model: str, max_tokens: int = 200, retries: int = 2)`
    - sans passage, l'appel de fidélité est sauté : `faithfulness` et `relevance` valent `None` ;
    - sans réponse de référence (chaîne vide), `correctness` vaut `None`.
  - `extract_json(text: str) -> dict` — tolère du texte autour de l'objet JSON.

- [ ] **Step 1: Écrire les tests**

```python
import pytest

from rag_eval_lab.judging import JudgeParseError, OllamaJudge, Verdict, extract_json
from rag_eval_lab.ollama_client import OllamaClient
from rag_eval_lab.types import Chunk

PASSAGES = [Chunk("d1#0", "d1", "Un volcan est une ouverture de la croûte terrestre.")]


class ScriptedTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def __call__(self, method, path, payload):
        self.prompts.append(payload["prompt"])
        return {"response": self.responses.pop(0)}


def _judge(transport, **kwargs):
    return OllamaJudge(OllamaClient(transport=transport, retries=0), "llama3.1:8B", **kwargs)


def test_extract_json_tolerates_surrounding_text():
    assert extract_json('Voici : {"correct": true} merci') == {"correct": True}


def test_extract_json_rejects_garbage():
    with pytest.raises(JudgeParseError):
        extract_json("pas de json ici")


def test_verdict_combines_both_calls():
    transport = ScriptedTransport([
        '{"faithful": true, "relevant": true, "reason": "appuyé par [1]"}',
        '{"correct": false, "reason": "contredit la référence"}',
    ])
    verdict = _judge(transport).verdict("Q ?", PASSAGES, "une réponse", "la référence")
    assert isinstance(verdict, Verdict)
    assert (verdict.faithfulness, verdict.relevance, verdict.correctness) == (True, True, False)
    assert verdict.rationales["faithfulness"] == "appuyé par [1]"
    assert verdict.rationales["correctness"] == "contredit la référence"
    assert verdict.latency_ms > 0
    assert len(transport.prompts) == 2


def test_verdict_without_passages_skips_grounding():
    transport = ScriptedTransport(['{"correct": true, "reason": "ok"}'])
    verdict = _judge(transport).verdict("Q ?", [], "une réponse", "la référence")
    assert (verdict.faithfulness, verdict.relevance) == (None, None)
    assert verdict.correctness is True
    assert len(transport.prompts) == 1


def test_verdict_without_reference_skips_correctness():
    transport = ScriptedTransport(['{"faithful": false, "relevant": true, "reason": "r"}'])
    verdict = _judge(transport).verdict("Q ?", PASSAGES, "une réponse", "")
    assert verdict.correctness is None
    assert verdict.faithfulness is False


def test_judge_retries_on_invalid_json_then_succeeds():
    transport = ScriptedTransport([
        "je ne sais pas",
        '{"faithful": true, "relevant": false, "reason": "r"}',
        '{"correct": true, "reason": "r"}',
    ])
    verdict = _judge(transport, retries=1).verdict("Q ?", PASSAGES, "a", "ref")
    assert verdict.relevance is False
    assert len(transport.prompts) == 3


def test_judge_raises_with_the_raw_output_after_retries():
    transport = ScriptedTransport(["bruit", "encore du bruit"])
    with pytest.raises(JudgeParseError) as excinfo:
        _judge(transport, retries=1).verdict("Q ?", PASSAGES, "a", "ref")
    assert excinfo.value.raw_output == "encore du bruit"


def test_judge_coerces_string_booleans():
    transport = ScriptedTransport([
        '{"faithful": "true", "relevant": "false", "reason": "r"}',
        '{"correct": "true", "reason": "r"}',
    ])
    verdict = _judge(transport).verdict("Q ?", PASSAGES, "a", "ref")
    assert (verdict.faithfulness, verdict.relevance, verdict.correctness) == (True, False, True)
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/test_judging.py -q`
Expected: FAIL avec `ModuleNotFoundError: No module named 'rag_eval_lab.judging'`

- [ ] **Step 3: Implémenter**

```python
"""LLM-as-a-judge: three binary verdicts per answer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from rag_eval_lab.ollama_client import OllamaClient
from rag_eval_lab.prompts import render_correctness_prompt, render_grounding_prompt
from rag_eval_lab.types import Chunk


class JudgeParseError(RuntimeError):
    def __init__(self, message: str, raw_output: str) -> None:
        super().__init__(message)
        self.raw_output = raw_output


@dataclass(frozen=True)
class Verdict:
    faithfulness: bool | None
    relevance: bool | None
    correctness: bool | None
    rationales: dict[str, str]
    latency_ms: float


def extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise JudgeParseError("no JSON object in judge output", text)
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise JudgeParseError(f"invalid JSON in judge output: {exc}", text) from exc
    if not isinstance(value, dict):
        raise JudgeParseError("judge output is not a JSON object", text)
    return value


def _as_bool(value: object, key: str, raw: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise JudgeParseError(f"key '{key}' is not a boolean: {value!r}", raw)


class Judge(Protocol):
    model: str

    def verdict(
        self, question: str, passages: list[Chunk], answer: str, reference: str
    ) -> Verdict: ...


class OllamaJudge:
    def __init__(
        self, client: OllamaClient, model: str, max_tokens: int = 200, retries: int = 2
    ) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.retries = retries

    def _ask(self, prompt: str, keys: tuple[str, ...]) -> tuple[dict, float]:
        last: JudgeParseError | None = None
        latency = 0.0
        for _ in range(self.retries + 1):
            completion = self.client.generate(
                self.model, prompt, temperature=0.0, max_tokens=self.max_tokens, json_mode=True
            )
            latency += completion.latency_ms
            try:
                payload = extract_json(completion.text)
                return ({key: payload[key] for key in keys} | {"raw": completion.text}, latency)
            except (JudgeParseError, KeyError) as exc:
                last = (
                    exc
                    if isinstance(exc, JudgeParseError)
                    else JudgeParseError(f"missing key {exc}", completion.text)
                )
        raise last  # type: ignore[misc]

    def verdict(
        self, question: str, passages: list[Chunk], answer: str, reference: str
    ) -> Verdict:
        faithfulness: bool | None = None
        relevance: bool | None = None
        correctness: bool | None = None
        rationales: dict[str, str] = {}
        latency = 0.0

        if passages:
            prompt = render_grounding_prompt(question, [p.text for p in passages], answer)
            payload, took = self._ask(prompt, ("faithful", "relevant", "reason"))
            latency += took
            faithfulness = _as_bool(payload["faithful"], "faithful", payload["raw"])
            relevance = _as_bool(payload["relevant"], "relevant", payload["raw"])
            rationales["faithfulness"] = str(payload["reason"])

        if reference:
            prompt = render_correctness_prompt(question, answer, reference)
            payload, took = self._ask(prompt, ("correct", "reason"))
            latency += took
            correctness = _as_bool(payload["correct"], "correct", payload["raw"])
            rationales["correctness"] = str(payload["reason"])

        return Verdict(faithfulness, relevance, correctness, rationales, latency)
```

- [ ] **Step 4: Vérifier le succès**

Run: `uv run pytest tests/test_judging.py -q`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add src/rag_eval_lab/judging.py tests/test_judging.py
git commit -m "feat: add the Ollama judge with tolerant JSON parsing"
```

---

### Task 7: Configuration de la v2

**Files:**
- Create: `src/rag_eval_lab/generation_config.py`
- Test: `tests/test_generation_config.py`

**Interfaces:**
- Consumes: `DatasetConfig` (config v1) ; `make_key` (cache) ; `PROMPT_VERSION` (Task 3) ; `parse_chunker` (chunking) ; `build_retriever` (factory).
- Produces :
  - `GeneratorConfig(model: str, temperature: float = 0.0, max_tokens: int = 300, timeout_s: float = 180.0)`
  - `JudgeConfig(model: str, temperature: float = 0.0, max_tokens: int = 200, timeout_s: float = 300.0)`
  - `ContextConfig(name: str, kind: Literal["empty", "retrieved", "oracle"], chunker: str | None = None, retriever: str | None = None)` — `retrieved` exige les deux, les autres les interdisent ; les noms de composants sont validés.
  - `AnnotationConfig(n: int = 60, seed: int = 7)`
  - `GenerationConfig(dataset, contexts, generator, judge, n_passages: int = 5, oracle_chunker: str = "fixed_1000_200", annotation, cache_dir: str = ".cache/embeddings")` avec `answer_id(context_name: str, query_id: str) -> str`
  - `load_generation_config(path: Path) -> GenerationConfig`
  - Toutes les classes refusent les champs inconnus.

- [ ] **Step 1: Écrire les tests**

```python
import pytest
import yaml
from pydantic import ValidationError

from rag_eval_lab.generation_config import GenerationConfig, load_generation_config

BASE = {
    "dataset": {"name": "alloprof", "n_queries": 80, "seed": 42},
    "n_passages": 5,
    "generator": {"model": "llama3.2:latest"},
    "judge": {"model": "llama3.1:8B"},
    "contexts": [
        {"name": "none", "kind": "empty"},
        {"name": "bm25", "kind": "retrieved", "chunker": "none", "retriever": "bm25"},
        {"name": "oracle", "kind": "oracle"},
    ],
    "annotation": {"n": 60, "seed": 7},
}


def _config(**overrides) -> GenerationConfig:
    return GenerationConfig.model_validate({**BASE, **overrides})


def test_defaults_and_accessors():
    cfg = _config()
    assert cfg.n_passages == 5
    assert cfg.generator.max_tokens == 300
    assert cfg.judge.timeout_s == 300.0
    assert cfg.oracle_chunker == "fixed_1000_200"
    assert [c.name for c in cfg.contexts] == ["none", "bm25", "oracle"]


def test_answer_id_is_stable_and_discriminates():
    cfg = _config()
    first = cfg.answer_id("bm25", "q1")
    assert first == _config().answer_id("bm25", "q1")
    assert len(first) == 12
    assert first != cfg.answer_id("oracle", "q1")
    assert first != cfg.answer_id("bm25", "q2")
    assert first != _config(generator={"model": "llama3.1:8B"}).answer_id("bm25", "q1")


def test_retrieved_context_requires_chunker_and_retriever():
    with pytest.raises(ValidationError, match="retrieved"):
        _config(contexts=[{"name": "x", "kind": "retrieved", "chunker": "none"}])


def test_empty_context_rejects_retrieval_fields():
    with pytest.raises(ValidationError, match="empty"):
        _config(contexts=[{"name": "x", "kind": "empty", "retriever": "bm25"}])


def test_unknown_retriever_is_rejected():
    with pytest.raises(ValidationError, match="Unknown encoder"):
        _config(contexts=[{"name": "x", "kind": "retrieved", "chunker": "none",
                           "retriever": "dense:nope"}])


def test_unknown_chunker_is_rejected():
    with pytest.raises(ValidationError, match="Unknown chunker"):
        _config(contexts=[{"name": "x", "kind": "retrieved", "chunker": "banana",
                           "retriever": "bm25"}])


def test_duplicate_context_names_are_rejected():
    with pytest.raises(ValidationError, match="unique"):
        _config(contexts=[{"name": "a", "kind": "empty"}, {"name": "a", "kind": "oracle"}])


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        _config(typo=1)


def test_load_generation_config(tmp_path):
    path = tmp_path / "v2.yaml"
    path.write_text(yaml.safe_dump(BASE), encoding="utf-8")
    assert load_generation_config(path).generator.model == "llama3.2:latest"
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/test_generation_config.py -q`
Expected: FAIL avec `ModuleNotFoundError: No module named 'rag_eval_lab.generation_config'`

- [ ] **Step 3: Implémenter**

```python
"""Configuration for the generation benchmark, validated before any model call."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_eval_lab.cache import make_key
from rag_eval_lab.chunking import parse_chunker
from rag_eval_lab.config import DatasetConfig
from rag_eval_lab.factory import build_retriever
from rag_eval_lab.prompts import PROMPT_VERSION


class GeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    temperature: float = 0.0
    max_tokens: int = Field(default=300, gt=0)
    timeout_s: float = Field(default=180.0, gt=0)


class JudgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    temperature: float = 0.0
    max_tokens: int = Field(default=200, gt=0)
    timeout_s: float = Field(default=300.0, gt=0)


class ContextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["empty", "retrieved", "oracle"]
    chunker: str | None = None
    retriever: str | None = None

    @model_validator(mode="after")
    def _check(self) -> ContextConfig:
        if self.kind == "retrieved":
            if not self.chunker or not self.retriever:
                raise ValueError(
                    f"context '{self.name}': kind 'retrieved' requires 'chunker' and 'retriever'"
                )
            parse_chunker(self.chunker)
            build_retriever(self.retriever)
        elif self.chunker or self.retriever:
            raise ValueError(
                f"context '{self.name}': kind '{self.kind}' takes no chunker or retriever"
            )
        return self


class AnnotationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: int = Field(default=60, gt=0)
    seed: int = 7


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: DatasetConfig
    contexts: list[ContextConfig] = Field(min_length=1)
    generator: GeneratorConfig
    judge: JudgeConfig
    n_passages: int = Field(default=5, gt=0)
    oracle_chunker: str = "fixed_1000_200"
    annotation: AnnotationConfig = Field(default_factory=AnnotationConfig)
    cache_dir: str = ".cache/embeddings"

    @model_validator(mode="after")
    def _check(self) -> GenerationConfig:
        names = [c.name for c in self.contexts]
        if len(set(names)) != len(names):
            raise ValueError(f"context names must be unique, got {names}")
        parse_chunker(self.oracle_chunker)
        return self

    def answer_id(self, context_name: str, query_id: str) -> str:
        return make_key(context_name, query_id, self.generator.model, PROMPT_VERSION)[:12]


def load_generation_config(path: Path) -> GenerationConfig:
    with Path(path).open(encoding="utf-8") as handle:
        return GenerationConfig.model_validate(yaml.safe_load(handle))
```

- [ ] **Step 4: Vérifier le succès**

Run: `uv run pytest tests/test_generation_config.py -q`
Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add src/rag_eval_lab/generation_config.py tests/test_generation_config.py
git commit -m "feat: add the validated generation config"
```

---

### Task 8: Étape de génération

**Files:**
- Create: `src/rag_eval_lab/generation_experiment.py`
- Test: `tests/test_generation_experiment.py`

**Interfaces:**
- Consumes: `GenerationConfig` (Task 7) ; `ContextSource` et ses trois implémentations (Task 4) ; `OllamaGenerator` (Task 5) ; `OllamaClient` (Task 2) ; `load_benchmark`, `sample_dataset` (data) ; `RetrievalPipeline`, `parse_chunker`, `build_retriever`, `EmbeddingCache` (v1) ; `environment_info` (experiment).
- Produces :
  - `V1_SAMPLE_SIZE = 300`
  - `build_dataset(cfg: GenerationConfig) -> Dataset` — échantillon v1 (300, graine 42) tronqué aux `n_queries` premières questions.
  - `build_context_sources(cfg: GenerationConfig, dataset: Dataset) -> list[ContextSource]`
  - `run_generation(cfg: GenerationConfig, out_dir: Path, client: OllamaClient | None = None, dataset: Dataset | None = None) -> list[dict]`
  - Enregistrement de `answers.jsonl` : `answer_id`, `context`, `query_id`, `question`, `reference` (la réponse du prof, copiée depuis `Query.answer`), `answer`, `passages` (liste de `{"id", "text"}`), `n_passage_chars`, `latency_ms`, `eval_count`, `generator` (`{"model", "digest", "max_tokens", "temperature"}`), `prompt_version`, `status`, `timestamp` ; `error` si échec.
  - Stocker `reference` ici rend les étapes suivantes autonomes : juger et annoter ne relisent jamais le jeu de données.

- [ ] **Step 1: Écrire les tests**

```python
import json

from rag_eval_lab import generation_experiment
from rag_eval_lab.generation_config import GenerationConfig
from rag_eval_lab.generation_experiment import build_dataset, run_generation
from rag_eval_lab.ollama_client import OllamaClient


class FakeTransport:
    def __init__(self, fail_on=()):
        self.fail_on = fail_on
        self.prompts = []

    def __call__(self, method, path, payload):
        if path == "/api/tags":
            return {"models": [{"name": "fake-model", "digest": "0123456789abcdef"}]}
        self.prompts.append(payload["prompt"])
        if any(marker in payload["prompt"] for marker in self.fail_on):
            raise TimeoutError("too slow")
        return {"response": "réponse", "eval_count": 5, "prompt_eval_count": 50}


def _config(mini_jsonl, tmp_path, **overrides) -> GenerationConfig:
    data = {
        "dataset": {"name": "jsonl", "path": str(mini_jsonl), "n_queries": 2},
        "n_passages": 2,
        "oracle_chunker": "fixed_200_50",
        "generator": {"model": "fake-model"},
        "judge": {"model": "fake-model"},
        "cache_dir": str(tmp_path / "cache"),
        "contexts": [
            {"name": "none", "kind": "empty"},
            {"name": "bm25", "kind": "retrieved", "chunker": "fixed_200_50", "retriever": "bm25"},
            {"name": "oracle", "kind": "oracle"},
        ],
    }
    data.update(overrides)
    return GenerationConfig.model_validate(data)


def _client(transport):
    return OllamaClient(transport=transport, retries=0)


def test_build_dataset_truncates_to_n_queries(mini_jsonl, tmp_path):
    dataset = build_dataset(_config(mini_jsonl, tmp_path))
    assert len(dataset.queries) == 2
    assert set(dataset.qrels) == {q.id for q in dataset.queries}


def test_run_generation_writes_one_record_per_context_and_query(mini_jsonl, tmp_path):
    out = tmp_path / "out"
    records = run_generation(_config(mini_jsonl, tmp_path), out, client=_client(FakeTransport()))
    assert len(records) == 6
    assert {r["status"] for r in records} == {"ok"}
    assert {r["context"] for r in records} == {"none", "bm25", "oracle"}
    lines = (out / "answers.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6
    record = json.loads(lines[0])
    assert record["generator"]["digest"] == "0123456789ab"
    assert record["prompt_version"]
    assert record["answer"] == "réponse"
    assert record["reference"] == "Le volcan est un sujet important."


def test_run_generation_stores_passage_texts(mini_jsonl, tmp_path):
    records = run_generation(
        _config(mini_jsonl, tmp_path), tmp_path / "out", client=_client(FakeTransport())
    )
    empty = next(r for r in records if r["context"] == "none")
    retrieved = next(r for r in records if r["context"] == "bm25")
    assert empty["passages"] == []
    assert len(retrieved["passages"]) == 2
    assert all(p["text"] for p in retrieved["passages"])
    assert retrieved["n_passage_chars"] == sum(len(p["text"]) for p in retrieved["passages"])


def test_run_generation_resumes(mini_jsonl, tmp_path):
    out = tmp_path / "out"
    cfg = _config(mini_jsonl, tmp_path)
    run_generation(cfg, out, client=_client(FakeTransport()))
    assert run_generation(cfg, out, client=_client(FakeTransport())) == []
    assert len((out / "answers.jsonl").read_text(encoding="utf-8").splitlines()) == 6


def test_run_generation_isolates_errors_and_retries_them(mini_jsonl, tmp_path):
    out = tmp_path / "out"
    cfg = _config(mini_jsonl, tmp_path)
    records = run_generation(cfg, out, client=_client(FakeTransport(fail_on=("connaissances",))))
    statuses = {r["context"]: r["status"] for r in records}
    assert statuses["none"] == "error"
    assert statuses["bm25"] == "ok"
    failed = next(r for r in records if r["status"] == "error")
    assert "OllamaError" in failed["error"]
    retried = run_generation(cfg, out, client=_client(FakeTransport()))
    assert {r["context"] for r in retried} == {"none"}
    assert {r["status"] for r in retried} == {"ok"}
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/test_generation_experiment.py -q`
Expected: FAIL avec `ModuleNotFoundError: No module named 'rag_eval_lab.generation_experiment'`

- [ ] **Step 3: Implémenter**

```python
"""Generation stage: one answer per (context, query), resumable and error-isolated."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from rag_eval_lab.cache import EmbeddingCache
from rag_eval_lab.chunking import parse_chunker
from rag_eval_lab.context import ContextSource, EmptyContext, OracleContext, RetrievedContext
from rag_eval_lab.data import load_benchmark
from rag_eval_lab.factory import build_retriever
from rag_eval_lab.generation import OllamaGenerator
from rag_eval_lab.generation_config import GenerationConfig
from rag_eval_lab.ollama_client import OllamaClient, OllamaError
from rag_eval_lab.pipeline import RetrievalPipeline
from rag_eval_lab.prompts import PROMPT_VERSION
from rag_eval_lab.types import Dataset

logger = logging.getLogger(__name__)

V1_SAMPLE_SIZE = 300
"""The v1 benchmark sampled this many queries; v2 keeps a prefix of that sample."""


def build_dataset(cfg: GenerationConfig) -> Dataset:
    requested = cfg.dataset.n_queries
    base = load_benchmark(cfg.dataset.model_copy(update={"n_queries": V1_SAMPLE_SIZE}))
    queries = base.queries if requested is None else base.queries[:requested]
    qrels = {q.id: base.qrels[q.id] for q in queries}
    return Dataset(name=base.name, documents=base.documents, queries=queries, qrels=qrels)


def build_context_sources(cfg: GenerationConfig, dataset: Dataset) -> list[ContextSource]:
    cache = EmbeddingCache(Path(cfg.cache_dir))
    sources: list[ContextSource] = []
    for context in cfg.contexts:
        if context.kind == "empty":
            sources.append(EmptyContext(context.name))
        elif context.kind == "oracle":
            sources.append(
                OracleContext(
                    context.name, dataset.qrels, dataset.documents,
                    parse_chunker(cfg.oracle_chunker), cfg.n_passages,
                )
            )
        else:
            pipeline = RetrievalPipeline(
                chunker=parse_chunker(context.chunker or ""),
                retriever=build_retriever(context.retriever or "", cache),
            )
            logger.info("indexing context '%s'", context.name)
            pipeline.index(dataset.documents)
            sources.append(RetrievedContext(context.name, pipeline, cfg.n_passages))
    return sources


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("status") == "ok":
                done.add(record["answer_id"])
    return done


def _append(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_generation(
    cfg: GenerationConfig,
    out_dir: Path,
    client: OllamaClient | None = None,
    dataset: Dataset | None = None,
) -> list[dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    answers_path = out_dir / "answers.jsonl"
    done = _completed_ids(answers_path)

    client = client or OllamaClient(timeout_s=cfg.generator.timeout_s)
    digest = client.model_digest(cfg.generator.model)
    generator = OllamaGenerator(
        client, cfg.generator.model, cfg.generator.max_tokens, cfg.generator.temperature
    )
    dataset = dataset or build_dataset(cfg)
    sources = build_context_sources(cfg, dataset)

    records: list[dict] = []
    for source in sources:
        for n, query in enumerate(dataset.queries, start=1):
            answer_id = cfg.answer_id(source.name, query.id)
            if answer_id in done:
                continue
            record = {
                "answer_id": answer_id,
                "context": source.name,
                "query_id": query.id,
                "question": query.text,
                "reference": query.answer,
                "prompt_version": PROMPT_VERSION,
                "generator": {
                    "model": cfg.generator.model,
                    "digest": digest,
                    "max_tokens": cfg.generator.max_tokens,
                    "temperature": cfg.generator.temperature,
                },
            }
            try:
                passages = source.passages(query)
                completion = generator.answer(query.text, passages)
                record.update(
                    status="ok",
                    answer=completion.text,
                    passages=[{"id": p.id, "text": p.text} for p in passages],
                    n_passage_chars=sum(len(p.text) for p in passages),
                    latency_ms=completion.latency_ms,
                    eval_count=completion.eval_count,
                )
            except (OllamaError, ValueError) as exc:
                logger.exception("generation failed for %s / %s", source.name, query.id)
                record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            record["timestamp"] = datetime.now(UTC).isoformat()
            _append(answers_path, record)
            records.append(record)
            if n % 10 == 0:
                logger.info("context %s: %d/%d", source.name, n, len(dataset.queries))
    return records
```

- [ ] **Step 4: Vérifier le succès**

Run: `uv run pytest tests/test_generation_experiment.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/rag_eval_lab/generation_experiment.py tests/test_generation_experiment.py
git commit -m "feat: add the resumable generation stage"
```

---

### Task 9: Étape de jugement

**Files:**
- Modify: `src/rag_eval_lab/generation_experiment.py`
- Test: `tests/test_generation_experiment.py` (ajouts)

**Interfaces:**
- Consumes: tout ce que produit la Task 8 ; `OllamaJudge`, `JudgeParseError`, `Verdict` (Task 6).
- Produces :
  - `load_answers(out_dir: Path) -> list[dict]` — dernier enregistrement par `answer_id`, uniquement `status == "ok"`.
  - `run_judging(cfg, out_dir, client=None) -> list[dict]` — la réponse de référence est lue dans le champ `reference` de `answers.jsonl`, donc aucune relecture du jeu de données.
  - Enregistrement de `judgments.jsonl` : `answer_id`, `judge` (`{"model", "digest"}`), `prompt_version`, `faithfulness`, `relevance`, `correctness` (`true`/`false`/`null`), `rationales`, `latency_ms`, `status`, `timestamp` ; `error` et `raw_output` si échec.

- [ ] **Step 1: Écrire les tests**

```python
import json

from rag_eval_lab.generation_experiment import load_answers, run_generation, run_judging


class JudgeTransport:
    def __init__(self, bad_json_for=()):
        self.bad_json_for = bad_json_for
        self.prompts = []

    def __call__(self, method, path, payload):
        if path == "/api/tags":
            return {"models": [{"name": "fake-model", "digest": "0123456789abcdef"}]}
        prompt = payload["prompt"]
        self.prompts.append(prompt)
        if any(marker in prompt for marker in self.bad_json_for):
            return {"response": "je ne sais pas"}
        if "faithful" in prompt:
            return {"response": '{"faithful": true, "relevant": true, "reason": "ok"}'}
        return {"response": '{"correct": false, "reason": "non"}'}


def test_run_judging_judges_every_answer(mini_jsonl, tmp_path):
    out = tmp_path / "out"
    cfg = _config(mini_jsonl, tmp_path)
    run_generation(cfg, out, client=_client(FakeTransport()))
    records = run_judging(cfg, out, client=_client(JudgeTransport()))
    assert len(records) == 6
    assert {r["status"] for r in records} == {"ok"}
    by_context = {json.loads(line)["answer_id"]: json.loads(line)
                  for line in (out / "judgments.jsonl").read_text(encoding="utf-8").splitlines()}
    answers = {a["answer_id"]: a for a in load_answers(out)}
    for answer_id, judgment in by_context.items():
        if answers[answer_id]["context"] == "none":
            assert judgment["faithfulness"] is None
            assert judgment["relevance"] is None
        else:
            assert judgment["faithfulness"] is True
        assert judgment["correctness"] is False
        assert judgment["judge"]["digest"] == "0123456789ab"


def test_run_judging_resumes(mini_jsonl, tmp_path):
    out = tmp_path / "out"
    cfg = _config(mini_jsonl, tmp_path)
    run_generation(cfg, out, client=_client(FakeTransport()))
    run_judging(cfg, out, client=_client(JudgeTransport()))
    assert run_judging(cfg, out, client=_client(JudgeTransport())) == []


def test_run_judging_records_parse_errors(mini_jsonl, tmp_path):
    out = tmp_path / "out"
    cfg = _config(mini_jsonl, tmp_path)
    run_generation(cfg, out, client=_client(FakeTransport()))
    records = run_judging(cfg, out, client=_client(JudgeTransport(bad_json_for=("faithful",))))
    failed = [r for r in records if r["status"] == "error"]
    assert failed, "grounding calls should have failed"
    assert failed[0]["raw_output"] == "je ne sais pas"
    assert all(r["status"] == "ok" for r in records if r["answer_id"] not in
               {f["answer_id"] for f in failed})


def test_load_answers_keeps_only_successful_records(mini_jsonl, tmp_path):
    out = tmp_path / "out"
    cfg = _config(mini_jsonl, tmp_path)
    run_generation(cfg, out, client=_client(FakeTransport(fail_on=("connaissances",))))
    answers = load_answers(out)
    assert {a["context"] for a in answers} == {"bm25", "oracle"}
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/test_generation_experiment.py -q -k judging`
Expected: FAIL avec `ImportError: cannot import name 'run_judging'`

- [ ] **Step 3: Implémenter**

Ajouter à `src/rag_eval_lab/generation_experiment.py` (et compléter les imports du module avec
`from rag_eval_lab.judging import JudgeParseError, OllamaJudge` et
`from rag_eval_lab.types import Chunk`) :

```python
def load_answers(out_dir: Path) -> list[dict]:
    """Last record per answer_id, successful ones only."""
    path = Path(out_dir) / "answers.jsonl"
    latest: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                latest[record["answer_id"]] = record
    return [r for r in latest.values() if r.get("status") == "ok"]


def run_judging(
    cfg: GenerationConfig, out_dir: Path, client: OllamaClient | None = None
) -> list[dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    judgments_path = out_dir / "judgments.jsonl"
    done = _completed_ids(judgments_path)

    client = client or OllamaClient(timeout_s=cfg.judge.timeout_s)
    digest = client.model_digest(cfg.judge.model)
    judge = OllamaJudge(client, cfg.judge.model, cfg.judge.max_tokens)

    records: list[dict] = []
    answers = [a for a in load_answers(out_dir) if a["answer_id"] not in done]
    for n, answer in enumerate(answers, start=1):
        record = {
            "answer_id": answer["answer_id"],
            "judge": {"model": cfg.judge.model, "digest": digest},
            "prompt_version": answer["prompt_version"],
        }
        try:
            passages = [Chunk(p["id"], p["id"].split("#")[0], p["text"]) for p in answer["passages"]]
            verdict = judge.verdict(
                answer["question"], passages, answer["answer"], answer.get("reference", "")
            )
            record.update(
                status="ok",
                faithfulness=verdict.faithfulness,
                relevance=verdict.relevance,
                correctness=verdict.correctness,
                rationales=verdict.rationales,
                latency_ms=verdict.latency_ms,
            )
        except JudgeParseError as exc:
            logger.warning("judge parse error on %s", answer["answer_id"])
            record.update(status="error", error=str(exc), raw_output=exc.raw_output)
        except OllamaError as exc:
            logger.exception("judge call failed on %s", answer["answer_id"])
            record.update(status="error", error=f"{type(exc).__name__}: {exc}", raw_output="")
        record["timestamp"] = datetime.now(UTC).isoformat()
        _append(judgments_path, record)
        records.append(record)
        if n % 10 == 0:
            logger.info("judged %d/%d", n, len(answers))
    return records
```

- [ ] **Step 4: Vérifier le succès**

Run: `uv run pytest tests/test_generation_experiment.py -q`
Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add src/rag_eval_lab/generation_experiment.py tests/test_generation_experiment.py
git commit -m "feat: add the resumable judging stage"
```

---

### Task 10: Calibration

**Files:**
- Create: `src/rag_eval_lab/calibration.py`
- Test: `tests/test_calibration.py`

**Interfaces:**
- Produces :
  - `wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]`
  - `cohen_kappa(pairs: list[tuple[bool, bool]]) -> float` — paires `(judge, human)`.
  - `confusion(pairs: list[tuple[bool, bool]]) -> dict[str, int]` — clés `tp`, `fp`, `fn`, `tn` (positif = `True`, « fp » = le juge dit vrai alors que l'humain dit faux).
  - `agreement(judgments: list[dict], annotations: list[dict], criteria: tuple[str, ...] = ("faithfulness", "correctness")) -> dict[str, dict]` — par critère : `n`, `agreement`, `ci_low`, `ci_high`, `kappa`, `confusion`, `judge_positive_rate`, `human_positive_rate`. Les paires où l'une des deux valeurs est `null` sont ignorées.

- [ ] **Step 1: Écrire les tests**

```python
import pytest

from rag_eval_lab.calibration import agreement, cohen_kappa, confusion, wilson_interval


def test_wilson_interval_brackets_the_rate():
    low, high = wilson_interval(45, 60)
    assert low < 0.75 < high
    assert 0.0 <= low < high <= 1.0


def test_wilson_interval_of_an_empty_sample():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_confusion_counts_each_cell():
    pairs = [(True, True), (True, False), (False, True), (False, False), (True, True)]
    assert confusion(pairs) == {"tp": 2, "fp": 1, "fn": 1, "tn": 1}


def test_cohen_kappa_perfect_and_chance():
    assert cohen_kappa([(True, True), (False, False)]) == pytest.approx(1.0)
    # judge always says True, human is split: raw agreement 0.5, kappa 0
    pairs = [(True, True), (True, False)]
    assert cohen_kappa(pairs) == pytest.approx(0.0)


def test_cohen_kappa_hand_computed():
    # 10 pairs: tp=6, fp=2, fn=1, tn=1 -> po=0.7
    # judge True 8/10, human True 7/10 -> pe = .8*.7 + .2*.3 = .62 -> kappa = .08/.38
    pairs = [(True, True)] * 6 + [(True, False)] * 2 + [(False, True)] + [(False, False)]
    assert cohen_kappa(pairs) == pytest.approx(0.08 / 0.38, abs=1e-6)


def test_cohen_kappa_of_a_single_class_is_zero():
    assert cohen_kappa([(True, True), (True, True)]) == 0.0


def test_agreement_skips_nulls_and_reports_both_criteria():
    judgments = [
        {"answer_id": "a", "status": "ok", "faithfulness": True, "correctness": True},
        {"answer_id": "b", "status": "ok", "faithfulness": None, "correctness": False},
        {"answer_id": "c", "status": "ok", "faithfulness": False, "correctness": True},
        {"answer_id": "d", "status": "error"},
    ]
    annotations = [
        {"answer_id": "a", "faithfulness": True, "correctness": False},
        {"answer_id": "b", "faithfulness": True, "correctness": False},
        {"answer_id": "c", "faithfulness": False, "correctness": None},
        {"answer_id": "d", "faithfulness": True, "correctness": True},
    ]
    result = agreement(judgments, annotations)
    assert result["faithfulness"]["n"] == 2
    assert result["faithfulness"]["agreement"] == pytest.approx(1.0)
    assert result["correctness"]["n"] == 2
    assert result["correctness"]["agreement"] == pytest.approx(0.5)
    assert result["correctness"]["confusion"] == {"tp": 0, "fp": 1, "fn": 0, "tn": 1}
    assert 0.0 <= result["faithfulness"]["ci_low"] <= result["faithfulness"]["ci_high"] <= 1.0


def test_agreement_without_overlap_returns_zero_sized_entries():
    result = agreement([], [])
    assert result["faithfulness"]["n"] == 0
    assert result["correctness"]["kappa"] == 0.0
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/test_calibration.py -q`
Expected: FAIL avec `ModuleNotFoundError: No module named 'rag_eval_lab.calibration'`

- [ ] **Step 3: Implémenter**

```python
"""Judge calibration: how well the LLM judge agrees with human labels."""

from __future__ import annotations

import math

CRITERIA = ("faithfulness", "correctness")


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion; (0.0, 1.0) when there is no data."""
    if total == 0:
        return (0.0, 1.0)
    phat = successes / total
    denominator = 1 + z**2 / total
    centre = phat + z**2 / (2 * total)
    spread = z * math.sqrt(phat * (1 - phat) / total + z**2 / (4 * total**2))
    return (max(0.0, (centre - spread) / denominator), min(1.0, (centre + spread) / denominator))


def confusion(pairs: list[tuple[bool, bool]]) -> dict[str, int]:
    """Counts with the judge as the predictor and the human as the reference."""
    cells = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for judge, human in pairs:
        if judge and human:
            cells["tp"] += 1
        elif judge and not human:
            cells["fp"] += 1
        elif not judge and human:
            cells["fn"] += 1
        else:
            cells["tn"] += 1
    return cells


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float:
    """Agreement corrected for chance; 0.0 when one rater never varies."""
    n = len(pairs)
    if n == 0:
        return 0.0
    observed = sum(1 for judge, human in pairs if judge == human) / n
    judge_positive = sum(1 for judge, _ in pairs if judge) / n
    human_positive = sum(1 for _, human in pairs if human) / n
    expected = judge_positive * human_positive + (1 - judge_positive) * (1 - human_positive)
    if expected >= 1.0:
        return 0.0
    return (observed - expected) / (1 - expected)


def agreement(
    judgments: list[dict], annotations: list[dict], criteria: tuple[str, ...] = CRITERIA
) -> dict[str, dict]:
    by_id = {j["answer_id"]: j for j in judgments if j.get("status") == "ok"}
    result: dict[str, dict] = {}
    for criterion in criteria:
        pairs: list[tuple[bool, bool]] = []
        for annotation in annotations:
            judgment = by_id.get(annotation["answer_id"])
            if judgment is None:
                continue
            judge_value, human_value = judgment.get(criterion), annotation.get(criterion)
            if isinstance(judge_value, bool) and isinstance(human_value, bool):
                pairs.append((judge_value, human_value))
        agreed = sum(1 for judge, human in pairs if judge == human)
        low, high = wilson_interval(agreed, len(pairs))
        result[criterion] = {
            "n": len(pairs),
            "agreement": agreed / len(pairs) if pairs else 0.0,
            "ci_low": low,
            "ci_high": high,
            "kappa": cohen_kappa(pairs),
            "confusion": confusion(pairs),
            "judge_positive_rate": (
                sum(1 for judge, _ in pairs if judge) / len(pairs) if pairs else 0.0
            ),
            "human_positive_rate": (
                sum(1 for _, human in pairs if human) / len(pairs) if pairs else 0.0
            ),
        }
    return result
```

- [ ] **Step 4: Vérifier le succès**

Run: `uv run pytest tests/test_calibration.py -q`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add src/rag_eval_lab/calibration.py tests/test_calibration.py
git commit -m "feat: add judge calibration metrics"
```

---

### Task 11: Annotation

**Files:**
- Create: `src/rag_eval_lab/annotation.py`
- Test: `tests/test_annotation.py`

**Interfaces:**
- Consumes: `load_answers` (Task 9).
- Produces :
  - `sample_for_annotation(answers: list[dict], n: int, seed: int) -> list[dict]` — tirage équilibré entre contextes (tour par tour sur les contextes triés par nom, chaque contexte mélangé avec la graine), ordre de sortie déterministe.
  - `load_annotations(out_dir: Path) -> list[dict]`
  - `append_annotation(out_dir: Path, record: dict) -> None`
  - `run_annotation_session(out_dir: Path, n: int, seed: int, read: Callable[[str], str] = input, write: Callable[[str], None] = print) -> int` — renvoie le nombre d'annotations ajoutées ; saute les réponses déjà annotées ; `o`/`n` répondent, `?` enregistre `null`, `q` quitte. Le contexte `none` ne pose pas la question de fidélité (`faithfulness` = `null`).
  - **La session n'affiche jamais le verdict du juge** (aucune lecture de `judgments.jsonl`).

- [ ] **Step 1: Écrire les tests**

```python
import json

import pytest

from rag_eval_lab.annotation import (
    append_annotation,
    load_annotations,
    run_annotation_session,
    sample_for_annotation,
)


def _answers(n_per_context=4, contexts=("none", "bm25", "oracle")):
    return [
        {
            "answer_id": f"{context}-{i}",
            "context": context,
            "query_id": f"q{i}",
            "question": f"question {i}",
            "answer": f"réponse {i}",
            "passages": [] if context == "none" else [{"id": "d1#0", "text": "extrait"}],
            "status": "ok",
        }
        for context in contexts
        for i in range(n_per_context)
    ]


def test_sample_is_balanced_across_contexts():
    sample = sample_for_annotation(_answers(), n=6, seed=7)
    assert len(sample) == 6
    counts = {}
    for record in sample:
        counts[record["context"]] = counts.get(record["context"], 0) + 1
    assert counts == {"none": 2, "bm25": 2, "oracle": 2}


def test_sample_is_deterministic():
    first = [a["answer_id"] for a in sample_for_annotation(_answers(), n=5, seed=7)]
    second = [a["answer_id"] for a in sample_for_annotation(_answers(), n=5, seed=7)]
    assert first == second
    assert first != [a["answer_id"] for a in sample_for_annotation(_answers(), n=5, seed=8)]


def test_sample_larger_than_available_returns_everything():
    assert len(sample_for_annotation(_answers(n_per_context=2), n=99, seed=1)) == 6


def test_append_and_load_annotations(tmp_path):
    append_annotation(tmp_path, {"answer_id": "a", "faithfulness": True, "correctness": False})
    append_annotation(tmp_path, {"answer_id": "b", "faithfulness": None, "correctness": True})
    loaded = load_annotations(tmp_path)
    assert [r["answer_id"] for r in loaded] == ["a", "b"]
    assert loaded[1]["faithfulness"] is None


def _write_answers(tmp_path, answers):
    with (tmp_path / "answers.jsonl").open("w", encoding="utf-8") as handle:
        for answer in answers:
            handle.write(json.dumps(answer, ensure_ascii=False) + "\n")


def test_session_records_answers_and_skips_faithfulness_without_passages(tmp_path):
    _write_answers(tmp_path, _answers(n_per_context=1, contexts=("none", "bm25")))
    replies = iter(["o", "o", "n"])  # none: correct=o ; bm25: faithful=o, correct=n
    added = run_annotation_session(tmp_path, n=2, seed=1, read=lambda _: next(replies),
                                   write=lambda _: None)
    assert added == 2
    records = {r["answer_id"]: r for r in load_annotations(tmp_path)}
    none_record = next(r for r in records.values() if r["answer_id"].startswith("none"))
    bm25_record = next(r for r in records.values() if r["answer_id"].startswith("bm25"))
    assert none_record["faithfulness"] is None and none_record["correctness"] is True
    assert bm25_record["faithfulness"] is True and bm25_record["correctness"] is False


def test_session_question_mark_records_null(tmp_path):
    _write_answers(tmp_path, _answers(n_per_context=1, contexts=("bm25",)))
    replies = iter(["?", "?"])
    run_annotation_session(tmp_path, n=1, seed=1, read=lambda _: next(replies),
                           write=lambda _: None)
    record = load_annotations(tmp_path)[0]
    assert record["faithfulness"] is None and record["correctness"] is None


def test_session_quit_stops_without_recording(tmp_path):
    _write_answers(tmp_path, _answers(n_per_context=2, contexts=("bm25",)))
    replies = iter(["o", "o", "q"])
    added = run_annotation_session(tmp_path, n=2, seed=1, read=lambda _: next(replies),
                                   write=lambda _: None)
    assert added == 1
    assert len(load_annotations(tmp_path)) == 1


def test_session_resumes_and_never_shows_the_judge(tmp_path):
    _write_answers(tmp_path, _answers(n_per_context=1, contexts=("bm25",)))
    (tmp_path / "judgments.jsonl").write_text(
        json.dumps({"answer_id": "bm25-0", "status": "ok", "faithfulness": True}) + "\n",
        encoding="utf-8",
    )
    shown: list[str] = []
    run_annotation_session(tmp_path, n=1, seed=1, read=lambda _: "o",
                           write=lambda text: shown.append(text))
    assert "juge" not in " ".join(shown).lower()
    assert run_annotation_session(tmp_path, n=1, seed=1, read=lambda _: "o",
                                  write=lambda _: None) == 0


def test_session_rejects_an_unreadable_reply(tmp_path):
    _write_answers(tmp_path, _answers(n_per_context=1, contexts=("bm25",)))
    replies = iter(["x", "o", "n"])
    run_annotation_session(tmp_path, n=1, seed=1, read=lambda _: next(replies),
                           write=lambda _: None)
    record = load_annotations(tmp_path)[0]
    assert record["faithfulness"] is True and record["correctness"] is False


def test_session_without_answers_raises(tmp_path):
    with pytest.raises(ValueError, match="no answers"):
        run_annotation_session(tmp_path, n=1, seed=1, read=lambda _: "o", write=lambda _: None)
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/test_annotation.py -q`
Expected: FAIL avec `ModuleNotFoundError: No module named 'rag_eval_lab.annotation'`

- [ ] **Step 3: Implémenter**

```python
"""Human annotation session used to calibrate the judge."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from rag_eval_lab.generation_experiment import load_answers

ANSWERS = {"o": True, "n": False, "?": None}


def sample_for_annotation(answers: list[dict], n: int, seed: int) -> list[dict]:
    """Round-robin over contexts so no context dominates the calibration set."""
    rng = random.Random(seed)
    by_context: dict[str, list[dict]] = {}
    for answer in answers:
        by_context.setdefault(answer["context"], []).append(answer)
    for records in by_context.values():
        records.sort(key=lambda r: r["answer_id"])
        rng.shuffle(records)
    selected: list[dict] = []
    position = 0
    while len(selected) < n:
        available = [c for c in sorted(by_context) if len(by_context[c]) > position]
        if not available:
            break
        for context in available:
            selected.append(by_context[context][position])
            if len(selected) == n:
                break
        position += 1
    return selected


def load_annotations(out_dir: Path) -> list[dict]:
    path = Path(out_dir) / "annotations.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_annotation(out_dir: Path, record: dict) -> None:
    path = Path(out_dir) / "annotations.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _ask(read: Callable[[str], str], write: Callable[[str], None], question: str) -> bool | None | str:
    while True:
        reply = read(f"{question} [o/n/? = je ne sais pas, q = quitter] ").strip().lower()
        if reply == "q":
            return "quit"
        if reply in ANSWERS:
            return ANSWERS[reply]
        write("Réponse attendue : o, n, ? ou q.")


def run_annotation_session(
    out_dir: Path,
    n: int,
    seed: int,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
) -> int:
    """Label sampled answers. The judge's verdict is never shown, to avoid anchoring."""
    out_dir = Path(out_dir)
    answers = load_answers(out_dir)
    if not answers:
        raise ValueError(f"no answers to annotate in {out_dir / 'answers.jsonl'}")
    already = {record["answer_id"] for record in load_annotations(out_dir)}
    todo = [a for a in sample_for_annotation(answers, n, seed) if a["answer_id"] not in already]

    added = 0
    for index, answer in enumerate(todo, start=1):
        started = time.perf_counter()
        write(f"\n--- {index}/{len(todo)} (contexte : {answer['context']})")
        write(f"Question : {answer['question']}")
        for position, passage in enumerate(answer["passages"], start=1):
            write(f"[{position}] {passage['text'][:600]}")
        if not answer["passages"]:
            write("(aucun passage fourni)")
        write(f"Réponse générée : {answer['answer']}")
        write(f"Réponse de référence : {answer.get('reference', '(absente)')}")

        faithfulness: bool | None = None
        if answer["passages"]:
            reply = _ask(read, write, "Fidèle aux passages ?")
            if reply == "quit":
                break
            faithfulness = reply
        reply = _ask(read, write, "Juste par rapport à la référence ?")
        if reply == "quit":
            break
        append_annotation(
            out_dir,
            {
                "answer_id": answer["answer_id"],
                "faithfulness": faithfulness,
                "correctness": reply,
                "seconds": round(time.perf_counter() - started, 1),
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
        added += 1
    return added
```

- [ ] **Step 4: Vérifier le succès**

Run: `uv run pytest tests/test_annotation.py -q`
Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
git add src/rag_eval_lab/annotation.py tests/test_annotation.py
git commit -m "feat: add the human annotation session"
```

---

### Task 12: Rapport de génération

**Files:**
- Create: `src/rag_eval_lab/report_generation.py`
- Test: `tests/test_report_generation.py`

**Interfaces:**
- Consumes: `load_answers` (Task 9) ; `load_annotations` (Task 11) ; `agreement` (Task 10).
- Produces :
  - `load_judgments(out_dir: Path) -> list[dict]` — dernier enregistrement par `answer_id`, `status == "ok"` seulement.
  - `summarise(answers: list[dict], judgments: list[dict]) -> list[dict]` — une entrée par contexte : `context`, `n`, `faithfulness_rate`, `relevance_rate`, `correctness_rate` (`None` quand le critère ne s'applique pas), `latency_ms_p50`, `mean_passage_chars`.
  - `build_generation_report(out_dir: Path) -> Path` — écrit `report-generation.md` et `generation.png` ; lève `ValueError` s'il n'y a aucun jugement.

- [ ] **Step 1: Écrire les tests**

```python
import json

import pytest

from rag_eval_lab.report_generation import build_generation_report, load_judgments, summarise


def _answer(answer_id, context, chars=100, latency=1000.0):
    return {"answer_id": answer_id, "context": context, "query_id": answer_id,
            "question": "q", "answer": "a", "status": "ok",
            "passages": [] if context == "none" else [{"id": "d#0", "text": "x" * chars}],
            "n_passage_chars": 0 if context == "none" else chars, "latency_ms": latency}


def _judgment(answer_id, faithful, correct, relevant=True):
    return {"answer_id": answer_id, "status": "ok", "faithfulness": faithful,
            "relevance": relevant, "correctness": correct, "latency_ms": 500.0,
            "rationales": {"correctness": "parce que"}}


def _write(tmp_path, name, records):
    with (tmp_path / name).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_load_judgments_keeps_last_ok_record(tmp_path):
    _write(tmp_path, "judgments.jsonl", [
        {"answer_id": "a", "status": "error", "error": "boom"},
        _judgment("a", True, True),
        {"answer_id": "b", "status": "error", "error": "boom"},
    ])
    judgments = load_judgments(tmp_path)
    assert [j["answer_id"] for j in judgments] == ["a"]


def test_summarise_computes_rates_per_context():
    answers = [_answer("a1", "bm25"), _answer("a2", "bm25"), _answer("n1", "none")]
    judgments = [_judgment("a1", True, True), _judgment("a2", False, True),
                 _judgment("n1", None, False, relevant=None)]
    rows = {row["context"]: row for row in summarise(answers, judgments)}
    assert rows["bm25"]["n"] == 2
    assert rows["bm25"]["faithfulness_rate"] == pytest.approx(0.5)
    assert rows["bm25"]["correctness_rate"] == pytest.approx(1.0)
    assert rows["none"]["faithfulness_rate"] is None
    assert rows["none"]["correctness_rate"] == pytest.approx(0.0)


def test_build_report_writes_markdown_and_chart(tmp_path):
    _write(tmp_path, "answers.jsonl", [_answer("a1", "bm25"), _answer("o1", "oracle"),
                                       _answer("n1", "none")])
    _write(tmp_path, "judgments.jsonl", [_judgment("a1", True, True),
                                         _judgment("o1", True, True),
                                         _judgment("n1", None, False, relevant=None)])
    _write(tmp_path, "annotations.jsonl", [{"answer_id": "a1", "faithfulness": True,
                                            "correctness": False}])
    path = build_generation_report(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert path.name == "report-generation.md"
    assert "oracle" in text and "bm25" in text
    assert "Calibration" in text
    assert "kappa" in text.lower()
    assert (tmp_path / "generation.png").exists()


def test_build_report_without_judgments_raises(tmp_path):
    _write(tmp_path, "answers.jsonl", [_answer("a1", "bm25")])
    _write(tmp_path, "judgments.jsonl", [{"answer_id": "a1", "status": "error", "error": "x"}])
    with pytest.raises(ValueError, match="No successful judgments"):
        build_generation_report(tmp_path)
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/test_report_generation.py -q`
Expected: FAIL avec `ModuleNotFoundError: No module named 'rag_eval_lab.report_generation'`

- [ ] **Step 3: Implémenter**

```python
"""Markdown report for the generation benchmark, including judge calibration."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rag_eval_lab.annotation import load_annotations  # noqa: E402
from rag_eval_lab.calibration import agreement  # noqa: E402
from rag_eval_lab.generation_experiment import load_answers  # noqa: E402

CRITERIA = ("faithfulness", "relevance", "correctness")


def load_judgments(out_dir: Path) -> list[dict]:
    path = Path(out_dir) / "judgments.jsonl"
    latest: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                latest[record["answer_id"]] = record
    return [r for r in latest.values() if r.get("status") == "ok"]


def _rate(values: list[bool | None]) -> float | None:
    kept = [v for v in values if isinstance(v, bool)]
    return sum(kept) / len(kept) if kept else None


def summarise(answers: list[dict], judgments: list[dict]) -> list[dict]:
    by_id = {j["answer_id"]: j for j in judgments}
    grouped: dict[str, list[dict]] = {}
    for answer in answers:
        if answer["answer_id"] in by_id:
            grouped.setdefault(answer["context"], []).append(answer)
    rows = []
    for context, records in sorted(grouped.items()):
        verdicts = [by_id[a["answer_id"]] for a in records]
        rows.append(
            {
                "context": context,
                "n": len(records),
                **{
                    f"{criterion}_rate": _rate([v.get(criterion) for v in verdicts])
                    for criterion in CRITERIA
                },
                "latency_ms_p50": statistics.median(a["latency_ms"] for a in records),
                "mean_passage_chars": statistics.mean(a["n_passage_chars"] for a in records),
            }
        )
    return rows


def _plot(rows: list[dict], path: Path) -> None:
    contexts = [row["context"] for row in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.35
    positions = range(len(rows))
    faithful = [(row["faithfulness_rate"] or 0.0) for row in rows]
    correct = [(row["correctness_rate"] or 0.0) for row in rows]
    ax.bar([p - width / 2 for p in positions], faithful, width, label="faithfulness",
           color="#9aa5b1")
    ax.bar([p + width / 2 for p in positions], correct, width, label="correctness",
           color="#2563eb")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(contexts, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("rate")
    ax.set_title("Answer quality by context")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def build_generation_report(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    judgments = load_judgments(out_dir)
    if not judgments:
        raise ValueError(f"No successful judgments in {out_dir / 'judgments.jsonl'}")
    answers = load_answers(out_dir)
    rows = summarise(answers, judgments)
    annotations = load_annotations(out_dir)
    calibration = agreement(judgments, annotations)

    lines = [
        "# Generation benchmark report",
        "",
        f"- Contexts: {len(rows)}",
        f"- Judged answers: {len(judgments)}",
        f"- Human annotations: {len(annotations)}",
        "",
        "| Context | n | faithfulness | relevance | correctness | latency p50 (ms) | mean passage chars |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['context']} | {row['n']} | {_format(row['faithfulness_rate'])} | "
            f"{_format(row['relevance_rate'])} | {_format(row['correctness_rate'])} | "
            f"{row['latency_ms_p50']:.0f} | {row['mean_passage_chars']:.0f} |"
        )
    lines += [
        "",
        "![Answer quality by context](generation.png)",
        "",
        "## Calibration — judge against human labels",
        "",
        "| Criterion | n | agreement | 95% CI | kappa | judge positive | human positive | tp/fp/fn/tn |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for criterion, stats in calibration.items():
        cells = stats["confusion"]
        lines.append(
            f"| {criterion} | {stats['n']} | {stats['agreement']:.3f} | "
            f"[{stats['ci_low']:.3f}, {stats['ci_high']:.3f}] | {stats['kappa']:.3f} | "
            f"{stats['judge_positive_rate']:.3f} | {stats['human_positive_rate']:.3f} | "
            f"{cells['tp']}/{cells['fp']}/{cells['fn']}/{cells['tn']} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- `faithfulness` and `relevance` are not judged for the `none` context: without passages,",
        "  grounding is undefined. Those cells read `n/a`.",
        "- Agreement counts only answers that carry both a human label and a judge verdict;",
        "  `?` answers during annotation are stored as null and excluded.",
        "- Confusion is reported judge-versus-human: `fp` means the judge said true where the",
        "  human said false (an indulgent judge), `fn` the opposite (a severe one).",
        "",
    ]
    _plot(rows, out_dir / "generation.png")
    report_path = out_dir / "report-generation.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
```

- [ ] **Step 4: Vérifier le succès**

Run: `uv run pytest tests/test_report_generation.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/rag_eval_lab/report_generation.py tests/test_report_generation.py
git commit -m "feat: add the generation report with calibration"
```

---

### Task 13: Commandes CLI

**Files:**
- Modify: `src/rag_eval_lab/cli.py`
- Test: `tests/test_cli_generation.py`

**Interfaces:**
- Consumes: `load_generation_config` (Task 7) ; `run_generation`, `run_judging` (Tasks 8-9) ; `run_annotation_session` (Task 11) ; `build_generation_report` (Task 12) ; `OllamaError` (Task 2).
- Produces, en plus des commandes `run` et `report` existantes :
  - `rel generate CONFIG [--out DIR] [--verbose]` — défaut `results/<nom du fichier>` ;
  - `rel judge RESULTS_DIR CONFIG [--verbose]` ;
  - `rel annotate RESULTS_DIR [--n 60] [--seed 7]` ;
  - `rel report-gen RESULTS_DIR`.
  - Toute `OllamaError` est affichée seule, sans trace Python, avec le code de sortie 1.

- [ ] **Step 1: Écrire les tests**

```python
import json

import yaml
from typer.testing import CliRunner

from rag_eval_lab.cli import app

runner = CliRunner()


def _write_config(tmp_path, mini_jsonl):
    config = {
        "dataset": {"name": "jsonl", "path": str(mini_jsonl), "n_queries": 2},
        "n_passages": 2,
        "oracle_chunker": "fixed_200_50",
        "generator": {"model": "fake-model"},
        "judge": {"model": "fake-model"},
        "cache_dir": str(tmp_path / "cache"),
        "contexts": [
            {"name": "none", "kind": "empty"},
            {"name": "bm25", "kind": "retrieved", "chunker": "fixed_200_50", "retriever": "bm25"},
        ],
        "annotation": {"n": 2, "seed": 1},
    }
    path = tmp_path / "v2.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_generate_judge_annotate_report(tmp_path, mini_jsonl, monkeypatch):
    from rag_eval_lab import cli

    class FakeTransport:
        def __call__(self, method, path, payload):
            if path == "/api/tags":
                return {"models": [{"name": "fake-model", "digest": "0123456789abcdef"}]}
            prompt = payload["prompt"]
            if "faithful" in prompt:
                return {"response": '{"faithful": true, "relevant": true, "reason": "ok"}'}
            if "correct" in prompt:
                return {"response": '{"correct": true, "reason": "ok"}'}
            return {"response": "réponse", "eval_count": 3, "prompt_eval_count": 10}

    monkeypatch.setattr(cli, "build_client", lambda timeout_s: cli.OllamaClient(
        transport=FakeTransport(), retries=0))

    config = _write_config(tmp_path, mini_jsonl)
    out = tmp_path / "results"
    assert runner.invoke(app, ["generate", str(config), "--out", str(out)]).exit_code == 0
    assert len((out / "answers.jsonl").read_text(encoding="utf-8").splitlines()) == 4

    result = runner.invoke(app, ["judge", str(out), str(config)])
    assert result.exit_code == 0, result.output
    assert len((out / "judgments.jsonl").read_text(encoding="utf-8").splitlines()) == 4

    result = runner.invoke(app, ["annotate", str(out), "--n", "2"], input="o\nn\no\n")
    assert result.exit_code == 0, result.output
    assert len((out / "annotations.jsonl").read_text(encoding="utf-8").splitlines()) == 2

    result = runner.invoke(app, ["report-gen", str(out)])
    assert result.exit_code == 0, result.output
    assert (out / "report-generation.md").exists()


def test_generate_reports_a_missing_ollama(tmp_path, mini_jsonl, monkeypatch):
    from rag_eval_lab import cli

    def explode(timeout_s):
        raise cli.OllamaError("cannot reach Ollama at http://127.0.0.1:11434. Start it with `ollama serve`.")

    monkeypatch.setattr(cli, "build_client", explode)
    result = runner.invoke(app, ["generate", str(_write_config(tmp_path, mini_jsonl))])
    assert result.exit_code == 1
    assert "ollama serve" in result.output
    assert "Traceback" not in result.output


def test_report_gen_without_judgments(tmp_path):
    (tmp_path / "judgments.jsonl").write_text("", encoding="utf-8")
    result = runner.invoke(app, ["report-gen", str(tmp_path)])
    assert result.exit_code == 1
```

- [ ] **Step 2: Vérifier l'échec**

Run: `uv run pytest tests/test_cli_generation.py -q`
Expected: FAIL avec `ImportError: cannot import name 'build_client' from 'rag_eval_lab.cli'`

- [ ] **Step 3: Implémenter**

Ajouter à `src/rag_eval_lab/cli.py` (après les imports existants) :

```python
from rag_eval_lab.annotation import run_annotation_session
from rag_eval_lab.generation_config import load_generation_config
from rag_eval_lab.generation_experiment import run_generation, run_judging
from rag_eval_lab.ollama_client import OllamaClient, OllamaError
from rag_eval_lab.report_generation import build_generation_report


def build_client(timeout_s: float) -> OllamaClient:
    """Seam for tests: every generation command builds its client here."""
    return OllamaClient(timeout_s=timeout_s)


def _load_generation_config(path: Path):
    try:
        return load_generation_config(path)
    except (ValidationError, yaml.YAMLError) as exc:
        typer.echo(f"Invalid config {path}:\n{exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def generate(
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="YAML config")],
    out: Annotated[Path | None, typer.Option(help="Output directory")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Generate one answer per (context, query); already generated ones are skipped."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    cfg = _load_generation_config(config)
    out_dir = out or Path("results") / config.stem
    try:
        records = run_generation(cfg, out_dir, client=build_client(cfg.generator.timeout_s))
    except OllamaError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    failed = [r for r in records if r["status"] == "error"]
    typer.echo(f"{len(records)} answer(s) generated, {len(failed)} failed -> {out_dir}")


@app.command()
def judge(
    results_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Judge every generated answer (already judged ones are skipped)."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    cfg = _load_generation_config(config)
    try:
        records = run_judging(cfg, results_dir, client=build_client(cfg.judge.timeout_s))
    except OllamaError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    failed = [r for r in records if r["status"] == "error"]
    typer.echo(f"{len(records)} answer(s) judged, {len(failed)} failed -> {results_dir}")


@app.command()
def annotate(
    results_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    n: Annotated[int, typer.Option(help="How many answers to label")] = 60,
    seed: Annotated[int, typer.Option(help="Sampling seed")] = 7,
) -> None:
    """Label sampled answers yourself; the judge's verdict is never shown."""
    try:
        added = run_annotation_session(results_dir, n=n, seed=seed)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"{added} annotation(s) added -> {results_dir / 'annotations.jsonl'}")


@app.command("report-gen")
def report_gen(
    results_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Build the generation report (quality per context + judge calibration)."""
    try:
        path = build_generation_report(results_dir)
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Report written to {path}")
```

- [ ] **Step 4: Vérifier le succès**

Run: `uv run pytest tests/test_cli_generation.py -q`
Expected: `3 passed`

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: `153 passed`, `All checks passed!`, aucun fichier à reformater.

Run: `uv run rel --help`
Expected: les commandes `run`, `report`, `generate`, `judge`, `annotate`, `report-gen` sont listées.

- [ ] **Step 5: Commit**

```bash
git add src/rag_eval_lab/cli.py tests/test_cli_generation.py
git commit -m "feat: add generate, judge, annotate and report-gen commands"
```

---

### Task 14: Mesure de vitesse réelle et configuration finale

Première exécution contre le vrai Ollama. Elle vérifie que tout fonctionne hors des doubles de
test et mesure les durées pour dimensionner la grille.

**Files:**
- Create: `configs/v2-smoke.yaml`, `configs/v2.yaml`

**Interfaces:**
- Consumes: la CLI de la Task 13.

- [ ] **Step 1: Vérifier qu'Ollama répond**

Run: `curl -s http://127.0.0.1:11434/api/tags`
Expected: un JSON listant `llama3.2:latest`, `llama3.2:1B`, `llama3.1:8B`. Si la commande échoue,
lancer `ollama serve` dans un terminal séparé et recommencer.

- [ ] **Step 2: Créer `configs/v2-smoke.yaml`**

```yaml
# Small run used to validate the setup and measure real speed on this CPU.
dataset: { name: alloprof, n_queries: 4, seed: 42 }
n_passages: 5
oracle_chunker: fixed_1000_200
generator: { model: "llama3.2:latest", temperature: 0, max_tokens: 300, timeout_s: 300 }
judge:     { model: "llama3.1:8B",     temperature: 0, max_tokens: 200, timeout_s: 600 }
contexts:
  - { name: none,     kind: empty }
  - { name: bm25,     kind: retrieved, chunker: none,           retriever: bm25 }
  - { name: e5-small, kind: retrieved, chunker: none,           retriever: dense:e5-small }
  - { name: bge-m3,   kind: retrieved, chunker: fixed_1000_200, retriever: dense:bge-m3 }
  - { name: oracle,   kind: oracle }
annotation: { n: 60, seed: 7 }
```

- [ ] **Step 3: Lancer la génération de validation**

Run: `uv run rel generate configs/v2-smoke.yaml`
Expected: `20 answer(s) generated, 0 failed -> results\v2-smoke`

Attention : l'indexation des contextes `e5-small` et `bge-m3` recalcule les embeddings si le
cache de la v1 ne correspond plus (la clé de cache inclut désormais l'empreinte de l'encodeur).
Compter jusqu'à 2 h pour bge-m3 dans ce cas ; c'est un coût unique, réutilisé par la grille
complète. Lancer la commande en arrière-plan avec un temps d'attente large.

- [ ] **Step 4: Lancer le jugement de validation**

Run: `uv run rel judge results/v2-smoke configs/v2-smoke.yaml`
Expected: `20 answer(s) judged, 0 failed -> results\v2-smoke`

Puis : `uv run rel report-gen results/v2-smoke`
Expected: `Report written to results\v2-smoke\report-generation.md`

- [ ] **Step 5: Relever les mesures et dimensionner**

Extraire des fichiers produits :

```bash
uv run python -c "
import json, statistics
rows=[json.loads(l) for l in open('results/v2-smoke/answers.jsonl',encoding='utf-8')]
print('generation p50 ms:', statistics.median(r['latency_ms'] for r in rows if r['status']=='ok'))
rows=[json.loads(l) for l in open('results/v2-smoke/judgments.jsonl',encoding='utf-8')]
print('judging p50 ms:', statistics.median(r['latency_ms'] for r in rows if r['status']=='ok'))
"
```

Calculer : `durée ≈ n_queries × (5 × génération + 5 × jugement)`. Choisir le plus grand
`n_queries` qui tient sous **8 heures**, borné à 80. Vérifier aussi dans
`results/v2-smoke/report-generation.md` que les verdicts ne sont pas tous identiques (un juge
qui répond toujours `true` serait un signal d'alarme à signaler).

- [ ] **Step 6: Créer `configs/v2.yaml`**

Copier `configs/v2-smoke.yaml` en remplaçant la ligne `dataset` par la valeur retenue, par
exemple :

```yaml
dataset: { name: alloprof, n_queries: 60, seed: 42 }
```

et ajouter en tête du fichier un commentaire donnant la mesure qui a conduit à ce choix, par
exemple : `# 60 queries: measured 38 s/answer and 52 s/judgment on Ryzen 7 8845HS -> ~7.5 h.`

- [ ] **Step 7: Commit**

```bash
git add configs/v2-smoke.yaml configs/v2.yaml
git commit -m "chore: add v2 configs sized from a measured smoke run"
```

(Les résultats de `results/v2-smoke/` ne sont pas commités ; `.gitignore` ignore déjà
`results/*/runs/`, donc ajouter `results/v2-smoke/` à `.gitignore` dans ce même commit.)

---

### Task 15: Exécution complète, annotation, analyse et README

**Files:**
- Create: `docs/results-v2.md`
- Modify: `README.md`, `.gitignore` (si nécessaire pour suivre `results/v2/`)
- Commit: `results/v2/answers.jsonl`, `results/v2/judgments.jsonl`, `results/v2/annotations.jsonl`, `results/v2/report-generation.md`, `results/v2/generation.png`

**Interfaces:**
- Consumes: la CLI de la Task 13, les configurations de la Task 14.

- [ ] **Step 1: Lancer la génération complète en arrière-plan**

Run: `uv run rel generate configs/v2.yaml`
Expected: `N answer(s) generated, 0 failed -> results\v2` avec N = 5 × `n_queries`.
En cas d'interruption, relancer la même commande : les réponses déjà produites sont sautées.

- [ ] **Step 2: Lancer le jugement complet**

Run: `uv run rel judge results/v2 configs/v2.yaml`
Expected: `N answer(s) judged, 0 failed -> results\v2`. Si des erreurs subsistent, relancer la
commande une fois ; si les mêmes échouent encore, relever leur `raw_output` et le signaler.

- [ ] **Step 3: Session d'annotation (utilisateur)**

Run: `uv run rel annotate results/v2 --n 60`
Cette étape est **manuelle** : c'est l'utilisateur qui répond. Lui indiquer qu'il peut quitter
avec `q` et reprendre plus tard, et que le verdict du juge n'est volontairement pas affiché.
Expected à la fin : `60 annotation(s) added -> results\v2\annotations.jsonl`

- [ ] **Step 4: Construire le rapport**

Run: `uv run rel report-gen results/v2`
Expected: `Report written to results\v2\report-generation.md`

- [ ] **Step 5: Extraire les enseignements**

Lire `results/v2/report-generation.md` et calculer explicitement :

1. l'écart de justesse entre `none` et `bge-m3` : ce que le RAG apporte réellement ;
2. l'écart entre `bge-m3` et `oracle` : ce qu'une recherche parfaite apporterait encore ;
3. la corrélation entre la qualité de recherche de la v1 (nDCG 0,328 / 0,351 / 0,554) et la
   justesse mesurée ici, pour répondre à la question centrale ;
4. l'accord juge/humain, son intervalle et son kappa, et le sens des erreurs du juge (indulgent
   ou sévère, d'après la colonne tp/fp/fn/tn).

Relever aussi trois réponses où le juge et l'humain sont en désaccord, avec la justification du
juge, pour la section d'analyse.

- [ ] **Step 6: Écrire `docs/results-v2.md`**

Structure imposée (en anglais), avec les **valeurs réelles** de l'étape 5 :

```markdown
# Results v2 — Generation on AlloprofRetrieval

## Setup
Dataset and query subset, models (generator and judge with their digests), n_passages,
prompt version, hardware, reproduction commands.

## Main results
The per-context table from results/v2/report-generation.md, plus generation.png.

## Findings
1. What retrieval buys — `none` vs `bge-m3` on correctness, with the measured gap.
2. What perfect retrieval would still buy — `bge-m3` vs `oracle`.
3. Does retrieval quality transfer to answer quality? — v1 nDCG vs v2 correctness per context.
4. Can a free local judge be trusted? — agreement, CI, kappa, and the direction of its errors.

## Disagreement analysis
Three answers where judge and human disagree: question, answer, judge rationale, why it is
wrong (or why the human label is debatable).

## Limitations
Single generator, single judge, one prompt per task, binary criteria, teacher answers are
conversational and sometimes partial, n queries, CPU-only latency.
```

- [ ] **Step 7: Ajouter une section au README**

Insérer, juste après la section « Key results » existante, une section « Generation (v2) »
contenant : une phrase de cadrage, le tableau réduit (contexte, correctness, faithfulness,
latence) pour `none`, `bge-m3` et `oracle`, les deux ou trois enseignements chiffrés, la ligne
de calibration du juge, et un lien vers `docs/results-v2.md`. Mettre aussi à jour la section
« Roadmap » : la v2 décrite devient « v2a — done », et la génération de questions synthétiques
devient la prochaine étape.

- [ ] **Step 8: Vérification finale**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: tous les tests passent, aucune erreur ruff.

Vérifier qu'aucune valeur du README ou de `docs/results-v2.md` n'est laissée entre crochets, et
que les chemins d'images pointent vers des fichiers existants.

- [ ] **Step 9: Commit**

```bash
git add .gitignore README.md docs/results-v2.md results/v2/answers.jsonl results/v2/judgments.jsonl results/v2/annotations.jsonl results/v2/report-generation.md results/v2/generation.png
git commit -m "docs: add v2a generation results, calibration and analysis"
```

La publication (`git push`) se fait uniquement avec l'accord explicite de l'utilisateur.
