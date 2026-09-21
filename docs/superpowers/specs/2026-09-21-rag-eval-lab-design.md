# rag-eval-lab — Design (v1)

Date : 2026-09-21
Statut : validé en brainstorming, en attente de relecture

## 1. Contexte et objectif

Projet portfolio GitHub pour une recherche de poste d'ingénieur IA (profil junior).
Il doit démontrer la capacité à **prendre des décisions techniques par la mesure** plutôt qu'à l'intuition.

Question centrale : *« Quelle configuration RAG choisir pour un corpus français, et que rapporte vraiment chaque composant par rapport à ce qu'il coûte ? »*

Contraintes :
- Aucun budget API. La v1 tourne 100 % en local.
- Matériel cible : portable Ryzen 7 8845HS, iGPU Radeon 780M (non utilisé), ~28 Go RAM, CPU uniquement. Ollama et Python 3.13 installés.
- Livrer vite : v1 publiable en 1 à 2 semaines, puis itérations.

## 2. Périmètre

### v1 — Benchmark retrieval (ce document)

- **Dataset** : jeu de retrieval public en français, candidat principal **AlloprofRetrieval** (MTEB), sous-échantillonné de façon déterministe. La première tâche du plan vérifie la licence, la taille et le format. Si le dataset est inutilisable, repli sur un autre jeu de retrieval français de MTEB avec qrels au niveau document.
- **Composants comparés** :
  - chunking : `none` (document entier), taille fixe avec chevauchement, récursif ;
  - retriever : BM25, dense (2 à 3 modèles multilingues, ex. `multilingual-e5-small`, `bge-m3`), hybride BM25 + dense par Reciprocal Rank Fusion ;
  - reranker : aucun, ou cross-encoder (ex. `bge-reranker-v2-m3`).
- **Métriques** : Recall@k (k = 5, 10), MRR@10, nDCG@10, latence par requête (p50, p95), temps d'indexation.
- **Sorties** : rapport Markdown (tableaux, graphe de Pareto qualité/latence, graphe d'ablation), README avec résultats clés.

### v2 — Génération et corpus personnalisé (hors de ce plan)

- Évaluation de la réponse finale (fidélité aux sources, pertinence) par un LLM-juge, en comparant un petit LLM local (Ollama) à un LLM via API gratuite (Groq ou Gemini).
- Calibration du juge sur environ 50 exemples annotés à la main.
- Génération de questions synthétiques sur n'importe quel corpus, avec relecture humaine.
- Exemple fourni : documentation Python en français.

### v3 — Options futures (hors de ce plan)

Démo Streamlit ou API FastAPI réutilisant `RetrievalPipeline`, et exemple de corpus Counter-Strike dans `examples/`.

### Hors périmètre

Fine-tuning, agents, interface web en v1, frameworks LangChain et LlamaIndex. Le pipeline est écrit en Python simple, avec des bibliothèques ciblées.

## 3. Architecture

```
configs/*.yaml ──> Experiment runner ──> results/<run>/results.jsonl ──> Report
                        │
     Dataset ──> Chunker ──> Retriever ──> Reranker (optionnel) ──> doc ids
```

Paquet `src/rag_eval_lab/` :

| Module | Rôle | Interface |
|---|---|---|
| `data/` | Chargement et sous-échantillonnage (seed) d'un dataset. Types `Document(id, text, metadata)`, `Query(id, text)` et `Qrels = dict[query_id, dict[doc_id, int]]`. | `load_dataset(name, n_queries, n_docs, seed) -> Dataset` |
| `chunking/` | Découpage : `none`, `fixed(size, overlap)`, `recursive(size)`. Chaque `Chunk` garde son `doc_id`. | `Chunker.chunk(doc) -> list[Chunk]` |
| `retrieval/` | `BM25Retriever`, `DenseRetriever(model_name)`, `HybridRetriever(retrievers, rrf_k=60)` | `index(chunks)`, `search(query, k) -> list[Hit]` |
| `reranking/` | `CrossEncoderReranker(model_name)` | `rerank(query, hits, k) -> list[Hit]` |
| `pipeline.py` | Assemble les briques et remonte des chunks aux documents (score du document = score maximal de ses chunks). | `RetrievalPipeline.search(query, k) -> list[str]` |
| `metrics.py` | Fonctions pures de métriques IR | `evaluate(run, qrels, ks) -> dict[str, float]` |
| `experiment.py` | Expansion de la grille, exécution, chronométrage, écriture des résultats, reprise | `run_experiment(config, out_dir)` |
| `cache.py` | Cache disque des embeddings (`.npy`), clé = hash(modèle, config de chunking, dataset et échantillon) | `get_or_compute(key, fn)` |
| `report/` | Tableaux Markdown et graphiques matplotlib | `build_report(results_dir) -> Path` |
| `cli.py` | CLI Typer : `rel run <config>`, `rel report <results_dir>` | — |

Outillage : `uv`, `pydantic` (validation des configs), `ruff`, `pytest`, `matplotlib`, `sentence-transformers`, `rank-bm25`, `datasets` (Hugging Face).

Les briques sont indépendantes et testables seules. La v2 ajoutera un module `generation/` après le reranker, et la v3 réutilisera `RetrievalPipeline`.

## 4. Déroulement d'une expérience

Exemple de configuration :

```yaml
dataset: { name: alloprof, n_queries: 500, n_docs: 5000, seed: 42 }
k: 10
candidates: 100          # nombre de candidats passés au reranker
grid:
  chunker:   [none, fixed_512_64, recursive_512]
  retriever: [bm25, dense:e5-small, dense:bge-m3, hybrid:bm25+bge-m3]
  reranker:  [none, bge-reranker-v2-m3]
exclude:
  - { chunker: none, reranker: bge-reranker-v2-m3 }
```

Déroulement de `rel run` :
1. Charger et sous-échantillonner le dataset (mis en cache localement). Le sous-échantillon de documents contient toujours tous les documents pertinents des requêtes retenues.
2. Pour chaque configuration de la grille : chunking, puis indexation (avec le cache d'embeddings), avec chronométrage.
3. Pour chaque requête : récupérer `candidates` chunks, rerank optionnel, remonter aux documents, garder le top-k.
4. Ajouter une ligne à `results/<run>/results.jsonl` : config, métriques, latences p50/p95, temps d'indexation, versions des modèles, commit git et infos matériel.
5. Écrire les classements bruts par requête dans `results/<run>/runs/<config_id>.json`, pour l'analyse des échecs.

**Reprise** : une configuration dont le `config_id` (hash de la config) est déjà présent dans `results.jsonl` est ignorée.

## 5. Gestion des erreurs

- La configuration est validée par pydantic avant tout calcul, avec des messages explicites.
- L'échec d'une configuration est journalisé et enregistré avec `status: "error"` dans `results.jsonl`, et les autres configurations continuent.
- Aucun appel réseau pendant l'expérience, sauf le téléchargement initial des modèles et des données.

## 6. Tests

- `metrics` : valeurs calculées à la main sur de petits cas, et comparaison avec `ranx` sur un exemple.
- `chunking` : chevauchement correct, aucun texte perdu, `doc_id` conservé.
- `pipeline` : remontée des chunks aux documents (max score, dédoublonnage).
- `retrieval` : fusion RRF sur des classements connus.
- Test de bout en bout : mini-dataset de 20 documents et 5 requêtes, avec un faux modèle d'embeddings déterministe (hors ligne, rapide).
- CI GitHub Actions : `ruff check` et `pytest`.

## 7. Livrables v1 et critères de réussite

- README orienté recruteur : problème, résultats clés (tableau et graphe de Pareto), trois conclusions chiffrées, commande de reproduction, architecture, limites, feuille de route v2 et v3.
- `docs/results-v1.md` : analyse détaillée et exemples d'échecs.
- **Critères** :
  - un inconnu clone le dépôt et reproduit le rapport en 3 commandes : `uv sync`, `uv run rel run configs/v1.yaml`, `uv run rel report results/v1` ;
  - la grille complète tourne en moins de 2 h sur la machine cible (sinon réduire `n_docs` et `n_queries`) ;
  - la CI est verte.
