# rag-eval-lab v2a — Évaluation de la génération (design)

Date : 2026-10-06
Statut : implémenté en v2a. Écarts avec ce document : le bloc `annotation:` de la configuration a
été retiré (`rel annotate` prend `--n` et `--seed`), `num_ctx` est explicite pour le générateur
et le juge, et l'empreinte des prompts est scindée en deux (réponse, juge). Résultats et limites :
`docs/results-v2.md`.
Spec précédente : `docs/design/specs/2026-09-21-rag-eval-lab-design.md` (v1, retrieval)

## 1. Contexte et objectif

La v1 a mesuré la **recherche** : quelle configuration retrouve les bons documents, à quel coût.
Elle s'arrête avant la réponse. La v2a répond à la question suivante :

> **Une meilleure recherche produit-elle de meilleures réponses, et jusqu'où ?**

Cet angle sert les deux objectifs du projet : il boucle la démonstration « je mesure avant de
régler », et il correspond au diagnostic qu'attend un prospect dont le RAG hallucine — faut-il
corriger la recherche ou le générateur ?

Contraintes inchangées : aucun budget API, portable Ryzen 7 8845HS, CPU seul, ~28 Go de RAM.
Ollama est installé avec `llama3.2:latest` (3B), `llama3.2:1B` et `llama3.1:8B`.

L'ancienne « v2 » de la spec v1 couvrait trois chantiers indépendants (évaluation de la
génération, génération de questions synthétiques, exemple sur corpus personnalisé). Elle est
découpée : **ce document ne couvre que l'évaluation de la génération.**

## 2. Périmètre

### Plan d'expérience

5 contextes × 1 générateur (`llama3.2:latest`), évalués par un juge local (`llama3.1:8B`),
lui-même calibré contre 60 annotations humaines.

| Contexte | `kind` | Rôle |
|---|---|---|
| `none` | `empty` | Borne basse : ce que le modèle répond sans passage |
| `bm25` | `retrieved` | Recherche lexicale faible (nDCG@10 = 0,351 en v1) |
| `e5-small` | `retrieved` | Le point faible de la v1 (0,328) |
| `bge-m3` | `retrieved` | Le meilleur de la v1 (0,554), `fixed_1000_200` |
| `oracle` | `oracle` | Borne haute : documents de référence du jeu de données |

### Critères mesurés

Trois critères **binaires** par réponse :

- **fidélité** : chaque affirmation de la réponse est appuyée par les passages fournis ;
- **pertinence** : la réponse répond à la question posée ;
- **justesse** : la réponse est d'accord avec la réponse rédigée par le prof (champ `answer`
  du jeu Alloprof).

La justesse sépare deux échecs que la fidélité seule confond : une réponse fidèle à des
passages hors sujet est fidèle **et** fausse. Réserve à documenter dans le rapport : les
réponses de prof sont conversationnelles et parfois partielles, donc ce critère mesure
l'accord avec une réponse humaine, pas une vérité absolue.

Pour le contexte `empty`, aucun passage n'est fourni : la fidélité et la pertinence n'ont pas
de sens et valent `null`. Seule la justesse est jugée. Les `null` sont exclus des moyennes et
des calculs d'accord.

**L'humain annote la fidélité et la justesse** (la pertinence est presque toujours vraie et
n'apporte pas d'information).

### Hors périmètre

- Juge via API (Groq, Gemini) : l'interface `Judge` le permettra, l'implémentation viendra
  plus tard.
- Génération de questions synthétiques : chantier séparé.
- Vérification des citations passage par passage.
- Comparaison de plusieurs générateurs ou de plusieurs prompts de génération : on compare des
  **contextes**, à générateur et prompt fixés.

### Résultat acceptable dans les deux sens

Si le juge local s'avère médiocre (accord faible avec l'humain), ce n'est pas un échec du
projet mais son résultat : « un juge local gratuit ne suffit pas pour ce critère, preuve à
l'appui ». Le rapport est publiable dans les deux cas.

## 3. Architecture

```
dataset ─┬─> ContextSource ─> Generator ─> answers.jsonl ─> Judge ─> judgments.jsonl ─┐
         │   (5 variantes)    (Ollama)                      (Ollama)                   ├─> report
         └─> annotate (humain) ──────────> annotations.jsonl ─> calibration ───────────┘
```

Nouveaux modules dans `src/rag_eval_lab/` :

| Module | Rôle | Interface |
|---|---|---|
| `context.py` | `RetrievedContext` (enveloppe une `RetrievalPipeline`), `OracleContext`, `EmptyContext` | `ContextSource.passages(query) -> list[Chunk]` |
| `ollama_client.py` | Client HTTP vers `127.0.0.1:11434` (bibliothèque standard), délai d'attente, relances, sortie JSON forcée, lecture de l'empreinte du modèle | `generate(model, prompt, ...) -> Completion`, `model_digest(model) -> str` |
| `generation.py` | `OllamaGenerator` : rendu du prompt, appel, mesure de latence | `Generator.answer(query, passages) -> Answer` |
| `judging.py` | `OllamaJudge` : deux appels par réponse — fidélité + pertinence (avec passages), justesse (avec réponse du prof) ; sortie JSON stricte | `Judge.verdict(answer, sample) -> Verdict` |
| `prompts.py` | Gabarits, `ANSWER_PROMPT_VERSION` et `JUDGE_PROMPT_VERSION` (hachages du contenu des gabarits de réponse, resp. de jugement) | `render_answer_prompt()`, `render_grounding_prompt()`, `render_correctness_prompt()` |
| `annotation.py` | Tirage équilibré des exemples, persistance des verdicts humains | `sample_for_annotation(...)`, `append_annotation(...)` |
| `calibration.py` | Accord brut, kappa de Cohen, intervalle de confiance de Wilson, matrice de confusion | `agreement(judgments, annotations) -> dict` |
| `generation_experiment.py` | Orchestration reprenable des deux étapes, isolation des erreurs | `run_generation(cfg, out)`, `run_judging(cfg, out)` |
| `report_generation.py` | Tableau par contexte, graphique, section calibration, exemples d'échecs | `build_generation_report(dir) -> Path` |

Modification ciblée de la v1 : `RetrievalPipeline` expose `search_passages(query, k) -> list[Hit]`
en plus de `search()`, car le prompt a besoin du texte des passages, pas des identifiants de
documents. `search()` est conservée telle quelle et réimplémentée au-dessus de la nouvelle
méthode.

Réutilisés sans changement : `RetrievalPipeline`, le cache d'embeddings, `data.py`, la
validation pydantic, le format `jsonl` reprenable, la CLI Typer.

### Quatre étapes séparées

```
rel generate configs/v2.yaml     # answers.jsonl
rel judge results/v2             # judgments.jsonl
rel annotate results/v2 --n 60   # annotations.jsonl (interactif, reprenable)
rel report-gen results/v2        # report.md + graphiques + calibration
```

Les prompts du juge vont changer plusieurs fois. Séparer juger de générer permet de rejuger
sans régénérer, et les annotations restent valables puisqu'elles portent sur les réponses.

## 4. Configuration et déroulement

```yaml
dataset: { name: alloprof, n_queries: 80, seed: 42 }   # voir ci-dessous : sous-ensemble des 300 de la v1
n_passages: 5
generator: { model: "llama3.2:latest", temperature: 0, max_tokens: 300, timeout_s: 180 }
judge:     { model: "llama3.1:8B",     temperature: 0, max_tokens: 200, timeout_s: 300 }
contexts:
  - { name: none,     kind: empty }
  - { name: bm25,     kind: retrieved, chunker: none,           retriever: bm25 }
  - { name: e5-small, kind: retrieved, chunker: none,           retriever: dense:e5-small }
  - { name: bge-m3,   kind: retrieved, chunker: fixed_1000_200, retriever: dense:bge-m3 }
  - { name: oracle,   kind: oracle }
annotation: { n: 60, seed: 7 }
```

Le jeu de questions est **le préfixe de celui de la v1** : on applique `sample_dataset`
avec `n_queries = 300` et `seed = 42` (donc exactement l'échantillon v1, dans son ordre de
tirage), puis on garde les `n_queries` premières questions. Les résultats de génération
portent ainsi sur des questions déjà mesurées côté recherche, et augmenter `n_queries` plus
tard n'invalide pas les réponses déjà produites.

`n_queries: 80` est une valeur de départ : la première tâche du plan mesure la vitesse réelle
de génération et de jugement sur la machine cible, puis fixe la valeur définitive pour tenir
dans le budget (voir §7).

**`answer_id`** identifie un couple (contexte, question) pour un générateur et une version de
prompt donnés : `make_key(context_name, query_id, generator_model, ANSWER_PROMPT_VERSION)[:12]`.
C'est la clé de reprise de `rel generate`, et la clé de jointure des trois fichiers.

Enregistrements :

- `answers.jsonl` : `answer_id`, `context`, `query_id`, `question`, `reference` (réponse du
  prof), `answer`, `passages` (liste de `{id, text}` — le texte est stocké pour que juger et
  annoter n'aient besoin ni de pipeline ni du jeu de données), `n_passage_chars`, `latency_ms`, `generator` (modèle, empreinte, options), `prompt_version`,
  `status`, `timestamp` ; en cas d'échec : `error`.
- `judgments.jsonl` : `answer_id`, `judge` (modèle, empreinte), `prompt_version`,
  `faithfulness`, `relevance`, `correctness` (booléens), `rationales` (trois chaînes courtes),
  `latency_ms`, `status`, `timestamp` ; en cas d'échec : `error` et `raw_output`.
- `annotations.jsonl` : `answer_id`, `faithfulness`, `correctness` (booléens ou `null` si
  passé), `note`, `seconds`, `timestamp`.

**Reprise** : un `answer_id` déjà présent avec `status: "ok"` n'est pas relancé ; un statut
`error` est retenté au lancement suivant. Même logique que la v1.

### Contexte oracle

Les documents de référence sont découpés avec le découpeur de la meilleure configuration
(`fixed_1000_200`), puis les premiers passages de chaque document sont pris en alternance,
jusqu'à `n_passages`. Les passages ne sont **pas** sélectionnés par similarité avec la
question : ce serait fabriquer le contexte parfait à partir de la question, donc une borne
haute trichée.

### Contexte vide

Aucun passage n'est fourni et le prompt demande explicitement de répondre à partir des
connaissances propres du modèle. La différence avec les autres contextes mesure ce que le RAG
apporte réellement.

## 5. Annotation et calibration

`rel annotate` affiche, pour chaque exemple tiré : la question, les passages fournis, la
réponse générée et la réponse du prof. Il demande « Fidèle ? » puis « Juste ? » en `o`/`n`,
avec une touche pour passer un cas douteux (enregistré `null`, exclu des calculs) et une pour
quitter. La reprise est automatique.

**Le verdict du juge n'est jamais affiché pendant l'annotation.** L'afficher produirait un
accord artificiellement élevé par ancrage, ce qui invaliderait la mesure.

Le tirage est **équilibré entre les 5 contextes** (12 exemples chacun pour `n = 60`), graine
fixe, afin que l'accord ne soit pas dominé par les contextes qui produisent surtout des
réponses fidèles.

`calibration.py` calcule, pour la fidélité et la justesse :

- l'accord brut avec l'humain et son intervalle de confiance de Wilson à 95 % ;
- le **kappa de Cohen**, qui corrige l'accord dû au hasard — indispensable ici puisqu'une
  classe domine ;
- la matrice de confusion, qui dit **dans quel sens** le juge se trompe (indulgent ou sévère).

## 6. Gestion des erreurs et tests

**Erreurs :**

- Ollama injoignable ou modèle absent : message explicite nommant la commande à lancer
  (`ollama serve`, `ollama pull <modèle>`), arrêt immédiat avec code 1.
- Délai dépassé, ou JSON du juge invalide après deux relances : l'élément est enregistré avec
  `status: "error"`, la sortie brute conservée, les autres éléments continuent.
- La configuration est validée par pydantic avant tout appel.

**Tests (pytest, hors ligne) :**

- client Ollama factice injecté : aucun appel réseau, aucun modèle téléchargé en CI ;
- rendu des prompts : les passages, la question et la réponse de référence apparaissent aux
  bons endroits ; `ANSWER_PROMPT_VERSION` change quand un gabarit de réponse change (et pas `JUDGE_PROMPT_VERSION`, qui est seulement inscrite dans les jugements) ;
- analyse des sorties du juge : JSON valide, JSON avec texte autour, JSON invalide (relance
  puis erreur) ;
- `ContextSource` : `empty` renvoie zéro passage, `oracle` ne dépend pas de la question et
  respecte `n_passages`, `retrieved` passe par la pipeline (encodeur `hashing`) ;
- calibration : accord, kappa et matrice de confusion calculés à la main sur un petit cas,
  dont un cas où l'accord brut est élevé mais le kappa proche de zéro ;
- reprise et isolation des erreurs de `run_generation` / `run_judging` ;
- bout en bout des quatre commandes avec un client factice, y compris une session
  d'annotation simulée sur l'entrée standard.

## 7. Livrables et critères de réussite

**Livrables :**

- `docs/results-v2.md` : plan d'expérience, tableau par contexte (taux de fidélité, pertinence
  et justesse, latence), graphique, section calibration, exemples d'échecs commentés, limites ;
- une section « Generation (v2) » dans le README, avec les trois enseignements chiffrés ;
- `results/v2/` versionné : `answers.jsonl`, `judgments.jsonl`, `annotations.jsonl`,
  `report.md` et les graphiques, pour que chaque affirmation reste auditable, comme en v1.

**Critères de réussite :**

- `rel generate` puis `rel judge` tiennent ensemble en moins de 8 heures sur la machine cible,
  après mesure de la vitesse réelle en première tâche ; si l'estimation dépasse, `n_queries`
  est réduit avant le lancement complet, et la valeur retenue est documentée dans
  `configs/v2.yaml` ;
- 60 annotations collectées, accord et kappa publiés avec leur intervalle de confiance ;
- reproduction en 4 commandes (`rel generate`, `rel judge`, `rel annotate`, `rel report-gen`) ;
- CI verte, suite de tests toujours hors ligne et sans téléchargement de modèle.
