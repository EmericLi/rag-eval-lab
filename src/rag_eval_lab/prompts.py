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
