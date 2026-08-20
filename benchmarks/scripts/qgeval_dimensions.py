"""QGEval's seven evaluation dimensions and their 1-3 scoring guidelines.

Rubric text is reproduced verbatim from Table 7 of:
  Fu, Wei, Hu, Cai, Liu. "QGEval: Benchmarking Multi-dimensional Evaluation
  for Question Generation." EMNLP 2024, pp. 11783-11803.

Kept in English (as in the paper) even though the passages/questions being
judged are Norwegian — the rubric defines the task, not the content language.
"""

LINGUISTIC = ["fluency", "clarity", "conciseness"]
TASK_ORIENTED = ["relevance", "consistency", "answerability", "answer_consistency"]
DIMENSIONS = LINGUISTIC + TASK_ORIENTED

DISPLAY_NAMES = {
    "fluency": "Fluency",
    "clarity": "Clarity",
    "conciseness": "Conciseness",
    "relevance": "Relevance",
    "consistency": "Consistency",
    "answerability": "Answerability",
    "answer_consistency": "Answer Consistency",
}

RUBRIC = {
    "fluency": (
        "Score 1: The question is incoherent, with imprecise wording or significant "
        "grammatical errors, making it difficult to comprehend its meaning.\n"
        "Score 2: The question is slightly incoherent or contains minor grammatical "
        "errors, but it does not hinder the understanding of the question's meaning.\n"
        "Score 3: The question is fluent and grammatically correct."
    ),
    "clarity": (
        "Score 1: The question is too broad or expressed in a confusing manner, making "
        "it difficult to understand or leading to ambiguity. Particularly, if the "
        "generated sentence is not a question but a declarative sentence, it should be "
        "considered in this situation.\n"
        "Score 2: The question is not expressed very clearly and specifically, but it is "
        "possible to infer the question's meaning based on the given passage.\n"
        "Score 3: The question is clear and specific, without any ambiguity."
    ),
    "conciseness": (
        "Score 1: The question contains too much redundant information, making it "
        "difficult to understand its intent.\n"
        "Score 2: The question includes some redundant information, but it does not "
        "impact the understanding of its meaning.\n"
        "Score 3: The question is concise and does not contain any unnecessary information."
    ),
    "relevance": (
        "Score 1: The question is completely unrelated to the passage.\n"
        "Score 2: The question is somewhat related to the passage but it asks for "
        "non-crucial information related to the passage.\n"
        "Score 3: The question is relevant to the context, and the information it seeks "
        "is crucial to the passage."
    ),
    "consistency": (
        "Score 1: The question contains factual contradictions with the passage or "
        "logical errors.\n"
        "Score 2: The information sought in the question is not fully described in the "
        "passage.\n"
        "Score 3: The information in the question is entirely consistent with the passage."
    ),
    "answerability": (
        "Score 1: The question cannot be answered based on the provided passage.\n"
        "Score 2: The question can be partially answered based on the provided passage, "
        "or the answer to the question can be inferred to some extent.\n"
        "Score 3: The question can be answered definitively based on the given passage."
    ),
    "answer_consistency": (
        "Score 1: The question cannot be answered by the provided answer.\n"
        "Score 2: The question can be partially answered using the provided answer.\n"
        "Score 3: The question can be answered directly using the provided answer."
    ),
}


def rubric_block():
    """Full rubric, formatted for inclusion in a judge prompt."""
    parts = []
    for dim in DIMENSIONS:
        parts.append(f"{DISPLAY_NAMES[dim]} ({dim}):\n{RUBRIC[dim]}")
    return "\n\n".join(parts)
