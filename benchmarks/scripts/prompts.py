"""Shared prompt template so every model (OpenRouter or Colab-hosted) is scored
on the exact same instruction.
"""

SYSTEM_PROMPT = (
    "Du er en norsk redaksjonell assistent som skriver korte, presise sammendrag "
    "av nyhetsartikler på bokmål."
)

USER_PROMPT_TEMPLATE = (
    "Skriv et sammendrag av følgende nyhetsartikkel på norsk bokmål. "
    "Sammendraget skal være 3-5 setninger, kun inneholde informasjon som "
    "fremkommer i artikkelen, og ikke inneholde noen innledende setning som "
    "\"Her er et sammendrag\".\n\n"
    "Artikkel:\n{article}\n\nSammendrag:"
)


# Applied identically by every generation script (OpenRouter and the
# Colab/Borealis runner) so no model gets a length advantage/disadvantage and
# every model sees the same input regardless of its context window or the
# host GPU's memory.
MAX_ARTICLE_CHARS = 6000


def truncate_article(text: str) -> str:
    return text[:MAX_ARTICLE_CHARS]


def build_messages(article: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(article=truncate_article(article))},
    ]
