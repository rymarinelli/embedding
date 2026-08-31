"""Shared QA prompt so every model (Anthropic native API or OpenRouter) is
scored on the exact same instruction — mirrors prompts.py's role for the
NorSumm summarization benchmark.
"""

SYSTEM_PROMPT = (
    "Du er en nøyaktig leseforståelsesassistent. Du får en tekst og et spørsmål. "
    "Svar KUN med det eksakte svaret slik det fremkommer i teksten, så kort "
    "som mulig — ikke skriv en fullstendig setning, ikke forklar, ikke "
    "inkluder noe annet enn selve svaret."
)
USER_PROMPT_TEMPLATE = "Tekst:\n{context}\n\nSpørsmål: {question}\n\nSvar:"


def build_messages(context: str, question: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(context=context, question=question)},
    ]
