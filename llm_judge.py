#!/usr/bin/env python3
"""
Shared LLM-as-a-judge helpers for the QA benchmarks in this repo
(benchmark_norquad.py, benchmark_normedqa.py).

Used as a *secondary* correctness signal alongside each task's primary
automatic metric (SQuAD EM/F1 for NorQuAD, letter-match accuracy for
NorMedQA), not as a replacement - this lets you compare judge-vs-metric
agreement rather than relying on the judge alone.
"""

import re

from openrouter_client import call_model

# Same judge model NorEval (Mikhailov et al. 2025, Section 5 / Appendix F)
# uses for its own LLM-as-a-judge evaluation. Override with any OpenRouter
# model id - a stronger model here gives a more reliable secondary signal.
DEFAULT_JUDGE_MODEL = "meta-llama/llama-3.3-70b-instruct"


def judge_norquad_answer(judge_model, headers, context, question, gold_answers, predicted_answer):
    """Ask the judge whether predicted_answer is a correct, text-grounded
    answer to question, given context and the accepted gold answer(s).
    Returns True/False, or None if the judge call failed or was unparseable.
    """
    gold_str = "; ".join(gold_answers)
    prompt = (
        "Du er en streng fasitsjekker for et norsk leseforståelsesspørsmål.\n\n"
        f"Tekst: {context}\n\n"
        f"Spørsmål: {question}\n\n"
        f"Godkjente fasitsvar (skilt med \";\"): {gold_str}\n\n"
        f"Modellens svar: {predicted_answer}\n\n"
        "Er modellens svar korrekt, dvs. semantisk ekvivalent med et av fasitsvarene "
        "og støttet av teksten over? Svar med kun ett ord: \"riktig\" eller \"feil\"."
    )
    response, error = call_model(judge_model, prompt, headers, max_tokens=5, temperature=0.0)
    if error or not response:
        return None
    verdict = response.strip().lower()
    if verdict.startswith("riktig") or verdict.startswith("korrekt"):
        return True
    if verdict.startswith("feil"):
        return False
    return None


def judge_normedqa_answer(judge_model, headers, question, options, letters, raw_response):
    """Ask the judge which lettered option a model's raw (possibly messy)
    response picked. Returns a letter in `letters`, or None if unclear/failed.
    """
    options_block = "\n".join(f"{letter}: {opt}" for letter, opt in zip(letters, options))
    prompt = (
        "Du skal avgjøre hvilket svaralternativ en AI-modell valgte på et norsk "
        "medisinsk flervalgsspørsmål.\n\n"
        f"Spørsmål: {question}\n\n"
        f"Svaralternativer:\n{options_block}\n\n"
        f"Modellens rå svar: {raw_response}\n\n"
        f"Hvilken bokstav ({', '.join(letters)}) valgte modellen? Svar med kun bokstaven. "
        "Hvis svaret ikke tydelig peker på ett av alternativene, svar \"uklart\"."
    )
    response, error = call_model(judge_model, prompt, headers, max_tokens=5, temperature=0.0)
    if error or not response:
        return None
    match = re.search(r"\b([A-Z])\b", response.strip().upper())
    if match and match.group(1) in letters:
        return match.group(1)
    return None
