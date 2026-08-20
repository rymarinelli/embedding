"""Compare Fable-generated higher-level questions (norquad_fable_questions.json,
one per passage) against NorQuAD's own human-authored questions (queries.jsonl,
~6 per passage) on the same 403 passages — same lexical-metric methodology used
for the Amedia question set: length, vocabulary/TTR, question-opener
distribution, and lexical overlap (ROUGE) with the source passage.

Usage:
    python3 compare_norquad_fable_questions.py
"""
import json
import re
import statistics as st
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_PNG = RESULTS_DIR / "norquad_fable_vs_original_questions.png"

WORD_RE = re.compile(r"[a-zæøåA-ZÆØÅ0-9]+", re.UNICODE)


def words(s):
    return WORD_RE.findall(s.lower())


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def lexical_report(name, questions, passages_by_doc):
    lens = [len(words(q)) for q in questions]
    all_words = [w for q in questions for w in words(q)]
    vocab = set(all_words)
    ttr = len(vocab) / len(all_words) if all_words else 0.0
    openers = Counter(words(q)[0] if words(q) else "?" for q in questions)
    print(f"\n=== {name} (n={len(questions)}) ===")
    print(f"lengde: mean={st.mean(lens):.1f} median={st.median(lens)} min={min(lens)} max={max(lens)}")
    print(f"TTR: {len(vocab)} unike / {len(all_words)} totalt = {ttr:.3f}")
    print("topp spørreord:", openers.most_common(6))
    return lens, ttr, openers


def main():
    corpus = load_jsonl(DATA_DIR / "corpus.jsonl")
    passages_by_doc = {c["doc_id"]: c["text"] for c in corpus}

    fable_path = RESULTS_DIR / "norquad_fable_questions.json"
    fable_data = json.loads(fable_path.read_text(encoding="utf-8"))
    fable_data = [d for d in fable_data if d.get("spørsmål") and not d.get("error")]
    fable_questions = [(d["doc_id"], d["spørsmål"]) for d in fable_data]

    queries = load_jsonl(DATA_DIR / "queries.jsonl")
    orig_questions = [(q["gold_doc_id"], q["question"]) for q in queries]

    fable_lens, fable_ttr, fable_openers = lexical_report(
        "Fable (1/passasje)", [q for _, q in fable_questions], passages_by_doc)
    orig_lens, orig_ttr, orig_openers = lexical_report(
        "NorQuAD original (~6/passasje)", [q for _, q in orig_questions], passages_by_doc)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)

    def overlap_scores(pairs):
        r1, rl = [], []
        for doc_id, q in pairs:
            passage = passages_by_doc.get(doc_id, "")
            if not passage:
                continue
            s = scorer.score(passage, q)
            r1.append(s["rouge1"].fmeasure)
            rl.append(s["rougeL"].fmeasure)
        return r1, rl

    fable_r1, fable_rl = overlap_scores(fable_questions)
    orig_r1, orig_rl = overlap_scores(orig_questions)

    print(f"\nFable overlapp m/passasje: ROUGE-1 mean={st.mean(fable_r1):.3f}  ROUGE-L mean={st.mean(fable_rl):.3f}")
    print(f"Original overlapp m/passasje: ROUGE-1 mean={st.mean(orig_r1):.3f}  ROUGE-L mean={st.mean(orig_rl):.3f}")

    # ---- plot ----
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), dpi=200)

    ax = axes[0]
    bins = range(0, max(max(fable_lens), max(orig_lens)) + 4, 2)
    ax.hist(orig_lens, bins=bins, color="#96712B", alpha=0.6, label=f"NorQuAD original (n={len(orig_lens)})", density=True, zorder=2)
    ax.hist(fable_lens, bins=bins, color="#1F6F78", alpha=0.75, label=f"Fable (n={len(fable_lens)})", density=True, zorder=3)
    ax.set_xlabel("Antall ord i spørsmålet")
    ax.set_ylabel("Andel (tetthet)")
    ax.set_title("Lengdefordeling")
    ax.legend(frameon=False, fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    top_fable = [w for w, _ in fable_openers.most_common(6)]
    top_orig = [w for w, _ in orig_openers.most_common(6)]
    all_top = list(dict.fromkeys(top_fable + top_orig))[:8]
    y = range(len(all_top))
    fable_tot = sum(fable_openers.values())
    orig_tot = sum(orig_openers.values())
    fable_pct = [100 * fable_openers.get(w, 0) / fable_tot for w in all_top]
    orig_pct = [100 * orig_openers.get(w, 0) / orig_tot for w in all_top]
    h = 0.38
    ax.barh([i + h / 2 for i in y], fable_pct, height=h, color="#1F6F78", label="Fable", zorder=3)
    ax.barh([i - h / 2 for i in y], orig_pct, height=h, color="#96712B", label="NorQuAD original", zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels([w.capitalize() for w in all_top])
    ax.invert_yaxis()
    ax.set_xlabel("Andel av spørsmål (%)")
    ax.set_title("Spørreord (første ord)")
    ax.legend(frameon=False, fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax = axes[2]
    ax.hist(orig_r1, bins=20, range=(0, 1), color="#96712B", alpha=0.6, label="NorQuAD original", density=True, zorder=2)
    ax.hist(fable_r1, bins=20, range=(0, 1), color="#1F6F78", alpha=0.75, label="Fable", density=True, zorder=3)
    ax.set_xlabel("ROUGE-1 F1 (spørsmål vs. passasje)")
    ax.set_ylabel("Andel (tetthet)")
    ax.set_title("Leksikalsk overlapp\nspørsmål ↔ passasje")
    ax.legend(frameon=False, fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Fable-genererte vs. NorQuADs originale spørsmål (samme 403 passasjer)", fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    print(f"\nsaved plot: {OUT_PNG}")


if __name__ == "__main__":
    main()
