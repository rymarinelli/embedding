"""Render the two results CSVs as markdown tables into results/REPORT.md."""
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def main():
    lines = ["# NorQuAD / NorSumm benchmark results\n"]

    retrieval_path = RESULTS_DIR / "norquad_retrieval_results.csv"
    if retrieval_path.exists():
        df = pd.read_csv(retrieval_path)
        df = df.sort_values("ndcg@10", ascending=False)
        lines.append("## NorQuAD news — retrieval benchmark\n")
        lines.append(
            f"{df['n_passages'].iloc[0]:.0f} passages, {df['n_queries'].iloc[0]:.0f} questions "
            "(combined news answers_1.json + answers_2.json).\n"
        )
        lines.append(md_table(df.drop(columns=["n_queries", "n_passages"], errors="ignore")))
        lines.append("")

    qa_path = RESULTS_DIR / "norquad_qa_results.csv"
    if qa_path.exists():
        df = pd.read_csv(qa_path).sort_values("f1", ascending=False)
        lines.append("\n## NorQuAD news — QA benchmark (Anthropic native API, 300-question sample)\n")
        lines.append(
            "Claude Opus 5 / Fable 5 can't take the embedding retrieval benchmark's role "
            "(Anthropic has no embeddings endpoint), so they're scored here as extractive QA "
            "models instead: context + question -> answer span, standard SQuAD-style "
            "Exact Match / F1 against the NorQuAD gold answers.\n"
        )
        lines.append(md_table(df))
        lines.append("")

    lexical_path = RESULTS_DIR / "norsumm_lexical_results.csv"
    if lexical_path.exists():
        df = pd.read_csv(lexical_path)
        lines.append("\n## NorSumm — lexical summarization metrics (OpenRouter models)\n")
        lines.append(md_table(df))
        lines.append("")

    out = RESULTS_DIR / "REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
