"""Download the NorSumm (Bokmål) test split and dump it to a plain JSON file
that both the OpenRouter generator and the Colab/Borealis generator can read
without needing `datasets`/HF auth.
"""
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    path = hf_hub_download(
        repo_id="SamiaT/NorSumm",
        filename="nb/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(path)

    records = []
    for _, row in df.iterrows():
        records.append({
            "id": row["id"],
            "article": row["article"],
            "reference_summaries": list(row["summaries"]),
        })

    out_path = DATA_DIR / "norsumm_test.json"
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} articles to {out_path}")


if __name__ == "__main__":
    main()
