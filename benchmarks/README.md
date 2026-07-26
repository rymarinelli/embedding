# NorQuAD / NorSumm benchmarks

Two independent benchmarks comparing model capability on Norwegian:

1. **Retrieval** — embedding models on the *news* portion of [NorQuAD](https://github.com/ltgoslo/NorQuAD).
2. **Summarization (lexical metrics)** — generative models on [NorSumm](https://huggingface.co/datasets/SamiaT/NorSumm), scored with ROUGE-1/2/L against the human reference summaries, comparing SOTA models via OpenRouter (plus NbAiLab/borealis-27b, generated separately — see below).

## 1. NorQuAD news retrieval benchmark

Data: `data/norquad_news_answers_1.json` and `data/norquad_news_answers_2.json`, downloaded verbatim from
`ltgoslo/NorQuAD` (`data/annotation/news/answers_{1,2}.json`). These are two disjoint sets of
annotated news paragraphs (255 + 148 = **403 unique passages**, **2,398 answerable questions**).

Pipeline:
```
python3 scripts/build_norquad_corpus.py     # -> data/corpus.jsonl, data/queries.jsonl
python3 scripts/retrieval_benchmark.py      # -> results/norquad_retrieval_results.csv
```

For each embedding model, the corpus and questions are encoded, cosine similarity is computed for
every (question, passage) pair, and the passage of origin is scored as the single relevant document
(Recall@1/5/10, MRR@10, nDCG@10). E5-family models use the required `query:`/`passage:` prefixes.

Models compared: `NbAiLab/nb-sbert-base`, `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`,
`intfloat/multilingual-e5-{small,base,large}`, `BAAI/bge-m3`.

## 2. NorSumm lexical summarization benchmark

Data: `data/norsumm_test.json`, derived from the `nb` (Bokmål) **test** split of `SamiaT/NorSumm`
(33 articles, 3 reference summaries each — the `dev`/`validation` split is reserved for training per
the project's `main.py` and intentionally not used here).

Every model is given the identical prompt (`scripts/prompts.py`), with articles truncated to the same
6,000 characters so no model gets a length advantage. ROUGE-1/2/L F1 is computed per article against
all 3 references (max over references, averaged over articles — standard multi-reference ROUGE).

```
export OPENROUTER_API_KEY=sk-or-...     # never commit this key
python3 scripts/prepare_norsumm.py                 # -> data/norsumm_test.json
python3 scripts/generate_summaries_openrouter.py    # -> results/generated_summaries/*.json
python3 scripts/score_summaries.py                  # -> results/norsumm_lexical_results.csv
```

Models compared via OpenRouter: `anthropic/claude-sonnet-5`, `openai/gpt-5.6-sol`,
`google/gemini-3.5-flash`, `google/gemma-3-27b-it` (Borealis's own base model), `qwen/qwen3.6-27b`,
`mistralai/mistral-small-3.2-24b-instruct`.

### NbAiLab/borealis-27b

**Borealis-27b is not hosted on OpenRouter** (checked against the full `/v1/models` catalog) and is
not enabled on any provider behind Hugging Face's Inference Router either. It requires a GPU to run
locally, which this benchmark's execution environment does not have.

`scripts/generate_summaries_borealis_colab.py` is a standalone script meant to be run in a separate
Colab GPU runtime: it 4-bit-quantizes `NbAiLab/borealis-27b` and generates summaries using the exact
same prompt/truncation as the OpenRouter run. Drop its output (`borealis-27b.json`) into
`results/generated_summaries/` and re-run `scripts/score_summaries.py` to fold it into the comparison
table.

## Results

See `results/norquad_retrieval_results.csv`, `results/norsumm_lexical_results.csv`, and the rendered
tables in `results/REPORT.md` (regenerate with `python3 scripts/make_report.py`).

## Notes

- No API keys are stored in this repo. `OPENROUTER_API_KEY` and any HF token must be exported as
  environment variables when running the scripts.
