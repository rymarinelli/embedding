# NorQuAD / NorSumm benchmarks

Three benchmarks comparing model capability on Norwegian:

1. **Retrieval** — embedding models on the *news* portion of [NorQuAD](https://github.com/ltgoslo/NorQuAD).
2. **QA** — Claude Opus 5 / Fable 5 via Anthropic's native API on the same NorQuAD news questions,
   scored as extractive QA (Anthropic has no embeddings endpoint, so they can't take part in #1).
3. **Summarization (lexical metrics)** — generative models on [NorSumm](https://huggingface.co/datasets/SamiaT/NorSumm), scored with ROUGE-1/2/L against the human reference summaries.

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
(Recall@1/5/10, MRR@10, nDCG@10). E5-family and Qwen3-Embedding models use the required
`query:`/`passage:` prefixes; `perplexity-ai/pplx-embed-v1-0.6b` (this repo's own `main.py`
fine-tuning base model) deliberately requires none.

Models compared: `NbAiLab/nb-sbert-base`, `NbAiLab/nb-sentence-bert-base-mnli-test`,
`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`,
`intfloat/multilingual-e5-{small,base,large}`, `BAAI/bge-m3`, `perplexity-ai/pplx-embed-v1-0.6b`,
`Qwen/Qwen3-Embedding-0.6B`, `ltg/norbert4-{base,large}`. The two Qwen3-based models need a much
smaller `batch_size` (8, vs. 64 for the rest) to run at reasonable speed on CPU — see the comment in
`retrieval_benchmark.py`.

### Why norbert4 scores near-random

`ltg/norbert4-base`/`-large` score at essentially random chance (recall@1 ≈ 1/403 passages) despite
being LTG's newest and largest Norwegian encoders. This is **not a bug** — verified directly: mean-
pooled embeddings from these models put "Oslo er hovedstaden i Norge" and "Jeg liker å spise pizza på
fredager" (two unrelated sentences) at 0.97 cosine similarity, barely below an actual paraphrase pair
(0.98). This is the well-documented anisotropy problem with raw MLM/causal-pretrained transformer
encoders: without contrastive/sentence-similarity fine-tuning (which is exactly what turns BERT into
SBERT), mean-pooled embeddings collapse into a narrow cone dominated by frequency/generic-content
signal rather than semantic content, and are unusable for retrieval as-is. `NbAiLab/nb-sbert-base`
and `NbAiLab/nb-sentence-bert-base-mnli-test` *are* fine-tuned this way and perform reasonably; the
NorBERT4 family (as released) simply hasn't had that step applied. Included for completeness/
documentation rather than as a fair capability comparison against purpose-built embedding models —
fixing this would mean fine-tuning norbert4 for sentence similarity, out of scope here.

`retrieval_benchmark.py`'s `MeanPoolingEncoder` class implements the standard approach (mean-pool
`last_hidden_state` over non-padding tokens via `attention_mask`, then L2-normalize) for models not
packaged as sentence-transformers.

### Data integrity note: NorQuAD `id` is not globally unique

NorQuAD's own `qa["id"]` field is only unique *within* each annotator file — combining
`answers_1.json` and `answers_2.json` produces hundreds of id collisions between completely
unrelated questions (e.g. id 2952 is "Hvem er Robert Næss?" in file 1 and "Hvor skal F-35 fly?" in
file 2). `build_norquad_corpus.py` reassigns `qid` as a fresh sequential counter across the combined
set and embeds `gold_answer` directly on each `queries.jsonl` row, rather than requiring a
downstream id-keyed lookup — the original lookup-by-id approach in `prepare_qa_sample.py` silently
attached the wrong answer to ~1/3 of the QA sample. The retrieval benchmark was unaffected (it never
used `qid` as a lookup key), but the QA benchmark's Exact Match / F1 scores below were re-scored after
this fix — the corrected numbers are substantially higher.

## 2. NorQuAD news QA benchmark (Anthropic API)

Anthropic has no embeddings API, so Claude Opus 5 / Fable 5 can't take the retrieval benchmark's
role. Instead they're scored as extractive QA models — the standard way NorQuAD/SQuAD-style datasets
are evaluated — on a seeded 300-question sample of the combined corpus.

```
export ANTHROPIC_API_KEY=sk-ant-...             # never commit this key
python3 scripts/prepare_qa_sample.py                    # -> data/norquad_qa_sample.json
python3 scripts/generate_qa_answers_anthropic.py        # -> results/qa_answers/*.json
python3 scripts/score_qa.py                             # -> results/norquad_qa_results.csv
```

Each model is given the gold passage + question and instructed to answer with the exact minimal
span; scored with standard SQuAD-style Exact Match and token-level F1 against the gold answer.

## 3. NorSumm lexical summarization benchmark

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

Models compared via OpenRouter: `anthropic/claude-{sonnet,opus,fable}-5`, `openai/gpt-5.6-sol`,
`google/gemini-3.5-flash`, `google/gemma-3-27b-it` (Borealis's own base model), `qwen/qwen3.6-27b`,
`mistralai/mistral-small-3.2-24b-instruct`.

### Norwegian-native models run locally (no GPU available)

`NbAiLab/borealis-27b`, `norallm/normistral-7b-warm-instruct`, and `norallm/normistral-11b-thinking`
are not hosted on OpenRouter or HF's Inference Router, and this benchmark's execution environment has
no GPU. All three were instead run as CPU inference via GGUF quantizations through `llama-cpp-python`
(`scripts/generate_summaries_local_gguf.py` for the first two; NorMistral-11B-thinking needed a
dedicated `scripts/generate_summaries_normistral11b.py` because llama-cpp-python can't parse its
embedded chat template — see that file's docstring). `scripts/generate_summaries_borealis_colab.py`
remains available as a much faster GPU alternative if you have Colab access.

Two things worth knowing if you re-run these:
- On a Firecracker microVM, `llama-cpp-python`'s default AVX512 codepath can crash with a silent
  SIGILL (`trap invalid opcode ... in libggml-cpu.so`) despite AVX512 being CPUID-advertised. Fix:
  `CMAKE_ARGS="-DGGML_AVX512=OFF -DGGML_AVX512_VBMI=OFF -DGGML_AVX512_VNNI=OFF -DGGML_AVX512_BF16=OFF" pip install --force-reinstall --no-cache-dir --no-binary llama-cpp-python llama-cpp-python`.
- NorMistral-11B-thinking (a "thinking" model) does not reliably close its `<think>` block under a
  plain zero-shot prompt at Q4_K_M — its recorded output is raw reasoning prose, not a concise
  summary, which pulls its ROUGE score down independent of underlying summarization quality.

## Results

See `results/norquad_retrieval_results.csv`, `results/norsumm_lexical_results.csv`, and the rendered
tables in `results/REPORT.md` (regenerate with `python3 scripts/make_report.py`).

## Notes

- No API keys are stored in this repo. `OPENROUTER_API_KEY` and any HF token must be exported as
  environment variables when running the scripts.
