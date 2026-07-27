# NorQuAD / NorSumm benchmark results

## NorQuAD news — retrieval benchmark

403 passages, 2398 questions (combined news answers_1.json + answers_2.json).

| model                                                       |   recall@1 |   recall@5 |   recall@10 |     mrr@10 |   ndcg@10 |   seconds |
|:------------------------------------------------------------|-----------:|-----------:|------------:|-----------:|----------:|----------:|
| multilingual-e5-large+rerank(mmarco-mMiniLMv2-L12-H384-v1)  | 0.801501   |  0.908257  |   0.929525  | 0.849118   | 0.868882  |    5645.6 |
| intfloat/multilingual-e5-large                              | 0.732277   |  0.879066  |   0.91618   | 0.797691   | 0.826575  |     638   |
| intfloat/multilingual-e5-base                               | 0.707673   |  0.863219  |   0.913261  | 0.775967   | 0.809146  |     237.6 |
| BAAI/bge-m3                                                 | 0.703086   |  0.857381  |   0.900751  | 0.77141    | 0.802838  |     691.1 |
| intfloat/multilingual-e5-small                              | 0.692244   |  0.852794  |   0.89241   | 0.762227   | 0.793906  |      96.5 |
| perplexity-ai/pplx-embed-v1-0.6b                            | 0.685571   |  0.849041  |   0.896163  | 0.75629    | 0.790148  |    1343.7 |
| Qwen/Qwen3-Embedding-0.6B                                   | 0.61593    |  0.773561  |   0.825271  | 0.684311   | 0.718238  |    3762.3 |
| NbAiLab/nb-sentence-bert-base-mnli-test                     | 0.537531   |  0.778982  |   0.851543  | 0.640003   | 0.691034  |     240.4 |
| NbAiLab/nb-sbert-base                                       | 0.454128   |  0.673478  |   0.751043  | 0.548875   | 0.597378  |      68.2 |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 0.415346   |  0.619266  |   0.690158  | 0.50396    | 0.548704  |      80.3 |
| ltg/norbert4-base                                           | 0.00250209 |  0.0145955 |   0.0321101 | 0.00845728 | 0.0138259 |     705.7 |
| ltg/norbert4-large                                          | 0.00250209 |  0.0145955 |   0.0321101 | 0.00845728 | 0.0138259 |    1159.8 |


## NorQuAD news — QA benchmark (Anthropic native API, 300-question sample)

Claude Opus 5 / Fable 5 can't take the embedding retrieval benchmark's role (Anthropic has no embeddings endpoint), so they're scored here as extractive QA models instead: context + question -> answer span, standard SQuAD-style Exact Match / F1 against the NorQuAD gold answers.

| model                 |   n_questions |   n_empty_predictions |   exact_match |     f1 |
|:----------------------|--------------:|----------------------:|--------------:|-------:|
| gemini-3.5-flash      |           300 |                     0 |        0.7333 | 0.9068 |
| claude-fable-5        |           300 |                     9 |        0.7233 | 0.8889 |
| claude-sonnet-5       |           300 |                     0 |        0.6667 | 0.8832 |
| gpt-5.6-sol           |           300 |                     0 |        0.6533 | 0.8779 |
| mistral-small-3.2-24b |           300 |                     0 |        0.6767 | 0.8656 |
| gemma-3-27b-it-base   |           300 |                     0 |        0.6233 | 0.8575 |
| claude-opus-5         |           300 |                    11 |        0.6567 | 0.8538 |
| qwen3.6-27b           |           300 |                    63 |        0.5967 | 0.7201 |


## NorSumm — lexical summarization metrics (OpenRouter models)

| model                       |   n_articles |   n_empty_outputs |   rouge1_f1 |   rouge2_f1 |   rougeL_f1 |
|:----------------------------|-------------:|------------------:|------------:|------------:|------------:|
| claude-fable-5              |           33 |                 0 |      0.5326 |      0.2626 |      0.3464 |
| claude-opus-5               |           33 |                 0 |      0.5137 |      0.2473 |      0.3224 |
| claude-sonnet-5             |           33 |                 0 |      0.4775 |      0.2103 |      0.3031 |
| mistral-small-3.2-24b       |           33 |                 0 |      0.4644 |      0.2064 |      0.3005 |
| gemini-3.5-flash            |           33 |                 0 |      0.4594 |      0.2016 |      0.2963 |
| gpt-5.6-sol                 |           33 |                 0 |      0.4517 |      0.1913 |      0.2904 |
| qwen3.6-27b                 |           33 |                 0 |      0.4475 |      0.1923 |      0.2844 |
| borealis-27b                |           33 |                 0 |      0.3814 |      0.1961 |      0.2702 |
| gemma-3-27b-it-base         |           33 |                 0 |      0.4507 |      0.1744 |      0.2701 |
| normistral-7b-warm-instruct |           33 |                 0 |      0.375  |      0.1827 |      0.2491 |
| normistral-11b-thinking     |           33 |                 0 |      0.2767 |      0.1485 |      0.178  |
