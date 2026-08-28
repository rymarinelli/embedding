# Borealis-27b: quantized vs. full-precision on NorSumm

`NbAiLab/borealis-27b` scored near the bottom of the NorSumm leaderboard on its
first run, which looked wrong for a 27B Norwegian-native model. This documents
the re-run at **full bf16 precision — nothing quantized** — and what it changed.

## The two runs

| | `borealis-27b-4bit-gguf` | `borealis-27b-bf16-full` |
|---|---|---|
| **Precision** | **GGUF-quantized** | **bf16 — unquantized, the model's native precision** |
| Inference stack | llama.cpp | transformers |
| Hardware | CPU (this container) | Colab A100, GPU + CPU offload |
| Script | `generate_summaries_local_gguf.py` | `generate_summaries_borealis_bf16_colab.py` |
| Weights on disk | ~10–29 GB depending on quant | 51.1 GiB |

Both files are named for their precision so the results table is unambiguous —
`score_summaries.py` takes the model label from the filename.

bf16 is the *native* precision of these weights (the model's own `config.json`
declares `"dtype": "bfloat16"`). Loading in fp32 would upcast to 102 GiB and
gain nothing, so bf16 is the ceiling, not a compromise.

**Offloading is not quantization.** The bf16 run does not fit on Colab's 40 GB
A100, so accelerate holds the overflow in CPU RAM. Those weights are bit-identical
bf16, just stored somewhere slower. It costs speed, not fidelity. The script
verifies this at runtime — it inspects loaded parameter dtypes and aborts if it
finds bitsandbytes layer classes or integer parameter dtypes.

## Result

| Metric | 4-bit GGUF | **bf16 full** | Δ |
|---|---|---|---|
| ROUGE-1 F1 | 0.3814 | **0.4252** | **+0.0438** |
| ROUGE-2 F1 | 0.1961 | **0.2190** | +0.0229 |
| ROUGE-L F1 | 0.2702 | **0.2972** | **+0.0270** |
| Leaderboard rank | 9th of 12 | **5th of 12** | +4 |

Same 33 articles, same prompts, same references, same scorer.

## The repetition artifact is gone

Measured as the share of 5-grams repeated *within* a single summary:

| | mean | max | summaries >5% repeated |
|---|---|---|---|
| 4-bit GGUF | 0.0068 | **0.225** | 1 |
| bf16 full | **0.0000** | **0.000** | **0** |

The clearest case, `bt~BT-20120916-2765289`:

> **4-bit:** *Dommer Tor Holger Bertelsen (57) i Bergen tingrett er anklaget for å sjikanere vitner, og for å favorisere fedre i familiefordelingssaker. Bertelsen mener arbeidsgiveren har spredd usanne rykter om ham.* **Han er anklaget for å sjikanere vitner, og for å favorisere fedre i familiefordelingssaker.**
>
> **bf16:** *Dommer Tor Holger Bertelsen (57) i Bergen tingrett er anklaget for å sjikanere vitner og favorisere fedre i familiefordelingssaker.* **Han er suspendert fra stillingen og kan bli den første dommeren i nyere tid som får sparken.** *Bertelsen mener han er utsatt for skitne triks og usanne rykter fra kollegaer.*

The quantized run spends its third sentence restating its first. The bf16 run
uses it for new information.

Note an exact-string sentence check does **not** catch this — the two sentences
differ in their opening words. Repetition has to be measured on n-grams.

## Two caveats, both load-bearing

**1. The gain is not attributable to precision alone.** The baseline ran through
llama.cpp on GGUF weights; the bf16 run ran through transformers on safetensors.
Tokenization, chat templating and sampling all differ between those runtimes, so
this is *"bf16-via-transformers vs quant-via-llama.cpp"*, not a clean precision
experiment.

To isolate precision, run `generate_summaries_borealis_colab.py` (4-bit, same
transformers stack, fits the A100 with no offload) and compare against that
instead. Then only the precision changes.

**2. Two outputs stop summarising and start transcribing.** `bt~BT-20120405-2681286`
(+1186 chars vs. the quantized run) and `spbm~20050822-508220321` (+795) drift
into copying article text, including quoted dialogue; the first is cut off
mid-word at the 500-token cap. That inflates ROUGE-1 on those items without
reflecting better summaries, so the +0.0438 is not uniformly earned.

## Decoding settings

Matched to the baseline so precision is the variable under test:

| | value | why |
|---|---|---|
| `temperature` | 0.2 | what the GGUF baseline used |
| `max_new_tokens` | 500 | what the GGUF baseline used |
| `repetition_penalty` | 1.0 (off) | the original diagnosis blamed quantization *and* a missing repetition penalty; changing both at once could not attribute the result |
| `MAX_ARTICLE_CHARS` | 6000 | matches `prompts.py` |

Prompts are asserted identical to `prompts.py`.

Set `REPETITION_PENALTY = 1.1` for a separate run to measure the other half of
the original hypothesis.

## Reproducing

Open the notebook in Colab (requires A100 + High-RAM; Pro+ background execution
strongly recommended, the run takes 1.5–4 hours):

```
https://colab.research.google.com/github/rymarinelli/embedding/blob/claude/norquad-norsumm-benchmarks-9qhyke/benchmarks/notebooks/borealis_bf16_colab.ipynb
```

Then place the output in `results/generated_summaries/` and run
`python3 scripts/score_summaries.py`.
