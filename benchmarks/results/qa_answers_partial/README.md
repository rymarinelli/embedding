# Partial QA runs — not scored

Files here are incomplete runs, kept so they can be resumed rather than
restarted. They are deliberately **outside** `results/qa_answers/`, because
`score_qa.py` globs that directory and would put a partial run in the results
table beside models that answered all 300 questions.

| File | Answered | Missing |
|---|---|---|
| `borealis-27b-instruct-preview-bf16-full.PARTIAL-159of300.json` | 159 | 141 |

To finish one: put it back at
`/content/drive/MyDrive/borealis_bench/qa/<model>-bf16-full.json` in Colab
(dropping the `.PARTIAL-...` suffix) and re-run the QA cell — the script skips
questions already answered and continues from there.
