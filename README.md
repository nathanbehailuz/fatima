# Fatima

Paired English and Amharic evaluation of [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) on [Uhura-ARC-Easy](https://huggingface.co/datasets/masakhane/uhura-arc-easy). Each item is asked once in each language. The model selects A, B, C, or D, then explains the choice in that language.

Answer letters are scored automatically. Explanation validity, answer–explanation consistency, and language adherence are filled in by hand. The study design is in `research_plan.md`. Prompt wording is in `prompt.md` and `prompts.json`.

## Layout

| Path | Role |
|---|---|
| `pair_uhura.py` | Join the English and Amharic splits on item id |
| `data/uhura_arc_easy_paired.jsonl` | Paired questions used by the eval |
| `run_eval.py` | Generate responses on a Modal T4 |
| `analyze_responses.py` | Score answer letters, write figures, and open a scoring sheet |
| `fatima.ipynb` | Colab notebook for trying the same model interactively |
| `prompts.json` | English and Amharic prompt templates |

Raw responses, scoring sheets, figures, and logs stay on disk and are listed in `.gitignore`.

## Setup

```bash
pip install datasets matplotlib modal
modal setup
```

`run_eval.py` installs PyTorch and Transformers inside its Modal image. Locally you only need `modal` to launch it, `datasets` to rebuild the pairs, and `matplotlib` to analyze responses.

## Rebuild the paired questions

```bash
python3 pair_uhura.py
```

This writes `data/uhura_arc_easy_paired.jsonl`. The file already in the repo is that join.

## Generate responses

```bash
modal run run_eval.py --split test --limit 30 --seed 0
```

Each run writes a new file under `data/` unless `--output` is set. Passing the same `--output` path resumes a file and skips item/language pairs already in it. Decoding is greedy, with thinking disabled and a 1024-token cap.

```bash
modal run run_eval.py --output data/responses_test_n30_seed0_20261021T155346Z.jsonl
```

## Score responses

```bash
python3 analyze_responses.py data/responses_test_n30_seed0_20261021T155346Z.jsonl
```

The script prints paired answer accuracy and writes a timestamped folder under `analysis/` with a summary, example pairs, and figures. It also creates `results/<response filename>` on the first run. Fill `explanation_validity` (0, 1, or 2), `consistency`, and `language_adherence` in that sheet, then run the same command again to include the explanation comparison.
