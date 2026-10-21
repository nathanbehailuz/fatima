# Fatima

Paired English and Amharic evaluation of [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) on [Uhura-ARC-Easy](https://huggingface.co/datasets/masakhane/uhura-arc-easy). Each item is asked once in each language. The model selects A, B, C, or D, then explains the choice in that language.

Answer letters are scored automatically. Explanation validity, answer–explanation consistency, and language adherence are scored by an OpenAI judge model, with human calibration and review. Judge labels are reported as LLM-assessed explanation validity, not ground truth, and are kept separate from human labels. The study design is in `research_plan.md` and the judging plan in `implementation_plan.md`. Prompt wording is in `prompt.md` and `prompts.json`.

## Layout

| Path | Role |
|---|---|
| `pair_uhura.py` | Join the English and Amharic splits on item id |
| `data/uhura_arc_easy_paired.jsonl` | Paired questions used by the eval |
| `run_eval.py` | Generate responses on a Modal T4 |
| `analyze_responses.py` | Score answer letters, compute explanation metrics, write figures, and open a scoring sheet |
| `judge_explanations.py` | Dry-run inventory, human sample sheets, and OpenAI judging with resume |
| `judge_schema.py` | Pydantic output schema and cross-field checks for the judge |
| `judge_prompt.md` | Fixed judge instructions and rubric (`Rubric version: v1`) |
| `tests/` | Synthetic fixtures and mocked API responses |
| `fatima.ipynb` | Colab notebook for trying the same model interactively |
| `prompts.json` | English and Amharic prompt templates |

Raw responses, scoring sheets, judgments, figures, and logs stay on disk and are listed in `.gitignore`.

## Setup

```bash
pip install datasets matplotlib modal openai pydantic
modal setup
```

`run_eval.py` installs PyTorch and Transformers inside its Modal image. Locally you only need `modal` to launch it, `datasets` to rebuild the pairs, `matplotlib` to analyze responses, and `openai` and `pydantic` to judge explanations.

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
modal run run_eval.py --output data/responses_test_n491_seed0_20261021T155346Z.jsonl
```

The `n` in the filename is the number of questions in the file. `--limit 0` selects the whole split. Resuming through `--output` keeps that path, and renames a file that still uses this pattern when the question count has changed. Each saved row still records the `--limit` and `--seed` used when that row was generated.

## Score responses

```bash
python3 analyze_responses.py data/responses_test_n491_seed0_20261021T155346Z.jsonl
```

The script prints paired answer accuracy and writes a timestamped folder under `analysis/` with a summary, example pairs, and figures. It reads the response file without rewriting it. The summary records run provenance from the records themselves (the file above holds 982 responses, 491 questions). Sixty of those rows were saved during a 30-question pass and still have `sample_limit` 30; the rest have `sample_limit` 0. Validation problems and a `parser_audit` of any answer outcomes that changed under the shared English/Amharic letter parser are included.

It also creates `results/<response filename>` on the first run. That sheet is for hand scores: fill `explanation_validity` (0, 1, or 2), `consistency`, and `language_adherence`, and they are reported as human labels.

## Judge explanations

Only responses whose letter matches the answer key are judged, in both languages, including truncated ones. Incorrect and ambiguous selections are not judged but stay in every denominator. The judge is `gpt-6-luna` (`JUDGE_MODEL` in `judge_explanations.py`). The API key is read from the environment, from `.env` (`OPENAI_API_KEY` or `open_ai`), or from a Colab secret. It is never written to any output. `--limit` counts questions: both languages of a selected question are judged when both are eligible. `0` means every eligible question.

```bash
R=data/responses_test_n491_seed0_20261021T155346Z.jsonl

# 1. Inventory and request count, no API calls
python3 judge_explanations.py $R --dry-run --limit 30 --seed 0

# 2. Blind calibration sheet: 30 completed both-correct pairs. Score it before judging.
python3 judge_explanations.py $R --sample calibration --pairs 30 --seed 0

# 3. Judge the calibration items, then compare with your scores.
#    --run-id keeps this separate from a --limit 30 --seed 0 pilot, which uses the same default name.
python3 judge_explanations.py $R --seed 0 --run-id calibration_seed0 \
    --items-from results/judge/samples/calibration_seed0.jsonl
python3 analyze_responses.py $R \
    --judgments results/judge/calibration_seed0/calibration_seed0.jsonl \
    --human results/judge/samples/calibration_seed0.jsonl

# 4. After freezing judge_prompt.md, a held-out agreement sheet (excludes calibration items
#    and adds 10 correct responses outside the both-correct subset). Score it blind.
python3 judge_explanations.py $R --sample agreement --pairs 30 --seed 1

# 5. Judge 30 questions. Rerun the same command to resume.
python3 judge_explanations.py $R --limit 30 --seed 0
```

The judgments file is named from the response file, the number of questions in the run, and the seed: `results/judge/<source>_n<questions>_seed<seed>/<source>_n<questions>_seed<seed>.jsonl`. A 30-question run of the file above with `--seed 0` is `responses_test_n491_seed0_20261021T155346Z_n30_seed0.jsonl`.

Each run folder `results/judge/<source>_n<questions>_seed<seed>/` holds:

- `<source>_n<questions>_seed<seed>.jsonl`: one flushed record per request, with the judgment or an explicit failure status (`api_error`, `refused`, `incomplete`, `validation_error`), plus source hashes, the exact judge input, rubric/prompt/schema hashes, requested and returned model ids, response id, token usage, and timestamp.
- `manifest.json`: configuration, inventory, run provenance, status counts, an artifact check, and token usage with a projection for the remaining requests (no prices applied).
- `review.jsonl`: every judgment the judge flagged for review plus a seeded random sample of unflagged ones. Fill the plain `explanation_validity`, `consistency`, and `language_adherence` fields; the `judge_*` fields are the original machine labels. Rows already in the file are never overwritten.
- copies of `judge_prompt.md` and the JSON schema.

Resume skips only successful judgments whose response record, rubric, prompt, schema, model, and settings all match. Failures are retried on the next run. A run folder refuses a different configuration; edit `Rubric version` in `judge_prompt.md` and use a new `--run-id` when the rubric changes. Transient API errors are retried with bounded backoff; permanent ones (authentication, bad request, exhausted quota) stop the run.

## Explanation analysis

```bash
python3 analyze_responses.py $R \
    --judgments results/judge/responses_test_n491_seed0_20261021T155346Z_n30_seed0/responses_test_n491_seed0_20261021T155346Z_n30_seed0.jsonl \
    --human results/judge/samples/agreement_seed1.jsonl \
    --human results/judge/responses_test_n491_seed0_20261021T155346Z_n30_seed0/review.jsonl
```

This adds `explanation_summary.json` to the analysis folder with separate `machine`, `human`, and `human_reviewed` (human label where present, otherwise machine) views. Each has a `completed` view (primary: both answers correct, both responses finished, both scored) and an `all_delivered` view that includes truncated responses. Reported per view: the paired fully-valid table, each language's fully-valid rate and their difference with a pair bootstrap, the paired 0/1/2 table and favor counts, and per-language counts of eligible, scored, unscored, flagged, and truncated correct answers. Correct-and-fully-valid uses all correct or incorrect responses as the denominator; correct answers without a score are reported as unresolved with bounds rather than counted as failures. Judge–human agreement is reported by language for each human source, and calibration agreement is marked as not independent.

## Tests

```bash
python3 -m unittest discover -s tests
```
