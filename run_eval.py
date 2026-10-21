"""Run paired Uhura-ARC-Easy questions on Qwen3.5-4B via Modal.

Loads prompts.json by language, renders one fresh user message per item, and
appends each response to a local JSONL file. A rerun skips item/language pairs
already present in that file.

Each run writes a new JSONL file unless --output is set. Pass the same
--output path to resume a file that already contains some responses.

Examples:
  modal run run_eval.py --split validation --limit 2
  modal run run_eval.py --split test --limit 30 --seed 0
  modal run run_eval.py --output data/responses_test_n30_seed0_20261021T120000Z.jsonl
"""

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "uhura_arc_easy_paired.jsonl"
PROMPTS_PATH = ROOT / "prompts.json"
MODEL_ID = "Qwen/Qwen3.5-4B"
CACHE_DIR = "/cache"
LETTERS = ("A", "B", "C", "D")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.57",
        "accelerate",
        "huggingface_hub>=1.32",
    )
    .env({"HF_HOME": CACHE_DIR, "HF_XET_HIGH_PERFORMANCE": "1"})
)
app = modal.App("fatima-uhura-eval")
hf_cache = modal.Volume.from_name("fatima-hf-cache", create_if_missing=True)


def load_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
    return rows


def load_prompts(path=PROMPTS_PATH):
    prompts = json.loads(path.read_text(encoding="utf-8"))
    missing = [language for language in ("am", "en") if language not in prompts]
    if missing:
        raise KeyError(f"{path} is missing language templates: {', '.join(missing)}")
    return prompts


def render_prompt(template, item, language):
    options = {option["label"]: option["text"] for option in item[f"options_{language}"]}
    missing = [letter for letter in LETTERS if letter not in options]
    if missing:
        raise KeyError(f"{item['item_id']} ({language}) is missing options {missing}")
    values = {f"question_{language}": item[f"question_{language}"]}
    values.update(
        {f"option_{letter.lower()}_{language}": options[letter] for letter in LETTERS}
    )
    return template.format(**values)


def select_items(rows, split, limit, seed):
    items = [row for row in rows if row["split"] == split]
    if limit <= 0 or limit >= len(items):
        return items
    by_id = {row["item_id"]: row for row in items}
    chosen_ids = random.Random(seed).sample(sorted(by_id), limit)
    return [by_id[item_id] for item_id in sorted(chosen_ids)]


def completed_pairs(path):
    if not path.exists():
        return set()
    return {(row["item_id"], row["language"]) for row in load_jsonl(path)}


def build_jobs(items, prompts, languages, done):
    jobs = []
    for item in items:
        for language in languages:
            key = (item["item_id"], language)
            if key in done:
                continue
            options = item[f"options_{language}"]
            jobs.append(
                {
                    "item_id": item["item_id"],
                    "split": item["split"],
                    "language": language,
                    "question": item[f"question_{language}"],
                    "options": options,
                    "correct_option": item["correct_option_am"]
                    if language == "am"
                    else item["correct_option"],
                    "prompt": render_prompt(prompts[language], item, language),
                }
            )
    return jobs


@app.cls(
    image=image,
    gpu="T4",
    memory=32768,
    timeout=12 * 60 * 60,
    startup_timeout=30 * 60,
    volumes={CACHE_DIR: hf_cache},
    scaledown_window=5,
)
class Evaluator:
    @modal.enter()
    def load(self):
        import torch
        import transformers
        from huggingface_hub import model_info
        from transformers import AutoModelForCausalLM, AutoTokenizer

        revision = model_info(MODEL_ID).sha
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=revision,
            dtype=torch.float16,
            device_map="cuda",
        )
        model.eval()
        hf_cache.commit()

        self.tokenizer = tokenizer
        self.model = model
        self.torch = torch
        self.meta = {
            "model": MODEL_ID,
            "revision": revision,
            "precision": "float16",
            "gpu": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        }

    def _answer(self, job, max_new_tokens):
        """Run one prompt and return JSON. Torch tensors stay inside this call."""
        settings = {
            "do_sample": False,
            "repetition_penalty": 1.0,
            "max_new_tokens": max_new_tokens,
            "enable_thinking": False,
        }
        messages = [{"role": "user", "content": job["prompt"]}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with self.torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        prompt_length = int(inputs["input_ids"].shape[1])
        token_count = int(outputs.shape[1]) - prompt_length
        answer = self.tokenizer.decode(
            outputs[0, prompt_length:],
            skip_special_tokens=True,
        )
        record = {
            **job,
            "answer": answer,
            "generated_tokens": token_count,
            "reached_token_limit": token_count >= max_new_tokens,
            "generation_settings": settings,
            **self.meta,
        }
        return json.dumps(record, ensure_ascii=False)

    @modal.method(is_generator=True)
    def generate(self, jobs, max_new_tokens):
        for index, job in enumerate(jobs, start=1):
            line = self._answer(job, max_new_tokens)
            print(f"{index}/{len(jobs)} {job['item_id']} {job['language']}", flush=True)
            yield line


@app.local_entrypoint()
def main(
    split: str = "test",
    limit: int = 30,
    seed: int = 0,
    languages: str = "en,am",
    max_new_tokens: int = 1024,
    output: str = "",
):
    language_list = [part.strip() for part in languages.split(",") if part.strip()]
    unknown = [language for language in language_list if language not in ("en", "am")]
    if unknown:
        raise SystemExit(f"Unknown language(s): {', '.join(unknown)}. Use en and am.")

    if output:
        output_path = Path(output)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        size = "all" if limit <= 0 else str(limit)
        output_path = ROOT / "data" / f"responses_{split}_n{size}_seed{seed}_{stamp}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    items = select_items(load_jsonl(DATA_PATH), split, limit, seed)
    jobs = build_jobs(items, load_prompts(), language_list, completed_pairs(output_path))
    print(
        f"{split}: {len(items)} items, {len(jobs)} remaining responses, "
        f"writing {output_path}"
    )
    if not jobs:
        print("Nothing to run.")
        return

    written = 0
    with output_path.open("a", encoding="utf-8") as handle:
        for line in Evaluator().generate.remote_gen(jobs, max_new_tokens):
            record = json.loads(line)
            record["sample_seed"] = seed
            record["sample_limit"] = limit
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1
            print(f"saved {written}/{len(jobs)} {record['item_id']} {record['language']}")
    print(f"wrote {written} responses to {output_path}")
