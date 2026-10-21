"""Load Uhura-ARC-Easy in English and Amharic and inner-join on item id.

Source: https://huggingface.co/datasets/masakhane/uhura-arc-easy
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

DATASET = "masakhane/uhura-arc-easy"
SPLITS = ("train", "validation", "test")


def choices_as_options(choices):
    labels = choices["label"]
    texts = choices["text"]
    if len(labels) != len(texts):
        raise ValueError(f"choice labels and texts differ in length: {labels!r} vs {texts!r}")
    return [{"label": label, "text": text} for label, text in zip(labels, texts)]


def index_by_id(rows):
    indexed = {}
    duplicates = []
    for row in rows:
        item_id = row["id"]
        if item_id in indexed:
            duplicates.append(item_id)
            continue
        indexed[item_id] = row
    return indexed, duplicates


def pair_split(am_rows, en_rows, split):
    am_by_id, am_dupes = index_by_id(am_rows)
    en_by_id, en_dupes = index_by_id(en_rows)
    shared_ids = sorted(set(am_by_id) & set(en_by_id))

    pairs = []
    key_mismatches = []
    for item_id in shared_ids:
        am = am_by_id[item_id]
        en = en_by_id[item_id]
        if am["answerKey"] != en["answerKey"]:
            key_mismatches.append(item_id)
        pairs.append(
            {
                "item_id": item_id,
                "split": split,
                "question_en": en["question"],
                "question_am": am["question"],
                "options_en": choices_as_options(en["choices"]),
                "options_am": choices_as_options(am["choices"]),
                "correct_option": en["answerKey"],
                "correct_option_am": am["answerKey"],
            }
        )

    stats = {
        "split": split,
        "am": len(am_by_id),
        "en": len(en_by_id),
        "paired": len(pairs),
        "am_only": len(set(am_by_id) - set(en_by_id)),
        "en_only": len(set(en_by_id) - set(am_by_id)),
        "duplicate_ids_am": am_dupes,
        "duplicate_ids_en": en_dupes,
        "answer_key_mismatches": key_mismatches,
    }
    return pairs, stats


def load_pairs(splits):
    am = load_dataset(DATASET, "am_multiple_choice")
    en = load_dataset(DATASET, "en_multiple_choice")
    pairs = []
    stats = []
    for split in splits:
        split_pairs, split_stats = pair_split(am[split], en[split], split)
        pairs.extend(split_pairs)
        stats.append(split_stats)
    return pairs, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=[*SPLITS, "all"],
        default="all",
        help="Which split to pair. Default: all.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/uhura_arc_easy_paired.jsonl"),
        help="JSONL path for the joined pairs.",
    )
    args = parser.parse_args()
    splits = SPLITS if args.split == "all" else (args.split,)

    pairs, stats = load_pairs(splits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")

    for row in stats:
        print(
            f"{row['split']}: am={row['am']} en={row['en']} "
            f"paired={row['paired']} am_only={row['am_only']} en_only={row['en_only']} "
            f"key_mismatches={len(row['answer_key_mismatches'])} "
            f"duplicate_ids={len(row['duplicate_ids_am']) + len(row['duplicate_ids_en'])}"
        )
    print(f"wrote {len(pairs)} pairs to {args.output}")


if __name__ == "__main__":
    main()
