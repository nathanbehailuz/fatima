"""Analyze paired English and Amharic responses from run_eval.py.

Answer selection is scored automatically. Explanation validity,
answer–explanation consistency, and language adherence follow the rubric in
research_plan.md and judge_prompt.md. Labels come from two kinds of source,
which are never mixed silently:

- machine: judgments.jsonl from judge_explanations.py (--judgments)
- human: the hand-scoring sheet results/<response filename>, plus any
  calibration, agreement, or review files passed with --human

The source response file is read but never rewritten.

Examples:
  python3 analyze_responses.py data/responses.jsonl
  python3 analyze_responses.py data/responses_test_n491_seed0_20261021T155346Z.jsonl \\
      --judgments results/judge/<source>_n<questions>_seed<seed>/<source>_n<questions>_seed<seed>.jsonl \\
      --human results/judge/samples/calibration_seed0.jsonl
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "fatima-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
LETTERS = ("A", "B", "C", "D")
EN_COLOR = "#3174A1"
AM_COLOR = "#E1812C"
MISSING_COLOR = "#B0B0B0"
CONSISTENCY = ("supports", "contradicts", "unclear")
ADHERENCE = ("maintained", "partly_maintained", "substantially_abandoned")
LANGUAGES = ("en", "am")
JUDGE_CONFIG_FIELDS = ("rubric_version", "prompt_sha256", "schema_sha256", "requested_model", "settings")
REQUIRED_FIELDS = ("item_id", "language", "question", "options", "correct_option")
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 0
SELECTION_LINE = re.compile(r"^(?:\*\*|__)?[\[(]?([ABCD])[\])]?(?:\*\*|__)?[.:]?$")
LEADING_OPTION = re.compile(r"^(?:\*\*|__)?([ABCD])\.\s*(\S.*?)\s*$")
BOLD_LETTER = re.compile(r"\*\*\s*([ABCD])\s*\*\*")
FIRST_LINE_LETTER = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])")
EMPHASIS = re.compile(r"\*\*|__")
LATIN_WORD = re.compile(r"[A-Za-z]+")
ANSWER_LABEL_WORDS = {"the", "correct", "answer", "is", "option", "choice"}


def load_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
    return rows


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_sha256(row):
    text = json.dumps(row, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_selection(answer):
    """Read a letter from the first line only, with the same rules for every language.

    Markdown emphasis is ignored. The line is accepted when it is a bare letter,
    'B. option text', or a line whose only Latin-script words are one option
    letter plus answer-label words (e.g. 'Answer: B', 'ትክክለኛው መልስ B ነው።').
    The explanation is never searched.
    """
    if not isinstance(answer, str) or not answer.strip():
        return None, "missing"
    first = EMPHASIS.sub("", answer.strip().splitlines()[0]).strip()
    if SELECTION_LINE.match(first):
        return re.sub(r"[^ABCD]", "", first), "parsed"
    leading = LEADING_OPTION.match(first)
    if leading and not re.fullmatch(r"[ABCD]", leading.group(2)):
        return leading.group(1), "parsed"
    words = LATIN_WORD.findall(first)
    letters = [word for word in words if word in LETTERS]
    others = [word for word in words if word not in LETTERS and word.lower() not in ANSWER_LABEL_WORDS]
    if len(letters) == 1 and not others:
        return letters[0], "parsed"
    return None, "ambiguous"


def legacy_parse_selection(answer, language):
    """The earlier rules: Amharic-only bold cleanup and letter search. Kept to report changes."""
    if not isinstance(answer, str) or not answer.strip():
        return None, "missing"
    if language == "am":
        answer = BOLD_LETTER.sub(r"\1", answer)
    first = answer.strip().splitlines()[0].strip()
    if SELECTION_LINE.match(first):
        return re.sub(r"[^ABCD]", "", first), "parsed"
    leading = LEADING_OPTION.match(first)
    if leading and not re.fullmatch(r"[ABCD]", leading.group(2)):
        return leading.group(1), "parsed"
    if language == "am":
        letters = list(dict.fromkeys(FIRST_LINE_LETTER.findall(first)))
        if len(letters) == 1:
            return letters[0], "parsed"
    return None, "ambiguous"


def record_problems(row):
    problems = [f"missing {field}" for field in REQUIRED_FIELDS if row.get(field) in (None, "")]
    if row.get("language") not in (None, *LANGUAGES):
        problems.append(f"unknown language {row.get('language')!r}")
    options = row.get("options")
    if options is not None:
        labels = [option.get("label") for option in options] if isinstance(options, list) and all(
            isinstance(option, dict) for option in options
        ) else None
        if labels is None or sorted(labels) != list(LETTERS):
            problems.append(f"option labels must be exactly A–D, got {labels}")
    key = str(row.get("correct_option") or "").strip().upper()
    if row.get("correct_option") not in (None, "") and key not in LETTERS:
        problems.append(f"answer key {row.get('correct_option')!r} is not A–D")
    return problems


def outcome_for(letter, status, correct_option):
    if status != "parsed":
        return "missing_or_ambiguous"
    return "correct" if letter == correct_option else "incorrect"


def parser_audit(rows):
    """Compare the shared parser with the earlier rules and list every changed outcome."""
    changed = []
    counts = {language: {"legacy": Counter(), "shared": Counter()} for language in LANGUAGES}
    for row in rows:
        legacy = outcome_for(*legacy_parse_selection(row["answer"], row["language"]), row["correct_option"])
        counts[row["language"]]["legacy"][legacy] += 1
        counts[row["language"]]["shared"][row["answer_outcome"]] += 1
        if legacy != row["answer_outcome"]:
            changed.append(
                {
                    "item_id": row["item_id"],
                    "language": row["language"],
                    "legacy_outcome": legacy,
                    "shared_outcome": row["answer_outcome"],
                    "first_line": first_line(row["answer"]),
                }
            )
    return {
        "note": "Answer outcomes under the earlier Amharic-only rules versus the shared parser.",
        "counts": {
            language: {name: dict(counter) for name, counter in pair.items()}
            for language, pair in counts.items()
        },
        "changed": changed,
    }


def script_share(text, language):
    letters = [char for char in text or "" if char.isalpha()]
    if not letters:
        return None
    if language == "am":
        expected = sum(
            "\u1200" <= char <= "\u137F"
            or "\u1380" <= char <= "\u139F"
            or "\u2D80" <= char <= "\u2DDF"
            for char in letters
        )
    else:
        expected = sum(("A" <= char <= "Z") or ("a" <= char <= "z") for char in letters)
    return expected / len(letters)


def termination(row):
    if not isinstance(row.get("answer"), str):
        return "execution_failure"
    if row.get("reached_token_limit"):
        return "token_limit_truncation"
    return "natural_end"


def prepare(rows):
    """Validate and parse response records without changing them.

    Returns (prepared rows, report). Invalid records and every copy of a
    conflicting duplicate are left out and listed in the report. Identical
    duplicates are kept once.
    """
    seen = {}
    hashes = {}
    report = {"invalid_records": [], "identical_duplicates": [], "conflicting_duplicates": []}
    conflicted = set()
    for index, row in enumerate(rows, start=1):
        problems = record_problems(row)
        if problems:
            report["invalid_records"].append({"line": index, "item_id": row.get("item_id"), "problems": problems})
            continue
        key = (row["item_id"], row["language"])
        digest = record_sha256(row)
        if key in hashes:
            if hashes[key] == digest:
                report["identical_duplicates"].append(list(key))
            else:
                conflicted.add(key)
            continue
        hashes[key] = digest
        answer = row.get("answer")
        letter, status = parse_selection(answer)
        correct_option = str(row["correct_option"]).strip().upper()
        seen[key] = {
            "item_id": row["item_id"],
            "split": row.get("split"),
            "language": row["language"],
            "question": row.get("question"),
            "options": row.get("options"),
            "correct_option": correct_option,
            "selected_option": letter,
            "selection_status": status,
            "answer_outcome": outcome_for(letter, status, correct_option),
            "answer": answer,
            "generated_tokens": row.get("generated_tokens"),
            "termination": termination(row),
            "expected_script_share": script_share(answer, row["language"]),
            "record_sha256": digest,
        }
    for key in sorted(conflicted):
        seen.pop(key, None)
        report["conflicting_duplicates"].append(list(key))
    return list(seen.values()), report


def run_provenance(source_rows):
    """Distinct values per run field, read from the records rather than the filename."""
    fields = (
        "split", "model", "revision", "precision", "gpu", "torch_version",
        "transformers_version", "sample_seed", "sample_limit",
    )
    provenance = {
        field: dict(Counter(json.dumps(row.get(field), ensure_ascii=False) for row in source_rows))
        for field in fields
    }
    provenance["generation_settings"] = dict(
        Counter(json.dumps(row.get("generation_settings"), sort_keys=True) for row in source_rows)
    )
    provenance["records"] = len(source_rows)
    provenance["item_ids"] = len({row.get("item_id") for row in source_rows})
    return provenance


def pair_items(rows):
    by_item = {}
    for row in rows:
        by_item.setdefault(row["item_id"], {})[row["language"]] = row
    paired = []
    unpaired = []
    for item_id, languages in sorted(by_item.items()):
        if "en" in languages and "am" in languages:
            paired.append({"item_id": item_id, "en": languages["en"], "am": languages["am"]})
        else:
            unpaired.append({"item_id": item_id, "languages": sorted(languages)})
    return paired, unpaired


def evaluable(row):
    return row["answer_outcome"] in ("correct", "incorrect")


def contingency(paired):
    """Cells a–d from the research plan. Missing selections are excluded."""
    included = []
    excluded = []
    for pair in paired:
        if evaluable(pair["en"]) and evaluable(pair["am"]):
            included.append(pair)
        else:
            excluded.append(pair)
    a = b = c = d = 0
    for pair in included:
        en_ok = pair["en"]["answer_outcome"] == "correct"
        am_ok = pair["am"]["answer_outcome"] == "correct"
        if en_ok and am_ok:
            a += 1
        elif en_ok and not am_ok:
            b += 1
        elif not en_ok and am_ok:
            c += 1
        else:
            d += 1
    return {"a": a, "b": b, "c": c, "d": d, "included": included, "excluded": excluded}


def rate(numerator, denominator):
    if denominator == 0:
        return {"numerator": numerator, "denominator": 0, "rate": None, "wilson_95": None}
    point = numerator / denominator
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": point,
        "wilson_95": wilson_interval(numerator, denominator),
    }


def wilson_interval(successes, total, z=1.96):
    if total == 0:
        return None
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def answer_summary(table):
    a, b, c, d = table["a"], table["b"], table["c"], table["d"]
    total = a + b + c + d
    return {
        "cells": {
            "english_correct_amharic_correct": a,
            "english_correct_amharic_incorrect": b,
            "english_incorrect_amharic_correct": c,
            "english_incorrect_amharic_incorrect": d,
        },
        "n_pairs": total,
        "english_accuracy": rate(a + b, total),
        "amharic_accuracy": rate(a + c, total),
        "amharic_correct_given_english_correct": rate(a, a + b),
        "english_correct_given_amharic_correct": rate(a, a + c),
        "excluded_missing_or_ambiguous": len(table["excluded"]),
    }


def bootstrap_answer_rates(included, draws=5000, seed=0):
    if not included:
        return None
    rng = random.Random(seed)
    flags = [
        (
            pair["en"]["answer_outcome"] == "correct",
            pair["am"]["answer_outcome"] == "correct",
        )
        for pair in included
    ]
    stored = {name: [] for name in ("english_accuracy", "amharic_accuracy", "accuracy_difference_am_minus_en")}
    size = len(flags)
    for _ in range(draws):
        sample = [flags[rng.randrange(size)] for _ in range(size)]
        a = sum(en and am for en, am in sample)
        b = sum(en and not am for en, am in sample)
        c = sum((not en) and am for en, am in sample)
        d = sum((not en) and (not am) for en, am in sample)
        total = a + b + c + d
        en_rate = (a + b) / total
        am_rate = (a + c) / total
        stored["english_accuracy"].append(en_rate)
        stored["amharic_accuracy"].append(am_rate)
        stored["accuracy_difference_am_minus_en"].append(am_rate - en_rate)
    return {
        name: percentile_interval(values)
        for name, values in stored.items()
    } | {"draws": draws, "seed": seed, "unit": "question pair"}


def percentile_interval(values, low=2.5, high=97.5):
    ordered = sorted(values)
    def pick(percent):
        index = (len(ordered) - 1) * percent / 100
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        weight = index - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight
    return [pick(low), pick(high)]


def score_filled(score):
    return any(
        score.get(field) is not None
        for field in ("explanation_validity", "consistency", "language_adherence")
    )


def human_labels(score_rows, source):
    """Filled rows from a human scoring file, keyed by (item_id, language)."""
    labels = {}
    for score in score_rows:
        if not score_filled(score):
            continue
        validity = score.get("explanation_validity")
        consistency = score.get("consistency")
        adherence = score.get("language_adherence")
        label = f"{score.get('item_id')} {score.get('language')} in {source}"
        if validity not in (0, 1, 2):
            raise ValueError(f"explanation_validity must be 0, 1, or 2 for {label}")
        if consistency not in CONSISTENCY:
            raise ValueError(f"consistency must be one of {CONSISTENCY} for {label}")
        if adherence not in ADHERENCE:
            raise ValueError(f"language_adherence must be one of {ADHERENCE} for {label}")
        labels[(score["item_id"], score["language"])] = {
            "explanation_validity": validity,
            "consistency": consistency,
            "language_adherence": adherence,
            "needs_human_review": False,
            "source": source,
        }
    return labels


def merge_human(sources):
    """Combine human sources. Keys scored differently in two sources are dropped and reported."""
    merged = {}
    conflicts = []
    fields = ("explanation_validity", "consistency", "language_adherence")
    for name, labels in sources.items():
        for key, label in labels.items():
            if key not in merged:
                merged[key] = label
                continue
            if any(merged[key][field] != label[field] for field in fields):
                conflicts.append({"item_id": key[0], "language": key[1], "sources": [merged[key]["source"], name]})
    for conflict in conflicts:
        merged.pop((conflict["item_id"], conflict["language"]), None)
    return merged, conflicts


def load_judgments(path, rows):
    """Machine labels from judge_explanations.py output.

    A judgment counts only when its status is ok, it still validates against
    the schema, and it was made for the same response record. One file must
    hold a single judge configuration.
    """
    from judge_schema import validate_judgment

    records = load_jsonl(path)
    configs = {
        json.dumps({field: record.get(field) for field in JUDGE_CONFIG_FIELDS}, sort_keys=True)
        for record in records
    }
    if len(configs) > 1:
        raise ValueError(f"{path} mixes judge configurations; analyze one run at a time")
    current = {(row["item_id"], row["language"]): row["record_sha256"] for row in rows}
    latest = {}
    stale = []
    for record in records:
        key = (record["item_id"], record["language"])
        if current.get(key) != record.get("source_record_sha256"):
            stale.append(list(key))
            continue
        if record.get("status") == "ok" or latest.get(key, {}).get("status") != "ok":
            latest[key] = record
    labels = {}
    failures = []
    for key, record in latest.items():
        if record.get("status") != "ok":
            failures.append({"item_id": key[0], "language": key[1], "status": record.get("status")})
            continue
        judgment, errors = validate_judgment(record.get("judgment"))
        if judgment is None or errors:
            failures.append({"item_id": key[0], "language": key[1], "status": "saved_judgment_invalid"})
            continue
        labels[key] = judgment.model_dump() | {"source": "machine"}
    config = json.loads(next(iter(configs))) if configs else {}
    return labels, {
        "path": str(path),
        "config": config,
        "records": len(records),
        "status_counts": dict(Counter(record.get("status") for record in latest.values())),
        "stale_judgments_ignored": len(stale),
        "failures": failures,
    }


def ensure_scores(response_path, rows):
    """Create results/<response filename> and append any new items without overwriting scores."""
    path = RESULTS_DIR / Path(response_path).name
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_jsonl(path) if path.exists() else []
    by_key = {(row["item_id"], row["language"]): row for row in existing}
    created = not path.exists()
    changed = created
    ordered = []
    for row in rows:
        key = (row["item_id"], row["language"])
        if key in by_key:
            kept = by_key.pop(key)
            if score_filled(kept):
                ordered.append(kept)
            else:
                fresh = scoring_template([row])[0]
                if any(
                    kept.get(field) != fresh.get(field)
                    for field in ("selected_option", "answer_outcome", "answer")
                ):
                    changed = True
                ordered.append(fresh)
        else:
            ordered.append(scoring_template([row])[0])
            changed = True
    ordered.extend(by_key.values())
    if changed:
        write_jsonl(path, ordered)
    return path, ordered, created


def validity_of(row, labels):
    label = labels.get((row["item_id"], row["language"]))
    return None if label is None else label["explanation_validity"]


def completed(row):
    return row["termination"] == "natural_end"


def paired_explanations(paired, labels, completed_only, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """Primary comparison: both answers correct (and completed, if asked), both explanations scored."""
    eligible = [
        pair
        for pair in paired
        if pair["en"]["answer_outcome"] == "correct" and pair["am"]["answer_outcome"] == "correct"
        and (not completed_only or (completed(pair["en"]) and completed(pair["am"])))
    ]
    scored = [
        (validity_of(pair["en"], labels), validity_of(pair["am"], labels))
        for pair in eligible
        if validity_of(pair["en"], labels) is not None and validity_of(pair["am"], labels) is not None
    ]
    n = len(scored)
    both = sum(en == 2 and am == 2 for en, am in scored)
    en_only = sum(en == 2 and am != 2 for en, am in scored)
    am_only = sum(en != 2 and am == 2 for en, am in scored)
    english = rate(both + en_only, n)
    amharic = rate(both + am_only, n)
    ordinal = {f"en_{en}": {f"am_{am}": 0 for am in (0, 1, 2)} for en in (0, 1, 2)}
    for en, am in scored:
        ordinal[f"en_{en}"][f"am_{am}"] += 1
    return {
        "eligible_pairs": len(eligible),
        "scored_pairs": n,
        "unscored_pairs": len(eligible) - n,
        "fully_valid_cells": {
            "both": both,
            "english_only": en_only,
            "amharic_only": am_only,
            "neither": n - both - en_only - am_only,
        },
        "english_fully_valid": english,
        "amharic_fully_valid": amharic,
        "fully_valid_difference_am_minus_en": (
            None if n == 0 else amharic["rate"] - english["rate"]
        ),
        "pair_bootstrap_95": bootstrap_fully_valid(scored, draws, seed),
        "ordinal_table": ordinal,
        "favor_english": sum(en > am for en, am in scored),
        "favor_amharic": sum(am > en for en, am in scored),
        "tie": sum(en == am for en, am in scored),
        "note": "Scores are ordinal. Favor counts compare order only, not distances.",
    }


def bootstrap_fully_valid(scored, draws, seed):
    if not scored:
        return None
    rng = random.Random(seed)
    flags = [(en == 2, am == 2) for en, am in scored]
    stored = {name: [] for name in ("english_fully_valid", "amharic_fully_valid", "difference_am_minus_en")}
    size = len(flags)
    for _ in range(draws):
        sample = [flags[rng.randrange(size)] for _ in range(size)]
        en_rate = sum(en for en, _ in sample) / size
        am_rate = sum(am for _, am in sample) / size
        stored["english_fully_valid"].append(en_rate)
        stored["amharic_fully_valid"].append(am_rate)
        stored["difference_am_minus_en"].append(am_rate - en_rate)
    return {name: percentile_interval(values) for name, values in stored.items()} | {
        "draws": draws,
        "seed": seed,
        "unit": "question pair",
    }


def language_explanations(rows, labels, language, completed_only):
    """Secondary per-language counts. These groups hold different questions per language."""
    in_view = [
        row for row in rows
        if row["language"] == language and (not completed_only or completed(row))
    ]
    evaluable_rows = [row for row in in_view if evaluable(row)]
    correct = [row for row in evaluable_rows if row["answer_outcome"] == "correct"]
    labelled = [labels[(row["item_id"], row["language"])] for row in correct if (row["item_id"], row["language"]) in labels]
    scored = [label for label in labelled if label["explanation_validity"] is not None]
    fully_valid = sum(label["explanation_validity"] == 2 for label in scored)
    unresolved = len(correct) - len(scored)
    denominator = len(evaluable_rows)
    joint = rate(fully_valid, denominator)
    joint["unresolved_correct_without_score"] = unresolved
    joint["provisional"] = unresolved > 0
    joint["bounds"] = (
        None if denominator == 0 else [fully_valid / denominator, (fully_valid + unresolved) / denominator]
    )
    joint["denominator_definition"] = (
        "Correct or incorrect responses in this view. Missing or ambiguous selections are excluded "
        "and counted separately. Correct answers without a score are unresolved, not failures."
    )
    fully = rate(fully_valid, len(scored))
    fully["denominator_definition"] = "Correct responses in this view with a 0/1/2 score"
    return {
        "responses": len(in_view),
        "missing_or_ambiguous_excluded": len(in_view) - len(evaluable_rows),
        "correct": len(correct),
        "correct_scored": len(scored),
        "correct_unscored_by_labeler": len(labelled) - len(scored),
        "correct_without_label": len(correct) - len(labelled),
        "correct_flagged_for_review": sum(bool(label.get("needs_human_review")) for label in labelled),
        "correct_truncated": sum(row["termination"] == "token_limit_truncation" for row in correct),
        "validity_counts": {str(score): sum(label["explanation_validity"] == score for label in scored) for score in (0, 1, 2)},
        "fully_valid_among_correct_scored": fully,
        "correct_and_fully_valid_among_evaluable": joint,
        "consistency_counts": dict(Counter(label["consistency"] for label in labelled)),
        "contradictions": sum(label["consistency"] == "contradicts" for label in labelled),
        "language_adherence_counts": dict(Counter(label["language_adherence"] for label in labelled)),
        "language_adherence_failures": sum(label["language_adherence"] == "substantially_abandoned" for label in labelled),
    }


def explanation_views(rows, paired, labels):
    views = {}
    for name, completed_only in (("completed", True), ("all_delivered", False)):
        views[name] = {
            "primary_paired": paired_explanations(paired, labels, completed_only),
            "by_language": {
                language: language_explanations(rows, labels, language, completed_only)
                for language in LANGUAGES
            },
        }
    return views


def truncation_summary(rows):
    summary = {}
    for language in LANGUAGES:
        in_language = [row for row in rows if row["language"] == language]
        correct = [row for row in in_language if row["answer_outcome"] == "correct"]
        summary[language] = {
            "all_responses": rate(sum(row["termination"] == "token_limit_truncation" for row in in_language), len(in_language)),
            "correct_responses": rate(sum(row["termination"] == "token_limit_truncation" for row in correct), len(correct)),
        }
    return summary


def agreement(machine, human, rows, examples=10):
    """Judge–human agreement on keys both scored, by language."""
    by_key = {(row["item_id"], row["language"]): row for row in rows}
    report = {}
    for language in LANGUAGES:
        keys = sorted(key for key in human if key[1] == language and key in machine and key in by_key)
        both_scored = [key for key in keys if machine[key]["explanation_validity"] is not None]
        confusion = {f"human_{h}": {f"machine_{m}": 0 for m in (0, 1, 2, None)} for h in (0, 1, 2)}
        disagreements = []
        for key in keys:
            h = human[key]["explanation_validity"]
            m = machine[key]["explanation_validity"]
            confusion[f"human_{h}"][f"machine_{m}"] += 1
            if h != m and len(disagreements) < examples:
                disagreements.append(
                    {
                        "item_id": key[0],
                        "human": h,
                        "machine": m,
                        "machine_justification": machine[key].get("brief_justification"),
                        "answer": by_key[key]["answer"],
                    }
                )
        report[language] = {
            "compared": len(keys),
            "machine_unscored": len(keys) - len(both_scored),
            "exact_validity": rate(sum(human[k]["explanation_validity"] == machine[k]["explanation_validity"] for k in both_scored), len(both_scored)),
            "fully_valid_vs_not": rate(
                sum((human[k]["explanation_validity"] == 2) == (machine[k]["explanation_validity"] == 2) for k in both_scored),
                len(both_scored),
            ),
            "consistency": rate(sum(human[k]["consistency"] == machine[k]["consistency"] for k in keys), len(keys)),
            "language_adherence": rate(sum(human[k]["language_adherence"] == machine[k]["language_adherence"] for k in keys), len(keys)),
            "confusion": confusion,
            "disagreement_examples": disagreements,
        }
    return report


def cell_name(pair):
    en_ok = pair["en"]["answer_outcome"] == "correct"
    am_ok = pair["am"]["answer_outcome"] == "correct"
    if not evaluable(pair["en"]) or not evaluable(pair["am"]):
        return "excluded_missing_or_ambiguous"
    return (
        f"english_{'correct' if en_ok else 'incorrect'}"
        f"_amharic_{'correct' if am_ok else 'incorrect'}"
    )


def examples_by_cell(paired):
    grouped = {}
    for pair in paired:
        grouped.setdefault(cell_name(pair), []).append(
            {
                "item_id": pair["item_id"],
                "en": brief(pair["en"]),
                "am": brief(pair["am"]),
            }
        )
    return grouped


def brief(row):
    kept = {
        "selected_option": row["selected_option"],
        "selection_status": row["selection_status"],
        "correct_option": row["correct_option"],
        "answer_outcome": row["answer_outcome"],
        "termination": row["termination"],
        "generated_tokens": row["generated_tokens"],
        "question": row["question"],
        "options": row["options"],
        "answer": row["answer"],
    }
    return kept


def scoring_template(rows):
    template = []
    for row in rows:
        template.append(
            {
                "item_id": row["item_id"],
                "language": row["language"],
                "selected_option": row["selected_option"],
                "correct_option": row["correct_option"],
                "answer_outcome": row["answer_outcome"],
                "explanation_validity": None,
                "consistency": None,
                "language_adherence": None,
                "answer": row["answer"],
            }
        )
    return template


def outcome_counts(rows, language):
    counts = Counter(row["answer_outcome"] for row in rows if row["language"] == language)
    return {name: counts.get(name, 0) for name in ("correct", "incorrect", "missing_or_ambiguous")}


def plot_answer_outcomes(rows, path):
    labels = ["Correct", "Incorrect", "Missing or ambiguous"]
    keys = ["correct", "incorrect", "missing_or_ambiguous"]
    colors = ["#2A9D8F", "#C44E52", MISSING_COLOR]
    languages = [("English", "en"), ("Amharic", "am")]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = list(range(len(languages)))
    width = 0.24
    series = []
    for offset, (label, key, color) in enumerate(zip(labels, keys, colors)):
        heights = [outcome_counts(rows, language)[key] for _, language in languages]
        positions = [position + (offset - 1) * width for position in x]
        drawn = [(position, height) for position, height in zip(positions, heights) if height]
        if drawn:
            series.append((label, color, drawn))
    max_height = max((height for _, _, drawn in series for _, height in drawn), default=1)
    for label, color, drawn in series:
        bars = ax.bar([position for position, _ in drawn], [height for _, height in drawn], width, label=label, color=color)
        for bar, (_, height) in zip(bars, drawn):
            ax.text(bar.get_x() + bar.get_width() / 2, height + max_height * 0.02, str(height), ha="center", va="bottom")
    ax.set_xticks(x, [name for name, _ in languages])
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_ylim(0, max_height * 1.18)
    ax.set_title("Answer selection by language")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, borderaxespad=0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_contingency(summary, path):
    cells = summary["cells"]
    matrix = [
        [cells["english_correct_amharic_correct"], cells["english_correct_amharic_incorrect"]],
        [cells["english_incorrect_amharic_correct"], cells["english_incorrect_amharic_incorrect"]],
    ]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["Amharic correct", "Amharic incorrect"])
    ax.set_yticks([0, 1], ["English correct", "English incorrect"])
    peak = max(value for row in matrix for value in row)
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            share = value / summary["n_pairs"] if summary["n_pairs"] else 0
            text_color = "white" if peak and value >= peak * 0.6 else "black"
            ax.text(
                col_index,
                row_index,
                f"{value}\n({share:.0%})",
                ha="center",
                va="center",
                color=text_color,
            )
    ax.set_title(f"Paired answer outcomes (N = {summary['n_pairs']})")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_tokens(paired, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for index, pair in enumerate(paired):
        en_tokens = pair["en"]["generated_tokens"] or 0
        am_tokens = pair["am"]["generated_tokens"] or 0
        jitter = (index % 5 - 2) * 0.015
        ax.plot([0 + jitter, 1 + jitter], [en_tokens, am_tokens], color="#C8C8C8", zorder=1)
        ax.scatter([0 + jitter], [en_tokens], color=EN_COLOR, zorder=2)
        ax.scatter([1 + jitter], [am_tokens], color=AM_COLOR, zorder=2)
    ax.set_xticks([0, 1], ["English", "Amharic"])
    ax.set_ylabel("Generated tokens")
    ax.set_title("Generated tokens for the same question")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_explanation_differences(primary, source, path):
    """English scores down the rows, Amharic scores across the columns, both on 0–2."""
    scores = (0, 1, 2)
    table = primary["ordinal_table"]
    matrix = [[table[f"en_{english}"][f"am_{amharic}"] for amharic in scores] for english in scores]
    scored = primary["scored_pairs"]
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    image = ax.imshow(matrix, cmap="Blues", origin="lower")
    ax.set_xticks(list(scores), [str(score) for score in scores])
    ax.set_yticks(list(scores), [str(score) for score in scores])
    ax.set_xlabel("Amharic")
    ax.set_ylabel("English")
    peak = max(value for row in matrix for value in row)
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            share = value / scored if scored else 0
            text_color = "white" if peak and value >= peak * 0.6 else "black"
            ax.text(col_index, row_index, f"{value}\n({share:.0%})", ha="center", va="center", color=text_color)
    ax.set_title(
        "Comparing LLM explanation scores\nfor correctly answered questions\n"
        f"({scored} of {primary['eligible_pairs']} completed pairs)"
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_accuracy_intervals(summary, bootstrap, path):
    names = ["English", "Amharic"]
    stats = [summary["english_accuracy"], summary["amharic_accuracy"]]
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    xs = [0, 1]
    colors = [EN_COLOR, AM_COLOR]
    for x, stat, color in zip(xs, stats, colors):
        if stat["rate"] is None:
            continue
        lower, upper = stat["wilson_95"]
        ax.errorbar(
            x,
            stat["rate"],
            yerr=[[stat["rate"] - lower], [upper - stat["rate"]]],
            fmt="o",
            color=color,
            capsize=4,
            markersize=8,
        )
        ax.text(x, upper + 0.04, f"{stat['numerator']}/{stat['denominator']}", ha="center")
    ax.set_xticks(xs, names)
    ax.set_xlim(-1.4, 2.4)
    ax.set_ylim(-0.05, 1.15)
    ax.set_ylabel("Accuracy")
    ax.set_title("Answer accuracy with Wilson 95% intervals")
    if bootstrap:
        difference = bootstrap["accuracy_difference_am_minus_en"]
        note = (
            "Pair bootstrap 95% for Amharic minus English: "
            f"[{difference[0]:.2f}, {difference[1]:.2f}]"
        )
        if abs(difference[0] - difference[1]) < 1e-9:
            note += "\nThe interval collapses because every pair has the same outcome."
        ax.set_xlabel(note)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def format_rate(stat):
    if stat["rate"] is None:
        return f"{stat['numerator']}/{stat['denominator']} (undefined)"
    interval = stat["wilson_95"]
    return f"{stat['numerator']}/{stat['denominator']} = {stat['rate']:.1%} (Wilson 95% {interval[0]:.1%}–{interval[1]:.1%})"


def first_line(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    return text.strip().splitlines()[0]


def problem_reasons(row):
    reasons = []
    if row["selection_status"] == "missing":
        reasons.append("missing letter")
    elif row["selection_status"] == "ambiguous":
        reasons.append("ambiguous letter")
    if row["termination"] == "token_limit_truncation":
        reasons.append("cut off at the token limit")
    elif row["termination"] == "execution_failure":
        reasons.append("execution failure")
    return reasons


def write_problem_log(rows, path, source):
    flagged = []
    for row in rows:
        reasons = problem_reasons(row)
        if reasons:
            flagged.append((row, reasons))
    lines = [
        f"source: {source}",
        f"flagged responses: {len(flagged)}",
        "A response is listed when the selected letter is missing or ambiguous, or when generation stopped before a natural end.",
        "A first line that starts with 'B.' is read as B. A line that is only 'A. D' stays ambiguous.",
        "",
    ]
    if not flagged:
        lines.append("No missing or ambiguous responses.")
    for index, (row, reasons) in enumerate(flagged, start=1):
        lines.append("=" * 72)
        lines.append(f"{index:02d}  {row['item_id']}  {row['language']}")
        lines.append(f"reasons: {', '.join(reasons)}")
        lines.append(f"correct option: {row['correct_option']}")
        lines.append(f"selected option: {row['selected_option']}")
        lines.append(f"termination: {row['termination']}  tokens: {row['generated_tokens']}")
        lines.append(f"first line: {first_line(row['answer'])}")
        lines.append("answer:")
        lines.append((row["answer"] or "").rstrip())
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(flagged)


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def explanation_report(rows, paired, judgments_path=None, human_paths=(), sheet=None):
    """Explanation metrics kept separate by label provenance."""
    machine, machine_info = ({}, None)
    if judgments_path:
        machine, machine_info = load_judgments(judgments_path, rows)
    sources = {}
    if sheet is not None:
        sources[sheet[0]] = human_labels(sheet[1], sheet[0])
    for path in human_paths:
        groups = {}
        for row in load_jsonl(path):
            groups.setdefault(row.get("sample") or Path(path).stem, []).append(row)
        for name, group in groups.items():
            name = name if name not in sources else f"{name}:{Path(path).stem}"
            sources[name] = human_labels(group, name)
    human, conflicts = merge_human(sources)
    if not machine and not human:
        return None
    reviewed = machine | human
    report = {
        "label_sources": {
            "machine": "LLM-assessed explanation validity from judge_explanations.py. Not ground truth.",
            "human": "Researcher labels from the files listed under human_sources.",
            "human_reviewed": "Human label where one exists, otherwise the machine label.",
        },
        "judgments": machine_info,
        "human_sources": {name: len(labels) for name, labels in sources.items()},
        "human_conflicts_dropped": conflicts,
        "human_reviewed_overrides": sum(key in machine for key in human),
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "unit": "question pair"},
        "truncation": truncation_summary(rows),
        "views": {},
        "agreement": {},
    }
    for name, labels in (("machine", machine), ("human", human), ("human_reviewed", reviewed if machine else {})):
        if labels:
            report["views"][name] = explanation_views(rows, paired, labels)
    if machine:
        for name, labels in sources.items():
            if labels:
                independent, note = agreement_status(name)
                report["agreement"][name] = {
                    "independent_validation": independent,
                    "note": note,
                    "by_language": agreement(machine, labels, rows),
                }
    return report


def agreement_status(name):
    if name.startswith("calibration"):
        return False, "Calibration examples shaped the rubric; their agreement is not an independent result."
    if name.startswith("review_flagged"):
        return False, "Selected because the judge flagged them; not a random sample."
    if name.startswith("review_random"):
        return False, "Random unflagged judgments, reviewed with the judge's labels visible."
    if name.startswith("agreement"):
        return True, "Held-out sample scored blind after the rubric was frozen."
    return None, "Selection and blinding not recorded for this source."


def analyze(response_path, output_dir=None, judgments_path=None, human_paths=()):
    source = load_jsonl(response_path)
    rows, prep_report = prepare(source)
    paired, unpaired = pair_items(rows)
    table = contingency(paired)
    summary = answer_summary(table)
    bootstrap = bootstrap_answer_rates(table["included"])

    scores_path, score_rows, scores_created = ensure_scores(response_path, rows)
    explanation = explanation_report(
        rows, paired, judgments_path, human_paths, sheet=("hand_scoring_sheet", score_rows)
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if output_dir is None:
        output_dir = ROOT / "analysis" / f"{Path(response_path).stem}_{stamp}"
    else:
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    plot_answer_outcomes(rows, figure_dir / "answer_selection.png")
    plot_contingency(summary, figure_dir / "paired_outcomes.png")
    plot_accuracy_intervals(summary, bootstrap, figure_dir / "accuracy_intervals.png")
    plot_tokens(paired, figure_dir / "generated_tokens.png")
    primary_source = None
    explanation_path = None
    if explanation:
        primary_source = "machine" if "machine" in explanation["views"] else "human"
        primary = explanation["views"][primary_source]["completed"]["primary_paired"]
        if primary["scored_pairs"]:
            plot_explanation_differences(primary, primary_source, figure_dir / "explanation_comparison.png")
        explanation_path = output_dir / "explanation_summary.json"
        write_json(explanation_path, {"source": str(response_path), "source_sha256": file_sha256(response_path)} | explanation)

    payload = {
        "source": str(response_path),
        "source_sha256": file_sha256(response_path),
        "responses": len(rows),
        "records_read": len(source),
        "validation": prep_report,
        "unpaired_items": unpaired,
        "run_provenance": run_provenance(source),
        "parser_audit": parser_audit(rows),
        "answer_selection": summary,
        "pair_bootstrap_95": bootstrap,
        "termination": dict(Counter(row["termination"] for row in rows)),
        "automatic_script_share": {
            "note": "Share of alphabetic characters in the expected script. This is not the human language-adherence rubric.",
            "en": [row["expected_script_share"] for row in rows if row["language"] == "en"],
            "am": [row["expected_script_share"] for row in rows if row["language"] == "am"],
        },
        "scores_file": str(scores_path),
        "explanation_summary": str(explanation_path) if explanation_path else None,
        "scoring_note": (
            f"Explanation metrics use {', '.join(explanation['views'])} labels; see explanation_summary."
            if explanation
            else f"Pass --judgments, or fill explanation_validity, consistency, and language_adherence in {scores_path.name}, then rerun this command."
        ),
    }
    summary_path = output_dir / f"summary_{stamp}.json"
    examples_path = output_dir / f"examples_{stamp}.json"
    problem_log_path = output_dir / f"missing_or_ambiguous_{stamp}.log"
    flagged_count = write_problem_log(rows, problem_log_path, response_path)
    payload["missing_or_ambiguous_log"] = str(problem_log_path)
    payload["flagged_responses"] = flagged_count
    write_json(summary_path, payload)
    write_json(examples_path, examples_by_cell(paired))
    return {
        "output_dir": output_dir,
        "summary_path": summary_path,
        "examples_path": examples_path,
        "problem_log_path": problem_log_path,
        "flagged_count": flagged_count,
        "scores_path": scores_path,
        "scores_created": scores_created,
        "summary": summary,
        "bootstrap": bootstrap,
        "explanation": explanation,
        "explanation_path": explanation_path,
        "primary_source": primary_source,
        "validation": prep_report,
        "parser_changes": len(payload["parser_audit"]["changed"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("responses", type=Path, help="Response JSONL from run_eval.py")
    parser.add_argument("--output-dir", type=Path, help="Directory for this analysis run. Default: a new timestamped folder.")
    parser.add_argument("--judgments", type=Path, help="judgments.jsonl from judge_explanations.py (machine labels)")
    parser.add_argument(
        "--human", type=Path, action="append", default=[],
        help="Human scoring file (calibration, agreement, or review). Repeatable.",
    )
    args = parser.parse_args()

    result = analyze(args.responses, args.output_dir, args.judgments, args.human)
    validation = result["validation"]
    if any(validation.values()):
        print(
            f"Validation: {len(validation['invalid_records'])} invalid, "
            f"{len(validation['conflicting_duplicates'])} conflicting duplicates excluded, "
            f"{len(validation['identical_duplicates'])} identical duplicates merged"
        )
    print(f"Answer outcomes changed by the shared parser: {result['parser_changes']} (see parser_audit)")
    summary = result["summary"]
    cells = summary["cells"]
    print(f"Wrote {result['output_dir']}")
    print(
        "Paired cells a/b/c/d "
        f"(en✓am✓ / en✓am✗ / en✗am✓ / en✗am✗): "
        f"{cells['english_correct_amharic_correct']}/"
        f"{cells['english_correct_amharic_incorrect']}/"
        f"{cells['english_incorrect_amharic_correct']}/"
        f"{cells['english_incorrect_amharic_incorrect']}"
    )
    print(f"English accuracy: {format_rate(summary['english_accuracy'])}")
    print(f"Amharic accuracy: {format_rate(summary['amharic_accuracy'])}")
    print(
        "Amharic correct given English correct: "
        f"{format_rate(summary['amharic_correct_given_english_correct'])}"
    )
    print(
        "English correct given Amharic correct: "
        f"{format_rate(summary['english_correct_given_amharic_correct'])}"
    )
    print(f"Pairs excluded for a missing or ambiguous letter: {summary['excluded_missing_or_ambiguous']}")
    print(f"Missing or ambiguous log ({result['flagged_count']} responses): {result['problem_log_path']}")
    print(f"Scoring sheet: {result['scores_path']}")
    if result["explanation"]:
        source = result["primary_source"]
        primary = result["explanation"]["views"][source]["completed"]["primary_paired"]
        print(
            f"{source.capitalize()} labels, completed both-correct pairs "
            f"(scored {primary['scored_pairs']} of {primary['eligible_pairs']}): "
            f"English fully valid {format_rate(primary['english_fully_valid'])}, "
            f"Amharic fully valid {format_rate(primary['amharic_fully_valid'])}"
        )
        print(
            f"English higher {primary['favor_english']}, "
            f"Amharic higher {primary['favor_amharic']}, tie {primary['tie']}"
        )
        print(f"Explanation summary: {result['explanation_path']}")


if __name__ == "__main__":
    main()
