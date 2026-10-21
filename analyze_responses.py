"""Analyze paired English and Amharic responses from run_eval.py.

Answer selection is scored automatically. Explanation validity,
answer–explanation consistency, and language adherence follow the rubric in
research_plan.md. The scoring sheet is results/<response filename>. It is
created on the first run and reused after the rubric fields are filled in.

Examples:
  python3 analyze_responses.py data/responses.jsonl
  python3 analyze_responses.py data/responses_test_n30_seed0_20261021T120000Z.jsonl
"""

import argparse
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
SELECTION_LINE = re.compile(r"^(?:\*\*|__)?[\[(]?([ABCD])[\])]?(?:\*\*|__)?[.:]?$")
LEADING_OPTION = re.compile(r"^(?:\*\*|__)?([ABCD])\.\s*(\S.*?)\s*$")
BOLD_LETTER = re.compile(r"\*\*\s*([ABCD])\s*\*\*")
FIRST_LINE_LETTER = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])")


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


def clean_bold_letters(answer):
    """Turn markdown **A** into A. The rest of the response stays as written."""
    if not isinstance(answer, str):
        return answer
    return BOLD_LETTER.sub(r"\1", answer)


def clean_response_file(path):
    rows = load_jsonl(path)
    changed = 0
    for row in rows:
        if row.get("language") != "am":
            continue
        cleaned = clean_bold_letters(row.get("answer"))
        if cleaned != row.get("answer"):
            row["answer"] = cleaned
            changed += 1
    if changed:
        write_jsonl(path, rows)
    return changed


def parse_selection(answer, language=None):
    """Read a letter from the first line only. Do not search the explanation."""
    if not isinstance(answer, str) or not answer.strip():
        return None, "missing"
    first = answer.strip().splitlines()[0].strip()
    if SELECTION_LINE.match(first):
        letter = re.sub(r"[^ABCD]", "", first)
        return letter, "parsed"
    leading = LEADING_OPTION.match(first)
    if leading and not re.fullmatch(r"[ABCD]", leading.group(2)):
        return leading.group(1), "parsed"
    if language == "am":
        letters = list(dict.fromkeys(FIRST_LINE_LETTER.findall(first)))
        if len(letters) == 1:
            return letters[0], "parsed"
    return None, "ambiguous"


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
    prepared = []
    seen = {}
    duplicates = []
    for row in rows:
        key = (row["item_id"], row["language"])
        if key in seen:
            duplicates.append(key)
        answer = clean_bold_letters(row.get("answer")) if row.get("language") == "am" else row.get("answer")
        letter, status = parse_selection(answer, row.get("language"))
        correct_option = str(row["correct_option"]).strip().upper()
        if status == "parsed":
            outcome = "correct" if letter == correct_option else "incorrect"
        else:
            outcome = "missing_or_ambiguous"
        prepared_row = {
            "item_id": row["item_id"],
            "split": row.get("split"),
            "language": row["language"],
            "question": row.get("question"),
            "options": row.get("options"),
            "correct_option": correct_option,
            "selected_option": letter,
            "selection_status": status,
            "answer_outcome": outcome,
            "answer": answer,
            "generated_tokens": row.get("generated_tokens"),
            "termination": termination(row),
            "expected_script_share": script_share(answer, row["language"]),
        }
        seen[key] = prepared_row
    return list(seen.values()), duplicates


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


def attach_scores(rows, score_rows):
    by_key = {(row["item_id"], row["language"]): row for row in score_rows}
    missing = []
    for row in rows:
        score = by_key.get((row["item_id"], row["language"]))
        if score is None or not score_filled(score):
            missing.append({"item_id": row["item_id"], "language": row["language"]})
            continue
        validity = score.get("explanation_validity")
        consistency = score.get("consistency")
        adherence = score.get("language_adherence")
        label = f"{row['item_id']} {row['language']}"
        if validity not in (0, 1, 2):
            raise ValueError(f"explanation_validity must be 0, 1, or 2 for {label}")
        if consistency not in CONSISTENCY:
            raise ValueError(f"consistency must be one of {CONSISTENCY} for {label}")
        if adherence not in ADHERENCE:
            raise ValueError(f"language_adherence must be one of {ADHERENCE} for {label}")
        row["explanation_validity"] = validity
        row["consistency"] = consistency
        row["language_adherence"] = adherence
    return missing


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


def explanation_summary(paired):
    both_correct = [
        pair
        for pair in paired
        if pair["en"]["answer_outcome"] == "correct" and pair["am"]["answer_outcome"] == "correct"
        and "explanation_validity" in pair["en"]
        and "explanation_validity" in pair["am"]
    ]
    favor_en = favor_am = tie = 0
    differences = []
    for pair in both_correct:
        difference = pair["en"]["explanation_validity"] - pair["am"]["explanation_validity"]
        differences.append(difference)
        if difference > 0:
            favor_en += 1
        elif difference < 0:
            favor_am += 1
        else:
            tie += 1
    by_language = {}
    for language in ("en", "am"):
        responses = [pair[language] for pair in paired if "explanation_validity" in pair[language]]
        correct = [row for row in responses if row["answer_outcome"] == "correct"]
        fully_valid_given_correct = sum(row["explanation_validity"] == 2 for row in correct)
        fully_valid_and_correct = sum(
            row["answer_outcome"] == "correct" and row["explanation_validity"] == 2 for row in responses
        )
        by_language[language] = {
            "fully_valid_among_correct": rate(fully_valid_given_correct, len(correct)),
            "correct_and_fully_valid_among_responses": rate(fully_valid_and_correct, len(responses)),
            "validity_counts": dict(Counter(row["explanation_validity"] for row in responses)),
            "contradictions": sum(row.get("consistency") == "contradicts" for row in responses),
            "language_adherence_failures": sum(
                row.get("language_adherence") == "substantially_abandoned" for row in responses
            ),
            "language_adherence_counts": dict(Counter(row.get("language_adherence") for row in responses)),
        }
    return {
        "both_correct_pairs": len(both_correct),
        "favor_english": favor_en,
        "favor_amharic": favor_am,
        "tie": tie,
        "score_difference_en_minus_am": dict(Counter(differences)),
        "by_language": by_language,
    }


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
    for field in ("explanation_validity", "consistency", "language_adherence"):
        if field in row:
            kept[field] = row[field]
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
    for offset, (label, key, color) in enumerate(zip(labels, keys, colors)):
        heights = [outcome_counts(rows, language)[key] for _, language in languages]
        positions = [position + (offset - 1) * width for position in x]
        bars = ax.bar(positions, heights, width, label=label, color=color)
        for bar, height in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width() / 2, height, str(height), ha="center", va="bottom")
    ax.set_xticks(x, [name for name, _ in languages])
    ax.set_ylabel("Responses")
    ax.set_title("Answer selection by language")
    ax.legend(frameon=False)
    ax.set_ylim(0, max(1, max(outcome_counts(rows, "en").values()) , max(outcome_counts(rows, "am").values())) + 1)
    fig.tight_layout()
    fig.savefig(path)
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


def plot_explanation_differences(explanation, path):
    labels = ["English higher", "Tie", "Amharic higher"]
    values = [explanation["favor_english"], explanation["tie"], explanation["favor_amharic"]]
    colors = [EN_COLOR, MISSING_COLOR, AM_COLOR]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(labels, values, color=colors)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom")
    ax.set_ylabel("Question pairs")
    ax.set_title(f"Explanation scores when both answers are correct (n = {explanation['both_correct_pairs']})")
    ax.set_ylim(0, max(values + [1]) + 1)
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


def analyze(response_path, output_dir=None):
    clean_response_file(response_path)
    rows, duplicates = prepare(load_jsonl(response_path))
    paired, unpaired = pair_items(rows)
    table = contingency(paired)
    summary = answer_summary(table)
    bootstrap = bootstrap_answer_rates(table["included"])
    source = load_jsonl(response_path)
    meta = {}
    if source:
        first = source[0]
        meta = {
            "model": first.get("model"),
            "revision": first.get("revision"),
            "precision": first.get("precision"),
            "gpu": first.get("gpu"),
            "sample_seed": first.get("sample_seed"),
            "sample_limit": first.get("sample_limit"),
            "generation_settings": first.get("generation_settings"),
        }

    scores_path, score_rows, scores_created = ensure_scores(response_path, rows)
    scores_missing = attach_scores(rows, score_rows)
    explanation = None if scores_created or len(scores_missing) == len(rows) else explanation_summary(paired)

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
    if explanation and explanation["both_correct_pairs"]:
        plot_explanation_differences(explanation, figure_dir / "explanation_comparison.png")

    payload = {
        "source": str(response_path),
        "responses": len(rows),
        "duplicate_item_language_pairs_kept_last": [list(key) for key in duplicates],
        "unpaired_items": unpaired,
        "run": meta,
        "answer_selection": summary,
        "pair_bootstrap_95": bootstrap,
        "termination": dict(Counter(row["termination"] for row in rows)),
        "automatic_script_share": {
            "note": "Share of alphabetic characters in the expected script. This is not the human language-adherence rubric.",
            "en": [row["expected_script_share"] for row in rows if row["language"] == "en"],
            "am": [row["expected_script_share"] for row in rows if row["language"] == "am"],
        },
        "scores_file": str(scores_path),
        "explanation": explanation,
        "scores_not_joined": scores_missing,
        "scoring_note": (
            "Explanation validity, consistency, and language adherence are human rubric scores."
            if explanation is not None
            else f"Fill explanation_validity, consistency, and language_adherence in {scores_path.name}, then rerun this command."
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
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("responses", type=Path, help="Response JSONL from run_eval.py")
    parser.add_argument("--output-dir", type=Path, help="Directory for this analysis run. Default: a new timestamped folder.")
    args = parser.parse_args()

    result = analyze(args.responses, args.output_dir)
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
        explanation = result["explanation"]
        print(
            "Both-correct explanation comparison: "
            f"English higher {explanation['favor_english']}, "
            f"Amharic higher {explanation['favor_amharic']}, "
            f"tie {explanation['tie']}"
        )


if __name__ == "__main__":
    main()
