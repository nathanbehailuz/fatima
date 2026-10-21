"""Synthetic fixtures and mocked API responses for the explanation-judging pipeline.

Run: python3 -m unittest discover -s tests
"""

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import httpx
import openai

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze_responses as ar
import judge_explanations as je
from judge_schema import ExplanationJudgment, validate_judgment

OPTIONS = {
    "en": [{"label": l, "text": t} for l, t in zip("ABCD", ["ice", "water", "steam", "sand"])],
    "am": [{"label": l, "text": t} for l, t in zip("ABCD", ["በረዶ", "ውሃ", "እንፋሎት", "አሸዋ"])],
}
GOOD = {
    "explanation_validity": 2,
    "consistency": "supports",
    "language_adherence": "maintained",
    "error_types": [],
    "brief_justification": "States the essential reason.",
    "needs_human_review": False,
    "review_reason": None,
}


def record(item_id, language, answer, truncated=False, correct="B"):
    return {
        "item_id": item_id,
        "split": "test",
        "language": language,
        "question": f"question {item_id}",
        "options": OPTIONS[language],
        "correct_option": correct,
        "answer": answer,
        "generated_tokens": 1024 if truncated else 50,
        "reached_token_limit": truncated,
        "sample_seed": 0,
        "sample_limit": 0,
    }


def fixture_rows():
    return [
        record("Q1", "en", "B\nWater is liquid."),
        record("Q1", "am", "B\nውሃ ፈሳሽ ነው።"),
        record("Q2", "en", "B\nBecause."),
        record("Q2", "am", "ትክክለኛው መልስ B ነው።\nምክንያቱም..."),
        record("Q3", "en", "B\nReason."),
        record("Q3", "am", "A\nተሳስቷል።"),
        record("Q4", "en", "A galaxy is big."),
        record("Q4", "am", "**መልስ፡ B**\nማብራሪያ"),
        record("Q5", "en", "B\nReason that runs out", truncated=True),
        record("Q5", "am", "B\nማብራሪያ"),
    ]


def config(model="judge-a", prompt="p1"):
    return {
        "rubric_version": "v1",
        "prompt_sha256": prompt,
        "schema_sha256": "s1",
        "requested_model": model,
        "settings": {"max_output_tokens": 100, "reasoning_effort": None, "store": False},
    }


def body(status="completed", text=None, refusal=None, reason=None):
    content = []
    if text is not None:
        content.append({"type": "output_text", "text": text})
    if refusal is not None:
        content.append({"type": "refusal", "refusal": refusal})
    return {
        "id": "resp_1",
        "model": "judge-a-2026",
        "status": status,
        "incomplete_details": {"reason": reason} if reason else None,
        "output": [{"type": "message", "content": content}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


class FakeParsed:
    def __init__(self, text):
        self.output_parsed = ExplanationJudgment.model_validate_json(text)


class FakeRaw:
    def __init__(self, payload):
        self.payload = payload
        self.http_response = self

    def json(self):
        return self.payload

    def parse(self):
        text, _ = je.output_text_and_refusals(self.payload)
        return FakeParsed(text)


class FakeClient:
    """Returns queued bodies or raises queued exceptions, one per call."""

    def __init__(self, queue):
        self.queue = list(queue)
        self.calls = 0
        self.responses = self
        self.with_raw_response = self

    def parse(self, **kwargs):
        self.calls += 1
        item = self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]
        if isinstance(item, Exception):
            raise item
        return FakeRaw(item)


def status_error(cls, code, error_code=None):
    response = httpx.Response(code, request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
    return cls("error", response=response, body={"code": error_code} if error_code else None)


class ParsingTests(unittest.TestCase):
    def test_same_rules_for_both_languages(self):
        cases = [
            ("B\nwhy", "B\nለምን", "B"),
            ("**B**\nwhy", "**B**\nለምን", "B"),
            ("Answer: C\nwhy", "መልስ: C\nለምን", "C"),
            ("B. water\nwhy", "B. ውሃ\nለምን", "B"),
        ]
        for en, am, letter in cases:
            self.assertEqual(ar.parse_selection(en), (letter, "parsed"))
            self.assertEqual(ar.parse_selection(am), (letter, "parsed"))

    def test_ambiguous_and_missing(self):
        self.assertEqual(ar.parse_selection("A galaxy is big.\nmore"), (None, "ambiguous"))
        self.assertEqual(ar.parse_selection("A. D"), (None, "ambiguous"))
        self.assertEqual(ar.parse_selection("B or C"), (None, "ambiguous"))
        self.assertEqual(ar.parse_selection("ትክክለኛው መልስ B ነው።"), ("B", "parsed"))
        self.assertEqual(ar.parse_selection(""), (None, "missing"))
        self.assertEqual(ar.parse_selection(None), (None, "missing"))

    def test_correct_only_eligibility(self):
        rows, _ = ar.prepare(fixture_rows())
        jobs = je.plan_jobs(rows)
        keys = {(row["item_id"], row["language"]) for row in jobs}
        self.assertEqual(
            keys,
            {("Q1", "en"), ("Q1", "am"), ("Q2", "en"), ("Q2", "am"), ("Q3", "en"), ("Q4", "am"), ("Q5", "en"), ("Q5", "am")},
        )

    def test_conflicting_duplicates_reported_and_excluded(self):
        source = fixture_rows() + [record("Q1", "en", "C\nOther."), record("Q2", "en", "B\nBecause.")]
        rows, report = ar.prepare(source)
        self.assertEqual(report["conflicting_duplicates"], [["Q1", "en"]])
        self.assertEqual(report["identical_duplicates"], [["Q2", "en"]])
        self.assertNotIn(("Q1", "en"), {(row["item_id"], row["language"]) for row in rows})

    def test_invalid_records_reported(self):
        bad = record("Q9", "en", "B")
        bad["options"] = bad["options"][:3]
        worse = record("Q8", "en", "B", correct="E")
        _, report = ar.prepare([bad, worse])
        self.assertEqual(len(report["invalid_records"]), 2)


class PreservationTests(unittest.TestCase):
    def test_prepare_and_analyze_do_not_touch_sources_or_scores(self):
        source = fixture_rows()
        before = copy.deepcopy(source)
        ar.prepare(source)
        self.assertEqual(source, before)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            path = tmp / "responses.jsonl"
            ar.write_jsonl(path, source)
            original = path.read_bytes()
            sheet = tmp / "results" / path.name
            sheet.parent.mkdir()
            ar.write_jsonl(sheet, [{
                "item_id": "Q1", "language": "en", "selected_option": "B", "correct_option": "B",
                "answer_outcome": "correct", "explanation_validity": 1, "consistency": "supports",
                "language_adherence": "maintained", "answer": "B\nWater is liquid.",
            }])
            saved_results = ar.RESULTS_DIR
            ar.RESULTS_DIR = tmp / "results"
            try:
                result = ar.analyze(path, tmp / "analysis")
            finally:
                ar.RESULTS_DIR = saved_results
            self.assertEqual(path.read_bytes(), original)
            kept = [row for row in ar.load_jsonl(sheet) if row["item_id"] == "Q1" and row["language"] == "en"]
            self.assertEqual(kept[0]["explanation_validity"], 1)
            self.assertEqual(result["explanation"]["human_sources"]["hand_scoring_sheet"], 1)

    def test_review_export_keeps_human_labels(self):
        rows, _ = ar.prepare(fixture_rows())
        cfg = config()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            flagged = GOOD | {"explanation_validity": None, "needs_human_review": True, "review_reason": "unclear"}
            with (run_dir / "judgments.jsonl").open("w") as handle:
                for row, judgment in ((rows[0], flagged), (rows[1], GOOD), (rows[2], GOOD)):
                    handle.write(json.dumps(je.build_record(
                        row, cfg, "x", {"path": "p", "sha256": "h"}, {"status": "ok", "judgment": judgment}
                    )) + "\n")
            path, count = je.write_review(run_dir, rows, 1, 0)
            reviewed = ar.load_jsonl(path)
            self.assertEqual(count, 2)
            self.assertEqual({row["sample"] for row in reviewed}, {"review_flagged", "review_random"})
            reviewed[0]["explanation_validity"] = 0
            reviewed[0]["consistency"] = "supports"
            reviewed[0]["language_adherence"] = "maintained"
            ar.write_jsonl(path, reviewed)
            je.write_review(run_dir, rows, 1, 0)
            self.assertEqual(ar.load_jsonl(path)[0]["explanation_validity"], 0)
            self.assertEqual(len(ar.load_jsonl(path)), 2)


class SchemaTests(unittest.TestCase):
    def test_valid_scores_accepted(self):
        for score in (0, 1, 2):
            judgment, errors = validate_judgment(GOOD | {"explanation_validity": score})
            self.assertIsNotNone(judgment)
            self.assertEqual(errors, [])
        judgment, errors = validate_judgment(
            GOOD | {"explanation_validity": None, "needs_human_review": True, "review_reason": "cannot read"}
        )
        self.assertEqual(errors, [])

    def test_invalid_fields_rejected(self):
        for bad in (
            GOOD | {"extra": 1},
            GOOD | {"explanation_validity": 3},
            GOOD | {"explanation_validity": True},
            GOOD | {"explanation_validity": "2"},
            GOOD | {"consistency": "agrees"},
            GOOD | {"error_types": ["typo"]},
            {key: value for key, value in GOOD.items() if key != "review_reason"},
        ):
            judgment, _ = validate_judgment(bad)
            self.assertIsNone(judgment, bad)

    def test_cross_field_rules(self):
        for bad in (
            GOOD | {"explanation_validity": None},
            GOOD | {"explanation_validity": None, "needs_human_review": True, "review_reason": " "},
            GOOD | {"needs_human_review": True},
            GOOD | {"error_types": ["other", "other"]},
        ):
            judgment, errors = validate_judgment(bad)
            self.assertIsNotNone(judgment)
            self.assertTrue(errors, bad)


class ApiTests(unittest.TestCase):
    def call(self, queue):
        client = FakeClient(queue)
        result = je.call_judge(client, config(), "instructions", "input", max_attempts=3, sleep=lambda _: None)
        return result, client

    def test_ok(self):
        result, _ = self.call([body(text=json.dumps(GOOD))])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["judgment"]["explanation_validity"], 2)
        self.assertEqual(result["returned_model"], "judge-a-2026")

    def test_failures_stay_unscored(self):
        cases = {
            "refused": body(refusal="I can't help with that."),
            "incomplete": body(status="incomplete", text='{"explanation_validity": 2', reason="max_output_tokens"),
            "validation_error": body(text='{"explanation_validity": 2}'),
            "api_error": body(status="failed"),
        }
        for expected, payload in cases.items():
            result, _ = self.call([payload])
            self.assertEqual(result["status"], expected)
            self.assertIsNone(result["judgment"])
        result, _ = self.call([body(text=json.dumps(GOOD | {"needs_human_review": True}))])
        self.assertEqual(result["status"], "validation_error")
        self.assertIsNone(result["judgment"])

    def test_transient_errors_retry_then_stop(self):
        result, client = self.call([status_error(openai.RateLimitError, 429), body(text=json.dumps(GOOD))])
        self.assertEqual((result["status"], result["attempts"], client.calls), ("ok", 2, 2))
        result, client = self.call([status_error(openai.InternalServerError, 500)])
        self.assertEqual((result["status"], result["attempts"], result["permanent"]), ("api_error", 3, False))
        self.assertIsNone(result.get("judgment"))

    def test_permanent_errors_not_retried(self):
        for error in (
            status_error(openai.AuthenticationError, 401),
            status_error(openai.BadRequestError, 400),
            status_error(openai.RateLimitError, 429, "insufficient_quota"),
        ):
            result, client = self.call([error])
            self.assertEqual((result["status"], client.calls, result["permanent"]), ("api_error", 1, True))

    def test_permanent_error_stops_run(self):
        rows, _ = ar.prepare(fixture_rows())
        jobs = je.plan_jobs(rows)
        client = FakeClient([status_error(openai.AuthenticationError, 401)])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "judgments.jsonl"
            counts = je.run_jobs(client, jobs, config(), "i", {"path": "p", "sha256": "h"}, out, 1, 3, log=lambda _: None)
            self.assertEqual(counts["api_error"], 1)
            self.assertEqual(counts["skipped_after_stop"], len(jobs) - 1)


class ResumeTests(unittest.TestCase):
    def test_resume_skips_matching_successes_only(self):
        rows, _ = ar.prepare(fixture_rows())
        jobs = je.plan_jobs(rows)
        cfg = config()
        client = FakeClient([body(text=json.dumps(GOOD))])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "judgments.jsonl"
            je.run_jobs(client, jobs[:3], cfg, "i", {"path": "p", "sha256": "h"}, out, 2, 1, log=lambda _: None)
            failing = FakeClient([status_error(openai.InternalServerError, 500)])
            je.run_jobs(failing, jobs[3:4], cfg, "i", {"path": "p", "sha256": "h"}, out, 1, 1, log=lambda _: None)
            done = je.successful_keys(out)
            pending = [row for row in jobs if je.resume_key(row, cfg) not in done]
            self.assertEqual(len(pending), len(jobs) - 3)
            self.assertIn(jobs[3], pending)
            for changed in (config(model="judge-b"), config(prompt="p2")):
                self.assertTrue(all(je.resume_key(row, changed) not in done for row in jobs))
            edited = copy.deepcopy(fixture_rows())
            for row in edited:
                if (row["item_id"], row["language"]) == (jobs[0]["item_id"], jobs[0]["language"]):
                    row["answer"] += " Edited."
            new_rows, _ = ar.prepare(edited)
            new_jobs = {(r["item_id"], r["language"]): r for r in je.plan_jobs(new_rows)}
            self.assertNotIn(je.resume_key(new_jobs[(jobs[0]["item_id"], jobs[0]["language"])], cfg), done)
            check = je.check_artifacts(out, cfg)
            self.assertEqual(check["problems"], [])
            self.assertEqual(check["status_counts"], {"ok": 3, "api_error": 1})


def labels_for(scores):
    return {
        key: GOOD | {"explanation_validity": score, "needs_human_review": score is None,
                     "review_reason": "unsure" if score is None else None}
        for key, score in scores.items()
    }


class DenominatorTests(unittest.TestCase):
    def setUp(self):
        self.rows, _ = ar.prepare(fixture_rows())
        self.paired, _ = ar.pair_items(self.rows)
        self.labels = labels_for({
            ("Q1", "en"): 2, ("Q1", "am"): 1,
            ("Q2", "en"): 2, ("Q2", "am"): 2,
            ("Q3", "en"): 0,
            ("Q4", "am"): None,
            ("Q5", "en"): 2, ("Q5", "am"): 0,
        })

    def test_primary_paired_counts(self):
        completed = ar.paired_explanations(self.paired, self.labels, True)
        self.assertEqual((completed["eligible_pairs"], completed["scored_pairs"]), (2, 2))
        self.assertEqual(completed["fully_valid_cells"], {"both": 1, "english_only": 1, "amharic_only": 0, "neither": 0})
        self.assertEqual(completed["english_fully_valid"]["rate"], 1.0)
        self.assertEqual(completed["amharic_fully_valid"]["rate"], 0.5)
        self.assertEqual(completed["fully_valid_difference_am_minus_en"], -0.5)
        self.assertEqual((completed["favor_english"], completed["favor_amharic"], completed["tie"]), (1, 0, 1))
        self.assertEqual(completed["ordinal_table"]["en_2"]["am_1"], 1)
        delivered = ar.paired_explanations(self.paired, self.labels, False)
        self.assertEqual((delivered["eligible_pairs"], delivered["scored_pairs"]), (3, 3))

    def test_missing_judgment_leaves_pair_unscored(self):
        labels = dict(self.labels)
        del labels[("Q2", "am")]
        primary = ar.paired_explanations(self.paired, labels, True)
        self.assertEqual((primary["eligible_pairs"], primary["scored_pairs"], primary["unscored_pairs"]), (2, 1, 1))

    def test_language_denominators(self):
        am = ar.language_explanations(self.rows, self.labels, "am", False)
        self.assertEqual(am["correct"], 4)
        self.assertEqual(am["correct_scored"], 3)
        self.assertEqual(am["correct_unscored_by_labeler"], 1)
        self.assertEqual(am["correct_flagged_for_review"], 1)
        joint = am["correct_and_fully_valid_among_evaluable"]
        self.assertEqual((joint["numerator"], joint["denominator"]), (1, 5))
        self.assertTrue(joint["provisional"])
        self.assertEqual(joint["bounds"], [0.2, 0.4])
        en = ar.language_explanations(self.rows, self.labels, "en", False)
        self.assertEqual(en["missing_or_ambiguous_excluded"], 1)
        self.assertEqual(en["correct_and_fully_valid_among_evaluable"]["denominator"], 4)
        self.assertEqual(en["correct_truncated"], 1)
        self.assertEqual(en["fully_valid_among_correct_scored"]["numerator"], 3)
        en_completed = ar.language_explanations(self.rows, self.labels, "en", True)
        self.assertEqual(en_completed["correct"], 3)
        self.assertEqual(en_completed["correct_truncated"], 0)
        self.assertEqual(en_completed["correct_and_fully_valid_among_evaluable"]["denominator"], 3)

    def test_truncation_rates(self):
        summary = ar.truncation_summary(self.rows)
        self.assertEqual(summary["en"]["all_responses"]["numerator"], 1)
        self.assertEqual(summary["en"]["correct_responses"]["denominator"], 4)
        self.assertEqual(summary["am"]["all_responses"]["numerator"], 0)

    def test_agreement_by_language(self):
        human = {key: GOOD | {"explanation_validity": 2} for key in self.labels}
        report = ar.agreement(self.labels, human, self.rows)
        self.assertEqual(report["en"]["exact_validity"]["numerator"], 3)
        self.assertEqual(report["en"]["exact_validity"]["denominator"], 4)
        self.assertEqual(report["am"]["machine_unscored"], 1)
        self.assertEqual(report["am"]["confusion"]["human_2"]["machine_None"], 1)


class NamingTests(unittest.TestCase):
    def test_limit_counts_questions_and_keeps_both_languages(self):
        rows, _ = ar.prepare(fixture_rows())
        jobs = je.plan_jobs(rows, seed=0)
        chosen, n_questions = je.select_question_jobs(jobs, 2)
        self.assertEqual(n_questions, 2)
        eligible = {}
        for row in jobs:
            eligible.setdefault(row["item_id"], set()).add(row["language"])
        kept = {}
        for row in chosen:
            kept.setdefault(row["item_id"], set()).add(row["language"])
        self.assertEqual(len(kept), 2)
        for item_id, languages in kept.items():
            self.assertEqual(languages, eligible[item_id])
        self.assertEqual(je.run_name("responses_test_n491_seed0_20261021T155346Z", n_questions, 0),
                         "responses_test_n491_seed0_20261021T155346Z_n2_seed0")

    def test_load_env_reads_aliases_without_overriding(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text('open_ai = from-file\n# comment\nOPENAI_API_KEY = "quoted"\n')
            saved = {name: os.environ.get(name) for name in ("open_ai", "OPENAI_API_KEY")}
            os.environ.pop("open_ai", None)
            os.environ.pop("OPENAI_API_KEY", None)
            try:
                je.load_env(path)
                self.assertEqual(os.environ["open_ai"], "from-file")
                self.assertEqual(os.environ["OPENAI_API_KEY"], "quoted")
                self.assertEqual(je.env_api_key(), "quoted")
                os.environ["OPENAI_API_KEY"] = "already"
                je.load_env(path)
                self.assertEqual(os.environ["OPENAI_API_KEY"], "already")
            finally:
                for name, value in saved.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value

    def test_response_file_renames_when_question_count_changes(self):
        import run_eval

        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / "responses_test_n30_seed0_20261021T155346Z.jsonl"
            old.write_text("{}\n")
            renamed = run_eval.align_output_path(old, "test", 491, 0, "unused")
            self.assertEqual(renamed.name, "responses_test_n491_seed0_20261021T155346Z.jsonl")
            self.assertTrue(renamed.exists())
            self.assertFalse(old.exists())
            fresh = run_eval.align_output_path(None, "validation", 2, 1, "20261021T120000Z")
            self.assertEqual(fresh.name, "responses_validation_n2_seed1_20261021T120000Z.jsonl")


if __name__ == "__main__":
    unittest.main()
