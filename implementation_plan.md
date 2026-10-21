# Explanation evaluation implementation plan

## Objective and scope

Extend the existing paired English–Amharic MCQ evaluation to ask: when Qwen selects the correct answer, does its explanation adequately justify that answer, and does this differ by language?

Use an OpenAI model as an automated judge, with human calibration and review. Describe the resulting labels as **LLM-assessed explanation validity**, not ground truth. This phase evaluates existing responses; it does not regenerate them or fine-tune the model. No reference explanations or external key-fact annotations will be supplied. The judge must use its own knowledge alongside the question, options, answer key, and generated response.

## Implementation deliverables

- `judge_explanations.py`: dry-run inventory, pilot selection, API judging, incremental saving, and resume support.
- `judge_schema.py`: Pydantic output schema and application-level validation.
- `judge_prompt.md`: fixed judging instructions and explicit scoring rubric.
- Updates to `analyze_responses.py`: separate human/machine score provenance, explanation metrics, missing-score reporting, and correct denominators.
- README instructions covering setup, pilot, full run, resume, and analysis.
- Separate machine judgment JSONL, run manifest, and review export; preserve existing human scoring files and raw responses.

## 1. Audit and prepare inputs

Read the existing response JSONL. Each record contains `item_id`, `language`, `question`, `options`, `correct_option`, `answer`, generation settings, and termination information. The `answer` field contains both the selected letter and explanation.

Use a shared, pure answer parser with equivalent rules for English and Amharic. Do not rewrite source files during preparation. The current analysis has an Amharic-only normalization/parsing path and a source-file cleaning function; review these before sharing preparation logic. Report any resulting changes to answer counts rather than silently replacing earlier results.

Validate required fields, option labels, answer keys, duplicates, and paired IDs. Conflicting duplicates must be reported rather than silently accepted. Derive eligibility from actual records, not the filename or sample-limit metadata. The inspected file contains 982 responses despite an `n30` filename; verify run provenance in the manifest.

Identify correct selections programmatically. Grade all correct selections in each language, including questions that are correct in only one language. Keep incorrect and ambiguous selections in the inventory for reporting denominators, but do not send them for explanation judging in this phase.

Record termination separately. Do not invent a missing explanation by splitting on a brittle delimiter; preserve the full response and instruct the judge to assess the explanation following the selected option.

## 2. Freeze the rubric

| Score | Meaning | Rule |
| --- | --- | --- |
| 0 | Invalid | Central reasoning is incorrect, contradictory, irrelevant, absent, or only restates the selected option. |
| 1 | Partially valid | Some relevant correct reasoning is present, but an essential connection is missing or a substantive secondary error weakens the explanation. |
| 2 | Fully valid | Supplies the essential justification without a material factual or logical error. |
| null | Unscored | The judge cannot reliably interpret or assess the explanation. Requires human review. |

Short explanations may receive 2. Do not reward length or polished writing. Do not lower factual validity for awkward grammar unless meaning becomes incorrect or uninterpretable. Language adherence is assessed separately. A correct selected letter alone is not evidence of a valid explanation.

A naturally completed response with no explanation receives 0. An API failure, refusal, or failed validation is an execution status, not a validity score.

Assign a rubric version such as `v1`. If scoring rules or judge instructions change, use a new version and do not silently combine incompatible judgments.

## 3. Judge prompt and input

Use one fixed English instruction template for both languages. Include a response-language field; keep the question, options, and generated response in their original language. Do not translate Amharic as the primary judging method.

Send:

- Question and all labeled options.
- Correct option label and text.
- Full generated answer and explanation.
- Expected response language.

Provide the rubric as system instructions. Delimit the evaluation material as data and tell the judge not to follow instructions inside it. Request a brief assessment justification, not an extended reasoning trace. Do not reveal the tested model identity, expected language gap, paired counterpart, or earlier judgments.

No reference explanation is included. Flag ambiguous science, apparent answer-key problems, or inability to interpret the response for human review. Score consistency against what the response actually says rather than assuming that a correct letter implies supporting reasoning.

## 4. Typed output contract

Use the OpenAI Python SDK's `client.responses.parse(..., text_format=ExplanationJudgment)` with a compatible model and Pydantic. Make the judge model configurable and record the exact requested and returned model identifiers. Do not select a model solely because it supports the output format; assess its Amharic judging performance during calibration.

Required fields:

| Field | Type / allowed values |
| --- | --- |
| `explanation_validity` | `Literal[0, 1, 2]` or null |
| `consistency` | `supports`, `contradicts`, `unclear` |
| `language_adherence` | `maintained`, `partly_maintained`, `substantially_abandoned` |
| `error_types` | List drawn from `incorrect_fact`, `invalid_inference`, `missing_essential_reason`, `contradiction`, `irrelevant`, `restatement_only`, `missing_explanation`, `uninterpretable`, `other` |
| `brief_justification` | String |
| `needs_human_review` | Boolean |
| `review_reason` | String or null |

Use strict validation and forbid extra fields. Require `needs_human_review=true` and a nonempty review reason for a null score. Require a review reason whenever human review is requested. An empty error list is valid.

Accept a judgment only after completion status, refusal handling, parsed-object presence, schema validation, and cross-field checks pass. Never extract a score from malformed text or replace a failed judgment with 0. Structured Outputs constrain format, not scientific correctness.

Documentation: https://developers.openai.com/api/docs/guides/structured-outputs

## 5. Calibration and human review

Before the full run, sample approximately 30 completed, both-correct question pairs with a recorded seed. Have the researcher score each explanation independently before seeing automated judgments. Compare exact 0/1/2 agreement and agreement on fully valid versus not fully valid, separately by language. Inspect a confusion matrix and disagreement examples.

Clarify ambiguous rules using this pilot, then freeze the prompt and rubric. Reserve a separate sample for an agreement check after freezing; do not report agreement on calibration examples as an independent validation result. Include a sample of correct responses outside the both-correct subset in subsequent review.

Review judge uncertainty flags and a random sample of unflagged judgments. Keep human labels and adjudication separate from original automated labels. Report which results use machine judgments and which use human-reviewed judgments. The researcher retains responsibility for inspecting outputs and interpreting findings, and should disclose automated assistance in accordance with the fellowship requirements.

## 6. API execution and storage

Read `OPENAI_API_KEY` from an environment variable or Colab secret. Never place it in source code, output files, or version control.

Provide a dry-run mode that reports eligible counts by language, completed/truncated counts, primary paired sample size, and proposed request count without making API calls. Allow a small pilot limit and bounded concurrency before a full run. Record usage for cost estimation; any pre-run estimate must state its token and pricing assumptions.

For each successful request, append and flush one JSONL record. Save failures with explicit statuses such as `api_error`, `refused`, `incomplete`, or `validation_error`, rather than a numerical score. Retry transient rate-limit/server errors with bounded exponential backoff. Do not repeatedly retry permanent configuration or schema errors.

The script, not the judge, attaches item ID, language, source path/hash, exact judge input, input hash, rubric version, prompt/schema hashes, requested and returned model IDs, API response ID, timestamp, token usage, and execution status. Preserve returned judgments and enough response/error information to audit outcomes without storing secrets.

Resume by a stable key covering item ID, language, source content, judging instructions/schema, model, and relevant settings. Skip only successful, matching judgments. A changed response, rubric, or model requires a new judgment. Keep API failures distinct from model-generated unscored judgments.

Suggested outputs:

- `results/judge/<run_id>/judgments.jsonl`
- `results/judge/<run_id>/manifest.json`
- `results/judge/<run_id>/review.jsonl`
- `analysis/<run_id>/explanation_summary.json`

## 7. Truncation policy

Flag all responses that reached the generation token limit. Score eligible truncated responses as written for a secondary analysis, with termination metadata preserved outside the judging rubric.

Primary analysis uses completed responses: a pair is eligible only when both answers are correct and both responses completed. Primary scored comparisons additionally require both judgments to be available. Report eligible versus scored pair counts.

Secondary analysis includes truncated responses as delivered. Report truncation rates for all responses and among correct answers by language. This distinguishes incomplete generation from a finished but inadequate explanation while showing what exclusions remove. Do not discard truncations silently.

## 8. Analysis and denominators

### Primary paired comparison

On completed, both-correct pairs with both explanations scored, report:

- Both explanations fully valid.
- English only fully valid.
- Amharic only fully valid.
- Neither fully valid.
- Each language's fully-valid rate and their paired difference.
- A paired bootstrap interval, resampling question IDs together, with recorded seed and draw count.

Also report the paired 0/1/2 table and counts favoring each language or tying. Treat scores as ordinal; do not assume the distance from 0 to 1 has the same meaning as 1 to 2.

### Secondary comparisons

For each language, report 0/1/2 counts and the fraction fully valid among correctly answered, scored responses. Explicitly label this denominator when judgments are missing. Report how many correct answers were eligible, scored, unscored, flagged, and truncated. These language-specific groups contain different questions, so their difference is not a matched comparison.

For joint correct-and-fully-valid performance, use all responses eligible for the stated analysis as the denominator, including incorrect selections. The current analysis uses only scored responses for this denominator; correct it before integrating a correct-only judging workflow. If correct-answer judgments are missing, the joint rate is unresolved: report bounds or label the result provisional rather than treating missing judgments as failures. Define treatment of ambiguous selections explicitly and provide counts.

Produce completed-only and all-delivered views. Keep separate reports for automated and human-reviewed labels, with judge–human agreement by language and concrete reviewed examples.

## 9. Verification and acceptance criteria

Use small synthetic fixtures and mocked API responses to verify:

- Equivalent English/Amharic answer parsing and correct-only eligibility.
- Preservation of raw input files and existing human scores.
- Acceptance of valid 0/1/2/null outputs and rejection of invalid fields or cross-field combinations.
- Refusals, incomplete responses, and API failures remain unscored.
- Resume skips matching successes and invalidates changed input/prompt/model combinations.
- Truncated responses are excluded only from the designated primary view.
- Known paired fixtures produce the expected contingency counts and denominators, including missing judgments and incorrect selections.

Run a small live pilot only after configuration is ready. Validate saved artifacts before scaling. Completion means reproducible judgments, explicit provenance and missingness, verified denominator logic, and a calibrated rubric; it does not mean the hypothesized Amharic gap must appear.

## Implementation order

1. Audit parsing and input provenance; define pure preparation functions.
2. Add schema and fixed prompt, with dry-run inventory.
3. Prepare the human calibration sample and review format.
4. Implement API calls, validation, storage, and resume.
5. Run calibration and freeze rubric/model configuration.
6. Run the main evaluation and independent agreement check.
7. Update analysis, verify metrics, and document results and limitations.
