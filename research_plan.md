# Beyond Correct Answers: Evaluating Amharic Explanations in a Multilingual Language Model

**Status:** Proposed exploratory study; no evaluation findings yet.  
**Model:** `Qwen/Qwen3.5-4B`  
**Author:** Nathan Behailu  
**Date:** October 21, 2026

## Abstract

This study will investigate whether multiple-choice answer accuracy adequately represents a model's ability to provide useful explanations in Amharic. We will evaluate Qwen3.5-4B on meaning-equivalent English and Amharic questions, asking it to select an answer and then provide a brief explanation in the question's language. Answer correctness, explanation validity, answer–explanation consistency, and language adherence will be assessed separately. Paired analyses will measure how often correctness carries over between languages and whether questions answered correctly in both languages receive equally valid explanations. The study is motivated by an early attempt to add Amharic support to tebebAI through supervised fine-tuning, after which repetitive generation was observed. That experience motivates the investigation but does not establish a failure in the checkpoint evaluated here. The intended contribution is a small, transparent, personally grounded analysis of what answer-only scores may overlook—not a claim that Amharic has never been evaluated.

## Background and motivation

During work on tebebAI, an initial attempt at supervised fine-tuning on Amharic data produced repeated sentence fragments. The experiment was stopped early, and effort shifted toward English. The cause was not established: data quality, training configuration, decoding, and the original model's capabilities remain possible explanations.

This raised a practical question: what evidence should an Amharic-speaking developer require before treating a model's declared language support as sufficient for an application? A correct answer choice alone does not establish that the accompanying explanation is accurate or useful.

Existing work must be acknowledged. Qwen3.5's release documentation claims expanded multilingual support, including Amharic in its family-level language list. Its model card reports aggregate multilingual benchmarks. Cohere's Tiny Aya report already publishes Amharic-specific results, including open-ended generation evaluations of Qwen3.5-4B. Therefore, this study will not claim that public Amharic evaluations are absent or that the topic is entirely unexplored.

## Research questions

1. For equivalent English and Amharic questions, how consistently does the model choose the correct answer?
2. Given a correct answer, how often is its explanation valid in each language?
3. On questions answered correctly in both languages, does explanation validity differ?
4. What observable patterns occur when answers or explanations disagree across languages?

## Hypotheses

**Primary hypothesis:** On questions answered correctly in both languages, Amharic explanations receive lower validity scores than English explanations.

**Supporting hypothesis:** A correct answer is less frequently accompanied by a fully valid explanation in Amharic than in English.

**Exploratory analysis:** Estimate correctness overlap and disagreement between languages without assuming English wins on every question.

These hypotheses concern observable outputs. Asking for an answer followed by an explanation does not establish that explaining causes answer accuracy to decline, nor does it reveal the model's internal reasoning.

## Model selection

Use the official `Qwen/Qwen3.5-4B` checkpoint from Hugging Face. It belongs to a major lab's general-purpose model family with an explicit Amharic-support claim and fits the challenge's 0.6–6B parameter range. Record the actual parameter count and clarify whether it includes multimodal components; the model name is not a complete parameter accounting.

Pin the checkpoint revision and record the software versions, loaded model class, precision, hardware, and device placement. Findings will apply to this checkpoint and configuration, not automatically to larger frontier models or the whole Qwen family.

## Dataset design

### Pilot

Author six paired questions to check translation equivalence, prompt formatting, output limits, and scoring rules. Pilot questions will not appear in the main evaluation. Resolve scoring ambiguities before collecting the main results.

### Main evaluation

Target 30 fresh question pairs, producing 60 responses in the primary run. This is a proposed scope and can be revised before the main evaluation begins.

Questions should reflect plausible tebebAI uses and subjects the evaluator can verify confidently. Prefer familiar science, everyday quantitative reasoning, and passage-grounded comprehension over obscure trivia. Record the category for each item; category-level findings will be exploratory because the sample is small.

For each question:

- Write natural, meaning-equivalent Amharic and English versions.
- Provide four options labeled A–D with the same order in both languages.
- Ensure exactly one defensible correct option.
- Balance correct-option positions across the dataset.
- Specify the correct option and essential explanation points before generating responses.
- Record factual sources where external verification is needed.
- Check that wording does not accidentally make one language easier or reveal the answer.

Suggested fields: `item_id`, `category`, `question_en`, `question_am`, `options_en`, `options_am`, `correct_option`, `required_explanation_points`, `reference_source`, and `translation_notes`.

A bilingual reviewer should check equivalence if available. If only one person authors and reviews the dataset, disclose that limitation.

## Generation procedure

For each question, start a fresh conversation and use the same prompt structure in both languages:

> First select one answer: A, B, C, or D. Then explain your answer in 2–4 sentences in [English/Amharic].

Translate the instruction naturally for the Amharic condition. Request a concise justification, not an exhaustive chain of thought.

Use fixed settings across languages. The current notebook uses non-thinking mode, greedy decoding (`do_sample=False`), and `repetition_penalty=1.0`. Treat these as a specific experimental configuration and record departures from the developer's recommended settings. Streaming affects display, not the intended scoring criteria.

The notebook's current 128-token limit is for diagnostics only. Use the pilot to choose an output budget that permits complete Amharic explanations. Set and document a uniform truncation rule before the main evaluation—for example, rerun any truncated response once with twice the budget, retaining both outputs. Do not label a response invalid merely because the configured budget cut it off.

Collect one response per item-language condition for the primary run. If resources permit, repeat the complete paired set with a prespecified sampled decoding configuration and fixed seeds as a robustness check. Do not rerun only failures until a preferred answer appears.

## Scoring rubric

Score answer and explanation separately. Shuffle outputs for scoring and avoid viewing the paired response first where practical. The language cannot be blinded, but model identity and paired outcomes can be hidden from an additional reviewer.

| Measure | Scoring |
|---|---|
| Answer correctness | Correct / incorrect / missing or ambiguous selection |
| Explanation validity | 2: essential justification is correct with no material error; 1: relevant correct reasoning but incomplete or materially flawed; 0: incorrect justification, contradiction, or mere restatement without justification |
| Answer–explanation consistency | Supports the selected option / contradicts it / unclear |
| Language adherence | Maintained / partly maintained / substantially abandoned |
| Termination | Natural end / token-limit truncation / execution failure |

Define item-specific examples of partial and valid explanations during rubric preparation. Awkward grammar alone is not a reasoning error. An explanation can contain the correct principle while the selected letter is wrong; preserve that distinction.

Infrastructure failures are missing observations, not wrong model answers. Rerun them consistently and report unresolved missingness. Report malformed answer selections rather than silently inferring a letter from the explanation.

If possible, have another bilingual reader independently score a prespecified subset. Preserve initial ratings, report disagreements, and document how final ratings were resolved.

## Analysis

### Paired answer outcomes

For evaluable pairs, construct:

| | Amharic correct | Amharic incorrect |
|---|---:|---:|
| English correct | a | b |
| English incorrect | c | d |

Report missing or ambiguous selections separately and state their treatment in accuracy calculations.

- English accuracy: `(a + b) / N`
- Amharic accuracy: `(a + c) / N`
- Amharic correct given English correct: `a / (a + b)`
- English correct given Amharic correct: `a / (a + c)`

Here `N = a + b + c + d` for the table's included pairs. A conditional rate with a zero denominator is undefined. Report numerators and denominators alongside percentages. The conditional rates share a numerator and are not independent findings.

### Explanation outcomes

Define a fully valid explanation as rubric score 2. Report:

1. Fully valid explanations among correct answers, separately by language.
2. Correct answer AND fully valid explanation among all evaluable responses, separately by language.
3. Explanation-score differences on the same questions answered correctly in both languages—the primary paired comparison.
4. Counts of answer–explanation contradictions and language-adherence failures.

For the both-correct subset, report how many pairs favor English, favor Amharic, or tie, alongside the score distribution. Show uncertainty where feasible, using methods that preserve question pairs, and avoid treating this small convenience sample as representative of all Amharic use.

### Qualitative inspection

Inspect all four answer-outcome groups. Include successes and counterexamples as well as failures. Possible descriptive patterns include misunderstanding a term, ignoring negation, selecting a letter inconsistent with an otherwise correct explanation, or providing a fluent but factually incorrect justification. Such labels do not establish internal causes.

## Optional follow-up: separating input and output language

The main comparison changes both question and explanation language. It cannot isolate comprehension from expression. A follow-up can cross English/Amharic question language with English/Amharic explanation language, producing four conditions per item. Treat any follow-up selected after inspecting results as exploratory and use fresh questions to confirm a revised hypothesis.

An answer-only condition is another possible extension, but is not required for the primary question. It would address a different question: whether requesting an explanation changes answer selection.

## Interpretation and limitations

- Similar answer accuracy but weaker Amharic explanations would support the claim that answer accuracy misses an important difference in user-facing reliability.
- Lower Amharic answer accuracy with similar explanations on the both-correct subset would support an answer-performance gap, but not the primary explanation hypothesis.
- Similar performance across conditions would leave the hypothesized gap unsupported in this setup.
- Translation problems or truncation may explain apparent differences and must be checked before attributing them to model capability.

Other limitations include a small researcher-authored dataset, subjective scoring, possible familiarity with benchmark-style questions, and dependence on decoding settings. Conditioning on both-correct questions improves item matching but restricts the conclusion to that subset. No result here diagnoses the earlier tebebAI SFT failure.

## Proposed path forward

Any intervention proposal should follow the observed errors. Candidate approaches include native-speaker-curated Amharic question–answer–explanation triples, explicit review of answer–explanation consistency, and targeted examples for repeatedly missed linguistic distinctions. These are hypotheses for future work, not demonstrated remedies. No fine-tuning intervention is part of the primary evaluation.

## Reproducibility and deliverables

Preserve:

- This plan and a dated record of changes made before and after the pilot.
- Pilot and main datasets, clearly separated.
- The Colab/Jupyter notebook and pinned environment details.
- Raw responses, model revision, prompts, generation settings, and termination status.
- Annotation rubric, individual ratings, and final scores.
- Summary tables and representative examples, including counterexamples.
- A README describing motivation, methods, results, limitations, and proposed next steps.

For the fellowship, publish the required artifacts in a public Hugging Face Bucket after checking dataset permissions and the model license. Record exactly which weights or references are included. Do not publish credentials or private user data.

## Authorship and fellowship requirement

This document is an AI-assisted planning draft, not completed research or a submission-ready personal statement. The fellowship explicitly asks applicants to conduct their own analysis. The applicant should author and review the examples, run and inspect the model, make and document scoring judgments, and write the final interpretation in their own words. Whether this level of assistance is permitted should be checked against the organizers' rules.

## References

- [Qwen3.5-4B official model card](https://huggingface.co/Qwen/Qwen3.5-4B)
- [Qwen3.5 official announcement](https://qwen.ai/blog?id=qwen3.5)
- [Tiny Aya technical report, including per-language evaluation results](https://arxiv.org/html/2603.11510v1)
