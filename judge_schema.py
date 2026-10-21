"""Structured output schema for the explanation judge.

ExplanationJudgment is passed to client.responses.parse as text_format.
cross_field_errors adds the rules that a JSON schema cannot express. A
judgment is accepted only when the schema and these checks both pass.
"""

import hashlib
import json
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

VALIDITY = (0, 1, 2)
CONSISTENCY = ("supports", "contradicts", "unclear")
ADHERENCE = ("maintained", "partly_maintained", "substantially_abandoned")
ERROR_TYPES = (
    "incorrect_fact",
    "invalid_inference",
    "missing_essential_reason",
    "contradiction",
    "irrelevant",
    "restatement_only",
    "missing_explanation",
    "uninterpretable",
    "other",
)


class ExplanationJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    explanation_validity: Optional[Literal[0, 1, 2]]
    consistency: Literal["supports", "contradicts", "unclear"]
    language_adherence: Literal["maintained", "partly_maintained", "substantially_abandoned"]
    error_types: List[
        Literal[
            "incorrect_fact",
            "invalid_inference",
            "missing_essential_reason",
            "contradiction",
            "irrelevant",
            "restatement_only",
            "missing_explanation",
            "uninterpretable",
            "other",
        ]
    ]
    brief_justification: str
    needs_human_review: bool
    review_reason: Optional[str]

    @field_validator("explanation_validity", mode="before")
    @classmethod
    def reject_bool(cls, value):
        if isinstance(value, bool):
            raise ValueError("explanation_validity must be 0, 1, 2, or null, not a boolean")
        return value


def cross_field_errors(judgment):
    errors = []
    reason = (judgment.review_reason or "").strip()
    if judgment.explanation_validity is None and not judgment.needs_human_review:
        errors.append("a null explanation_validity requires needs_human_review=true")
    if judgment.needs_human_review and not reason:
        errors.append("needs_human_review=true requires a nonempty review_reason")
    if not judgment.brief_justification.strip():
        errors.append("brief_justification is empty")
    if len(set(judgment.error_types)) != len(judgment.error_types):
        errors.append("error_types contains duplicates")
    return errors


def validate_judgment(payload):
    """Return (judgment, errors) for a dict or JSON string. Never coerces a failure into a score."""
    try:
        if isinstance(payload, str):
            judgment = ExplanationJudgment.model_validate_json(payload)
        else:
            judgment = ExplanationJudgment.model_validate(payload)
    except ValidationError as exc:
        return None, [error["msg"] + f" at {'.'.join(map(str, error['loc']))}" for error in exc.errors()]
    return judgment, cross_field_errors(judgment)


def schema_hash():
    text = json.dumps(ExplanationJudgment.model_json_schema(), sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
