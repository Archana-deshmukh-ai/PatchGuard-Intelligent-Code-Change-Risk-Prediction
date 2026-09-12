"""
LLM Response parsing and validation module for Phase 6 LLM Code-Change Analysis.
Parses raw LLM JSON text responses into validated LLMAnalysisResult objects.
"""

import json
import re
from typing import Dict, Any, Optional
from src.analysis.schema import LLMAnalysisResult, ALLOWED_REVIEW_PRIORITIES
from src.analysis.exceptions import MalformedLLMResponseError

class ResponseParser:
    """
    Parses raw LLM text outputs into structured LLMAnalysisResult dataclass instances.
    Handles markdown code fence stripping, JSON deserialization, and strict nested schema validation.
    """
    REQUIRED_KEYS = {
        "summary",
        "key_changes",
        "potential_risk_factors",
        "affected_areas",
        "testing_observations",
        "recommended_review_actions"
    }

    def parse(
        self,
        raw_text: str,
        expected_commit_hash: Optional[str] = None
    ) -> LLMAnalysisResult:
        """
        Parses raw text into an LLMAnalysisResult object.
        Raises MalformedLLMResponseError if raw_text is invalid JSON or violates schema rules.
        """
        if not raw_text or not raw_text.strip():
            raise MalformedLLMResponseError("LLM response string is empty.")

        clean_text = self._strip_markdown_fences(raw_text.strip())

        try:
            data = json.loads(clean_text)
        except json.JSONDecodeError as e:
            raise MalformedLLMResponseError(f"Failed to parse LLM response as JSON: {e}") from e

        if not isinstance(data, dict):
            raise MalformedLLMResponseError(
                f"Expected JSON object (dict) in LLM response, got {type(data).__name__}."
            )

        missing_keys = self.REQUIRED_KEYS - set(data.keys())
        if missing_keys:
            raise MalformedLLMResponseError(
                f"LLM response missing required top-level fields: {sorted(missing_keys)}"
            )

        if not isinstance(data.get("summary"), str):
            raise MalformedLLMResponseError(
                f"Field 'summary' must be a string, got {type(data.get('summary')).__name__}."
            )

        if expected_commit_hash:
            data["commit_hash"] = expected_commit_hash
        elif "commit_hash" not in data or not data["commit_hash"]:
            data["commit_hash"] = "unknown"

        # Validate string lists
        for list_field in ["key_changes", "affected_areas", "testing_observations"]:
            val = data.get(list_field)
            if not isinstance(val, list):
                raise MalformedLLMResponseError(
                    f"Field '{list_field}' must be a list, got {type(val).__name__}."
                )
            for idx, elem in enumerate(val):
                if not isinstance(elem, str):
                    raise MalformedLLMResponseError(
                        f"Elements in '{list_field}' must be strings, got {type(elem).__name__} at index {idx}."
                    )

        # Validate nested risk factors
        risk_factors_raw = data.get("potential_risk_factors")
        if not isinstance(risk_factors_raw, list):
            raise MalformedLLMResponseError(
                f"Field 'potential_risk_factors' must be a list, got {type(risk_factors_raw).__name__}."
            )

        for idx, rf in enumerate(risk_factors_raw):
            if not isinstance(rf, dict):
                raise MalformedLLMResponseError(
                    f"Each element in 'potential_risk_factors' must be a dictionary, got {type(rf).__name__} at index {idx}."
                )
            rf_req = {"category", "description", "review_priority", "file_path"}
            rf_missing = rf_req - set(rf.keys())
            if rf_missing:
                raise MalformedLLMResponseError(
                    f"Risk factor at index {idx} missing required fields: {sorted(rf_missing)}"
                )
            if not isinstance(rf["category"], str) or not isinstance(rf["description"], str) or not isinstance(rf["file_path"], str):
                raise MalformedLLMResponseError(
                    f"Fields 'category', 'description', and 'file_path' in risk factor at index {idx} must be strings."
                )
            if not isinstance(rf["review_priority"], str) or rf["review_priority"] not in ALLOWED_REVIEW_PRIORITIES:
                raise MalformedLLMResponseError(
                    f"Invalid review_priority '{rf.get('review_priority')}' in risk factor at index {idx}. Must be one of {sorted(ALLOWED_REVIEW_PRIORITIES)}"
                )
            if "line_range" in rf and rf["line_range"] is not None and not isinstance(rf["line_range"], str):
                raise MalformedLLMResponseError(
                    f"Field 'line_range' in risk factor at index {idx} must be a string or null."
                )
            if "evidence_snippet" in rf and not isinstance(rf["evidence_snippet"], str):
                raise MalformedLLMResponseError(
                    f"Field 'evidence_snippet' in risk factor at index {idx} must be a string."
                )

        # Validate nested review actions
        review_actions_raw = data.get("recommended_review_actions")
        if not isinstance(review_actions_raw, list):
            raise MalformedLLMResponseError(
                f"Field 'recommended_review_actions' must be a list, got {type(review_actions_raw).__name__}."
            )

        for idx, ra in enumerate(review_actions_raw):
            if not isinstance(ra, dict):
                raise MalformedLLMResponseError(
                    f"Each element in 'recommended_review_actions' must be a dictionary, got {type(ra).__name__} at index {idx}."
                )
            if "action" not in ra or not isinstance(ra["action"], str):
                raise MalformedLLMResponseError(
                    f"Review action at index {idx} missing required string field 'action'."
                )
            if "target_file" in ra and ra["target_file"] is not None and not isinstance(ra["target_file"], str):
                raise MalformedLLMResponseError(
                    f"Field 'target_file' in review action at index {idx} must be a string or null."
                )

        try:
            result = LLMAnalysisResult.from_dict(data)
        except (TypeError, ValueError, KeyError) as e:
            raise MalformedLLMResponseError(
                f"Schema validation error while creating LLMAnalysisResult: {e}"
            ) from e

        return result

    def _strip_markdown_fences(self, text: str) -> str:
        """Strips ```json ... ``` or ``` ... ``` markdown code block fences if present."""
        pattern = r'^```(?:json)?\s*\n?(.*?)\n?```$'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines).strip()

        return text
