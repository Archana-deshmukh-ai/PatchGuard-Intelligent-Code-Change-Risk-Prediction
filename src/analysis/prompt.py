"""
Prompt construction module for Phase 6 LLM Code-Change Analysis.
Formats structured git diffs and ML prediction results into LLM prompts with context budgeting.
"""

from typing import Optional, List, Tuple
from dataclasses import dataclass
from src.analysis.diff_extractor import CommitDiff, FileDiff, is_generated_file
from src.engine.schema import PredictionResult

ML_DISCLAIMER_PROMPT = """### Machine Learning Risk Context (Phase 5 ML Engine)
- Predicted Risk Score: {raw_probability:.4f} (Decision Threshold: {decision_threshold:.2f})
- ML Risk Level: {risk_level} ({prediction_label})
- Top Contributing Statistical Factors:
{top_factors_str}

IMPORTANT NOTE ON ML RISK CONTEXT:
The ML risk score above is a statistical pattern score derived from historical engineering metrics (such as code churn, file dispersion, and author experience). It represents mathematical model feature contributions, NOT a causal explanation of software bugs or a guarantee of defects.
You must analyze the actual code diff independently for qualitative defect patterns, logic errors, missing edge-case handling, security issues, and structural design concerns.
You are permitted and encouraged to disagree with or refine the statistical ML risk assessment if the actual code changes appear safe, benign, or well-tested, or vice versa."""

PROMPT_TEMPLATE = """You are an expert software reviewer and static code analyst.
Analyze the following Git commit diff and produce a structured qualitative review in JSON format.

## Commit Metadata
- Commit Hash: {commit_hash}
- Author: {author}
- Date: {date}
- Commit Message: {message}
- Stats: {files_changed_count} files changed, {additions} additions (+), {deletions} deletions (-)

{ml_section}

## Summary of Changed Files
{files_summary_str}

## Git Code Diff
{diff_str}

## Required JSON Output Schema
You MUST respond with valid JSON matching the following structure:
```json
{{
  "commit_hash": "{commit_hash}",
  "summary": "High-level description of what this commit does (2-3 sentences).",
  "key_changes": [
    "List of key functional/architectural changes made in this commit"
  ],
  "potential_risk_factors": [
    {{
      "category": "e.g. Error Handling | Concurrency | Logic Error | Performance | Security | Edge Case | Test Coverage",
      "description": "Specific observation or risk hypothesis based on code changes",
      "review_priority": "LOW | MEDIUM | HIGH",
      "file_path": "path/to/file.py",
      "line_range": "e.g. L45-L52 (or null if general)",
      "evidence_snippet": "Exact line or snippet of code from diff that supports this risk factor"
    }}
  ],
  "affected_areas": [
    "Modules, APIs, or system components affected by these changes"
  ],
  "testing_observations": [
    "Observations about added/modified tests or recommendations for test cases"
  ],
  "recommended_review_actions": [
    {{
      "action": "Specific action item for developer/reviewer",
      "target_file": "path/to/file.py"
    }}
  ],
  "confidence_notes": "Notes on code clarity or any limitations of the analysis."
}}
```

Rules:
1. 'review_priority' in potential_risk_factors MUST be strictly one of: "LOW", "MEDIUM", or "HIGH".
2. 'file_path' and 'evidence_snippet' in potential_risk_factors MUST accurately reference actual files and code snippets present in the diff.
3. Output ONLY the JSON object. Do not include markdown code block backticks unless returning pure JSON.
"""

@dataclass
class PromptResult:
    prompt_text: str
    is_truncated: bool
    original_diff_bytes: int
    included_diff_bytes: int
    truncated_files_count: int

class PromptBuilder:
    """
    Constructs contextual, structured prompts for LLM code change analysis.
    Supports diff context budgeting, source file prioritization, and non-causal ML context integration.
    """
    def __init__(self, max_diff_bytes: int = 50_000):
        self.max_diff_bytes = max_diff_bytes
        self.last_is_truncated = False
        self.last_original_diff_bytes = 0
        self.last_included_diff_bytes = 0

    def build_prompt(
        self,
        commit_diff: CommitDiff,
        prediction_result: Optional[PredictionResult] = None
    ) -> str:
        """
        Builds a complete text prompt string from commit diff and optional ML prediction result.
        Sets self.last_is_truncated indicating if diff content was truncated due to budget constraints.
        """
        res = self.build(commit_diff, prediction_result)
        self.last_is_truncated = res.is_truncated
        self.last_original_diff_bytes = res.original_diff_bytes
        self.last_included_diff_bytes = res.included_diff_bytes
        return res.prompt_text

    def build(
        self,
        commit_diff: CommitDiff,
        prediction_result: Optional[PredictionResult] = None
    ) -> PromptResult:
        """
        Builds a PromptResult containing prompt text, diff size metadata, and truncation metadata.
        """
        if prediction_result is not None:
            top_pos = prediction_result.model_signals.top_positive_factors
            top_neg = prediction_result.model_signals.top_negative_factors
            
            factors = []
            for k, v in top_pos.items():
                factors.append(f"  - Positive risk factor ({k}): +{v:.4f}")
            for k, v in top_neg.items():
                factors.append(f"  - Negative risk factor ({k}): {v:.4f}")
            
            factors_str = "\n".join(factors) if factors else "  - None"
            
            ml_section = ML_DISCLAIMER_PROMPT.format(
                raw_probability=prediction_result.raw_probability,
                decision_threshold=prediction_result.decision_threshold,
                risk_level=prediction_result.risk_level,
                prediction_label=prediction_result.prediction_label,
                top_factors_str=factors_str
            )
        else:
            ml_section = "### Machine Learning Risk Context\n- Machine learning risk prediction: Not provided for this analysis."

        # ALWAYS include complete changed-file inventory summary regardless of truncation
        files_summary_lines = []
        for file in commit_diff.files_changed:
            gen_note = " [GENERATED]" if file.is_generated else ""
            bin_note = " [BINARY]" if file.is_binary else ""
            files_summary_lines.append(
                f"- {file.new_path} ({file.status}): +{file.additions} -{file.deletions}{gen_note}{bin_note}"
            )
        files_summary_str = "\n".join(files_summary_lines) if files_summary_lines else "No files changed."

        diff_str, is_truncated, original_diff_bytes, included_diff_bytes, truncated_files_count = self._format_diff(commit_diff.files_changed)

        author_str = f"{commit_diff.author} <{commit_diff.author_email}>" if commit_diff.author_email else commit_diff.author

        prompt_text = PROMPT_TEMPLATE.format(
            commit_hash=commit_diff.short_hash or commit_diff.full_hash,
            author=author_str,
            date=commit_diff.timestamp,
            message=commit_diff.commit_message.strip(),
            files_changed_count=len(commit_diff.files_changed),
            additions=commit_diff.total_additions,
            deletions=commit_diff.total_deletions,
            ml_section=ml_section,
            files_summary_str=files_summary_str,
            diff_str=diff_str
        )

        res = PromptResult(
            prompt_text=prompt_text,
            is_truncated=is_truncated,
            original_diff_bytes=original_diff_bytes,
            included_diff_bytes=included_diff_bytes,
            truncated_files_count=truncated_files_count
        )
        self.last_is_truncated = res.is_truncated
        self.last_original_diff_bytes = res.original_diff_bytes
        self.last_included_diff_bytes = res.included_diff_bytes
        return res

    def _get_file_priority(self, file: FileDiff) -> int:
        """
        Determines formatting priority for diff context budgeting.
        Lower values indicate higher priority for inclusion in prompt diff.
        1: Source code files
        2: Other text files
        3: Generated / lock files
        4: Binary files
        """
        if file.is_binary:
            return 4
        if file.is_generated or is_generated_file(file.new_path) or is_generated_file(file.old_path):
            return 3
        return 1

    def _format_diff(self, files: List[FileDiff]) -> Tuple[str, bool, int, int, int]:
        """
        Formats files and diff hunks into a string within max_diff_bytes budget.
        Prioritizes source files over generated/lock/binary files.
        Returns: (result_diff, is_truncated, original_diff_bytes, included_diff_bytes, truncated_files_count)
        """
        sorted_files = sorted(files, key=self._get_file_priority)

        file_texts = []
        original_diff_bytes = 0

        for file in sorted_files:
            file_header = f"--- {file.old_path}\n+++ {file.new_path}\n"
            if file.is_binary:
                file_text = file_header + "Binary files differ\n"
            elif file.is_generated or is_generated_file(file.new_path) or is_generated_file(file.old_path):
                file_text = file_header + f"[Generated file omitted from diff: {file.new_path}]\n"
            else:
                hunk_texts = []
                for hunk in file.hunks:
                    hunk_lines = [hunk.header]
                    for line in hunk.lines:
                        hunk_lines.append(f"{line.line_type}{line.content}")
                    hunk_texts.append("\n".join(hunk_lines))
                file_text = file_header + "\n".join(hunk_texts) + ("\n" if hunk_texts else "")

            b_len = len(file_text.encode('utf-8'))
            file_texts.append((file, file_text, b_len))
            original_diff_bytes += b_len

        formatted_files = []
        included_diff_bytes = 0
        is_truncated = False
        truncated_files_count = 0

        for file_idx, (file, file_text, file_bytes) in enumerate(file_texts):
            if included_diff_bytes + file_bytes > self.max_diff_bytes:
                is_truncated = True
                truncated_files_count = len(file_texts) - file_idx
                remaining_budget = self.max_diff_bytes - included_diff_bytes
                if remaining_budget > 20:
                    partial_bytes = file_text.encode('utf-8')[:remaining_budget]
                    partial_text = partial_bytes.decode('utf-8', errors='ignore')
                    formatted_files.append(partial_text + "\n[TRUNCATED: File diff cut off due to context size limit]")
                    included_diff_bytes += len(partial_bytes)
                break
            else:
                formatted_files.append(file_text)
                included_diff_bytes += file_bytes

        result_diff = "\n".join(formatted_files)
        if is_truncated:
            result_diff += f"\n\n[TRUNCATED: Total diff exceeded context budget of {self.max_diff_bytes} bytes. {truncated_files_count} files/hunks omitted or truncated.]"

        return result_diff, is_truncated, original_diff_bytes, included_diff_bytes, truncated_files_count
