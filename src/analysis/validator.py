"""
Evidence verification module for Phase 6 LLM Code-Change Analysis.
Audits risk factor citations (file path, evidence snippet, and line range) against actual CommitDiff content.
"""

import re
from dataclasses import replace
from typing import List, Optional, Tuple
from src.analysis.schema import LLMAnalysisResult, RiskFactor
from src.analysis.diff_extractor import CommitDiff, FileDiff, DiffHunk, DiffLine

class EvidenceValidator:
    """
    Non-destructively audits citation claims in LLM risk factors against actual git diffs.
    Validates file paths, evidence snippets, and structural line ranges.
    
    DELETED CODE SEMANTICS:
    Evidence from a deleted diff line (line_type == '-') is valid evidence about WHAT CHANGED
    in the commit. It grounds the citation in the commit's diff. However, verifying a deleted line
    does NOT imply that the deleted code exists in the post-commit repository state.
    """
    def validate_analysis(
        self,
        analysis: LLMAnalysisResult,
        commit_diff: CommitDiff
    ) -> LLMAnalysisResult:
        """
        Validates all risk factors in an LLMAnalysisResult against a CommitDiff.
        Returns a new LLMAnalysisResult with updated evidence_verified flags on risk factors.
        """
        validated_risk_factors: List[RiskFactor] = []

        for rf in analysis.potential_risk_factors:
            is_verified = self.verify_risk_factor(rf, commit_diff)
            validated_rf = replace(rf, evidence_verified=is_verified)
            validated_risk_factors.append(validated_rf)

        return replace(analysis, potential_risk_factors=validated_risk_factors)

    def verify_risk_factor(self, risk_factor: RiskFactor, commit_diff: CommitDiff) -> bool:
        """
        Verifies whether a RiskFactor's file_path, evidence_snippet, and line_range match the diff.
        """
        target_path = risk_factor.file_path.strip().replace('\\', '/')
        if not target_path:
            return False

        matching_file: Optional[FileDiff] = None
        for f in commit_diff.files_changed:
            norm_new = f.new_path.strip().replace('\\', '/')
            norm_old = f.old_path.strip().replace('\\', '/')
            if target_path == norm_new or target_path == norm_old:
                matching_file = f
                break
            if norm_new and (norm_new.endswith(target_path) or target_path.endswith(norm_new)):
                matching_file = f
                break

        if not matching_file:
            return False

        # 1. Verify line range if specified
        if risk_factor.line_range and risk_factor.line_range.strip():
            parsed_range, is_valid_format = self._parse_line_range(risk_factor.line_range)
            if not is_valid_format or parsed_range is None:
                return False
            if not self._verify_line_range_overlap(parsed_range, matching_file):
                return False

        # 2. Verify evidence snippet if specified
        snippet = risk_factor.evidence_snippet.strip() if risk_factor.evidence_snippet else ""
        if snippet:
            if not self._verify_snippet(snippet, matching_file):
                return False

        return True

    def _parse_line_range(self, line_range_str: str) -> Tuple[Optional[Tuple[int, int]], bool]:
        """
        Parses line_range string like "L40-L45", "L10", "40-45", "10".
        Returns ((start, end), is_valid_format).
        """
        cleaned = line_range_str.strip()
        pattern = r'^\s*L?(\d+)(?:\s*-\s*L?(\d+))?\s*$'
        match = re.match(pattern, cleaned, re.IGNORECASE)
        if not match:
            return None, False

        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        if start > end:
            start, end = end, start
        return (start, end), True

    def _verify_line_range_overlap(self, line_range: Tuple[int, int], file_diff: FileDiff) -> bool:
        """
        Checks whether the cited line range [start, end] is structurally consistent
        with any diff hunk or line in the FileDiff.
        """
        start, end = line_range

        for hunk in file_diff.hunks:
            # Check individual diff lines (handles added +, deleted -, and context lines)
            for line in hunk.lines:
                if line.old_lineno is not None and start <= line.old_lineno <= end:
                    return True
                if line.new_lineno is not None and start <= line.new_lineno <= end:
                    return True

            # Check hunk metadata boundaries
            if hunk.old_lines > 0:
                old_hunk_end = hunk.old_start + hunk.old_lines - 1
                if max(start, hunk.old_start) <= min(end, old_hunk_end):
                    return True

            if hunk.new_lines > 0:
                new_hunk_end = hunk.new_start + hunk.new_lines - 1
                if max(start, hunk.new_start) <= min(end, new_hunk_end):
                    return True

        return False

    def _verify_snippet(self, snippet: str, file_diff: FileDiff) -> bool:
        """
        Checks whether evidence snippet is present in any diff line (added, deleted, or context).
        """
        file_lines = []
        for hunk in file_diff.hunks:
            for line in hunk.lines:
                file_lines.append(line.content)

        full_file_diff_text = "\n".join(file_lines)

        if snippet in full_file_diff_text:
            return True

        for line_text in file_lines:
            if snippet in line_text or line_text.strip() in snippet:
                return True

        return False
