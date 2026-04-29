"""Phase 3 — Grounding, Citation & Accuracy System

Provides citation extraction and grounding verification for LLM responses.
All functions are pure (no I/O side-effects) and designed to run in < 10 ms.
"""

import re
import time
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Uncited clinical claim patterns ──────────────────────────────────────────
# Each tuple is (regex_pattern, human_readable_label).
# Patterns match numeric values that must be attributed to a source.
_UNCITED_CLINICAL_PATTERNS: list[tuple[str, str]] = [
    (r"\d+(?:\.\d+)?\s*(?:mg|mcg|µg|mEq|mL|L|g|units?|mmol)\b", "medication/dose value"),
    (r"\d+(?:\.\d+)?\s*(?:mg/dL|mmol/L|pg/mL|ng/mL|IU/L|U/L|mEq/L|g/dL|%)(?!\w)", "lab value"),
    (r"\bBP\s+\d{2,3}/\d{2,3}\b|\b\d{2,3}/\d{2,3}\s*mmHg\b", "blood pressure"),
    (r"\bHR\s+\d+\b|\bpulse\s+of\s+\d+\b|\bheart\s+rate\s+(?:of\s+|was\s+)?\d+\b", "heart rate"),
    (r"\bSpO2\s+(?:of\s+)?\d+\b|\bO2\s+sat(?:uration)?\s+(?:of\s+)?\d+\b", "oxygen saturation"),
    (r"\btemperature\s+(?:of\s+)?\d+(?:\.\d+)?\s*(?:°[CF]|degrees?)\b", "temperature"),
    (r"\bweight\s+(?:of\s+)?\d+(?:\.\d+)?\s*(?:kg|lbs?|pounds?)\b", "weight"),
    (r"\bHbA1c\s+(?:of\s+)?\d+(?:\.\d+)?%?\b|\bA1c\s+(?:of\s+)?\d+(?:\.\d+)?%?\b", "HbA1c value"),
]

# Pre-compile for speed
_COMPILED_UNCITED = [(re.compile(p, re.IGNORECASE), lbl) for p, lbl in _UNCITED_CLINICAL_PATTERNS]
_CITATION_RE = re.compile(r"\[S(\d+)(?:[,\s][^\]]+)?\]")  # matches [S1] and [S22, description]
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# ── Numeric hallucination detection patterns ──────────────────────────────────
# These patterns capture numeric clinical values in the LLM response that must
# appear verbatim (normalized for whitespace) in the retrieved tool content.
# A value present in the response but absent in tool content is a hallucination.
_HALLUCINATION_PATTERNS: list[tuple[str, str]] = [
    (r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|mEq|mL|g|units?|IU)\b", "medication/dose"),
    (r"\b\d+(?:\.\d+)?\s*(?:mg/dL|mmol/L|mEq/L|g/dL|pg/mL|ng/mL|IU/L|U/L)\b", "lab value with unit"),
    (r"\bBP\s+\d{2,3}/\d{2,3}\b|\b\d{2,3}/\d{2,3}\s*mmHg\b", "blood pressure"),
    (r"\bHR\s+\d+\b|\bpulse\s+(?:of\s+)?\d+\b|\bheart\s+rate\s+(?:of\s+|was\s+)?\d+\b", "heart rate"),
    (r"\bSpO2\s+(?:of\s+)?\d+\b|\bO2\s+sat(?:uration)?\s+(?:of\s+)?\d+\b", "oxygen saturation"),
    (r"\btemperature\s+(?:of\s+)?\d+(?:\.\d+)?\s*(?:°[CF]|degrees?)\b", "temperature"),
    (r"\bweight\s+(?:of\s+)?\d+(?:\.\d+)?\s*(?:kg|lbs?|pounds?)\b", "weight"),
    (r"\bHbA1c\s+(?:of\s+)?\d+(?:\.\d+)?%?\b|\bA1c\s+(?:of\s+)?\d+(?:\.\d+)?%?\b", "HbA1c"),
    (r"\b(?:BNP|NT-proBNP|troponin|creatinine|eGFR|LVEF)\s+(?:of\s+|was\s+|is\s+)?\d+(?:\.\d+)?\b", "key lab"),
]
_COMPILED_HALLUCINATION = [(re.compile(p, re.IGNORECASE), lbl) for p, lbl in _HALLUCINATION_PATTERNS]


def _normalize_numeric(val: str) -> str:
    """Normalize a numeric clinical value for loose comparison.

    Strips internal whitespace and lowercases so "1842 pg/mL" matches "1842pg/ml".
    """
    return re.sub(r"\s+", "", val.lower())


def detect_numeric_hallucinations(response: str, tool_contents: List[str]) -> List[str]:
    """Cross-check every numeric clinical value in *response* against tool outputs.

    A value is flagged as a hallucination when it:
    1. Matches one of the clinical numeric patterns, AND
    2. Does NOT appear (normalized for whitespace) anywhere in the combined tool content.

    Returns a deduplicated list of flagged value strings in the order they appear
    in the response. Empty list means no hallucinations detected (or no tool content
    was supplied to compare against).

    Args:
        response:      The synthesized LLM response text.
        tool_contents: List of raw tool output strings (search results, summaries).

    >>> detect_numeric_hallucinations("BNP was 9999 pg/mL [S1]", ["BNP 1842 pg/mL [Source 1]"])
    ['9999 pg/mL']
    >>> detect_numeric_hallucinations("BNP was 1842 pg/mL [S1]", ["BNP 1842 pg/mL [Source 1]"])
    []
    """
    if not tool_contents:
        return []

    combined_content = " ".join(tool_contents).lower()
    # Also build a whitespace-normalized version for looser matching
    combined_normalized = re.sub(r"\s+", "", combined_content)

    flagged: list[str] = []
    seen_normalized: set[str] = set()

    seen_numeric_keys: set[str] = set()  # deduplicate by first numeric substring in match

    for pattern, _label in _COMPILED_HALLUCINATION:
        for m in pattern.finditer(response, re.IGNORECASE if False else 0):
            val = m.group().strip()
            norm = _normalize_numeric(val)

            # Build a dedup key from the first numeric token in the match (e.g. "9999")
            _num_match = re.search(r"\d+(?:\.\d+)?", val)
            _num_key = _num_match.group() if _num_match else norm

            if _num_key in seen_numeric_keys:
                continue  # Already flagged or confirmed this numeric value

            seen_numeric_keys.add(_num_key)

            # Accept if either verbatim (lowercased) or whitespace-normalized form
            # appears anywhere in the tool content
            if norm not in combined_normalized and val.lower() not in combined_content:
                flagged.append(val)

    return flagged


def extract_citations(response_text: str) -> list[int]:
    """Extract all [SN] inline citation numbers from a response string.

    Returns a sorted, deduplicated list of citation source numbers.

    >>> extract_citations("Patient has HbA1c 8.2% [S1] and BNP 1842 pg/mL [S3].")
    [1, 3]
    """
    return sorted(set(int(m) for m in _CITATION_RE.findall(response_text)))


def verify_grounding(
    response_text: str,
    source_count: int,
    *,
    check_uncited: bool = True,
    tool_contents: Optional[List[str]] = None,
) -> dict:
    """Verify that LLM response citations are grounded in retrieved sources.

    Checks:
    1. All [SN] references must be within [1, source_count].
    2. Numeric clinical claims (lab values, doses, vitals) must be accompanied
       by at least one [SN] citation in the same sentence / clause.
    3. (When tool_contents provided) Numeric values in the response are
       cross-checked against the raw tool output to detect hallucinations.

    Args:
        response_text:  The synthesized LLM response text.
        source_count:   Number of source chunks that were retrieved (max [Source N]).
        check_uncited:  Whether to flag uncited numeric clinical claims.
        tool_contents:  Optional list of raw tool output strings for hallucination
                        detection (Phase 2.1). When provided, every numeric clinical
                        value in the response is verified against this content.

    Returns:
        {
            "is_grounded": bool,
            "issues": list[str],              # human-readable problem descriptions
            "citation_count": int,            # total unique [SN] citations in response
            "source_count": int,              # sources available
            "citation_coverage": float,       # fraction of sources actually cited
            "hallucinated_numerics": list[str],  # numeric values not found in tool outputs
            "elapsed_ms": float,
        }
    """
    t0 = time.perf_counter()
    issues: list[str] = []

    cited_numbers = extract_citations(response_text)
    citation_count = len(cited_numbers)

    # ── Check 1: invalid citation references ─────────────────────────────────
    if source_count > 0:
        invalid_refs = [n for n in cited_numbers if n < 1 or n > source_count]
        if invalid_refs:
            issues.append(
                f"Invalid citation(s) {invalid_refs} — only {source_count} source(s) available"
            )

    # ── Check 2: uncited numeric clinical claims ──────────────────────────────
    if check_uncited and source_count > 0:
        sentences = _SENTENCE_SPLIT_RE.split(response_text)
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            # Skip sentence if it already has a citation
            if _CITATION_RE.search(sent):
                continue
            for pattern, label in _COMPILED_UNCITED:
                if pattern.search(sent):
                    excerpt = sent[:100]
                    issues.append(f'Uncited {label}: "{excerpt}{"..." if len(sent) > 100 else ""}"')
                    break  # one issue per sentence is sufficient

    # ── Check 3: numeric hallucination detection (Phase 2.1) ─────────────────
    hallucinated_numerics: list[str] = []
    if tool_contents:
        hallucinated_numerics = detect_numeric_hallucinations(response_text, tool_contents)
        if hallucinated_numerics:
            issues.append(
                f"Numeric value(s) not found in source documents: {', '.join(hallucinated_numerics[:5])}"
            )

    coverage = (len(set(cited_numbers)) / source_count) if source_count > 0 else 0.0
    elapsed_ms = (time.perf_counter() - t0) * 1000

    result = {
        "is_grounded": len(issues) == 0,
        "issues": issues,
        "citation_count": citation_count,
        "source_count": source_count,
        "citation_coverage": round(coverage, 3),
        "hallucinated_numerics": hallucinated_numerics,
        "elapsed_ms": round(elapsed_ms, 2),
    }

    if issues:
        logger.warning(
            "Grounding issues (%d) [%.1fms]: %s",
            len(issues),
            elapsed_ms,
            " | ".join(issues[:3]),
        )
    else:
        logger.debug(
            "Grounding OK: %d citation(s), coverage=%.0f%%, hallucinated_numerics=0 [%.1fms]",
            citation_count,
            coverage * 100,
            elapsed_ms,
        )

    return result


def count_sources_in_tool_messages(tool_messages: list) -> int:
    """Count the number of distinct [Source N] blocks across all tool messages.

    Returns the maximum source number seen (= total source count when sources
    are numbered sequentially starting from 1, which is how search() formats
    them).
    """
    combined = " ".join(
        str(getattr(m, "content", "") or "") for m in tool_messages
    )
    nums = {int(n) for n in re.findall(r"\[Source (\d+)", combined)}
    return max(nums) if nums else 0
