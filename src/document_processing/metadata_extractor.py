"""
metadata_extractor.py — Robust multi-pattern metadata extraction for EHR documents.

Extracts:
  - Patient MRN (Medical Record Number) — 6+ patterns
  - Patient name — 5+ patterns
  - Document type classification — clinical_note, lab_report, radiology, etc.
  - Encounter/service date

All extractors are format-agnostic and handle variations across EHR systems.
"""

import re
import logging
from datetime import datetime
import os
from typing import Optional, TypedDict, List, Dict, Any

logger = logging.getLogger(__name__)


def extract_mrn(text: str) -> str:
    """Extract MRN (Medical Record Number) from document text.

    Handles multiple EHR formats:
      - "MRN: MRN-2026-004782"
      - "Medical Record Number: 123456"
      - "Med Rec No.: ABC-12345"
      - "Record Number: 7890"
      - "MRN is MRN-2026-004782"
      - "MR#: 123456"
      - Tabular: "MRN | 123456" or "MRN  123456"
    """
    # Try first 3000 chars (header area) first, then full text
    search_text = text[:3000]

    patterns = [
        # Standard MRN labels with colon or "is"
        r"\b(?:mrn|medical\s+record\s+(?:number|no\.?|#))\s*(?:is\s*|:\s*|#?\s*)([A-Z0-9][\w\-]+)",
        # Abbreviated forms
        r"\b(?:med\.?\s*rec\.?\s*(?:no\.?|number|#)|mr\s*#|mr\s*no\.?)\s*:?\s*([A-Z0-9][\w\-]+)",
        # Record Number
        r"\b(?:record\s+(?:number|no\.?|#))\s*:?\s*([A-Z0-9][\w\-]+)",
        # Tabular format: "MRN  value" or "MRN | value"
        r"\bMRN\s*\|?\s+([A-Z0-9][\w\-]{3,})",
        # Standalone MRN pattern (MRN-YYYY-NNNNNN)
        r"\b(MRN-\d{4}-\d{4,})\b",
    ]

    for pat in patterns:
        match = re.search(pat, search_text, re.IGNORECASE)
        if match:
            mrn = match.group(1).strip()
            if mrn and len(mrn) >= 3:
                logger.debug("MRN extracted: %s (pattern: %s)", mrn, pat[:40])
                return mrn

    # If not found in header, try full text with the most specific patterns
    for pat in patterns[:2]:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            mrn = match.group(1).strip()
            if mrn and len(mrn) >= 3:
                return mrn

    return ""


def extract_patient_name(text: str) -> str:
    """Extract patient name from document text.

    Handles multiple EHR formats:
      - "Patient Name: Robert James Whitfield"
      - "Full Name: John Doe"
      - "Patient: Jane Smith"
      - "Pt. Name: Bob Wilson"
      - "Name: Alice Johnson"
      - Structured headers with "Patient" on a label line
    """
    search_text = text[:3000]

    patterns = [
        # "Patient Name:" / "Full Name:" / "Pt. Name:" / "Pt Name:"
        r"(?:patient\s+name|full\s+name|pt\.?\s*name)\s*:\s*(.+?)(?:\n|$)",
        # "Name:" at start of line (but not "File Name:", "Drug Name:", etc.)
        r"(?:^|\n)\s*(?<!file\s)(?<!drug\s)(?<!brand\s)name\s*:\s*([A-Za-z][A-Za-z .\-,]+?)(?:\n|$)",
        # "Patient: Name" (inpatient header)
        r"(?:^|\n)\s*patient\s*:\s*([A-Za-z][A-Za-z .\-,]+?)(?:\n|$)",
        # "Pt:" or "Pt.:" followed by name
        r"(?:^|\n)\s*pt\.?\s*:\s*([A-Za-z][A-Za-z .\-,]+?)(?:\n|$)",
        # Markdown bold: **Patient Name:** or **Name:**
        r"\*\*(?:patient\s+name|name|patient)\*\*\s*:?\s*([A-Za-z][A-Za-z .\-,]+?)(?:\n|$)",
    ]

    for pat in patterns:
        match = re.search(pat, search_text, re.IGNORECASE)
        if match:
            name = match.group(1).strip().rstrip(",.")
            # Validate: must be 2+ chars, start with letter, contain at least 2 words
            if name and len(name) > 2 and name[0].isalpha():
                words = [w for w in name.split() if len(w) >= 2 and w[0].isalpha()]
                if len(words) >= 2:
                    logger.debug("Patient name extracted: %s", name)
                    return name

    return ""


# Document type classification keywords
_DOC_TYPE_PATTERNS = {
    "lab_report": [
        r"\b(?:laboratory|lab)\s+(?:report|results|values|test)",
        r"\bcomplete\s+blood\s+count\b",
        r"\bcomprehensive\s+metabolic\s+panel\b",
        r"\blab\s+(?:order|panel)\b",
    ],
    "radiology": [
        r"\b(?:radiology|imaging)\s+report\b",
        r"\b(?:x-ray|xray|ct\s+scan|mri|ultrasound|echocardiogram)\s+(?:report|findings)\b",
        r"\bimpression\s*:\s*\n",
        r"\bradiolog",
    ],
    "discharge_summary": [
        r"\bdischarge\s+summar",
        r"\bdischarge\s+(?:instructions|plan|disposition)\b",
    ],
    "operative_note": [
        r"\b(?:operative|surgical|procedure)\s+(?:note|report)\b",
        r"\bpreoperative\s+diagnosis\b",
        r"\bpostoperative\s+diagnosis\b",
    ],
    "progress_note": [
        r"\b(?:progress|daily)\s+note",
        r"\bSOAP\s+note\b",
        r"\bsubjective\s*:\s*\n",
        r"\bassessment\s+(?:and|&)\s+plan\b",
    ],
    "clinical_note": [
        r"\bclinical\s+note\b",
        r"\b(?:history\s+and\s+physical|h\s*&\s*p)\b",
        r"\bchief\s+complaint\b",
        r"\bhistory\s+of\s+present\s+illness\b",
    ],
    "billing": [
        r"\b(?:billing|insurance|claim)\s+(?:statement|summary|report)\b",
        r"\bcpt\s+code\b.*\bcharge\b",
        r"\btotal\s+charges\b",
    ],
}


def classify_document_type(text: str) -> str:
    """Classify the document type based on content analysis.

    Returns one of: lab_report, radiology, discharge_summary, operative_note,
    progress_note, clinical_note, billing, or general.
    """
    # Check first 5000 chars for type indicators
    sample = text[:5000].lower()
    scores = {}

    for doc_type, patterns in _DOC_TYPE_PATTERNS.items():
        score = 0
        for pat in patterns:
            matches = re.findall(pat, sample, re.IGNORECASE)
            score += len(matches)
        if score > 0:
            scores[doc_type] = score

    if scores:
        best_type = max(scores, key=scores.get)
        logger.debug("Document classified as: %s (score=%d)", best_type, scores[best_type])
        return best_type

    return "general"


def extract_encounter_date(text: str) -> Optional[str]:
    """Extract the primary encounter/service date from the document.

    Looks for: admission date, date of service, encounter date, visit date.
    Returns date string as found in text, or None.
    """
    search_text = text[:3000]

    patterns = [
        # "Admission Date: 01/15/2026" or "Date of Admission: ..."
        r"(?:admission|admit)\s+date\s*:\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
        # "Date of Service: ..."
        r"date\s+of\s+service\s*:\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
        # "Encounter Date: ..."
        r"encounter\s+date\s*:\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
        # "Visit Date: ..."
        r"visit\s+date\s*:\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
        # "Date: MM/DD/YYYY" at start of line
        r"(?:^|\n)\s*date\s*:\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
    ]

    for pat in patterns:
        match = re.search(pat, search_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


class ClinicalValues(TypedDict, total=False):
    """
    Typed container for structured clinical values stored in Qdrant payloads.

    Note: we keep this intentionally small and extensible; missing values are
    omitted (not set to None) so Qdrant payload filters can behave cleanly.
    """

    # Labs (chunk-scoped)
    hba1c: float
    hba1c_date: str  # ISO-8601 date (YYYY-MM-DD)

    # Diagnosis / medication tags (document-scoped, propagated to chunks)
    diagnoses_text: List[str]
    active_medications: List[str]

    # Optional patient-level metadata (document-scoped)
    patient_location: str


_NEGATION_CUES = (
    "no",
    "not",
    "denies",
    "denied",
    "without",
    "negative for",
    "rule out",
    "r/o",
)


def _chunk_confidence_negated(context: str) -> bool:
    """Heuristic negation detection around the matched lab text."""
    c = (context or "").lower()
    # Keep this fast: token-based checks avoid expensive NLP tooling.
    return any(kw in c for kw in _NEGATION_CUES)


def _parse_date_candidate(raw: str) -> Optional[str]:
    """
    Parse common clinical date formats into ISO-8601 (YYYY-MM-DD).

    Supported examples:
      - 01/15/2026
      - 1/5/26
      - 01-15-2026
    """
    s = (raw or "").strip()
    if not s:
        return None

    # Normalize separators
    s_norm = re.sub(r"[.]", "/", s)
    s_norm = s_norm.replace("-", "/")
    parts = s_norm.split("/")
    if len(parts) != 3:
        return None

    try:
        m, d, y = parts
        m_i = int(m)
        d_i = int(d)
        y_i = int(y)
        if y_i < 100:
            # Assume 2-digit years are in 2000s for EHR-style notes.
            y_i += 2000
        dt = datetime(y_i, m_i, d_i)
        # Qdrant DATETIME payloads are happiest with an ISO timestamp.
        return dt.isoformat()
    except Exception:
        return None


def extract_patient_location(text: str) -> str:
    """
    Best-effort extraction of patient location/facility field.

    This is primarily to support cohort UIs (location cards) where PDFs include
    a "Location" field. If not found, returns an empty string.
    """
    if not text:
        return ""
    sample = text[:8000]
    # Examples that appear in exports:
    #   "Location: Get Well"
    #   "Get Well Location"
    patterns = [
        r"(?:location|unit)\s*:\s*([A-Za-z][A-Za-z0-9 /&\-]{2,80})",
        r"(?:get\s*well\s*location)\s*[:\-]?\s*([A-Za-z][A-Za-z0-9 /&\-]{2,80})",
    ]
    for pat in patterns:
        m = re.search(pat, sample, re.IGNORECASE)
        if m:
            loc = m.group(1).strip()
            if loc:
                return loc
    return ""


def extract_doc_level_diagnoses_and_meds(text: str) -> Dict[str, Any]:
    """
    Document-scoped diagnosis and medication tag extraction.

    Production note:
    - We use high-precision keyword/tag extraction for common diabetes terms.
    - For meds we use a lightweight "dose-pattern" approach.
    - LLM fallback is intentionally disabled by default (env-gated) to keep
      indexing deterministic and cost-bounded.
    """
    if not text:
        return {}

    lower = text.lower()

    diagnoses: list[str] = []

    diabetes_synonyms = [
        "diabetes",
        "t2dm",
        "type 2 diabetes",
        "type ii diabetes",
        "type 1 diabetes",
        "t1dm",
        "hba1c",
    ]
    if any(term in lower for term in diabetes_synonyms):
        # Keep both generic + specific where possible.
        if "type 2 diabetes" in lower or "t2dm" in lower:
            diagnoses.extend(["diabetes", "type 2 diabetes"])
        elif "type 1 diabetes" in lower or "t1dm" in lower:
            diagnoses.extend(["diabetes", "type 1 diabetes"])
        else:
            diagnoses.append("diabetes")

    # Lightweight medication extraction: capture "Name <number> mg/mcg/g"
    # from sections that look like medication lists.
    meds: list[str] = []

    medication_section_hint = bool(
        re.search(r"(medication|meds|rx|discharge medication|discharge rx|current medication|home medication)", lower)
    )
    dose_re = re.compile(
        r"\b([A-Za-z][A-Za-z0-9\-_]{1,30})\s*(\d+(?:\.\d+)?)\s*(mg|mcg|g)\b",
        flags=re.IGNORECASE,
    )
    if medication_section_hint:
        for m in dose_re.finditer(text):
            name = m.group(1).strip().lower()
            if name and name not in meds:
                meds.append(name)

    # Deduplicate while preserving order
    diagnoses = list(dict.fromkeys(diagnoses))
    meds = list(dict.fromkeys(meds))

    # Optional LLM fallback (document-level only).
    # Gate it behind an env var so indexing stays deterministic by default.
    # Example env var: MEDGRAPH_CLINICAL_VALUES_DOC_LLM_FALLBACK=1
    enable_llm = str(os.environ.get("MEDGRAPH_CLINICAL_VALUES_DOC_LLM_FALLBACK", "")).strip().lower() in (
        "1",
        "true",
        "yes",
    )
    has_any_clinical_hints = bool(
        re.search(r"(assessment|impression|diagnos|problem|medication|meds|rx|discharge medication)", lower)
    )
    low_confidence = enable_llm and has_any_clinical_hints and (not diagnoses or not meds)

    if low_confidence:
        try:
            # Lazy imports so local indexing doesn't require LLM deps at import time.
            import json as _json
            from agent_graph.load_tools_config import (
                LoadToolsConfig,
                get_google_api_key,
                get_active_llm_provider,
            )
            from langchain_openai import ChatOpenAI
            from langchain_google_genai import ChatGoogleGenerativeAI

            tools_cfg = LoadToolsConfig()
            provider = get_active_llm_provider(getattr(tools_cfg, "llm_provider", "openai"))
            model = getattr(tools_cfg, "clinical_notes_rag_llm", "gpt-4.1-mini")
            if provider == "google" or str(model).startswith("gemini"):
                llm = ChatGoogleGenerativeAI(
                    model=model,
                    google_api_key=get_google_api_key(),
                    temperature=0.0,
                    max_output_tokens=2048,
                    timeout=90,
                )
            else:
                llm = ChatOpenAI(
                    model=model,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    temperature=0.0,
                    max_tokens=2048,
                    request_timeout=90,
                )

            prompt = (
                "Extract the following from this medical document text. "
                "Return STRICT JSON with keys: diagnoses_text (list[str]), "
                "active_medications (list[str]), patient_location (string or empty).\n"
                "Rules:\n"
                "- Only include items explicitly documented in the text.\n"
                "- For diagnoses, include diabetes-related terms when present; normalize to simple phrases.\n"
                "- For medications, include medication names only (no doses).\n"
                "- If a key is not found, return an empty list/string.\n\n"
                f"DOCUMENT TEXT (truncated):\n{text[:12000]}"
            )
            raw = llm.invoke(prompt).content

            # Best-effort JSON extraction (handle minor formatting issues).
            t = (raw or "").strip()
            t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE).strip()
            t = re.sub(r"\s*```$", "", t).strip()
            m = re.search(r"\{[\s\S]*\}", t)
            if m:
                t = m.group(0)
            obj = _json.loads(t)
            if isinstance(obj, dict):
                if not diagnoses and isinstance(obj.get("diagnoses_text"), list):
                    diagnoses = [str(x).strip().lower() for x in obj["diagnoses_text"] if str(x).strip()]
                if not meds and isinstance(obj.get("active_medications"), list):
                    meds = [str(x).strip().lower() for x in obj["active_medications"] if str(x).strip()]

        except Exception:
            # Any LLM issues should never break indexing; we keep heuristic results.
            pass

    out: Dict[str, Any] = {}
    if diagnoses:
        out["diagnoses_text"] = diagnoses
    if meds:
        out["active_medications"] = meds
    loc = extract_patient_location(text)
    if loc:
        out["patient_location"] = loc
    return out


def _extract_date_candidates_with_pos(text: str) -> list[tuple[int, str]]:
    date_re = re.compile(
        r"\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b"
    )
    candidates: list[tuple[int, str]] = []
    for m in date_re.finditer(text):
        raw = m.group(1)
        candidates.append((m.start(1), raw))
    return candidates


def extract_hba1c_from_chunk_text(text: str) -> Dict[str, Any]:
    """
    Extract HbA1c value and nearest date from a single chunk of text.

    Returns an empty dict if HbA1c evidence is not present.
    """
    if not text:
        return {}

    # Match: HbA1c / A1C / Hemoglobin A1C with a numeric value.
    # Capture value and match position to locate nearest date.
    hba1c_re = re.compile(
        r"\b(?:hba1c|a1c|hemoglobin\s*a1c|glycated\s*hemoglobin)\b\s*[:=]?\s*"
        r"(?P<val>\d+(?:\.\d+)?)\s*%?",
        flags=re.IGNORECASE,
    )

    date_candidates = _extract_date_candidates_with_pos(text)
    if not date_candidates:
        # We still might extract value (some PDFs omit dates), but cohort
        # queries requiring date ranges will naturally exclude those chunks.
        pass

    matches: list[tuple[float, Optional[str], int]] = []
    for m in hba1c_re.finditer(text):
        raw_val = m.group("val")
        try:
            val = float(raw_val)
        except Exception:
            continue

        # Negation window: if the preceding context indicates negation, skip.
        match_start = m.start()
        neg_window_start = max(0, match_start - 100)
        context_before = text[neg_window_start:match_start]
        if _chunk_confidence_negated(context_before):
            continue

        nearest_iso: Optional[str] = None
        nearest_dist: Optional[int] = None
        if date_candidates:
            for date_pos, raw_date in date_candidates:
                iso = _parse_date_candidate(raw_date)
                if not iso:
                    continue
                d = abs(date_pos - match_start)
                if nearest_dist is None or d < nearest_dist:
                    nearest_dist = d
                    nearest_iso = iso

        matches.append((val, nearest_iso, match_start))

    if not matches:
        return {}

    # Prefer highest value; if tie, prefer with a date.
    best = sorted(matches, key=lambda t: (t[0], bool(t[1]) is True), reverse=True)[0]
    best_val, best_date, _ = best

    out: Dict[str, Any] = {"hba1c": float(best_val)}
    if best_date:
        out["hba1c_date"] = best_date
    return out


def extract_clinical_values_for_chunk(chunk_text: str, doc_text: Optional[str] = None) -> ClinicalValues:
    """
    Extract clinically relevant payload fields for Qdrant.

    Tiered logic (no heavy local ML required here):
    - Chunk scope: HbA1c + nearest date (rule-based extraction).
    - Document scope (optional): diagnoses_text, active_medications, patient_location.
    """
    out: Dict[str, Any] = {}
    out.update(extract_hba1c_from_chunk_text(chunk_text))

    if doc_text:
        out.update(extract_doc_level_diagnoses_and_meds(doc_text))

    return out  # type: ignore[return-value]
