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
            # Phase 1.6: require at least 4 characters — 3-char values are too
            # short to be valid MRNs and are often parsing artefacts (e.g. "MRN").
            if mrn and len(mrn) >= 4:
                logger.debug("MRN extracted: %s (pattern: %s)", mrn, pat[:40])
                return mrn

    # If not found in header, try full text with the most specific patterns
    for pat in patterns[:2]:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            mrn = match.group(1).strip()
            if mrn and len(mrn) >= 4:
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


def extract_doc_level_diagnoses_and_meds(
    text: str,
    use_llm_fallback: bool = True,
) -> Dict[str, Any]:
    """
    Document-scoped diagnosis and medication tag extraction.

    Production note:
    - We use high-precision keyword/tag extraction for common diabetes terms.
    - For meds we use a lightweight "dose-pattern" approach.
    - LLM fallback is env-gated by default AND requires use_llm_fallback=True.
      Pass use_llm_fallback=False in unit tests to avoid LLM mocking.
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
    enable_llm = use_llm_fallback and str(os.environ.get("MEDGRAPH_CLINICAL_VALUES_DOC_LLM_FALLBACK", "")).strip().lower() in (
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


# ── Phase 1.4 + 1.6 — New payload field extractors ───────────────────────────

# Mapping from section heading keywords to normalised section_category values.
# Keys are lowercased regex patterns; values are category labels.
_SECTION_CATEGORY_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\ballerg", re.IGNORECASE), "allergies"),
    (re.compile(r"\bmedication|\bcurrent\s+med|\bdischarge\s+med|\bmed\s+list|\bpharmac|\bprescri", re.IGNORECASE), "medications"),
    (re.compile(r"\blab\b|\blaboratory|\bblood\s+(?:test|result|panel)|\bcbc|\bcmp|\bbmp|\bculture|\bcoagulat", re.IGNORECASE), "labs"),
    (re.compile(r"\bvital|\bblood\s+pressure|\bheart\s+rate|\btemperature|\bpulse\s+ox|\bspo2", re.IGNORECASE), "vitals"),
    (re.compile(r"\bassessment|\bimpression|\bproblem\s+list|\bdiagnos", re.IGNORECASE), "assessment"),
    (re.compile(r"\bprocedure|\boperat|\bsurgical|\bintervention|\bimaging|\bradiol|\bct\b|\bmri\b|\becho", re.IGNORECASE), "procedures"),
    (re.compile(r"\bdischarge|\bsummary", re.IGNORECASE), "discharge"),
    (re.compile(r"\bdemographic|\bpatient\s+info|\bidentifi|\bregistration|\badmission\s+info", re.IGNORECASE), "demographics"),
]


def extract_section_category(section_title: str) -> str:
    """Map a section heading to a normalised category label for Qdrant payload filtering.

    Returns one of: allergies | medications | labs | vitals | assessment |
    procedures | discharge | demographics | other.
    """
    if not section_title:
        return "other"
    t = (section_title or "").strip()
    for pattern, category in _SECTION_CATEGORY_RULES:
        if pattern.search(t):
            return category
    return "other"


def extract_clinical_date(text: str) -> Optional[str]:
    """Extract a specific clinical event date from chunk text (e.g., lab draw date).

    This is distinct from encounter_date (admission date on the document header).
    Looks for dates adjacent to clinical measurements, test results, or procedure notes.
    Returns the first found ISO-8601 date string, or None.

    The result should be stored in the chunk's `clinical_date` payload field.
    """
    if not text:
        return None

    # Clinical event date patterns — more specific than encounter_date patterns.
    # These look for dates that appear directly adjacent to clinical data.
    patterns = [
        # "Collected: MM/DD/YYYY", "Collected on: ...", "Collection Date: ..."
        r"(?:collected(?:\s+on)?|collection\s+date|collection\s+time)\s*:?\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
        # "Resulted: MM/DD/YYYY", "Result Date: ..."
        r"(?:resulted?(?:\s+date)?|result\s+date|reported)\s*:?\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
        # "Ordered: MM/DD/YYYY"
        r"(?:ordered(?:\s+on)?|order\s+date)\s*:?\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
        # "Procedure Date: ...", "Study Date: ..."
        r"(?:procedure|study|test|event)\s+date\s*:?\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
        # Date at the start of a SOAP entry: "01/28/2026 —" or "01/28/2026:"
        r"^(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\s*(?:[:\-\u2014]|$)",
    ]

    sample = text[:2000]
    for pat in patterns:
        m = re.search(pat, sample, re.IGNORECASE | re.MULTILINE)
        if m:
            raw = m.group(1).strip()
            iso = _parse_date_candidate(raw)
            if iso:
                return iso

    return None


def _clean_table_cell(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\\|", "|")).strip()


def _split_markdown_row(line: str) -> list[str]:
    raw = (line or "").strip()
    if raw.startswith("|"):
        raw = raw[1:]
    if raw.endswith("|"):
        raw = raw[:-1]
    sentinel = "\u0000PIPE\u0000"
    raw = raw.replace("\\|", sentinel)
    return [_clean_table_cell(c.replace(sentinel, "|")) for c in raw.split("|")]


def _defragment_pdf_table_cells(cells: list[str]) -> str:
    """Best-effort reconstruction for PDF text that was split into table cells."""
    compact = "".join(c for c in cells if c).strip()
    if not compact:
        return ""
    compact = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", compact)
    compact = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", compact)
    compact = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", compact)
    compact = re.sub(r"\b(mg|mcg|g|iu|units?)(?=(oral|orally|intranasal|nasal|daily|spray|bid|tid|qd|qhs))", r"\1 ", compact, flags=re.I)
    compact = re.sub(r"\b(intranasal|nasal)(?=spray)", r"\1 ", compact, flags=re.I)
    compact = re.sub(r"\b(each)(?=nostril)", r"\1 ", compact, flags=re.I)
    compact = re.sub(r"\s+", " ", compact).strip()
    return compact


def _is_markdown_separator(line: str) -> bool:
    cells = _split_markdown_row(line)
    return bool(cells) and all(re.fullmatch(r"[:\-\s]+", c or "") for c in cells)


def _iter_markdown_tables(text: str) -> list[tuple[list[str], list[list[str]]]]:
    """Return markdown pipe tables as (headers, rows)."""
    tables: list[tuple[list[str], list[list[str]]]] = []
    lines = (text or "").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not (line.startswith("|") and "|" in line[1:]):
            i += 1
            continue

        block: list[str] = []
        while i < len(lines):
            current = lines[i].strip()
            if current.startswith("|") and "|" in current[1:]:
                block.append(current)
                i += 1
            else:
                break

        sep_idx = next((idx for idx, row in enumerate(block) if _is_markdown_separator(row)), -1)
        if sep_idx <= 0 or sep_idx >= len(block) - 1:
            continue

        headers = _split_markdown_row(block[sep_idx - 1])
        rows = [_split_markdown_row(row) for row in block[sep_idx + 1:]]
        col_count = max([len(headers)] + [len(r) for r in rows])
        headers = (headers + [""] * col_count)[:col_count]
        normalized_rows = [(r + [""] * col_count)[:col_count] for r in rows]
        tables.append((headers, normalized_rows))
    return tables


def _header_index(headers: list[str], patterns: list[str], *, exclude: list[str] | None = None) -> Optional[int]:
    exclude = exclude or []
    for idx, header in enumerate(headers):
        h = re.sub(r"[^a-z0-9]+", " ", (header or "").lower()).strip()
        if not h:
            continue
        if any(re.search(p, h) for p in exclude):
            continue
        if any(re.search(p, h) for p in patterns):
            return idx
    return None


def _first_nonempty(cells: list[str], *indices: Optional[int]) -> str:
    for idx in indices:
        if idx is None:
            continue
        if 0 <= idx < len(cells) and cells[idx]:
            return cells[idx]
    return ""


def _fact_date_from_row(cells: list[str], headers: list[str], fallback_meta: Dict[str, Any]) -> str:
    date_idx = _header_index(headers, [r"\bdate\b", r"\bcollected\b", r"\bresulted\b", r"\bservice\b"])
    raw = cells[date_idx] if date_idx is not None and date_idx < len(cells) else ""
    if raw:
        iso = _parse_date_candidate(raw)
        if iso:
            return iso
        return raw
    return str(
        fallback_meta.get("clinical_date")
        or fallback_meta.get("encounter_date")
        or fallback_meta.get("document_date")
        or ""
    )


_CLINICAL_TABLE_CONTEXT_RE = re.compile(
    r"\b(lab|laboratory|cbc|cmp|bmp|metabolic|chemistry|hematology|differential|"
    r"thyroid|lipid|vitamin|ige|allergen|a1c|hba1c|glucose|creatinine|egfr|bun|"
    r"electrolyte|vital|blood pressure|heart rate|pulse|temperature|spo2|oxygen|"
    r"height|weight|bmi)\b",
    re.IGNORECASE,
)

_ADMIN_TABLE_CONTEXT_RE = re.compile(
    r"\b(benefit|coverage|copay|co-?insurance|deductible|policy|eligibility|"
    r"allowance|covered|insurance|claim|premium|out[- ]of[- ]pocket|authorization|"
    r"in[- ]network|out[- ]of[- ]network)\b",
    re.IGNORECASE,
)

_CLINICAL_ROW_LABEL_RE = re.compile(
    r"\b(wbc|rbc|hemoglobin|hgb|hematocrit|hct|platelet|plt|mcv|mch|mchc|rdw|"
    r"neutrophil|lymphocyte|monocyte|eosinophil|basophil|sodium|potassium|chloride|"
    r"bicarbonate|co2|bun|creatinine|egfr|glucose|calcium|albumin|bilirubin|ast|alt|"
    r"alkaline|cholesterol|ldl|hdl|triglyceride|tsh|t4|inr|pt|ptt|a1c|hba1c|"
    r"vitamin|ige|allergen|blood pressure|bp|heart rate|pulse|respiratory|spo2|"
    r"temperature|height|weight|bmi)\b",
    re.IGNORECASE,
)

_MEASUREMENT_VALUE_RE = re.compile(
    r"^\s*(?:[<>]=?\s*)?\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?"
    r"(?:\s*(?:%|mg/dl|g/dl|mmol/l|meq/l|iu/ml|iu/l|ku/l|ng/ml|pg/ml|"
    r"ml/min(?:/1\.73m2)?|mmhg|bpm|beats/min|breaths/min|kg/m2|lbs?|kg|cm|"
    r"mg|mcg|g|iu|units?))?"
    r"(?:\s*\([^)]+\))?\s*$",
    re.IGNORECASE,
)


def _is_plausible_clinical_value(value: str) -> bool:
    """Reject PDF table-fragment artefacts while keeping numeric clinical values."""
    v = _clean_table_cell(value or "")
    if not v or len(v) > 80 or not re.search(r"\d", v):
        return False
    if "):" in v or "(:" in v:
        return False
    # Common OCR/parser failure: date fragments such as 3/18/2 appear as a
    # result value when a narrative sentence is mistaken for a table row.
    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{1,4}", v):
        return False
    return bool(_MEASUREMENT_VALUE_RE.fullmatch(v))


def _is_clinical_table_context(headers: list[str], section: str, rows: list[list[str]]) -> bool:
    context = " ".join(headers + [section])
    if _CLINICAL_TABLE_CONTEXT_RE.search(context):
        return True
    sample_labels = " ".join((row[0] if row else "") for row in rows[:12])
    return bool(_CLINICAL_ROW_LABEL_RE.search(sample_labels))


def _is_admin_table_context(headers: list[str], section: str, rows: list[list[str]]) -> bool:
    context = " ".join(headers + [section] + [" ".join(r) for r in rows[:6]])
    return bool(_ADMIN_TABLE_CONTEXT_RE.search(context))


def extract_table_row_facts(chunk_text: str, metadata: Optional[Dict[str, Any]] = None) -> list[Dict[str, Any]]:
    """Extract row-level facts from markdown tables.

    The extractor is intentionally conservative. It creates additive fact
    records for lab/vital-style and administrative/benefit-style tables while
    leaving the original chunk untouched for narrative context.
    """
    metadata = dict(metadata or {})
    section = str(metadata.get("section_title", "") or "")
    facts: list[Dict[str, Any]] = []
    tables = _iter_markdown_tables(chunk_text or "")
    max_facts = int(os.environ.get("MEDGRAPH_MAX_TABLE_FACTS_PER_CHUNK", "40"))

    for table_idx, (headers, rows) in enumerate(tables):
        name_idx = _header_index(headers, [
            r"\btest\b", r"\bcomponent\b", r"\banalyte\b", r"\blab\b", r"\bname\b",
            r"\bitem\b", r"\bservice\b", r"\bbenefit\b", r"\bdescription\b",
            r"\bfield\b",
        ])
        result_idx = _header_index(headers, [
            r"\bresult\b", r"\bvalue\b", r"\breading\b", r"\bobserved\b", r"\bmeasurement\b",
            r"\bcoverage\b", r"\bcovered\b", r"\bamount\b", r"\bdetails?\b",
        ], exclude=[r"\breference\b", r"\bref\b", r"\brange\b", r"\bnormal\b"])
        unit_idx = _header_index(headers, [r"\bunit\b"])
        ref_idx = _header_index(headers, [r"\breference\b", r"\bref\b", r"\brange\b", r"\bnormal\b"])
        flag_idx = _header_index(headers, [r"\bflag\b", r"\babnormal\b", r"\bstatus\b"])
        class_idx = _header_index(headers, [r"\bclass\b", r"\bgrade\b", r"\bseverity\b"])
        interpretation_idx = _header_index(headers, [
            r"\binterpretation\b", r"\binterpret\b", r"\bcategory\b", r"\bcomment\b",
        ])
        copay_idx = _header_index(headers, [r"\bcopay\b", r"\bco pay\b"])
        coins_idx = _header_index(headers, [r"\bcoinsurance\b", r"\bco insurance\b"])
        limit_idx = _header_index(headers, [r"\blimit\b", r"\bmax\b", r"\bmaximum\b"])

        is_admin = _is_admin_table_context(headers, section, rows)
        is_clinical = _is_clinical_table_context(headers, section, rows)
        if not is_admin and not is_clinical:
            continue

        if name_idx is None and len(headers) >= 2:
            name_idx = 0
        if result_idx is None and len(headers) >= 2 and not is_admin:
            candidate = 1 if ref_idx != 1 else None
            result_idx = candidate

        for row_idx, cells in enumerate(rows):
            row_text = " ".join(c for c in cells if c).strip()
            if not row_text:
                continue

            label = _first_nonempty(cells, name_idx)
            value = _first_nonempty(cells, result_idx)
            reference = _first_nonempty(cells, ref_idx)
            unit = _first_nonempty(cells, unit_idx)
            flag = _first_nonempty(cells, flag_idx)
            result_class = _first_nonempty(cells, class_idx)
            interpretation = _first_nonempty(cells, interpretation_idx)
            row_date = _fact_date_from_row(cells, headers, metadata)

            if is_admin:
                benefit_name = label or row_text[:80]
                coverage = value or _first_nonempty(cells, copay_idx, coins_idx, limit_idx)
                if not benefit_name or not coverage:
                    continue
                fact_meta = {
                    "chunk_kind": "table_row",
                    "fact_type": "administrative_benefit",
                    "table_id": f"{metadata.get('document_id', '')}:table:{table_idx}",
                    "row_index": row_idx,
                    "benefit_name": benefit_name,
                    "coverage_value": coverage,
                    "copay": _first_nonempty(cells, copay_idx),
                    "coinsurance": _first_nonempty(cells, coins_idx),
                    "limit_value": _first_nonempty(cells, limit_idx),
                    "row_date": row_date,
                    "source_chunk_index": metadata.get("chunk_index", -1),
                    "section_category": "administrative",
                }
                content = (
                    f"ADMINISTRATIVE_BENEFIT | Benefit: {benefit_name} | Coverage: {coverage} | "
                    f"Copay: {fact_meta['copay']} | Coinsurance: {fact_meta['coinsurance']} | "
                    f"Limit: {fact_meta['limit_value']} | Date: {row_date}"
                )
            else:
                if not label or not value or not _is_plausible_clinical_value(value):
                    continue
                if not (_CLINICAL_ROW_LABEL_RE.search(label) or unit or reference or flag):
                    continue
                fact_meta = {
                    "chunk_kind": "table_row",
                    "fact_type": "lab_or_vital_result",
                    "table_id": f"{metadata.get('document_id', '')}:table:{table_idx}",
                    "row_index": row_idx,
                    "test_name": label,
                    "result_value": value,
                    "unit": unit,
                    "reference_range": reference,
                    "result_flag": flag,
                    "result_class": result_class,
                    "result_interpretation": interpretation,
                    "row_date": row_date,
                    "source_chunk_index": metadata.get("chunk_index", -1),
                    "section_category": "labs",
                }
                content = (
                    f"LAB_OR_VITAL_RESULT | Test: {label} | Value: {value} | Unit: {unit} | "
                    f"Reference range: {reference} | Flag: {flag} | "
                    f"Document class: {result_class} | Interpretation: {interpretation} | "
                    f"Date: {row_date}"
                )

            facts.append({"page_content": content, "metadata": fact_meta})
            if len(facts) >= max_facts:
                return facts

    return facts


def extract_clinical_event_facts(chunk_text: str, metadata: Optional[Dict[str, Any]] = None) -> list[Dict[str, Any]]:
    """Extract conservative event facts for timeline/change retrieval."""
    metadata = dict(metadata or {})
    text = chunk_text or ""
    facts: list[Dict[str, Any]] = []
    max_events = int(os.environ.get("MEDGRAPH_MAX_EVENT_FACTS_PER_CHUNK", "8"))
    section_category = metadata.get("section_category") or extract_section_category(metadata.get("section_title", ""))
    doc_type = str(metadata.get("document_type", "") or "").lower()
    if section_category in {"demographics", "administrative"} or doc_type in {
        "billing",
        "consent_form",
        "insurance",
        "benefits",
        "physical_form",
    }:
        return []
    event_date = str(
        metadata.get("clinical_date")
        or metadata.get("encounter_date")
        or metadata.get("document_date")
        or ""
    )

    med_name = r"[A-Za-z][A-Za-z0-9/\-]{2,40}(?:\s+[A-Za-z][A-Za-z0-9/\-]{2,40}){0,2}"
    dose_signal = (
        r"(?:\d+(?:\.\d+)?\s*(?:mcg/spray|international\s+units?|mg|mcg|g|iu|"
        r"units?|sprays?|ml)|po|oral(?:ly)?|intranasal|nasal|iv|bid|tid|qhs|"
        r"daily|once\s+daily|twice\s+daily|dose|dosage)"
    )
    patterns: list[tuple[str, str, re.Pattern]] = [
        ("medication_change", "started", re.compile(rf"\b(start(?:ed)?|initiated|added|began)\b.{{0,80}}?\b({med_name})\b.{{0,70}}?\b({dose_signal})\b", re.I)),
        ("medication_change", "stopped", re.compile(rf"\b(stop(?:ped)?|discontinued|dc(?:'d)?|held|withheld)\b.{{0,80}}?\b({med_name})\b.{{0,70}}?\b({dose_signal})\b", re.I)),
        ("medication_change", "changed", re.compile(rf"\b(increase(?:d)?|decrease(?:d)?|reduce(?:d)?|titrate(?:d)?|change(?:d)?|switch(?:ed)?)\b.{{0,100}}?\b({med_name})\b.{{0,90}}?\b({dose_signal})\b", re.I)),
        ("treatment_action", "initiated", re.compile(r"\b(start(?:ed)?|initiated|began|ordered|recommended|referred|scheduled)\b.{0,140}\b(therapy|treatment|follow[- ]?up|consult|referral|monitor|plan|education)\b", re.I)),
        ("measurement", "documented", re.compile(r"\b(bp|blood pressure|a1c|hba1c|creatinine|egfr|glucose|ldl|inr|weight)\b.{0,80}\b\d+(?:\.\d+)?(?:/\d{2,3})?\b", re.I)),
    ]

    seen: set[str] = set()
    candidate_lines: list[tuple[str, bool]] = []
    for raw_line in text.splitlines():
        raw = raw_line.strip()
        if raw.startswith("|") and "|" in raw[1:]:
            if _is_markdown_separator(raw):
                continue
            cells = [c for c in _split_markdown_row(raw) if c]
            if cells:
                reconstructed = _defragment_pdf_table_cells(cells)
                candidate_lines.append((reconstructed or " ".join(cells), True))
            continue
        candidate_lines.append((raw_line, False))

    for raw_line, from_table in candidate_lines:
        line = re.sub(r"\s+", " ", raw_line).strip(" -\t")
        if len(line) < 12:
            continue
        for event_type, action, pattern in patterns:
            if from_table and event_type == "measurement":
                continue
            match = pattern.search(line)
            if not match:
                continue
            event_name = line[:120]
            medication_name = ""
            event_value = ""
            if event_type == "medication_change":
                medication_name = (match.group(2) if len(match.groups()) >= 2 else "").strip()
                event_value = (match.group(3) if len(match.groups()) >= 3 else "").strip()
                medication_name = re.sub(r"\s+(?:from|to|dose)$", "", medication_name, flags=re.I).strip()
                action_section = bool(re.search(
                    r"\b(plan|assessment|recommendation|orders?|instructions?|pharmacist|prescri)\b",
                    str(metadata.get("section_title", "") or ""),
                    re.IGNORECASE,
                ))
                historical_start = bool(re.search(
                    r"\b(as prescribed|already|previously|prior|home medication|current(?:ly)?|"
                    r"has been|have been|has started|had started|confirmed|reported starting|since last)\b",
                    line,
                    re.IGNORECASE,
                ))
                if action == "started" and historical_start and not action_section:
                    continue
                if action == "changed":
                    target = re.search(
                        r"\bto\s+(\d+(?:\.\d+)?\s*(?:mcg/spray|international\s+units?|mg|mcg|g|iu|units?|ml))\b",
                        line,
                        re.IGNORECASE,
                    )
                    if target:
                        event_value = target.group(1).strip()
            if event_type == "medication_change":
                norm_name = re.sub(r"[^a-z0-9]+", "", medication_name.lower())
                norm_value = re.sub(r"\s+", "", event_value.lower())
                key = f"{event_type}|{action}|{norm_name}|{norm_value}|{event_date}"
            else:
                key = f"{event_type}|{action}|{line.lower()[:160]}"
            if key in seen:
                continue
            seen.add(key)
            fact_meta = {
                "chunk_kind": "clinical_event",
                "fact_type": event_type,
                "event_type": event_type,
                "event_action": action,
                "event_name": event_name,
                "medication_name": medication_name,
                "event_value": event_value,
                "event_date": event_date,
                "date_source": "clinical_date" if metadata.get("clinical_date") else "encounter_or_document_date",
                "source_chunk_index": metadata.get("chunk_index", -1),
                "section_category": metadata.get("section_category") or extract_section_category(metadata.get("section_title", "")),
            }
            detail = f" | Medication: {medication_name} | Value: {event_value}" if event_type == "medication_change" else ""
            content = f"CLINICAL_EVENT | Type: {event_type} | Action: {action}{detail} | Date: {event_date} | Evidence: {line}"
            facts.append({"page_content": content, "metadata": fact_meta})
            if len(facts) >= max_events:
                return facts
            break
    return facts


def extract_facility_name(text: str) -> str:
    """Best-effort extraction of the healthcare facility / hospital name.

    Supports multi-facility deployments where different buildings or departments
    generate separate documents. Returns empty string if not found (never errors).
    """
    if not text:
        return ""
    sample = text[:4000]
    patterns = [
        # "Hospital: St. Mary Medical Center"
        r"(?:hospital|facility|institution|health\s+system|medical\s+center|clinic)\s*:\s*([A-Za-z][A-Za-z0-9 &\-,.']{4,80})",
        # "Discharge from: <facility>"
        r"(?:discharge\s+from|admitted\s+to|treated\s+at)\s*:?\s*([A-Za-z][A-Za-z0-9 &\-,.']{4,80})",
        # First line of documents often starts with the facility name on its own line,
        # followed by an address or phone number.
        r"^([A-Z][A-Za-z0-9 &\-,.']{4,60})\s*\n(?:\d{1,5}|\([0-9]{3}\))",
    ]
    for pat in patterns:
        m = re.search(pat, sample, re.IGNORECASE | re.MULTILINE)
        if m:
            name = m.group(1).strip().rstrip(",.")
            if name and len(name) > 4:
                return name
    return ""
