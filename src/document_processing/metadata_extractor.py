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
from typing import Optional

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
