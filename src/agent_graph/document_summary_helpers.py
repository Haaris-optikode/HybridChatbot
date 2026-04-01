import json
import re
from typing import List, Tuple


def safe_json_loads(text: str) -> dict:
    """Best-effort JSON parse: trims fences, extracts first {...} block."""
    if not text:
        return {}
    t = text.strip()
    # Strip markdown fences if present
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"\s*```$", "", t).strip()
    # If there's leading commentary, extract first JSON object
    if not t.startswith("{"):
        m = re.search(r"\{[\s\S]*\}", t)
        if m:
            t = m.group(0)
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def norm_key(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def merge_lists_unique(items: list, key_fields: List[str]) -> list:
    """Merge a list of dict items using a composite normalized key."""
    out = []
    seen = set()
    for it in items or []:
        if not isinstance(it, dict):
            continue
        key_parts = []
        for f in key_fields:
            key_parts.append(norm_key(str(it.get(f, ""))))
        key = "||".join(key_parts)
        if not key.strip("|"):
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def merge_doc_extractions(extractions: list[dict]) -> dict:
    """Deterministically merge per-group extraction JSONs into one."""
    merged: dict = {
        "document": {},
        "patient": {},
        "allergies": [],
        "problems": [],
        "medications": [],
        "labs": [],
        "vitals": [],
        "imaging": [],
        "procedures": [],
        "hospital_course": [],
        "disposition_followup": [],
        "other_key_findings": [],
        "missing_or_unclear": [],
    }

    for ex in extractions or []:
        if not isinstance(ex, dict):
            continue
        merged["document"] = {**merged.get("document", {}), **(ex.get("document") or {})}
        merged["patient"] = {**merged.get("patient", {}), **(ex.get("patient") or {})}

        merged["allergies"].extend(ex.get("allergies") or [])
        merged["problems"].extend(ex.get("problems") or [])
        merged["medications"].extend(ex.get("medications") or [])
        merged["labs"].extend(ex.get("labs") or [])
        merged["vitals"].extend(ex.get("vitals") or [])
        merged["imaging"].extend(ex.get("imaging") or [])
        merged["procedures"].extend(ex.get("procedures") or [])
        merged["hospital_course"].extend(ex.get("hospital_course") or [])
        merged["disposition_followup"].extend(ex.get("disposition_followup") or [])
        merged["other_key_findings"].extend(ex.get("other_key_findings") or [])
        merged["missing_or_unclear"].extend(ex.get("missing_or_unclear") or [])

    # Deduplicate with conservative keys (avoid accidental drops)
    merged["allergies"] = merge_lists_unique(merged["allergies"], ["substance", "reaction", "severity"])
    merged["problems"] = merge_lists_unique(merged["problems"], ["problem", "status"])
    merged["medications"] = merge_lists_unique(merged["medications"], ["name", "dose", "route", "frequency", "context"])
    merged["labs"] = merge_lists_unique(merged["labs"], ["test", "value", "unit", "date_time"])
    merged["vitals"] = merge_lists_unique(merged["vitals"], ["type", "value", "unit", "date_time"])
    merged["imaging"] = merge_lists_unique(merged["imaging"], ["study", "date", "impression"])
    merged["procedures"] = merge_lists_unique(merged["procedures"], ["procedure", "date", "result"])
    merged["hospital_course"] = merge_lists_unique(merged["hospital_course"], ["date", "event"])
    merged["disposition_followup"] = merge_lists_unique(merged["disposition_followup"], ["item", "date", "details"])
    merged["other_key_findings"] = merge_lists_unique(merged["other_key_findings"], ["finding"])
    merged["missing_or_unclear"] = merge_lists_unique(merged["missing_or_unclear"], ["topic", "note"])

    return merged


def build_sources_for_groups(chunks: List[Tuple[int, str, dict]], max_group_chars: int = 22_000) -> List[str]:
    """Create grouped source strings with globally stable Source IDs."""
    groups: list[list[tuple[int, str, dict]]] = []
    cur: list[tuple[int, str, dict]] = []
    cur_chars = 0
    for global_idx, txt, meta in chunks:
        t = txt or ""
        if cur and cur_chars + len(t) > max_group_chars:
            groups.append(cur)
            cur = []
            cur_chars = 0
        cur.append((int(global_idx), t, meta))
        cur_chars += len(t)
    if cur:
        groups.append(cur)

    rendered = []
    for g in groups:
        parts = []
        for global_idx, txt, meta in g:
            sid = f"S{global_idx}"
            sec = (meta or {}).get("section_title", "Section")
            page = (meta or {}).get("page_number", "?")
            parts.append(f"[{sid} | Section: {sec} | Page {page}]\n{txt}")
        rendered.append("\n\n---\n\n".join(parts))
    return rendered

