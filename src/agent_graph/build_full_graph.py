import logging


from langgraph.graph import StateGraph, START
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage as _ToolMessage, AIMessage as _AIMessage
from agent_graph.tool_clinical_notes_rag import (
    lookup_clinical_notes,
    lookup_patient_orders,
    cohort_patient_search,
    summarize_patient_record,
    summarize_uploaded_document,
    list_available_patients,
    list_uploaded_documents,
    get_rag_tool_instance,
)
from agent_graph.tool_tavily_search import load_tavily_search_tool
from agent_graph.load_tools_config import LoadToolsConfig, get_google_api_key, get_active_llm_provider
from agent_graph.agent_backend import State, BasicToolNode, route_tools, plot_agent_schema
import os
import json as _json
import re as _re
from datetime import datetime as _dt
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

TOOLS_CFG = LoadToolsConfig()
_UNSCOPED_PATIENT_QUERY_MSG = (
    "I need the patient name or MRN before searching clinical notes. "
    "Please specify which patient you want me to review."
)
_STRICT_GROUNDING_NOT_FOUND_MSG = "Not found after targeted review of the retrieved record."


def _safe_json_object(text: str) -> dict:
    """Best-effort JSON object parse."""
    if not text:
        return {}
    t = str(text).strip()
    t = _re.sub(r"^```(?:json)?\s*", "", t, flags=_re.IGNORECASE).strip()
    t = _re.sub(r"\s*```$", "", t).strip()
    if not t.startswith("{"):
        m = _re.search(r"\{[\s\S]*\}", t)
        if m:
            t = m.group(0)
    try:
        obj = _json.loads(t)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _normalize_match_text(text: str) -> str:
    """Normalize text for quote containment checks."""
    return _re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _extract_query_scope(query: str) -> dict:
    """Derive a conservative answer scope from the user's question."""
    q = (query or "").strip().lower()
    requested_topics: list[str] = []
    focus = "general"
    exclude_sections: list[str] = []

    if (
        "past medical history" in q
        or _re.search(r"\bpmh\b", q)
        or (
            "medical history" in q
            and "family history" not in q
            and "surgical history" not in q
            and "social history" not in q
        )
    ):
        focus = "past_medical_history"
        requested_topics.append("other")
        exclude_sections.extend([
            "family history",
            "past surgical history",
            "surgical history",
            "social history",
            "preventive care",
            "immunization",
            "screening",
        ])
    elif "family history" in q or _re.search(r"\bfh\b", q):
        focus = "family_history"
        requested_topics.append("other")
        exclude_sections.extend([
            "past medical history",
            "past surgical history",
            "surgical history",
            "social history",
            "preventive care",
        ])
    elif "past surgical history" in q or "surgical history" in q:
        focus = "surgical_history"
        requested_topics.append("other")
        exclude_sections.extend([
            "past medical history",
            "family history",
            "social history",
            "preventive care",
        ])

    if any(k in q for k in ["follow-up", "follow up", "appointment", "after discharge", "discharge follow"]):
        requested_topics.append("follow_up")
    if any(k in q for k in ["long-term", "long term", "planned", "future", "repeat echo", "repeat echocardiogram", "icd", "cardiac decision"]):
        requested_topics.append("long_term_plan")
    if any(k in q for k in ["medication", "medications", "meds", "drug", "prescription"]):
        requested_topics.append("medications")
    if any(k in q for k in ["lab", "labs", "bloodwork", "creatinine", "glucose", "bnp", "a1c", "hba1c"]):
        requested_topics.append("labs")
    if focus == "general" and any(k in q for k in ["diagnosis", "diagnoses", "problem list", "condition", "conditions", "history"]):
        requested_topics.append("diagnoses")
    if any(k in q for k in ["procedure", "procedures", "surgery", "operation", "thoracentesis", "catheterization"]):
        requested_topics.append("procedures")
    if any(k in q for k in ["imaging", "echo", "echocardiogram", "ct", "mri", "x-ray", "ultrasound", "ecg", "ekg"]):
        requested_topics.append("imaging")
    if any(k in q for k in ["instruction", "instructions", "diet", "activity", "fluid restriction", "weights"]):
        requested_topics.append("instructions")
    if any(k in q for k in ["home health", "care logistics", "pt", "ot", "transport", "services"]):
        requested_topics.append("care_logistics")

    requested_topics = list(dict.fromkeys(requested_topics or ["other"]))

    # Guard against over-answering for common discharge/follow-up asks.
    if "follow_up" in requested_topics and "instructions" not in requested_topics:
        exclude_sections.extend(["discharge instructions", "activity", "diet"])
    if "follow_up" in requested_topics and "care_logistics" not in requested_topics:
        exclude_sections.extend(["home health", "physical therapy", "occupational therapy"])

    return {
        "requested_topics": requested_topics,
        "focus": focus,
        "exclude_sections": [s.lower() for s in exclude_sections],
        "query": query or "",
    }


def _recent_assistant_texts(state: State) -> list[str]:
    """Return assistant messages in reverse chronological order."""
    out: list[str] = []
    for m in reversed(state.get("messages", [])):
        if getattr(m, "type", "") == "ai" or m.__class__.__name__ == "AIMessage":
            content = getattr(m, "content", "") or ""
            if content:
                out.append(content)
    return out


def _extract_query_scope_with_context(state: State, query: str) -> dict:
    """Resolve scope using the current turn plus recent conversational context."""
    scope = _extract_query_scope(query)
    if scope.get("focus") != "general":
        return scope

    q = (query or "").strip().lower()
    recent_humans = _recent_human_texts(state)
    recent_assistant = _recent_assistant_texts(state)
    context_text = "\n".join((recent_humans[:3] + recent_assistant[:2]))
    context_lower = context_text.lower()

    history_followup_markers = [
        "what about",
        "you missed",
        "missed that",
        "anything else",
        "complete",
        "full",
        "entire",
        "all of it",
    ]
    pmh_context = (
        "past medical history" in context_lower
        or "the past medical history includes:" in context_lower
        or "medical history includes:" in context_lower
    )

    if pmh_context and (
        any(marker in q for marker in history_followup_markers)
        or any(term in q for term in ["prediabetes", "asthma", "chickenpox", "hyperlipidemia"])
    ):
        return _extract_query_scope("past medical history")

    return scope


def _parse_lookup_tool_content(content: str, source_start_idx: int = 1) -> tuple[list[dict], dict[str, dict]]:
    """Parse lookup_clinical_notes content into stable source records."""
    ordered_sources: list[dict] = []
    source_map: dict[str, dict] = {}
    source_idx = max(1, int(source_start_idx or 1))
    header_re = _re.compile(
        r"^\[Source\s+\d+\s+\|\s+Section:\s*(?P<section>.*?)\s+\|\s+Page\s+(?P<page>[^|\]]+)"
        r"(?:\s+\|\s+MRN:\s*(?P<mrn>[^|\]]*))?"
        r"(?:\s+\|\s+Date:\s*(?P<encounter_date>[^|\]]*))?"
        r"(?:\s+\|\s+DocType:\s*(?P<document_type>[^\]]*))?"
        r"\]\s*(?P<body>[\s\S]*)$",
        flags=_re.IGNORECASE,
    )

    content = str(content or "")
    for part in [p.strip() for p in content.split("\n\n---\n\n") if p.strip()]:
        match = header_re.match(part)
        if not match:
            continue
        body = (match.group("body") or "").strip()
        first_line = body.splitlines()[0] if body else ""
        doc_type = (match.group("document_type") or "").strip()
        if not doc_type:
            m_doc = _re.search(r"DocType:\s*([^|\]]+)", first_line)
            doc_type = (m_doc.group(1).strip() if m_doc else "")
        sid = f"S{source_idx}"
        source = {
            "citation": sid,
            "section_title": (match.group("section") or "Section").strip(),
            "page_number": (match.group("page") or "?").strip(),
            "patient_mrn": (match.group("mrn") or "").strip(),
            "encounter_date": (match.group("encounter_date") or "").strip(),
            "document_type": doc_type,
            "body": body,
        }
        ordered_sources.append(source)
        source_map[sid] = source
        source_idx += 1

    return ordered_sources, source_map


def _parse_lookup_tool_sources(messages: list) -> tuple[list[dict], dict[str, dict]]:
    """Parse lookup_clinical_notes tool outputs into a global source map."""
    ordered_sources: list[dict] = []
    source_map: dict[str, dict] = {}
    next_idx = 1

    for m in messages:
        if not isinstance(m, _ToolMessage) or (getattr(m, "name", "") or "") != "lookup_clinical_notes":
            continue
        parsed_sources, parsed_map = _parse_lookup_tool_content(
            str(getattr(m, "content", "") or ""),
            source_start_idx=next_idx,
        )
        ordered_sources.extend(parsed_sources)
        source_map.update(parsed_map)
        next_idx += len(parsed_sources)

    return ordered_sources, source_map


def _source_identity(source: dict) -> tuple[str, str, str, str, str, str]:
    """Stable identity for deduplicating grounded sources across retries."""
    return (
        str(source.get("patient_mrn", "") or "").strip(),
        str(source.get("section_title", "") or "").strip().lower(),
        str(source.get("page_number", "") or "").strip(),
        str(source.get("encounter_date", "") or "").strip(),
        str(source.get("document_type", "") or "").strip().lower(),
        _normalize_match_text(source.get("body", "") or ""),
    )


def _merge_lookup_sources(
    ordered_sources: list[dict],
    source_map: dict[str, dict],
    new_sources: list[dict],
) -> tuple[list[dict], dict[str, dict]]:
    """Merge retry sources without duplicating equivalent records."""
    merged_ordered = list(ordered_sources)
    merged_map = dict(source_map)
    seen = {_source_identity(src) for src in merged_ordered}

    next_idx = len(merged_ordered) + 1
    for source in new_sources:
        source_id = _source_identity(source)
        if source_id in seen:
            continue
        seen.add(source_id)
        new_source = dict(source)
        citation = f"S{next_idx}"
        new_source["citation"] = citation
        merged_ordered.append(new_source)
        merged_map[citation] = new_source
        next_idx += 1

    return merged_ordered, merged_map


def _build_targeted_retry_query(user_query: str, scope: dict, topic: str) -> str:
    """Create a focused follow-up retrieval query for a missing topic."""
    focus = str(scope.get("focus", "general") or "general")
    normalized_topic = str(topic or "").strip().lower()

    if normalized_topic == "other":
        if focus == "past_medical_history":
            q = (user_query or "").strip()
            return f"past medical history {q}".strip()
        if focus == "family_history":
            q = (user_query or "").strip()
            return f"family history {q}".strip()
        if focus == "surgical_history":
            q = (user_query or "").strip()
            return f"past surgical history {q}".strip()
        return (user_query or "").strip()

    topic_queries = {
        "follow_up": "follow-up appointments after discharge",
        "long_term_plan": "long-term plan after discharge",
        "medications": "current medications",
        "labs": "relevant lab results",
        "diagnoses": "diagnoses and problem list",
        "procedures": "procedures and surgeries",
        "imaging": "imaging and echocardiogram findings",
        "instructions": "discharge instructions",
        "care_logistics": "home health and care logistics",
    }
    return topic_queries.get(normalized_topic, (user_query or "").strip())


def _run_targeted_lookup_retry(
    active_mrn: str,
    user_query: str,
    scope: dict,
    missing_topics: list[str],
    ordered_sources: list[dict],
    source_map: dict[str, dict],
) -> tuple[list[dict], dict[str, dict]]:
    """Run one focused retrieval retry for each still-missing topic."""
    merged_ordered = list(ordered_sources)
    merged_map = dict(source_map)

    for topic in missing_topics:
        retry_query = _build_targeted_retry_query(user_query, scope, topic)
        if not retry_query:
            continue
        try:
            retry_content = lookup_clinical_notes.func(
                retry_query,
                patient_mrn=active_mrn,
                document_id="",
            )
        except Exception:
            continue
        parsed_sources, _ = _parse_lookup_tool_content(
            str(retry_content or ""),
            source_start_idx=len(merged_ordered) + 1,
        )
        if not parsed_sources:
            continue
        merged_ordered, merged_map = _merge_lookup_sources(
            merged_ordered,
            merged_map,
            parsed_sources,
        )

    return merged_ordered, merged_map


def _format_sources_for_grounding(ordered_sources: list[dict]) -> str:
    """Render parsed sources with global stable citation IDs."""
    parts = []
    for src in ordered_sources:
        header = f"[{src['citation']} | Section: {src['section_title']} | Page {src['page_number']}"
        if src.get("patient_mrn"):
            header += f" | MRN: {src['patient_mrn']}"
        if src.get("encounter_date"):
            header += f" | Date: {src['encounter_date']}"
        if src.get("document_type"):
            header += f" | DocType: {src['document_type']}"
        header += "]"
        parts.append(f"{header}\n{src.get('body', '')}")
    return "\n\n---\n\n".join(parts)


def _build_grounding_prompt(scope: dict, sources_blob: str, missing_topics: list[str] | None = None) -> str:
    """Build a strict scoped extraction prompt."""
    requested = missing_topics or list(scope.get("requested_topics") or ["other"])
    focus = scope.get("focus", "general")
    exclusions = scope.get("exclude_sections") or []
    exclusion_text = ", ".join(exclusions) if exclusions else "none"
    exhaustive_history = focus in {"past_medical_history", "family_history", "surgical_history"}
    history_rule = (
        "- For history-focused questions, extract all distinct documented items from the relevant section, not just a few examples.\n"
        if exhaustive_history else ""
    )
    return (
        "You are a clinical evidence extraction engine.\n\n"
        f"USER QUESTION: {scope.get('query', '')}\n"
        f"REQUESTED TOPICS: {', '.join(requested)}\n"
        f"FOCUS: {focus}\n"
        f"EXCLUDED SECTIONS/CONTENT: {exclusion_text}\n\n"
        "Return STRICT JSON only in this format:\n"
        "{\n"
        '  "items": [\n'
        "    {\n"
        '      "topic": "follow_up|long_term_plan|medications|labs|diagnoses|procedures|imaging|instructions|care_logistics|other",\n'
        '      "claim": "",\n'
        '      "supports": [\n'
        '        {"citation": "S1", "evidence_quote": "", "encounter_date": "", "document_type": ""}\n'
        "      ]\n"
        "    }\n"
        "  ],\n"
        '  "missing_topics": ["topic_name"]\n'
        "}\n\n"
        "RULES:\n"
        "- Only include facts that directly answer the user's question.\n"
        "- Do not include related but unasked sections.\n"
        "- For past_medical_history focus, exclude family history, surgical history, preventive care, social history, and immunizations unless explicitly asked.\n"
        f"{history_rule}"
        "- Each claim must be atomic and directly supported by an exact quote copied from one cited source.\n"
        "- If unsure, omit the claim and mark the topic as missing.\n\n"
        f"SOURCES:\n{sources_blob}"
    )


def _document_type_authority(document_type: str, section_title: str = "") -> int:
    """Rank support authority for same-date tie breaking."""
    doc_type = str(document_type or "").strip().lower()
    section = str(section_title or "").strip().lower()
    if doc_type == "discharge_summary":
        return 4
    if "follow-up" in section or "follow up" in section or "plan" in section:
        return 3
    if doc_type in {"progress_note", "clinical_note"}:
        return 2
    return 1


def _parse_support_date(value: str):
    """Parse an ISO-like support date for sorting."""
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%Y/%m/%d"):
        try:
            return _dt.strptime(text, fmt).date()
        except Exception:
            continue
    return None


def _sort_supports(supports: list[dict]) -> list[dict]:
    """Sort supports by newest date, then by document authority."""
    def _key(support: dict):
        parsed = _parse_support_date(support.get("encounter_date", ""))
        ordinal = parsed.toordinal() if parsed else -1
        authority = _document_type_authority(
            support.get("document_type", ""),
            support.get("section_title", ""),
        )
        return (ordinal, authority, support.get("citation", ""))

    return sorted(supports, key=_key, reverse=True)


def _sort_items(items: list[dict]) -> list[dict]:
    """Sort claims by their strongest supporting source."""
    def _item_key(item: dict):
        supports = _sort_supports(list(item.get("supports") or []))
        if not supports:
            return (-1, -1, item.get("claim", ""))
        top = supports[0]
        parsed = _parse_support_date(top.get("encounter_date", ""))
        ordinal = parsed.toordinal() if parsed else -1
        authority = _document_type_authority(
            top.get("document_type", ""),
            top.get("section_title", ""),
        )
        return (ordinal, authority, item.get("claim", ""))

    normalized = []
    for item in items:
        item_copy = dict(item)
        item_copy["supports"] = _sort_supports(list(item.get("supports") or []))
        normalized.append(item_copy)
    return sorted(normalized, key=_item_key, reverse=True)


def _source_is_excluded(source: dict, scope: dict) -> bool:
    """Return True when a source section is excluded by scope policy."""
    section = (source.get("section_title", "") or "").lower()
    body = (source.get("body", "") or "").lower()
    haystack = f"{section}\n{body[:300]}"
    for token in scope.get("exclude_sections") or []:
        if token and token in haystack:
            return True
    return False


def _validate_grounded_payload(payload: dict, source_map: dict[str, dict], scope: dict) -> dict:
    """Mechanically validate extracted claims against the parsed source map."""
    requested = set(scope.get("requested_topics") or ["other"])
    seen = set()
    items = []

    for raw in payload.get("items") or []:
        if not isinstance(raw, dict):
            continue
        topic = str(raw.get("topic", "") or "").strip().lower() or "other"
        if topic not in requested:
            continue
        claim = str(raw.get("claim", "") or "").strip()
        if not claim:
            continue

        supports = []
        for sup in raw.get("supports") or []:
            if not isinstance(sup, dict):
                continue
            citation = str(sup.get("citation", "") or "").strip().upper()
            evidence_quote = str(sup.get("evidence_quote", "") or "").strip()
            source = source_map.get(citation)
            if not citation or not evidence_quote or not source:
                continue
            if _source_is_excluded(source, scope):
                continue
            if _normalize_match_text(evidence_quote) not in _normalize_match_text(source.get("body", "")):
                continue
            supports.append({
                "citation": citation,
                "evidence_quote": evidence_quote,
                "encounter_date": source.get("encounter_date", "") or str(sup.get("encounter_date", "") or "").strip(),
                "document_type": source.get("document_type", "") or str(sup.get("document_type", "") or "").strip(),
                "section_title": source.get("section_title", ""),
                "page_number": source.get("page_number", ""),
            })

        if not supports:
            continue

        key = (topic, _normalize_match_text(claim))
        if key in seen:
            continue
        seen.add(key)
        items.append({"topic": topic, "claim": claim, "supports": _sort_supports(supports)})

    extracted_missing = []
    for topic in payload.get("missing_topics") or []:
        t = str(topic or "").strip().lower()
        if t and t in requested:
            extracted_missing.append(t)

    covered = {item["topic"] for item in items}
    computed_missing = sorted((requested - covered) | set(extracted_missing))
    return {"items": _sort_items(items), "missing_topics": computed_missing}


def _semantic_filter_grounded_items(verifier_llm, query: str, scope: dict, grounded: dict) -> dict:
    """Use a lightweight verifier to reject claim/quote mismatches."""
    items = list(grounded.get("items") or [])
    if not verifier_llm or not items:
        return grounded

    verifier_payload = {
        "query": query,
        "focus": scope.get("focus", "general"),
        "requested_topics": scope.get("requested_topics") or [],
        "items": [
            {
                "index": idx,
                "topic": item.get("topic", ""),
                "claim": item.get("claim", ""),
                "supports": [
                    {
                        "citation": s.get("citation", ""),
                        "section_title": s.get("section_title", ""),
                        "evidence_quote": s.get("evidence_quote", ""),
                    }
                    for s in item.get("supports", [])
                ],
            }
            for idx, item in enumerate(items)
        ],
    }
    prompt = (
        "You are a clinical grounding verifier.\n"
        "Return STRICT JSON only in this format:\n"
        '{"verdicts":[{"index":0,"supported":true}]}\n\n'
        "Mark supported=true only if the evidence quotes directly support the claim for the user's question.\n"
        "If the claim adds details not present in the quotes, mark it false.\n\n"
        f"PAYLOAD:\n{_json.dumps(verifier_payload, ensure_ascii=True)}"
    )
    try:
        raw = verifier_llm.invoke(prompt).content
        verdict_obj = _safe_json_object(raw)
    except Exception:
        return grounded

    supported_indexes = {
        int(v.get("index"))
        for v in verdict_obj.get("verdicts", [])
        if isinstance(v, dict) and v.get("supported") is True and str(v.get("index", "")).isdigit()
    }
    filtered = [item for idx, item in enumerate(items) if idx in supported_indexes]
    covered = {item["topic"] for item in filtered}
    missing = sorted(set(scope.get("requested_topics") or []) - covered)
    return {"items": _sort_items(filtered), "missing_topics": missing}


def _render_grounded_answer(scope: dict, grounded: dict) -> str:
    """Deterministically render the validated grounded answer."""
    items = list(grounded.get("items") or [])
    missing_topics = list(grounded.get("missing_topics") or [])
    if not items:
        return _STRICT_GROUNDING_NOT_FOUND_MSG

    topic_order = ["follow_up", "long_term_plan", "medications", "labs", "diagnoses", "procedures", "imaging", "instructions", "care_logistics", "other"]
    grouped: dict[str, list[dict]] = {k: [] for k in topic_order}
    for item in items:
        grouped.setdefault(item["topic"], []).append(item)

    def _cite_text(item: dict) -> str:
        cites = []
        for s in item.get("supports", []):
            section = str(s.get("section_title", "") or "").strip()
            page = str(s.get("page_number", "") or "").strip()
            parts = []
            if section:
                parts.append(section)
            if page:
                parts.append(f"p. {page}")
            if not parts:
                parts.append(str(s.get("citation", "") or "").strip())
            display = ", ".join(parts)
            if display and display not in cites:
                cites.append(display)
        return f" ({'; '.join(cites)})" if cites else ""

    focus = scope.get("focus", "general")
    if focus == "past_medical_history":
        lines = ["The past medical history includes:"]
        for item in grouped.get("other", []):
            lines.append(f"- {item['claim']} {_cite_text(item)}".rstrip())
        if not grouped.get("other") and missing_topics:
            lines.append(f"- {_STRICT_GROUNDING_NOT_FOUND_MSG}")
        return "\n".join(lines)
    if focus == "family_history":
        lines = ["The family history includes:"]
        for item in grouped.get("other", []):
            lines.append(f"- {item['claim']} {_cite_text(item)}".rstrip())
        return "\n".join(lines)
    if focus == "surgical_history":
        lines = ["The past surgical history includes:"]
        for item in grouped.get("other", []):
            lines.append(f"- {item['claim']} {_cite_text(item)}".rstrip())
        return "\n".join(lines)

    labels = {
        "follow_up": "Follow-Up",
        "long_term_plan": "Long-Term Plan",
        "medications": "Medications",
        "labs": "Labs",
        "diagnoses": "Diagnoses",
        "procedures": "Procedures",
        "imaging": "Imaging",
        "instructions": "Instructions",
        "care_logistics": "Care Logistics",
        "other": "Relevant Findings",
    }
    sections = []
    for topic in topic_order:
        topic_items = grouped.get(topic) or []
        if not topic_items:
            if topic in missing_topics and len(scope.get("requested_topics") or []) > 1:
                sections.append(f"{labels.get(topic, topic.title())}:")
                sections.append(f"- {_STRICT_GROUNDING_NOT_FOUND_MSG}")
            continue
        if len(scope.get("requested_topics") or []) > 1:
            sections.append(f"{labels.get(topic, topic.title())}:")
        for item in topic_items:
            sections.append(f"- {item['claim']} {_cite_text(item)}".rstrip())
    return "\n".join(sections) if sections else _STRICT_GROUNDING_NOT_FOUND_MSG


def _is_reasoning_model(model_name: str) -> bool:
    """Check if a model name refers to a reasoning model (no temperature param)."""
    _reasoning_prefixes = ("o1", "o3", "o4", "gpt-5")
    return any(model_name.lower().startswith(p) for p in _reasoning_prefixes)


def _make_llm(model_name: str, timeout: int = 60, max_tokens: int = None, thinking_budget: int = 0):
    """Create a Chat LLM instance — auto-selects Google Gemini or OpenAI based on config."""
    provider = get_active_llm_provider(getattr(TOOLS_CFG, "llm_provider", "openai"))

    if provider == "google" or model_name.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        kwargs = dict(
            model=model_name,
            google_api_key=get_google_api_key(),
            timeout=timeout,
            max_retries=2,
        )
        if thinking_budget > 0:
            # Thinking mode: temperature must be omitted or set by the API
            kwargs["thinking_budget"] = thinking_budget
        else:
            kwargs["temperature"] = TOOLS_CFG.primary_agent_llm_temperature
            kwargs["thinking_budget"] = 0
        if max_tokens:
            kwargs["max_output_tokens"] = max_tokens
        llm = ChatGoogleGenerativeAI(**kwargs)
        return llm

    # OpenAI path (fallback)
    kwargs = dict(
        model=model_name,
        api_key=os.getenv("OPENAI_API_KEY"),
        request_timeout=timeout,
        max_retries=2,
    )
    if _is_reasoning_model(model_name):
        kwargs["max_completion_tokens"] = 16384
    else:
        kwargs["temperature"] = TOOLS_CFG.primary_agent_llm_temperature
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)


def _last_user_text(state: State) -> str:
    """Return the most recent user utterance from the graph state."""
    for m in reversed(state.get("messages", [])):
        if getattr(m, "type", "") == "human" or m.__class__.__name__ == "HumanMessage":
            return getattr(m, "content", "") or ""
    return ""


def _recent_human_texts(state: State) -> list[str]:
    """Return human messages in reverse chronological order."""
    out: list[str] = []
    for m in reversed(state.get("messages", [])):
        if getattr(m, "type", "") == "human" or m.__class__.__name__ == "HumanMessage":
            content = getattr(m, "content", "") or ""
            if content:
                out.append(content)
    return out


def _query_has_explicit_patient_identifier(query: str) -> bool:
    """Return True when the user explicitly mentions a patient identifier."""
    q = (query or "").strip().lower()
    if not q:
        return False
    if any(marker in q for marker in ("mrn", "patient id", "patient name")):
        return True
    return False


def _resolve_active_patient_mrn(state: State, current_query: str = "") -> str:
    """Resolve the active patient MRN from the current query or prior chat history."""
    existing = (state.get("active_patient_mrn", "") or "").strip()

    try:
        rag = get_rag_tool_instance()
    except Exception:
        return existing

    current = (current_query or "").strip()
    if current:
        try:
            current_mrn = rag.resolve_to_mrn(current)
        except Exception:
            current_mrn = None
        if current_mrn:
            return str(current_mrn).strip()
        if _query_has_explicit_patient_identifier(current):
            # User explicitly named a different patient/identifier; do not fall
            # back to stale session context if resolution failed.
            return ""

    if existing:
        return existing

    seen = {current} if current else set()
    for text in _recent_human_texts(state):
        norm = (text or "").strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        try:
            mrn = rag.resolve_to_mrn(norm)
        except Exception:
            mrn = None
        if mrn:
            return str(mrn).strip()
    return ""


def _copy_ai_message_with_tool_calls(message: _AIMessage, tool_calls: list[dict]) -> _AIMessage:
    """Clone an AIMessage while replacing tool calls."""
    try:
        if hasattr(message, "model_copy"):
            return message.model_copy(update={"tool_calls": tool_calls})
        return message.copy(update={"tool_calls": tool_calls})
    except Exception:
        return _AIMessage(content=str(getattr(message, "content", "") or ""), tool_calls=tool_calls)


def _scope_lookup_tool_calls(message: _AIMessage, active_mrn: str) -> _AIMessage:
    """Inject MRN scoping into lookup_clinical_notes calls or block unscoped calls."""
    tool_calls = getattr(message, "tool_calls", []) or []
    if not tool_calls:
        return message

    updated_calls = []
    for tool_call in tool_calls:
        if tool_call.get("name") != "lookup_clinical_notes":
            updated_calls.append(tool_call)
            continue

        args = dict(tool_call.get("args") or {})
        has_document_scope = bool(str(args.get("document_id", "") or "").strip())
        if active_mrn:
            args["patient_mrn"] = active_mrn
        elif not has_document_scope:
            return _AIMessage(content=_UNSCOPED_PATIENT_QUERY_MSG)

        updated_call = dict(tool_call)
        updated_call["args"] = args
        updated_calls.append(updated_call)

    return _copy_ai_message_with_tool_calls(message, updated_calls)


def build_graph(thinking_mode: bool = False):
    """
    Builds an agent decision-making graph with a split router/synthesizer
    architecture for optimal latency:

      router  (gemini-flash) — fast tool selection (~1s)
      synthesizer (gemini-flash / gemini-2.5-pro) — answer generation

    When thinking_mode=True, the synthesizer uses a deeper reasoning model
    with extended thinking budget for complex clinical queries.

    The synthesizer always produces the final answer, even for summaries,
    so it can tailor the response to the user's actual question.
    """

    # ── Build the LLMs ────────────────────────────────────────────────
    active_provider = get_active_llm_provider(getattr(TOOLS_CFG, "llm_provider", "openai"))
    router_model = os.getenv("MEDGRAPH_ROUTER_MODEL_OVERRIDE", "").strip() or getattr(
        TOOLS_CFG, "primary_agent_router_llm", TOOLS_CFG.primary_agent_llm
    )

    # Router is always fast (same regardless of thinking mode)
    router_llm = _make_llm(router_model, timeout=20, max_tokens=300)

    if thinking_mode:
        synth_model = os.getenv("MEDGRAPH_THINKING_MODEL_OVERRIDE", "").strip() or getattr(
            TOOLS_CFG, "primary_agent_thinking_llm", "gemini-2.5-pro"
        )
        # GPT-4.1 is a non-reasoning model (no "reasoning effort" knob). In
        # practice, "deep thinking mode" here is prompt + higher accuracy
        thinking_max_out = int(os.getenv("MEDGRAPH_THINKING_MAX_OUTPUT_TOKENS", "4096"))
        synth_llm = _make_llm(synth_model, timeout=180, max_tokens=thinking_max_out, thinking_budget=8192)
        logger.info("Router LLM: %s (timeout=20s)", router_model)
        logger.info("Synthesizer LLM [THINKING]: %s (timeout=180s, max_tokens=%s, thinking=8192)", synth_model, thinking_max_out)
    else:
        synth_model = os.getenv("MEDGRAPH_SYNTH_MODEL_OVERRIDE", "").strip() or TOOLS_CFG.primary_agent_llm
        synth_llm = _make_llm(synth_model, timeout=60, max_tokens=4096)
        logger.info("Router LLM: %s (timeout=20s)", router_model)
        logger.info("Synthesizer LLM: %s (timeout=60s, max_tokens=4096)", synth_model)

    semantic_verifier_llm = None
    if os.getenv("OPENAI_API_KEY"):
        try:
            semantic_verifier_llm = ChatOpenAI(
                model="gpt-4.1-mini",
                api_key=os.getenv("OPENAI_API_KEY"),
                request_timeout=30,
                max_retries=1,
                max_tokens=600,
                temperature=0,
            )
        except Exception:
            semantic_verifier_llm = None

    # ── Build the graph ───────────────────────────────────────────────
    graph_builder = StateGraph(State)
    search_tool = load_tavily_search_tool(TOOLS_CFG.tavily_search_max_results)
    tools = [search_tool,
             lookup_clinical_notes,
             lookup_patient_orders,
             cohort_patient_search,
             summarize_patient_record,
             summarize_uploaded_document,
             list_available_patients,
             list_uploaded_documents,
             ]

    # Router: fast model with tools bound — decides WHICH tool to call
    router_with_tools = router_llm.bind_tools(tools)

    def router(state: State):
        """Fast tool-routing + deterministic decomposition for multi-part questions."""
        last_user_text = _last_user_text(state)
        active_mrn = _resolve_active_patient_mrn(state, current_query=last_user_text)
        q_lower = (last_user_text or "").lower()

        # Heuristic: multi-part analytical questions often contain multiple domains.
        domains = set()
        if any(k in q_lower for k in ["medication", "medications", "meds", "drug", "rx", "prescription"]):
            domains.add("meds")
        if any(k in q_lower for k in ["lab", "labs", "hba1c", "a1c", "cbc", "bmp", "cmp", "creatinine", "glucose", "test", "ordered"]):
            domains.add("labs")
        if any(k in q_lower for k in ["diagnosis", "diagnoses", "impression", "differential", "icd"]):
            domains.add("dx")
        if any(k in q_lower for k in ["procedure", "procedures", "surgery", "imaging", "ct", "mri", "x-ray", "ultrasound", "referral", "consult"]):
            domains.add("proc")

        should_decompose = (
            (" and " in q_lower or "&" in q_lower)
            and len(domains) >= 2
            and ("patients" not in q_lower)
            and ("cohort" not in q_lower)
            and ("diabetic" not in q_lower or "hba1c" in q_lower)
        )

        if should_decompose and last_user_text.strip():
            # Ask the router LLM for sub-queries, then trigger multiple lookup_clinical_notes calls.
            try:
                from langchain_core.messages import SystemMessage as _SysMsg2, HumanMessage as _HumanMsg2

                decomp_prompt = _SysMsg2(
                    content=(
                        "You are a clinical query decomposition engine. "
                        "Decompose the user's question into focused sub-queries optimized for lookup_clinical_notes. "
                        "Each sub-query must be a standalone search for one clinical domain "
                        "(e.g., medications, labs, diagnoses, procedures) and keep the original time context "
                        "(e.g., 'last encounter', 'during hospitalization'). "
                        "Return STRICT JSON only in this format: {\"sub_queries\": [\"...\", \"...\" ]}."
                    )
                )
                decomp_user = _HumanMsg2(content=last_user_text)
                raw = router_llm.invoke([decomp_prompt, decomp_user]).content

                # Safe JSON extraction (strip fences).
                t = str(raw or "").strip()
                t = t.replace("```json", "```").replace("```", "")
                if not t.startswith("{"):
                    m = _json.search(r"\{[\s\S]*\}", t) if hasattr(_json, "search") else None
                    # Fallback to no match.
                try:
                    obj = _json.loads(t)
                except Exception:
                    # Last chance: extract first JSON object substring.
                    import re as _re2
                    m2 = _re2.search(r"\{[\s\S]*\}", str(raw or ""))
                    if not m2:
                        raise
                    obj = _json.loads(m2.group(0))

                sub_queries = obj.get("sub_queries") or []
                sub_queries = [str(s).strip() for s in sub_queries if str(s).strip()]
                if len(sub_queries) >= 2:
                    if not active_mrn:
                        return {
                            "messages": [_AIMessage(content=_UNSCOPED_PATIENT_QUERY_MSG)],
                            "active_patient_mrn": "",
                        }
                    tool_calls = []
                    for i, sq in enumerate(sub_queries[:4]):
                        tool_calls.append(
                            {
                                "name": "lookup_clinical_notes",
                                "args": {
                                    "query": sq,
                                    "patient_mrn": active_mrn,
                                    "document_id": "",
                                },
                                "id": f"decomp_{i}",
                            }
                        )
                    return {
                        "messages": [_AIMessage(content="", tool_calls=tool_calls)],
                        "active_patient_mrn": active_mrn,
                    }
            except Exception:
                # Fall back to normal router behavior.
                pass

        # Default: rely on the router's tool selection.
        routed = router_with_tools.invoke(state["messages"])
        routed = _scope_lookup_tool_calls(routed, active_mrn)
        return {
            "messages": [routed],
            "active_patient_mrn": active_mrn,
        }

    # Synthesizer: generates final answer from tool output.
    # Binds tools so it CAN call more if needed (multi-hop), but we prefer
    # it to synthesize from the tool results already in the message history.
    synth_with_tools = synth_llm.bind_tools(tools)

    # Prepend a lightweight synthesis instruction to guide the synthesizer.
    from langchain_core.messages import SystemMessage as _SysMsg
    _SYNTH_HINT = _SysMsg(content=(
        "Tool results are now in the conversation above. "
        "Synthesize a clear, grounded answer to the user's original question "
        "using those results. Only call additional tools if the existing results "
        "are truly insufficient to answer. When in doubt, answer from what you have."
    ))

    _OPENAI_THINKING_HINT = _SysMsg(content=(
        "Deep thinking mode (OpenAI GPT-4.1):\n"
        "- Verify every factual claim is directly supported by the provided tool sources.\n"
        "- If a key detail is missing, explicitly state it is not documented in the retrieved context.\n"
        "- If multiple retrieved sources disagree, report the discrepancy rather than averaging.\n"
        "- Then provide the final clinician-ready answer."
    ))

    def synthesizer(state: State):
        """High-quality answer synthesis from retrieved context."""
        msgs = state["messages"]
        active_mrn = (state.get("active_patient_mrn", "") or "").strip()
        user_query = _last_user_text(state)

        # OpenAI thinking mode can hit "input limit exceeded" when tool outputs
        # are large. Truncate ToolMessage contents to keep the prompt within
        # safe bounds while preserving the most informative parts.
        if thinking_mode and active_provider == "openai":
            max_tool_chars = int(os.getenv("MEDGRAPH_OPENAI_THINKING_MAX_TOOL_CHARS", "8000"))
            max_total_tool_chars = int(os.getenv("MEDGRAPH_OPENAI_THINKING_MAX_TOTAL_TOOL_CHARS", "22000"))

            total_tool_chars = 0
            truncated_msgs = []
            for m in msgs:
                if isinstance(m, _ToolMessage):
                    content = getattr(m, "content", "") or ""
                    content_str = str(content)
                    if len(content_str) > max_tool_chars:
                        half = max_tool_chars // 2
                        content_str = content_str[:half] + "\n\n...[TRUNCATED]...\n\n" + content_str[-half:]

                    # Soft cap: if we already used the budget, replace further tool
                    # content with an indicator instead of leaving it huge.
                    if total_tool_chars + len(content_str) > max_total_tool_chars:
                        allowed = max(0, max_total_tool_chars - total_tool_chars)
                        if allowed <= 200:
                            content_str = "...[TRUNCATED - tool context omitted]..."
                        else:
                            content_str = content_str[:allowed] + "\n...[TRUNCATED]..."

                    total_tool_chars += len(content_str)
                    try:
                        truncated_msgs.append(m.copy(update={"content": content_str}))
                    except Exception:
                        # Fallback: keep message object but with no tool content
                        truncated_msgs.append(m)
                else:
                    truncated_msgs.append(m)
            msgs = truncated_msgs

        # Accuracy-first pass-through for full-summary tools:
        # if a dedicated summarizer tool already produced a comprehensive,
        # citation-grounded summary, avoid re-summarizing it (which can drop details).
        for m in reversed(msgs):
            if isinstance(m, _ToolMessage):
                tool_name = getattr(m, "name", "") or ""
                if tool_name in ("summarize_patient_record", "summarize_uploaded_document"):
                    return {"messages": [_AIMessage(content=str(getattr(m, "content", "") or ""))]}
                # Stop at the first tool message seen in reverse order
                break

        ordered_sources, source_map = _parse_lookup_tool_sources(msgs)
        if active_mrn and source_map:
            initial_source_count = len(source_map)
            scope = _extract_query_scope_with_context(state, user_query)
            sources_blob = _format_sources_for_grounding(ordered_sources)

            grounding_prompt = _build_grounding_prompt(scope, sources_blob)
            grounded_payload = _safe_json_object(str(getattr(synth_llm.invoke(grounding_prompt), "content", "") or ""))
            grounded = _validate_grounded_payload(grounded_payload, source_map, scope)
            grounded = _semantic_filter_grounded_items(semantic_verifier_llm, user_query, scope, grounded)

            if grounded.get("missing_topics"):
                retry_prompt = _build_grounding_prompt(
                    scope,
                    sources_blob,
                    missing_topics=grounded.get("missing_topics") or [],
                ) + (
                    "\n\nIMPORTANT: A prior extraction missed or over-generated content. "
                    "Return only directly supported claims for the listed missing topics."
                )
                retry_payload = _safe_json_object(str(getattr(synth_llm.invoke(retry_prompt), "content", "") or ""))
                retry_grounded = _validate_grounded_payload(retry_payload, source_map, scope)
                retry_grounded = _semantic_filter_grounded_items(semantic_verifier_llm, user_query, scope, retry_grounded)
                if retry_grounded.get("items"):
                    existing_keys = {
                        (item.get("topic", ""), _normalize_match_text(item.get("claim", "")))
                        for item in grounded.get("items", [])
                    }
                    for item in retry_grounded.get("items", []):
                        key = (item.get("topic", ""), _normalize_match_text(item.get("claim", "")))
                        if key not in existing_keys:
                            grounded.setdefault("items", []).append(item)
                            existing_keys.add(key)
                    covered = {item.get("topic", "") for item in grounded.get("items", [])}
                    grounded["missing_topics"] = sorted(set(scope.get("requested_topics") or []) - covered)

            if grounded.get("missing_topics"):
                ordered_sources, source_map = _run_targeted_lookup_retry(
                    active_mrn=active_mrn,
                    user_query=user_query,
                    scope=scope,
                    missing_topics=list(grounded.get("missing_topics") or []),
                    ordered_sources=ordered_sources,
                    source_map=source_map,
                )
                if len(source_map) > initial_source_count:
                    retry_sources_blob = _format_sources_for_grounding(ordered_sources)
                    targeted_prompt = _build_grounding_prompt(
                        scope,
                        retry_sources_blob,
                        missing_topics=grounded.get("missing_topics") or [],
                    ) + (
                        "\n\nIMPORTANT: This is a targeted retrieval retry for missing topics only. "
                        "Return only directly supported claims from the cited sources."
                    )
                    targeted_payload = _safe_json_object(
                        str(getattr(synth_llm.invoke(targeted_prompt), "content", "") or "")
                    )
                    targeted_grounded = _validate_grounded_payload(targeted_payload, source_map, scope)
                    targeted_grounded = _semantic_filter_grounded_items(
                        semantic_verifier_llm,
                        user_query,
                        scope,
                        targeted_grounded,
                    )
                    if targeted_grounded.get("items"):
                        existing_keys = {
                            (item.get("topic", ""), _normalize_match_text(item.get("claim", "")))
                            for item in grounded.get("items", [])
                        }
                        for item in targeted_grounded.get("items", []):
                            key = (item.get("topic", ""), _normalize_match_text(item.get("claim", "")))
                            if key not in existing_keys:
                                grounded.setdefault("items", []).append(item)
                                existing_keys.add(key)
                        covered = {item.get("topic", "") for item in grounded.get("items", [])}
                        grounded["missing_topics"] = sorted(set(scope.get("requested_topics") or []) - covered)

            rendered = _render_grounded_answer(scope, grounded)
            return {
                "messages": [_AIMessage(content=rendered)],
                "active_patient_mrn": active_mrn,
            }
        # Inject synthesis hint right before synthesizer's call (idempotent for multi-hop).
        if not any(
            isinstance(m, _SysMsg) and "Synthesize a clear" in str(getattr(m, "content", ""))
            for m in msgs
        ):
            msgs = [msgs[0], _SYNTH_HINT] + list(msgs[1:])

        if thinking_mode and active_provider == "openai":
            if not any(
                isinstance(m, _SysMsg) and "Deep thinking mode" in str(getattr(m, "content", ""))
                for m in msgs
            ):
                # Insert after _SYNTH_HINT to keep the system ordering stable.
                # We expect msgs[1] to be _SYNTH_HINT due to the previous block.
                insert_at = 2 if len(msgs) > 2 else len(msgs)
                msgs = list(msgs[:insert_at]) + [_OPENAI_THINKING_HINT] + list(msgs[insert_at:])

        synth_msg = synth_with_tools.invoke(msgs)
        synth_msg = _scope_lookup_tool_calls(synth_msg, active_mrn)
        return {
            "messages": [synth_msg],
            "active_patient_mrn": active_mrn,
        }


    graph_builder.add_node("router", router)
    graph_builder.add_node("synthesizer", synthesizer)
    tool_node = BasicToolNode(
        tools=[
            search_tool,
            lookup_clinical_notes,
            lookup_patient_orders,
            cohort_patient_search,
            summarize_patient_record,
            summarize_uploaded_document,
            list_available_patients,
            list_uploaded_documents,
        ])
    graph_builder.add_node("tools", tool_node)

    # Router decides: call a tool or end directly
    # When no tools needed (greetings, simple conversation), router's response
    # IS the final answer — skip synthesizer to avoid redundant LLM call
    graph_builder.add_conditional_edges(
        "router",
        route_tools,
        {"tools": "tools", "__end__": "__end__"},
    )

    # After tools: always go to synthesizer so it can answer the user's actual question
    graph_builder.add_edge("tools", "synthesizer")

    # Synthesizer may call additional tools (multi-hop) or finish
    graph_builder.add_conditional_edges(
        "synthesizer",
        route_tools,
        {"tools": "tools", "__end__": "__end__"},
    )

    graph_builder.add_edge(START, "router")
    graph = graph_builder.compile()
    plot_agent_schema(graph)
    return graph
