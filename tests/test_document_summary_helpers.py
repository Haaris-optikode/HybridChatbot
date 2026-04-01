import json


from src.agent_graph.document_summary_helpers import build_sources_for_groups, merge_doc_extractions


def test_build_sources_global_ids_stable_across_groups():
    # Build > max_group_chars so it splits into multiple groups
    chunks = []
    for i in range(1, 6):
        chunks.append((i, "X" * 9000, {"section_title": f"Sec{i}", "page_number": i}))

    groups = build_sources_for_groups(chunks, max_group_chars=22_000)
    assert len(groups) >= 2

    # Ensure global IDs appear (S1..S5) across the grouped renderings
    rendered = "\n\n".join(groups)
    for i in range(1, 6):
        assert f"[S{i} " in rendered


def test_merge_doc_extractions_dedupes_conservatively():
    ex1 = {
        "document": {"document_id": "d1"},
        "patient": {"mrn": "MRN-1"},
        "allergies": [{"substance": "Penicillin", "reaction": "Anaphylaxis", "severity": "Severe", "citations": ["S1"]}],
        "medications": [{"name": "Metformin", "dose": "500 mg", "route": "PO", "frequency": "BID", "context": "home", "citations": ["S2"]}],
        "labs": [{"test": "Creatinine", "value": "1.8", "unit": "mg/dL", "date_time": "2026-01-01", "citations": ["S3"]}],
        "vitals": [],
        "imaging": [],
        "procedures": [],
        "problems": [],
        "hospital_course": [],
        "disposition_followup": [],
        "other_key_findings": [],
        "missing_or_unclear": [],
    }
    ex2 = {
        "document": {"doc_type": "discharge_summary"},
        "patient": {"name": "John Doe"},
        "allergies": [{"substance": "penicillin", "reaction": "anaphylaxis", "severity": "severe", "citations": ["S4"]}],
        "medications": [{"name": "Metformin", "dose": "500 mg", "route": "PO", "frequency": "BID", "context": "home", "citations": ["S5"]}],
        "labs": [{"test": "Creatinine", "value": "1.8", "unit": "mg/dL", "date_time": "2026-01-01", "citations": ["S6"]}],
        "vitals": [],
        "imaging": [],
        "procedures": [],
        "problems": [],
        "hospital_course": [],
        "disposition_followup": [],
        "other_key_findings": [],
        "missing_or_unclear": [],
    }

    merged = merge_doc_extractions([ex1, ex2])

    assert merged["document"]["document_id"] == "d1"
    assert merged["document"]["doc_type"] == "discharge_summary"
    assert merged["patient"]["mrn"] == "MRN-1"
    assert merged["patient"]["name"] == "John Doe"

    # Deduped
    assert len(merged["allergies"]) == 1
    assert len(merged["medications"]) == 1
    assert len(merged["labs"]) == 1

    # Still JSON serializable (critical for reduce prompt)
    json.dumps(merged, ensure_ascii=False)

