from types import SimpleNamespace
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_table_row_extractor_separates_result_from_reference_range():
    from document_processing.metadata_extractor import extract_table_row_facts

    text = """
| Test | Result | Unit | Reference Range | Flag |
| --- | --- | --- | --- | --- |
| Creatinine | 1.4 | mg/dL | 0.7-1.2 | H |
| eGFR | 58 | mL/min | >=60 | L |
"""
    facts = extract_table_row_facts(
        text,
        {
            "document_id": "doc-lab",
            "chunk_index": 3,
            "section_title": "Lab Results",
            "encounter_date": "2025-08-10T00:00:00",
        },
    )

    creatinine = next(f for f in facts if f["metadata"]["test_name"] == "Creatinine")
    assert creatinine["metadata"]["result_value"] == "1.4"
    assert creatinine["metadata"]["reference_range"] == "0.7-1.2"
    assert "Value: 1.4" in creatinine["page_content"]
    assert "Reference range: 0.7-1.2" in creatinine["page_content"]


def test_table_row_extractor_ignores_appointment_tables():
    from document_processing.metadata_extractor import extract_table_row_facts

    text = """
| Date | Time | Status |
| --- | --- | --- |
| 02/15/2025 | 3:00-3:30 PM | Completed |
"""
    facts = extract_table_row_facts(
        text,
        {"document_id": "doc-appt", "chunk_index": 1, "section_title": "Appointment History"},
    )

    assert facts == []


def test_table_row_extractor_treats_copay_table_as_admin_not_lab():
    from document_processing.metadata_extractor import extract_table_row_facts

    text = """
| Service | Value |
| --- | --- |
| In-Network Specialist Copay | $55 |
"""
    facts = extract_table_row_facts(
        text,
        {"document_id": "doc-admin", "chunk_index": 1, "section_title": "Primary Insurance"},
    )

    assert len(facts) == 1
    assert facts[0]["metadata"]["fact_type"] == "administrative_benefit"
    assert "test_name" not in facts[0]["metadata"]


def test_table_row_extractor_preserves_row_provided_allergen_class():
    from document_processing.metadata_extractor import extract_table_row_facts

    text = """
| Allergen | Result (kU/L) | Class | Interpretation | Flag |
| --- | --- | --- | --- | --- |
| Timothy Grass | 1.6 | 1 | Low | POS |
| Oak | 2.1 | 2 | Low-Moderate | POS |
"""
    facts = extract_table_row_facts(
        text,
        {"document_id": "doc-ige", "chunk_index": 2, "section_title": "Allergen IgE Panel"},
    )

    timothy = next(f for f in facts if f["metadata"]["test_name"] == "Timothy Grass")
    assert timothy["metadata"]["result_value"] == "1.6"
    assert timothy["metadata"]["result_class"] == "1"
    assert timothy["metadata"]["result_interpretation"] == "Low"
    assert "Document class: 1" in timothy["page_content"]


def test_table_row_extractor_rejects_fragmented_narrative_rows():
    from document_processing.metadata_extractor import extract_table_row_facts

    text = """
| Finding | Value |
| --- | --- |
| Total IgE (03/18/20 | 26): 3 |
| 25-OH Vitamin D (0 | 3/18/2 |
"""
    facts = extract_table_row_facts(
        text,
        {"document_id": "doc-fragment", "chunk_index": 7, "section_title": "Review of Systems"},
    )

    assert facts == []


def test_admin_table_row_extractor_creates_benefit_facts():
    from document_processing.metadata_extractor import extract_table_row_facts

    text = """
| Benefit | Coverage | Copay | Limit |
| --- | --- | --- | --- |
| Physical therapy | Covered after authorization | $20 | 20 visits/year |
"""
    facts = extract_table_row_facts(
        text,
        {"document_id": "doc-admin", "chunk_index": 1, "section_title": "Benefits"},
    )

    assert len(facts) == 1
    meta = facts[0]["metadata"]
    assert meta["fact_type"] == "administrative_benefit"
    assert meta["benefit_name"] == "Physical therapy"
    assert meta["coverage_value"] == "Covered after authorization"
    assert meta["copay"] == "$20"


def test_admin_table_row_extractor_handles_field_details_tables():
    from document_processing.metadata_extractor import extract_table_row_facts

    text = """
| Field | Details |
| --- | --- |
| In-Network Specialist Copay | $55 |
| Annual Deductible | $1,000 Individual / $2,000 Family |
| Telehealth Benefit | Covered at PCP copay rate ($30) |
"""
    facts = extract_table_row_facts(
        text,
        {"document_id": "doc-insurance", "chunk_index": 2, "section_title": "Primary Insurance"},
    )

    by_name = {f["metadata"]["benefit_name"]: f["metadata"]["coverage_value"] for f in facts}
    assert by_name["In-Network Specialist Copay"] == "$55"
    assert by_name["Annual Deductible"] == "$1,000 Individual / $2,000 Family"
    assert by_name["Telehealth Benefit"] == "Covered at PCP copay rate ($30)"


def test_clinical_event_extractor_captures_iu_and_intranasal_med_changes():
    from document_processing.metadata_extractor import extract_clinical_event_facts

    text = """
## Plan
Start Cholecalciferol (Vitamin D3) 2000 IU orally, once daily.
Increase Cholecalciferol to 4000 IU daily.
Start Fluticasone Propionate 50 mcg/spray intranasal daily.
"""
    facts = extract_clinical_event_facts(
        text,
        {
            "document_id": "doc-progress",
            "chunk_index": 5,
            "section_title": "Plan",
            "document_type": "progress_note",
            "document_date": "2026-03-26",
        },
    )

    evidence = "\n".join(f["page_content"] for f in facts)
    assert "Cholecalciferol" in evidence
    assert "4000 IU" in evidence
    assert "Fluticasone Propionate" in evidence
    assert all(f["metadata"]["event_type"] == "medication_change" for f in facts)


def test_fact_retrieval_prefers_scoped_row_facts_for_numeric_queries():
    from agent_graph.tool_clinical_notes_rag import ClinicalNotesRAGTool

    rag = object.__new__(ClinicalNotesRAGTool)
    rag.collection_name = "PatientData"
    point = SimpleNamespace(payload={
        "page_content": "LAB_OR_VITAL_RESULT | Test: Creatinine | Value: 1.4 | Unit: mg/dL | Reference range: 0.7-1.2",
        "metadata": {
            "chunk_kind": "table_row",
            "fact_type": "lab_or_vital_result",
            "document_id": "doc-lab",
            "patient_mrn": "MRN001",
            "test_name": "Creatinine",
            "result_value": "1.4",
            "reference_range": "0.7-1.2",
        },
    })
    rag._scroll_all_points = lambda scroll_filter=None, limit=512: [point]

    docs = rag._retrieve_fact_documents(
        "What was the creatinine value?",
        patient_mrn="MRN001",
        document_ids=["doc-lab"],
    )

    assert len(docs) == 1
    assert docs[0].metadata["result_value"] == "1.4"
    assert docs[0].metadata["reference_range"] == "0.7-1.2"


def test_fact_retrieval_prefers_office_vitals_over_home_bp_measurement():
    from agent_graph.tool_clinical_notes_rag import ClinicalNotesRAGTool

    rag = object.__new__(ClinicalNotesRAGTool)
    rag.collection_name = "PatientData"
    home = SimpleNamespace(payload={
        "page_content": "CLINICAL_EVENT | Type: measurement | Action: documented | Date: 2026-03-26 | Evidence: Home BP average 126/78.",
        "metadata": {
            "chunk_kind": "clinical_event",
            "fact_type": "measurement",
            "event_type": "measurement",
            "document_id": "doc-home",
            "event_date": "2026-03-26",
        },
    })
    office = SimpleNamespace(payload={
        "page_content": "LAB_OR_VITAL_RESULT | Test: Blood Pressure | Value: 128/76 | Unit: mmHg | Date: 2026-03-26",
        "metadata": {
            "chunk_kind": "table_row",
            "fact_type": "lab_or_vital_result",
            "document_id": "doc-office",
            "section_title": "VITALS",
            "test_name": "Blood Pressure",
            "result_value": "128/76",
            "unit": "mmHg",
            "row_date": "2026-03-26",
        },
    })
    rag._scroll_all_points = lambda scroll_filter=None, limit=512: [home, office]

    docs = rag._retrieve_fact_documents("What was the office blood pressure?", document_ids=["doc-home", "doc-office"])

    assert docs[0].metadata["document_id"] == "doc-office"


def test_chronology_controls_inject_missing_selected_document():
    from langchain_core.documents import Document
    from agent_graph.tool_clinical_notes_rag import ClinicalNotesRAGTool

    rag = object.__new__(ClinicalNotesRAGTool)
    existing = Document(
        page_content="Earlier BP 130/80",
        metadata={"document_id": "doc-1", "encounter_date": "2025-07-01T00:00:00", "chunk_index": 1},
    )
    later = Document(
        page_content="Later BP 128/78",
        metadata={"document_id": "doc-2", "encounter_date": "2025-08-01T00:00:00", "chunk_index": 1},
    )
    rag._bm25_fallback_all_chunks = lambda *args, **kwargs: [later] if kwargs.get("document_ids") == ["doc-2"] else []

    docs = rag._apply_chronology_controls(
        "Compare blood pressure trend over time",
        [existing],
        patient_mrn="MRN001",
        document_ids=["doc-1", "doc-2"],
    )

    assert [d.metadata["document_id"] for d in docs] == ["doc-1", "doc-2"]


def test_source_header_includes_authoritative_dates_and_chunk_kind():
    from agent_graph.tool_clinical_notes_rag import ClinicalNotesRAGTool

    header = ClinicalNotesRAGTool._format_source_header(1, {
        "document_id": "doc-1",
        "patient_mrn": "MRN001",
        "document_type": "progress_note",
        "section_title": "Assessment",
        "page_number": 2,
        "encounter_date": "2025-08-01T00:00:00",
        "document_date": "2025-08-02",
        "chunk_kind": "clinical_event",
    })

    assert "Encounter Date: 2025-08-01T00:00:00" in header
    assert "Document Date: 2025-08-02" in header
    assert "Chunk Kind: clinical_event" in header


def test_process_document_adds_fact_documents_without_removing_text_chunks(monkeypatch):
    import document_processing as dp

    markdown = """
MRN: MRN001
Date of Service: 08/10/2025

## Lab Results
| Test | Result | Unit | Reference Range |
| --- | --- | --- | --- |
| Creatinine | 1.4 | mg/dL | 0.7-1.2 |

## Plan
Started lisinopril 10 mg PO daily for hypertension.
"""
    monkeypatch.setattr(dp, "extract_text", lambda path: [{"markdown": markdown, "page_number": 1}])
    monkeypatch.setattr(dp, "validate_extracted_text", lambda *args, **kwargs: (True, "ok"))

    docs = dp.process_document(
        file_path="fake.pdf",
        chunk_size=1200,
        chunk_overlap=80,
        metadata_overrides={"document_id": "doc-1", "patient_mrn": "MRN001"},
    )

    kinds = [d.metadata.get("chunk_kind") for d in docs]
    assert "text_chunk" in kinds
    assert "table_row" in kinds
    fact = next(d for d in docs if d.metadata.get("chunk_kind") == "table_row")
    assert fact.metadata["test_name"] == "Creatinine"
    assert fact.metadata["result_value"] == "1.4"
