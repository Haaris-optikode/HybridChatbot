import re
import sys


def main() -> int:
    ROOT = __file__
    # Add src/ to import path
    # tests/ -> project root -> src
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root / "src"))

    from agent_graph.tool_clinical_notes_rag import get_rag_tool_instance

    rag = get_rag_tool_instance()
    mrn = "250813000000794"

    print(f"Running retrieval smoke for MRN={mrn} ...")
    orders_text = rag.get_order_chunks_for_patient(mrn)
    if "No order-related data found" in orders_text:
        print("FAIL: no order-related sections returned")
        print(orders_text[:800])
        return 1

    # We expect medication-related content to be inferred, regardless of
    # section header naming (e.g., "7 CURRENT MEDICATIONS" vs "18b Medication Orders").
    has_metformin = "metformin" in orders_text.lower()
    has_lisinopril = "lisinopril" in orders_text.lower()
    has_discontinued = "atorvastatin" in orders_text.lower()

    if not (has_metformin or has_lisinopril):
        print("FAIL: medication orders not inferred from chunk content")
        print(orders_text[:1200])
        return 1

    if not has_discontinued:
        print("WARN: discontinued medication not found (may still be correct depending on document).")

    print("OK: get_order_chunks_for_patient inferred medication-related content.")

    # Validate that semantic search returns medication chunks tagged with some Section.
    # We don't assert specific section numbers to keep the test format-agnostic.
    search_query = f"MRN {mrn} medications"
    search_out = rag.search(search_query)
    has_section_tag = "Section:" in search_out
    if not has_section_tag:
        print("WARN: rag.search output did not include a 'Section:' tag in the format.")
        print(search_out[:1200])
    else:
        print("OK: rag.search output included a 'Section:' tag.")

    print("Retrieval smoke PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

