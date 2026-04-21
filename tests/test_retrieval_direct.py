"""
Validate 5 critical retrieval queries directly against Qdrant (server must be STOPPED).
Tests the Phase 2.6 fixes: query rewriting + BM25 fallback.
"""
import sys
sys.path.insert(0, "src")
from agent_graph.tool_clinical_notes_rag import get_rag_tool_instance

rag = get_rag_tool_instance()
MRN = "MRN-2026-004782"

tests = [
    ("advance directive",  "what is the advance directive and code status?",
     ["full code", "linda"]),
    ("weight gain",        "how many pounds did patient gain before admission?",
     ["18"]),
    ("home medications",   "what medications were being taken at home before admission?",
     ["lisinopril", "carvedilol", "metformin"]),
    ("BNP admission",      "what was the BNP on admission?",
     ["1842"]),
    ("total weight loss",  "what was the total weight loss during hospitalization?",
     ["18.4"]),
]

passed = 0
for name, query, terms in tests:
    result = rag.search(query, patient_mrn=MRN, document_id=None)
    rl = result.lower()
    no_evidence = "no_evidence" in rl or "no sufficiently" in rl
    all_found = all(t.lower() in rl for t in terms)
    ok = not no_evidence and all_found
    passed += ok
    status = "PASS" if ok else "FAIL"
    missing = [t for t in terms if t.lower() not in rl]
    banner = next((l.strip() for l in result.split("\n")[:3] if "RETRIEVAL" in l), "")
    print(f"[{status}] {name}")
    if banner:
        print(f"  {banner}")
    if not ok:
        print(f"  no_evidence={no_evidence}  missing={missing if not no_evidence else terms}")
    print()

print(f"Total: {passed}/{len(tests)} PASS")
