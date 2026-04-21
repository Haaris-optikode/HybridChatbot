"""
Memory profile for ClinicalNotesRAGTool.
Measures RSS before/after import, init, and query bursts.
Must be run with server STOPPED (Qdrant embedded lock).
"""
import sys, os, gc, time
sys.path.insert(0, "src")

import psutil
proc = psutil.Process(os.getpid())

def mb():
    return proc.memory_info().rss / 1024 / 1024

print("=== Memory Profile: ClinicalNotesRAGTool ===")
baseline = mb()
print(f"Baseline (Python interpreter):  {baseline:.1f} MB")

from agent_graph.tool_clinical_notes_rag import get_rag_tool_instance
after_import = mb()
print(f"After module imports:            {after_import:.1f} MB  (+{after_import-baseline:.1f} MB)")

rag = get_rag_tool_instance()
after_init = mb()
print(f"After RAG init (models loaded): {after_init:.1f} MB  (+{after_init-baseline:.1f} MB from baseline)")
print(f"  Components: PubMedBERT embed + cross-encoder + Qdrant client")

MRN = "MRN-2026-004782"
queries = [
    "Is this patient allergic to penicillin?",
    "What is the patient's code status?",
    "What medications was the patient on at home?",
    "What was the BNP on admission?",
    "What was the total weight loss during hospitalization?",
]

print(f"\n--- 5-query warm-up burst ---")
warmup_start = mb()
for i, q in enumerate(queries):
    before_q = mb()
    _ = rag.search(q, patient_mrn=MRN, document_id=None)
    after_q = mb()
    print(f"  Q{i+1}: {before_q:.1f} → {after_q:.1f} MB  ({after_q-before_q:+.1f} MB)")
    gc.collect()

after_5q = mb()
print(f"\nAfter 5 queries:  {after_5q:.1f} MB  (+{after_5q-after_init:.1f} MB vs init)")

print(f"\n--- 20 more queries (steady-state check) ---")
for _ in range(4):
    for q in queries:
        rag.search(q, patient_mrn=MRN, document_id=None)
    gc.collect()

steady = mb()
growth = steady - after_5q
print(f"After 25 total:   {steady:.1f} MB  ({growth:+.1f} MB vs warm-up)")
print(f"\n--- Summary ---")
print(f"  Init footprint:  {after_init - baseline:.1f} MB")
print(f"  Per-query leak:  {growth / 20:.2f} MB/query (over 20 post-warmup queries)")
if abs(growth) < 50:
    print("  RESULT: PASS — no significant memory leak (<50 MB growth over 20 queries)")
else:
    print(f"  RESULT: WARN — {growth:.1f} MB growth detected, investigate BM25 or embedding cache")
