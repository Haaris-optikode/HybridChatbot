"""Comprehensive chunk quality audit — checks all 79 Qdrant chunks for known failure patterns."""
import requests
import json
import re

r = requests.post(
    "http://localhost:6333/collections/PatientData/points/scroll",
    json={"limit": 100, "with_payload": True, "with_vector": False},
)
pts = r.json()["result"]["points"]
print(f"Total chunks: {len(pts)}\n")

issues = []
section_titles = []

for p in pts:
    pc = p["payload"].get("page_content", "")
    meta = p["payload"].get("metadata", {})
    sec = meta.get("section_title", "?")
    section_titles.append(sec)

    # --- BAD: ICD-10 codes as section titles ---
    if re.match(r"^[A-Z]\d+[\.\-]", sec) or re.match(r"^\([A-Z]\d+", sec):
        issues.append(f"[ICD section title] [{sec}]")

    # --- BAD: Insurance/alphanumeric codes as section titles ---
    if re.match(r"^\d+[A-Z0-9]{2,}[\-]", sec):
        issues.append(f"[Insurance code section] [{sec}]")

    # --- BAD: Time values as section titles ---
    if re.match(r"^\d{1,2}:\d{2}", sec):
        issues.append(f"[Time as section title] [{sec}]")

    # --- BAD: Empty/very short content ---
    if len(pc.strip()) < 60:
        issues.append(f"[Tiny chunk] section=[{sec}] len={len(pc.strip())}")

    # --- BAD: Mid-sentence start (body starts with lowercase, not list/table/header) ---
    # Skip the [prefix] header line; inspect the first real content line
    lines = [l for l in pc.split("\n") if l.strip() and not l.strip().startswith("[")]
    if lines:
        first_real = lines[0].strip()
        if (
            re.match(r"^[a-z]", first_real)
            and not first_real.startswith("|")
            and not first_real.startswith("-")
            and not first_real.startswith("*")
        ):
            issues.append(f"[Mid-sentence start] [{sec}]: \"{first_real[:70]}\"")

    # --- BAD: Linearized vitals (sequences of bare numbers) ---
    if re.search(r"\b\d{2,3}\s+\d{2,3}\s+\d{2,3}\s+\d{2,3}\b", pc):
        issues.append(f"[Linearized number blob] [{sec}]")

    # --- BAD: Section title is just a number or punctuation ---
    if re.match(r"^[\d\s\.\-\(\)]+$", sec.strip()):
        issues.append(f"[Numeric-only section title] [{sec}]")

print("=" * 60)
print("SECTION TITLES (all 79):")
for i, t in enumerate(section_titles, 1):
    print(f"  {i:3d}. {t}")

print()
print("=" * 60)
if issues:
    print(f"ISSUES FOUND: {len(issues)}")
    for iss in issues:
        print(f"  - {iss}")
else:
    print("NO ISSUES FOUND — all chunks pass quality checks.")
print("=" * 60)

# --- Content spot-checks ---
checks = [
    ("BNP", "1,842"),
    ("sacubitril", "Sacubitril"),
    ("lisinopril", "lisinopril"),
    ("ejection fraction", "ejection fraction"),
    ("thoracentesis", "thoracentesis"),
    ("discharge", "discharge"),
]
print("\nCONTENT SPOT-CHECKS:")
for keyword, display in checks:
    hits = [p for p in pts if keyword.lower() in p["payload"].get("page_content", "").lower()]
    print(f"  '{display}': {len(hits)} chunk(s) contain it")

# --- Table integrity check ---
print("\nTABLE INTEGRITY:")
# Only consider chunks that have pipe chars in body lines (skip the [prefix] header line)
table_chunks = []
for p in pts:
    pc = p["payload"].get("page_content", "")
    body_lines = [l for l in pc.split("\n") if l.strip().startswith("|")]
    if body_lines:  # has actual markdown table rows
        table_chunks.append(p)
print(f"  {len(table_chunks)} chunks contain markdown tables")
broken = []
for p in table_chunks:
    pc = p["payload"].get("page_content", "")
    sec = p["payload"].get("metadata", {}).get("section_title", "?")
    pipe_lines = [l for l in pc.split("\n") if l.strip().startswith("|")]
    # Table must have header + separator (|---|) + at least one data row = 3+ lines
    has_separator = any(re.match(r"\|[-:\s]+\|", l) for l in pipe_lines)
    if len(pipe_lines) < 3 or not has_separator:
        broken.append(f"[{sec}] ({len(pipe_lines)} pipe lines, sep={has_separator})")
if broken:
    print(f"  WARNING: {len(broken)} possible partial table chunks:")
    for b in broken:
        print(f"    - {b}")
else:
    print("  All table chunks: header + separator + data rows present")
