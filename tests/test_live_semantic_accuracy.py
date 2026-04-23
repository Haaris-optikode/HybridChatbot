"""
Live API semantic regression tests.

These tests are designed for a running backend at localhost:7860 and validate:
  - health/auth endpoints
  - semantic response quality (concept-group scoring, not exact match)
  - streaming chat endpoint behavior
  - upload/list/status/delete flows for DOCX and image medical notes
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path

import pytest
import requests


BASE_URL = "http://localhost:7860"


def _token_headers(user_id: str = "semantic-live-test") -> dict:
    r = requests.post(
        f"{BASE_URL}/api/auth/token",
        json={"user_id": user_id, "role": "clinician"},
        timeout=20,
    )
    r.raise_for_status()
    token = r.json()["token"]
    return {"Authorization": f"Bearer {token}"}


def _chat(message: str, headers: dict, timeout: int = 200) -> str:
    r = requests.post(
        f"{BASE_URL}/api/chat",
        json={"message": message, "session_id": f"live-sem-{uuid.uuid4()}"},
        headers=headers,
        timeout=timeout,
    )
    r.raise_for_status()
    payload = r.json()
    return payload.get("reply", "")


def _chat_stream(message: str, headers: dict, timeout: int = 200) -> dict:
    r = requests.post(
        f"{BASE_URL}/api/chat/stream",
        json={"message": message, "session_id": f"live-sem-stream-{uuid.uuid4()}"},
        headers=headers,
        stream=True,
        timeout=timeout,
    )
    r.raise_for_status()

    content = ""
    done_event = {}
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        try:
            event = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        if event.get("type") == "token":
            content += event.get("content", "")
        elif event.get("type") == "done":
            done_event = event
        elif event.get("type") == "error":
            pytest.fail(f"Streaming endpoint returned error: {event.get('content')}")

    return {
        "content": content,
        "done": done_event,
    }


def _score_concept_groups(reply: str, groups: list[list[str]]) -> int:
    text = (reply or "").lower()
    score = 0
    for group in groups:
        if any(token in text for token in group):
            score += 1
    return score


def _upload_document(path: Path, headers: dict) -> str:
    with path.open("rb") as f:
        files = {"file": (path.name, f)}
        r = requests.post(
            f"{BASE_URL}/api/documents/upload",
            headers=headers,
            files=files,
            timeout=60,
        )
    r.raise_for_status()
    payload = r.json()
    return payload["document_id"]


def _wait_terminal_status(document_id: str, headers: dict, timeout_s: int = 240) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = requests.get(
            f"{BASE_URL}/api/documents/{document_id}",
            headers=headers,
            timeout=20,
        )
        r.raise_for_status()
        payload = r.json()
        status = payload.get("status")
        if status in {"ready", "duplicate", "error"}:
            return payload
        time.sleep(2)
    pytest.fail(f"Document {document_id} did not reach terminal status within timeout")


def _local_tesseract_available() -> bool:
    if shutil.which("tesseract"):
        return True

    windows_candidates = [
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        Path.home() / "AppData" / "Local" / "Tesseract-OCR" / "tesseract.exe",
    ]
    if any(candidate.exists() for candidate in windows_candidates):
        return True

    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def live_server_headers():
    try:
        health = requests.get(f"{BASE_URL}/api/health", timeout=10)
        health.raise_for_status()
    except Exception as exc:
        pytest.skip(f"Live server is not reachable at {BASE_URL}: {exc}")
    return _token_headers(user_id=f"semantic-suite-{uuid.uuid4()}")


def test_live_health_and_auth_contract(live_server_headers):
    r = requests.get(f"{BASE_URL}/api/health", timeout=15)
    r.raise_for_status()
    data = r.json()

    assert data.get("status") in {"ok", "degraded"}
    assert "qdrant" in data
    assert data["qdrant"].get("status") == "ok"
    assert data["qdrant"].get("vectors", 0) >= 1

    # Verify token works against a protected endpoint.
    docs_r = requests.get(f"{BASE_URL}/api/documents", headers=live_server_headers, timeout=20)
    docs_r.raise_for_status()
    assert "documents" in docs_r.json()


def test_semantic_queries_use_concept_group_accuracy(live_server_headers):
    cases = [
        {
            "name": "surrogate_and_code_status",
            "query": "For patient Robert James Whitfield, if he loses decision capacity, who is surrogate and what is his code status?",
            "groups": [["linda"], ["full code", "code status"], ["power of attorney", "surrogate"]],
            "min_groups": 2,
        },
        {
            "name": "net_weight_change",
            "query": "For patient Robert James Whitfield, how much net weight was removed during this stay (kg or pounds)?",
            "groups": [["18.2", "18.4", "40.5", "40.6"], ["kg", "pound"], ["weight", "fluid", "diures"]],
            "min_groups": 2,
        },
        {
            "name": "gdmt_discharge_meds",
            "query": "For patient Robert James Whitfield, which discharge heart-failure medications reflect guideline-directed therapy including ARNI-based treatment?",
            "groups": [
                ["sacubitril", "valsartan", "entresto"],
                ["carvedilol", "beta-blocker"],
                ["spironolactone", "mra"],
                ["empagliflozin", "sglt2"],
            ],
            "min_groups": 3,
        },
        {
            "name": "inpatient_procedures",
            "query": "For patient Robert James Whitfield, what in-hospital procedures or diagnostic studies were performed for cardiac and volume status evaluation?",
            "groups": [
                ["bnp", "troponin", "laboratory"],
                ["telemetry", "rhythm", "atrial fibrillation"],
                ["echocardi", "ejection fraction"],
                ["intake", "output", "daily weights", "diures"],
            ],
            "min_groups": 3,
        },
        {
            "name": "post_discharge_followup",
            "query": "For patient Robert James Whitfield, which specialty follow-ups were arranged after discharge?",
            "groups": [
                ["cardiology"],
                ["nephrology", "kidney"],
                ["primary care", "pcp"],
                ["icd", "echocardiogram", "diabetes education"],
            ],
            "min_groups": 2,
        },
        {
            "name": "ef_trend",
            "query": "For patient Robert James Whitfield, how did ejection fraction evolve over the hospitalization?",
            "groups": [
                ["ejection fraction", "ef"],
                ["25%"],
                ["30%"],
                ["improv", "declin", "trend"],
            ],
            "min_groups": 3,
        },
    ]

    failures = []
    for case in cases:
        reply = _chat(case["query"], live_server_headers)
        lower = reply.lower()

        if "need the patient name or mrn" in lower:
            failures.append(f"{case['name']}: router asked for patient ID despite explicit patient anchor")
            continue

        score = _score_concept_groups(reply, case["groups"])
        if score < case["min_groups"]:
            failures.append(
                f"{case['name']}: concept-group score {score}/{len(case['groups'])} below required {case['min_groups']}"
            )

    assert not failures, "\n".join(failures)


def test_streaming_chat_semantic_response(live_server_headers):
    result = _chat_stream(
        "For patient Robert James Whitfield, summarize the trajectory of his heart failure admission in concise bullet points.",
        live_server_headers,
        timeout=220,
    )
    content = result["content"].lower()
    done = result["done"]

    assert len(content) > 120
    assert done.get("session_id")
    assert any(token in content for token in ["heart failure", "diures", "ejection fraction", "discharge"])


def test_upload_docx_status_list_and_delete(live_server_headers, tmp_path: Path):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    uid = uuid.uuid4().hex[:10]
    d.add_heading("Clinical Follow-Up Note", level=1)
    d.add_paragraph(f"Patient Name: Test Patient {uid}")
    d.add_paragraph(f"MRN: MRN-LIVE-{uid}")
    d.add_paragraph(
        "Assessment: Acute on chronic systolic heart failure with pulmonary congestion and volume overload. "
        "Plan includes intensified diuresis, electrolyte checks, and cardiology follow-up."
    )
    d.add_paragraph("Medications: sacubitril/valsartan, carvedilol, spironolactone, empagliflozin.")
    d.add_paragraph("Follow-up: cardiology in one week, PCP in two weeks.")

    path = tmp_path / "live_upload_note.docx"
    d.save(path)

    document_id = _upload_document(path, live_server_headers)
    status_payload = _wait_terminal_status(document_id, live_server_headers)

    assert status_payload["status"] in {"ready", "duplicate"}
    if status_payload["status"] == "error":
        pytest.fail(f"DOCX upload ended in error: {status_payload.get('error')}")

    get_r = requests.get(f"{BASE_URL}/api/documents/{document_id}", headers=live_server_headers, timeout=20)
    get_r.raise_for_status()
    assert get_r.json().get("document_id") == document_id

    list_r = requests.get(f"{BASE_URL}/api/documents", headers=live_server_headers, timeout=20)
    list_r.raise_for_status()
    docs = list_r.json().get("documents", [])
    assert any(d.get("document_id") == document_id for d in docs)

    del_r = requests.delete(f"{BASE_URL}/api/documents/{document_id}", headers=live_server_headers, timeout=30)
    del_r.raise_for_status()
    assert del_r.json().get("status") == "deleted"


def test_upload_image_medical_note_ingestion(live_server_headers, tmp_path: Path):
    if not _local_tesseract_available():
        pytest.skip("Tesseract not available; live image upload ingestion cannot be validated")

    image_mod = pytest.importorskip("PIL.Image")
    draw_mod = pytest.importorskip("PIL.ImageDraw")

    uid = uuid.uuid4().hex[:8]
    image = image_mod.new("RGB", (1800, 1200), color="white")
    draw = draw_mod.Draw(image)
    lines = [
        f"Patient Name: OCR Test {uid}",
        f"MRN: MRN-OCR-{uid}",
        "Assessment: CHF exacerbation with edema and dyspnea.",
        "Plan: Daily weights, strict I/O, loop diuretic optimization.",
        "Medications: furosemide and carvedilol.",
        "Allergies: penicillin rash.",
    ]
    y = 40
    for line in lines:
        draw.text((40, y), line, fill="black")
        y += 60

    path = tmp_path / "live_upload_note.png"
    image.save(path)

    document_id = _upload_document(path, live_server_headers)
    status_payload = _wait_terminal_status(document_id, live_server_headers)

    assert status_payload["status"] in {"ready", "duplicate"}
    if status_payload["status"] == "error":
        pytest.fail(f"Image upload ended in error: {status_payload.get('error')}")

    del_r = requests.delete(f"{BASE_URL}/api/documents/{document_id}", headers=live_server_headers, timeout=30)
    del_r.raise_for_status()
    assert del_r.json().get("status") == "deleted"
