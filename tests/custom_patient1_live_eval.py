"""
Custom live accuracy evaluation for patient1_clinical_note.pdf.

This is a new, document-specific production gate (not using pre-made test suites).

Coverage:
1) Factual quality across demographics, allergies, meds, PMH/PSH, family/social, vitals/labs.
2) Complex reasoning quality.
3) Complex-query model compliance: verifies GPT-5.4-mini appears in usage telemetry.

Usage:
    c:/Users/dell/Desktop/HybridChatbot/venv/Scripts/python.exe tests/custom_patient1_live_eval.py
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests


BASE_URL = "http://localhost:7860"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "logs" / "custom_patient1_live_eval_results.json"


def _concept_group_score(reply: str, groups: list[list[str]]) -> int:
    r = (reply or "").lower()
    score = 0
    for group in groups:
        if any(token.lower() in r for token in group):
            score += 1
    return score


@dataclass
class EvalCase:
    case_id: int
    label: str
    question: str
    groups: list[list[str]]
    min_groups: int
    is_complex: bool = False
    require_gpt54mini: bool = False
    extra_check: Callable[[str], bool] | None = None


def _get_token() -> str:
    r = requests.post(
        f"{BASE_URL}/api/auth/token",
        json={"user_id": "custom-p1-eval", "role": "clinician"},
        timeout=20,
    )
    r.raise_for_status()
    return r.json()["token"]


def _ask_chat(question: str, token: str, thinking: bool = False) -> tuple[str, dict, float]:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "message": question,
        "session_id": f"custom-p1-{uuid.uuid4()}",
        "thinking": thinking,
        "llm_provider": "openai",
    }
    t0 = time.perf_counter()
    r = requests.post(f"{BASE_URL}/api/chat", json=payload, headers=headers, timeout=240)
    r.raise_for_status()
    wall_s = round(time.perf_counter() - t0, 2)
    data = r.json()
    return data.get("reply", ""), (data.get("usage") or {}), wall_s


def _default_cases() -> list[EvalCase]:
    p = "For patient MRN 250813000000794, "

    return [
        EvalCase(
            1,
            "demographics_core",
            p + "what are age, gender, race, and ethnicity?",
            groups=[["45"], ["male"], ["asian"], ["south asian"]],
            min_groups=4,
        ),
        EvalCase(
            2,
            "anthropometrics_blood_group",
            p + "what are the height, weight, and blood group?",
            groups=[["5'8", "172.7"], ["200", "90.7"], ["o positive", "o+"]],
            min_groups=3,
        ),
        EvalCase(
            3,
            "penicillin_allergy_severity",
            p + "summarize the penicillin allergy severity and emergency management details.",
            groups=[
                ["penicillin"],
                ["severe", "anaphylaxis"],
                ["hives", "facial", "lip swelling", "shortness of breath"],
                ["epinephrine", "auto-injector", "medical alert"],
            ],
            min_groups=3,
        ),
        EvalCase(
            4,
            "active_diagnoses",
            p + "what are the active diagnoses with ICD codes?",
            groups=[
                ["e11.9", "type 2 diabetes"],
                ["i10", "hypertension"],
                ["e66.9", "overweight"],
            ],
            min_groups=3,
        ),
        EvalCase(
            5,
            "resolved_conditions",
            p + "which conditions are resolved and what time windows are documented?",
            groups=[
                ["prediabetes", "2018", "2020"],
                ["hyperlipidemia", "2018", "2020"],
            ],
            min_groups=2,
        ),
        EvalCase(
            6,
            "current_meds",
            p + "what are the current ongoing medications with dose and frequency?",
            groups=[
                ["metformin", "500", "twice"],
                ["lisinopril", "10", "once daily"],
            ],
            min_groups=2,
        ),
        EvalCase(
            7,
            "lisinopril_safety",
            p + "when was lisinopril started and what renal/electrolyte safety labs are documented?",
            groups=[
                ["03/15/2022", "march", "2022"],
                ["creatinine", "0.9"],
                ["potassium", "4.2"],
            ],
            min_groups=2,
        ),
        EvalCase(
            8,
            "discontinued_statin",
            p + "which medication was discontinued and why?",
            groups=[
                ["atorvastatin"],
                ["12/2020", "2020"],
                ["resolved hyperlipidemia", "lipid targets"],
            ],
            min_groups=2,
        ),
        EvalCase(
            9,
            "surgical_history",
            p + "what past surgeries/procedures are documented and outcomes?",
            groups=[
                ["appendectomy", "2015", "laparoscopic", "no complications"],
                ["skin biopsy", "2017", "benign", "no recurrence"],
            ],
            min_groups=2,
        ),
        EvalCase(
            10,
            "family_history",
            p + "summarize major family history risks relevant to cardiometabolic disease.",
            groups=[
                ["father", "type 2 diabetes", "myocardial infarction"],
                ["brother", "prediabetes"],
            ],
            min_groups=2,
        ),
        EvalCase(
            11,
            "social_history",
            p + "what are smoking, alcohol, and support-system details?",
            groups=[
                ["never smoker", "non-smoker"],
                ["1-2 beers", "alcohol"],
                ["wife", "support", "meal prep", "monitoring"],
            ],
            min_groups=2,
        ),
        EvalCase(
            12,
            "vitals_and_labs",
            p + "what were the key vitals and labs from the July 2025 diabetes follow-up?",
            groups=[
                ["130/80", "bp"],
                ["a1c", "7.2"],
                ["fasting glucose", "115"],
                ["ldl", "110"],
            ],
            min_groups=3,
        ),
        EvalCase(
            13,
            "complex_trend",
            p + "compare trends over time between the July and August 2025 follow-ups for blood pressure and A1C.",
            groups=[
                ["july", "130/80"],
                ["august", "128/78"],
                ["a1c", "7.2", "unchanged"],
                ["trend", "compare", "over time"],
            ],
            min_groups=3,
            is_complex=True,
            require_gpt54mini=True,
        ),
        EvalCase(
            14,
            "complex_medication_reasoning",
            p + "compare current medications versus discontinued medications and explain why those choices were made.",
            groups=[
                ["metformin"],
                ["lisinopril"],
                ["atorvastatin", "discontinued"],
                ["resolved hyperlipidemia", "lipid targets"],
            ],
            min_groups=3,
            is_complex=True,
            require_gpt54mini=True,
        ),
        EvalCase(
            15,
            "complex_risk_prognosis",
            p + "which findings are most important for future cardiometabolic risk and prognosis, and why?",
            groups=[
                ["a1c", "7.2", "above target"],
                ["overweight", "bmi 28.5"],
                ["family history", "father", "myocardial infarction"],
                ["blood pressure", "stable", "hypertension"],
            ],
            min_groups=3,
            is_complex=True,
            require_gpt54mini=True,
        ),
        EvalCase(
            16,
            "complex_plan_prioritization",
            p + "what were the prioritized interventions in the diabetes management plan and what triggered them?",
            groups=[
                ["nutrition", "low-carb", "south asian"],
                ["exercise", "150 minutes"],
                ["endocrinology", "09/05/2025"],
                ["sglt2", "if a1c remains >7"],
            ],
            min_groups=3,
            is_complex=True,
            require_gpt54mini=True,
        ),
    ]


def run() -> int:
    print("=" * 72)
    print("CUSTOM PATIENT1 LIVE EVAL (NEW HARNESS)")
    print(f"Server: {BASE_URL}")
    print("=" * 72)

    hr = requests.get(f"{BASE_URL}/api/health", timeout=20)
    hr.raise_for_status()
    health = hr.json()
    print(f"Health status: {health.get('status')} | Qdrant: {(health.get('qdrant') or {}).get('status')}")

    token = _get_token()
    cases = _default_cases()

    rows: list[dict] = []
    passed = 0
    simple_total = 0
    simple_pass = 0
    complex_total = 0
    complex_pass = 0
    complex_model_ok = 0

    for c in cases:
        reply, usage, wall_s = _ask_chat(c.question, token=token, thinking=False)
        score = _concept_group_score(reply, c.groups)
        base_ok = score >= c.min_groups

        extra_ok = True if c.extra_check is None else bool(c.extra_check(reply))

        models = [str(m) for m in (usage.get("models") or [])]
        model_ok = True
        if c.require_gpt54mini:
            model_ok = any("gpt-5.4-mini" in m.lower() for m in models)

        ok = base_ok and extra_ok and model_ok
        passed += int(ok)

        if c.is_complex:
            complex_total += 1
            complex_pass += int(ok)
            complex_model_ok += int(model_ok)
        else:
            simple_total += 1
            simple_pass += int(ok)

        status = "PASS" if ok else "FAIL"
        print(f"Case {c.case_id:02d} [{status}] {c.label} | score={score}/{len(c.groups)} | t={wall_s:.2f}s")
        if c.is_complex:
            print(f"         complex model check (gpt-5.4-mini): {'OK' if model_ok else 'MISS'} | models={models}")
        if not ok:
            print(f"         question: {c.question}")
            print(f"         reply: {reply[:420].replace(chr(10), ' ')}")

        rows.append(
            {
                "case_id": c.case_id,
                "label": c.label,
                "question": c.question,
                "is_complex": c.is_complex,
                "score": score,
                "max_score": len(c.groups),
                "min_groups": c.min_groups,
                "base_ok": base_ok,
                "model_ok": model_ok,
                "passed": ok,
                "latency_s": wall_s,
                "usage": usage,
                "reply": reply,
            }
        )

    total = len(cases)
    overall_pct = round((passed / total) * 100.0, 1)
    simple_pct = round((simple_pass / simple_total) * 100.0, 1) if simple_total else 0.0
    complex_pct = round((complex_pass / complex_total) * 100.0, 1) if complex_total else 0.0
    complex_model_pct = round((complex_model_ok / complex_total) * 100.0, 1) if complex_total else 0.0

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": BASE_URL,
        "document_target": "data/unstructured_docs/clinical_notes/patient1_clinical_note.pdf",
        "total_cases": total,
        "passed_cases": passed,
        "overall_accuracy_pct": overall_pct,
        "simple": {
            "total": simple_total,
            "passed": simple_pass,
            "accuracy_pct": simple_pct,
        },
        "complex": {
            "total": complex_total,
            "passed": complex_pass,
            "accuracy_pct": complex_pct,
            "gpt_5_4_mini_usage_ok": complex_model_ok,
            "gpt_5_4_mini_usage_pct": complex_model_pct,
        },
        "results": rows,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"Overall accuracy: {passed}/{total} ({overall_pct}%)")
    print(f"Simple accuracy : {simple_pass}/{simple_total} ({simple_pct}%)")
    print(f"Complex accuracy: {complex_pass}/{complex_total} ({complex_pct}%)")
    print(
        "Complex model compliance (gpt-5.4-mini): "
        f"{complex_model_ok}/{complex_total} ({complex_model_pct}%)"
    )
    print(f"Saved detailed report: {OUTPUT_PATH}")
    print("=" * 72)

    if overall_pct < 80.0:
        return 2
    if complex_model_pct < 100.0:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
