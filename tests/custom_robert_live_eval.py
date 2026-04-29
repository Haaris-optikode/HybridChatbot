"""
Custom live accuracy evaluation for Robert inpatient record.

This is a brand-new test harness (not reusing existing regression scripts).

What it measures:
1) Factual answer quality for Robert-specific questions.
2) Complex-query handling quality.
3) Complex-query model compliance: verifies GPT-5.4-mini appears in usage telemetry.

Usage:
    c:/Users/dell/Desktop/HybridChatbot/venv/Scripts/python.exe tests/custom_robert_live_eval.py
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
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "logs" / "custom_robert_live_eval_results.json"


def _contains_all(reply: str, words: list[str]) -> bool:
    r = (reply or "").lower()
    return all(w.lower() in r for w in words)


def _contains_any(reply: str, words: list[str]) -> bool:
    r = (reply or "").lower()
    return any(w.lower() in r for w in words)


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


def _ask_chat(question: str, thinking: bool = False) -> tuple[str, dict, float]:
    headers = {"Content-Type": "application/json"}
    payload = {
        "message": question,
        "session_id": f"custom-eval-{uuid.uuid4()}",
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
    p = "For patient Robert James Whitfield (MRN: MRN-2026-004782), "

    return [
        EvalCase(
            1,
            "demographics",
            p + "what is the full name and date of birth?",
            groups=[["robert james whitfield"], ["june 14, 1963", "06/14/1963"]],
            min_groups=2,
        ),
        EvalCase(
            2,
            "drug_allergies",
            p + "list the drug allergies and severe non-drug allergy.",
            groups=[["penicillin"], ["sulfonamide", "sulfa"], ["contrast"], ["shellfish", "anaphylaxis"]],
            min_groups=3,
        ),
        EvalCase(
            3,
            "code_status",
            p + "what is the code status and who is healthcare power of attorney?",
            groups=[["full code"], ["linda"], ["power of attorney", "advance directive", "surrogate"]],
            min_groups=2,
        ),
        EvalCase(
            4,
            "admission_bnp",
            p + "what was the BNP level on admission?",
            groups=[["1,842", "1842"], ["bnp"]],
            min_groups=2,
        ),
        EvalCase(
            5,
            "home_meds",
            p + "what medications was he taking at home before admission?",
            groups=[
                ["lisinopril"], ["carvedilol"], ["metformin"], ["furosemide"],
                ["atorvastatin"], ["aspirin"],
            ],
            min_groups=4,
        ),
        EvalCase(
            6,
            "discharge_meds_core",
            p + "what are the key discharge heart-failure and anticoagulation medications?",
            groups=[
                ["sacubitril", "valsartan", "entresto"],
                ["carvedilol"],
                ["spironolactone"],
                ["empagliflozin"],
                ["apixaban"],
            ],
            min_groups=4,
        ),
        EvalCase(
            7,
            "ef_trajectory",
            p + "how did ejection fraction change during hospitalization?",
            groups=[["25"], ["30"], ["ejection fraction", "ef"], ["improv", "increase"]],
            min_groups=3,
        ),
        EvalCase(
            8,
            "follow_up",
            p + "what post-discharge follow-up specialties were planned?",
            groups=[["cardiology"], ["nephrology"], ["primary care", "pcp"]],
            min_groups=2,
        ),
        EvalCase(
            9,
            "lisinopril_stop_reason",
            p + "why was lisinopril stopped at discharge?",
            groups=[["sacubitril", "entresto", "arni"], ["washout", "36-hour", "36 hour"], ["lisinopril"]],
            min_groups=2,
        ),
        EvalCase(
            10,
            "complex_med_changes_why",
            p + "what changed in his medications during admission and why were those changes made?",
            groups=[
                ["sacubitril", "entresto", "arni"],
                ["carvedilol", "uptitrate", "increase"],
                ["furosemide", "diures", "volume"],
                ["apixaban", "atrial fibrillation", "stroke prevention"],
            ],
            min_groups=3,
            is_complex=True,
            require_gpt54mini=True,
        ),
        EvalCase(
            11,
            "complex_trajectory_critical",
            p + "describe the trajectory over time and the most critical turning points in the hospital course.",
            groups=[
                ["trajectory", "over time", "hospital course"],
                ["decompensated", "heart failure", "volume overload"],
                ["diures", "weight", "fluid"],
                ["discharge", "follow-up", "outpatient"],
            ],
            min_groups=3,
            is_complex=True,
            require_gpt54mini=True,
        ),
        EvalCase(
            12,
            "complex_risk_reasoning",
            p + "which findings were most critical for readmission risk and prognosis?",
            groups=[
                ["readmission", "prognosis", "risk"],
                ["heart failure", "ejection fraction", "ef"],
                ["volume", "weight", "bnp"],
                ["follow-up", "medication adherence", "warning signs"],
            ],
            min_groups=3,
            is_complex=True,
            require_gpt54mini=True,
        ),
        EvalCase(
            13,
            "procedures_and_studies",
            p + "what key procedures or diagnostic studies were done during hospitalization?",
            groups=[
                ["echocardi", "echo"],
                ["thoracentesis", "pleural"],
                ["catheterization", "cardiac cath", "coronary"],
            ],
            min_groups=2,
        ),
        EvalCase(
            14,
            "discharge_diet",
            p + "what dietary and fluid restrictions were advised at discharge?",
            groups=[
                ["sodium", "2g", "2 g"],
                ["fluid", "1.5", "restriction"],
                ["diabetic", "heart-healthy", "heart healthy"],
            ],
            min_groups=2,
        ),
        EvalCase(
            15,
            "complex_admit_vs_discharge_meds",
            p + "compare admission/home medications versus discharge medications and explain the clinical rationale for the major changes.",
            groups=[
                ["home", "admission"],
                ["discharge"],
                ["sacubitril", "entresto", "arni"],
                ["furosemide", "diures", "volume overload"],
                ["apixaban", "atrial fibrillation", "stroke prevention"],
            ],
            min_groups=4,
            is_complex=True,
            require_gpt54mini=True,
        ),
        EvalCase(
            16,
            "return_precautions",
            p + "what red-flag symptoms should trigger emergency evaluation after discharge?",
            groups=[
                ["chest pain"],
                ["shortness of breath", "sob", "dyspnea"],
                ["weight gain", "2 lbs", "5 lbs"],
                ["syncope", "faint", "palpit"],
                ["blood sugar", "glucose"],
            ],
            min_groups=3,
        ),
    ]


def run() -> int:
    print("=" * 72)
    print("CUSTOM ROBERT LIVE EVAL (NEW HARNESS)")
    print(f"Server: {BASE_URL}")
    print("=" * 72)

    # Health first
    hr = requests.get(f"{BASE_URL}/api/health", timeout=20)
    hr.raise_for_status()
    health = hr.json()
    print(f"Health status: {health.get('status')} | Qdrant: {(health.get('qdrant') or {}).get('status')}")

    token = None
    cases = _default_cases()

    rows: list[dict] = []
    passed = 0
    simple_total = 0
    simple_pass = 0
    complex_total = 0
    complex_pass = 0
    complex_model_ok = 0

    for c in cases:
        reply, usage, wall_s = _ask_chat(c.question, thinking=False)
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
        "document_target": "data/unstructured_docs/clinical_notes/patient_inpatient_record.pdf",
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

    # Return non-zero if either quality or model-compliance fails.
    if overall_pct < 80.0:
        return 2
    if complex_model_pct < 100.0:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
