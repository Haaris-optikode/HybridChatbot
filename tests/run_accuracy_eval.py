"""
13-question gold-standard accuracy evaluation.

Runs every question from SampleQuestions.txt against the live server and scores
each answer using the same pass criteria used in logs/sample_questions_eval_results.json.

Usage:
    python tests/run_accuracy_eval.py [--save]

Flags:
    --save   Overwrite logs/sample_questions_eval_results.json with the new results.
"""
import json
import sys
import time
from pathlib import Path

import requests

BASE_URL = "http://localhost:7860"
LOG_PATH = Path(__file__).parent.parent / "logs" / "sample_questions_eval_results.json"

# ---------------------------------------------------------------------------
# Evaluation definitions — (question, life_critical, rule_label, pass_fn)
# ---------------------------------------------------------------------------

def _lower_all(*words):
    """Return a function that checks all words appear in the response (case-insensitive)."""
    def _check(reply: str) -> bool:
        r = reply.lower()
        return all(w.lower() in r for w in words)
    return _check


def _lower_any(*words):
    """Return a function that checks any word appears in the response (case-insensitive)."""
    def _check(reply: str) -> bool:
        r = reply.lower()
        return any(w.lower() in r for w in words)
    return _check


def _hit_count(words, min_hits: int):
    """Return a function that counts how many words appear; passes if >= min_hits."""
    def _check(reply: str) -> bool:
        r = reply.lower()
        hits = sum(1 for w in words if w.lower() in r)
        return hits >= min_hits
    return _check


QUESTIONS = [
    {
        "id": 1,
        "question": "Is this patient allergic to penicillin?",
        "life_critical": True,
        "expected": "YES — urticaria and angioedema. Moderate. Documented since 1985. Avoid all penicillin derivatives.",
        "rule": "penicillin + reaction + allergy/avoidance",
        "pass_fn": _lower_all("penicillin", "urticaria"),
    },
    {
        "id": 2,
        "question": "Can this patient receive iodinated contrast?",
        "life_critical": True,
        "expected": "NOT without premedication — flushing, nausea, mild bronchospasm (2021 cath). Premedicate with steroids and diphenhydramine.",
        "rule": "contrast + premedication caution",
        "pass_fn": lambda r: ("contrast" in r.lower() and
                              any(w in r.lower() for w in ["premedic", "caution", "steroid", "diphenhydramine", "avoid"])),
    },
    {
        "id": 3,
        "question": "Does this patient carry an EpiPen?",
        "life_critical": True,
        "expected": "Yes — shellfish anaphylaxis (throat swelling, hypotension). Last reaction 2018, required ED visit.",
        "rule": "epipen + shellfish/anaphylaxis context",
        "pass_fn": lambda r: ("epipen" in r.lower() and
                              any(w in r.lower() for w in ["shellfish", "anaphylaxis"])),
    },
    {
        "id": 4,
        "question": "List all drug allergies",
        "life_critical": True,
        "expected": "Penicillin (moderate), Sulfonamides (mild — maculopapular rash 2010), Iodinated contrast (moderate)",
        "rule": "all three drug-allergy categories present",
        "pass_fn": _lower_all("penicillin", "sulfonamide", "contrast"),
    },
    {
        "id": 5,
        "question": "What is the patient's code status?",
        "life_critical": True,
        "expected": "Full Code. Healthcare POA: Linda Whitfield (wife). Advance directive on file.",
        "rule": "full code + Linda + POA/directive",
        "pass_fn": lambda r: ("full code" in r.lower() and "linda" in r.lower()),
    },
    {
        "id": 6,
        "question": "When should the patient call 911 or go to the ED?",
        "life_critical": True,
        "expected": "Chest pain, worsening SOB, weight gain >2 lbs/day or >5 lbs/week, syncope, palpitations >15 min, blood sugar >300 or <70 with symptoms, new leg swelling.",
        "rule": "return precaution markers >=4",
        "pass_fn": _hit_count(
            ["chest pain", "weight gain", "sob", "shortness of breath", "syncope",
             "palpitation", "blood sugar", "leg swelling", "dyspnea"],
            4,
        ),
    },
    {
        "id": 7,
        "question": "What is the patient's CHA2DS2-VASc score?",
        "life_critical": False,
        "expected": "4 — documented in Day 2 SOAP note and discharge prescriptions. Indication for apixaban anticoagulation.",
        "rule": "CHA2DS2-VASc 4 + apixaban",
        "pass_fn": lambda r: (" 4" in r and any(w in r.lower() for w in ["apixaban", "anticoagul"])),
    },
    {
        "id": 8,
        "question": "Does this patient have an ICD?",
        "life_critical": False,
        "expected": "No. ICD evaluation planned for April 2026 — 3 months post-GDMT optimization per ACC/AHA guidelines.",
        "rule": "no current ICD + future evaluation",
        "pass_fn": lambda r: (
            any(n in r.lower() for n in ["does not", "no icd", "not currently", "not yet", "not implanted"])
            and any(w in r.lower() for w in ["3 month", "three month", "gdmt", "april 2026", "planned"])
        ),
    },
    {
        "id": 9,
        "question": "What dietary restrictions apply at discharge?",
        "life_critical": False,
        "expected": "Sodium <2g/day, fluid restriction 1.5L/day, diabetic meal plan, heart-healthy diet. Avoid processed foods, canned soups, fast food.",
        "rule": "sodium + fluid + diabetic/heart-healthy",
        "pass_fn": lambda r: (
            "sodium" in r.lower()
            and any(w in r.lower() for w in ["fluid", "1.5", "restriction"])
            and any(w in r.lower() for w in ["diabet", "heart-healthy", "heart healthy"])
        ),
    },
    {
        "id": 10,
        "question": "What medications was the patient on at home?",
        "life_critical": False,
        "expected": "Lisinopril 20mg, Carvedilol 12.5mg BID, Furosemide 40mg, Metformin 1000mg BID, Atorvastatin 40mg, Aspirin 81mg, Tiotropium 2.5mcg, Omeprazole 20mg, Allopurinol 100mg, KCl 20mEq",
        "rule": "home meds hits >=7",
        "pass_fn": _hit_count(
            ["lisinopril", "carvedilol", "metformin", "furosemide", "atorvastatin",
             "aspirin", "tiotropium", "omeprazole", "allopurinol", "potassium"],
            7,
        ),
    },
    {
        "id": 11,
        "question": "What new medications were started during hospitalization?",
        "life_critical": False,
        "expected": "Sacubitril/Valsartan (ARNI), Spironolactone 25mg, Apixaban 5mg BID, Insulin Glargine, Empagliflozin 10mg, Metolazone, IV Furosemide, Heparin drip, Magnesium Oxide 400mg",
        "rule": "new inpatient meds hits >=5",
        "pass_fn": _hit_count(
            ["sacubitril", "entresto", "spironolactone", "apixaban", "insulin",
             "empagliflozin", "metolazone", "heparin", "magnesium"],
            5,
        ),
    },
    {
        "id": 12,
        "question": "What are the discharge medications?",
        "life_critical": False,
        "expected": "15 total: Entresto 24/26mg BID (new), Carvedilol 25mg BID (uptitrated), Furosemide 80mg (increased from 40mg), Spironolactone 25mg (new), Empagliflozin 10mg (new), Apixaban 5mg BID (new), Metformin 1000mg BID, Insulin Glargine 18u (new), Atorvastatin 40mg, Aspirin 81mg, Tiotropium, Omeprazole 20mg, Allopurinol 100mg, KCl 20mEq, Magnesium Oxide 400mg (new), Hydralazine 25mg PRN SBP>160",
        "rule": "discharge meds hits >=8 + Entresto/Sacubitril + apixaban",
        "pass_fn": lambda r: (
            any(w in r.lower() for w in ["entresto", "sacubitril"])
            and "apixaban" in r.lower()
            and sum(1 for w in [
                "carvedilol", "furosemide", "spironolactone", "empagliflozin",
                "metformin", "insulin", "atorvastatin", "aspirin", "tiotropium",
                "omeprazole", "allopurinol", "potassium", "magnesium", "hydralazine",
            ] if w in r.lower()) >= 8
        ),
    },
    {
        "id": 13,
        "question": "Why was lisinopril stopped at discharge?",
        "life_critical": False,
        "expected": "Replaced by Sacubitril/Valsartan (Entresto). Require 36-hour washout from ACE inhibitor before starting ARNI. Lisinopril was also held on admission due to AKI.",
        "rule": "lisinopril stop reason + ARNI + 36-hour washout",
        "pass_fn": lambda r: (
            any(w in r.lower() for w in ["sacubitril", "entresto", "arni"])
            and any(w in r.lower() for w in ["washout", "36-hour", "36 hour"])
        ),
    },
]


# ---------------------------------------------------------------------------

def get_token() -> str:
    r = requests.post(
        f"{BASE_URL}/api/auth/token",
        json={"user_id": "eval_runner", "role": "clinician"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()["token"]


def ask(question: str, token: str, session_id: str, timeout: int = 120) -> tuple[str, float]:
    """Send question through streaming endpoint; return (content, latency_s)."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"message": question, "session_id": session_id}

    t0 = time.perf_counter()
    resp = requests.post(
        f"{BASE_URL}/api/chat/stream",
        json=payload,
        headers=headers,
        stream=True,
        timeout=timeout,
    )
    resp.raise_for_status()

    content = ""
    server_latency_ms = 0
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        try:
            ev = json.loads(line[6:])
            if ev["type"] == "token":
                content += ev["content"]
            elif ev["type"] == "done":
                server_latency_ms = ev.get("latency_ms", 0)
        except json.JSONDecodeError:
            pass

    wall = time.perf_counter() - t0
    latency_s = server_latency_ms / 1000.0 if server_latency_ms else wall
    return content, round(latency_s, 2)


def run_eval(save: bool = False):
    print("=" * 70)
    print("ACCURACY EVALUATION — 13 Gold-Standard Questions")
    print(f"Server: {BASE_URL}")
    print("=" * 70)

    try:
        token = get_token()
        print("Auth: OK\n")
    except Exception as e:
        print(f"Auth FAILED: {e}")
        return 1

    # Each question is self-contained with explicit patient identity prefix —
    # matching the quick_regression.py approach (which achieves 100%).
    # This is also the correct clinical pattern: always specify which patient.
    PATIENT_PREFIX = (
        "For patient Robert James Whitfield (MRN: MRN-2026-004782), "
    )

    results = []
    passed_total = 0
    lc_passed = 0
    lc_total = 0
    latencies = []

    for item in QUESTIONS:
        sid = f"eval-{item['id']}-{int(time.time())}"
        full_question = PATIENT_PREFIX + item["question"][0].lower() + item["question"][1:]
        try:
            content, latency_s = ask(full_question, token, sid)
        except Exception as e:
            content = ""
            latency_s = 0.0
            print(f"  Q{item['id']:02d} [ERR ] → {e}")

        ok = item["pass_fn"](content)
        passed_total += ok
        latencies.append(latency_s)

        if item["life_critical"]:
            lc_total += 1
            lc_passed += ok

        status = "PASS" if ok else "FAIL"
        tag = " [LIFE-CRITICAL]" if item["life_critical"] else ""
        print(f"  Q{item['id']:02d} [{status}]{tag}")
        print(f"       {item['question'][:80]}")
        if not ok:
            print(f"       Rule : {item['rule']}")
            print(f"       Reply: {content[:300]}")
        print()

        results.append({
            "id": item["id"],
            "question": item["question"],
            "life_critical": item["life_critical"],
            "expected": item["expected"],
            "passed": bool(ok),
            "rule": item["rule"],
            "latency_s": latency_s,
            "answer": content,
        })

    total = len(QUESTIONS)
    accuracy_pct = round(100 * passed_total / total, 1)
    lc_accuracy_pct = round(100 * lc_passed / lc_total, 1) if lc_total else 0.0
    avg_latency = round(sum(latencies) / len(latencies), 2)

    print("=" * 70)
    print(f"RESULTS: {passed_total}/{total} passed  ({accuracy_pct}%)")
    print(f"LIFE-CRITICAL: {lc_passed}/{lc_total} passed  ({lc_accuracy_pct}%)")
    print(f"AVG LATENCY: {avg_latency}s")
    print("=" * 70)

    # Baseline comparison
    BASELINE_ACCURACY = 84.6
    BASELINE_LC = 83.3
    delta = accuracy_pct - BASELINE_ACCURACY
    print(f"\nBaseline accuracy: {BASELINE_ACCURACY}%")
    print(f"Current  accuracy: {accuracy_pct}%  ({'+' if delta >= 0 else ''}{delta:.1f}%)")
    if accuracy_pct >= BASELINE_ACCURACY:
        print("✓ ACCURACY MAINTAINED OR IMPROVED vs baseline")
    else:
        print("✗ ACCURACY REGRESSED vs baseline")

    output = {
        "base_url": BASE_URL,
        "total": total,
        "passed": passed_total,
        "accuracy_pct": accuracy_pct,
        "life_critical_total": lc_total,
        "life_critical_passed": lc_passed,
        "life_critical_accuracy_pct": lc_accuracy_pct,
        "avg_latency_s": avg_latency,
        "baseline_accuracy_pct": BASELINE_ACCURACY,
        "delta_vs_baseline_pct": round(delta, 1),
        "results": results,
    }

    if save:
        LOG_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"\nResults saved → {LOG_PATH}")

    return 0 if accuracy_pct >= BASELINE_ACCURACY else 1


if __name__ == "__main__":
    save_flag = "--save" in sys.argv
    sys.exit(run_eval(save=save_flag))
