"""Quick regression test — runs 10 key questions to verify no regressions after changes."""
import requests
import json
import time
import sys

BASE = "http://localhost:7860"
QUESTIONS = [
    # Q1: Demographics
    ("For patient Robert James Whitfield (MRN: MRN-2026-004782), what is the full name and date of birth?",
     ["Robert James Whitfield", "June 14, 1963"]),
    # Q6: Advance directive
    ("For patient Robert James Whitfield (MRN: MRN-2026-004782), what is the advance directive status?",
     ["Full Code", "Linda Whitfield"]),
    # Q11: Weight gain
    ("For patient Robert James Whitfield (MRN: MRN-2026-004782), how many pounds were gained before admission?",
     ["18"]),
    # Q20: Admission meds
    ("For patient Robert James Whitfield (MRN: MRN-2026-004782), what medications were being taken at home before admission?",
     ["lisinopril", "carvedilol", "metformin", "furosemide"]),
    # Q25: Lab abnormals
    ("For patient Robert James Whitfield (MRN: MRN-2026-004782), what was the BNP level on admission?",
     ["1,842", "1842"]),
    # Q35: Discharge prescriptions
    ("For patient Robert James Whitfield (MRN: MRN-2026-004782), what are the discharge prescriptions?",
     ["Sacubitril", "Carvedilol", "Furosemide", "Spironolactone", "Empagliflozin"]),
    # Q40: Procedures
    ("For patient Robert James Whitfield (MRN: MRN-2026-004782), what procedures were performed during hospitalization?",
     ["echocardiogram", "thoracentesis", "catheterization"]),
    # Q45: Follow-up
    ("For patient Robert James Whitfield (MRN: MRN-2026-004782), what follow-up appointments are scheduled after discharge?",
     ["cardiology", "nephrology"]),
    # Q50: Weight loss
    ("For patient Robert James Whitfield (MRN: MRN-2026-004782), how much total weight was lost during hospitalization?",
     ["40.5", "18.4"]),
    # Q60: EF improvement
    ("For patient Robert James Whitfield (MRN: MRN-2026-004782), what was the ejection fraction improvement during hospital stay?",
     ["25%", "30%"]),
]


def main():
    # Auth
    r = requests.post(f"{BASE}/api/auth/token", json={"user_id": "admin", "password": "admin123"})
    token = r.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    passed = 0
    failed = 0
    times = []

    for i, (q, expected_keywords) in enumerate(QUESTIONS, 1):
        sid = f"regression-{i}-{int(time.time())}"
        t0 = time.time()
        try:
            resp = requests.post(
                f"{BASE}/api/chat",
                json={"message": q, "session_id": sid},
                headers=headers,
                timeout=60,
            )
            elapsed = time.time() - t0
            times.append(elapsed)

            reply = resp.json().get("reply", "")
            reply_lower = reply.lower()

            # Check if ANY expected keyword is present (case-insensitive)
            found = [kw for kw in expected_keywords if kw.lower() in reply_lower]
            ok = len(found) >= max(1, len(expected_keywords) // 2)

            status = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            else:
                failed += 1

            print(f"  Q{i:2d} [{status}] {elapsed:5.1f}s | {q[:60]}")
            if not ok:
                print(f"       Expected: {expected_keywords}")
                print(f"       Found: {found}")
                print(f"       Reply: {reply[:200]}")
        except Exception as e:
            failed += 1
            print(f"  Q{i:2d} [ERR ] | {q[:60]} → {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{passed+failed} passed ({100*passed/(passed+failed):.0f}%)")
    print(f"Avg latency: {sum(times)/len(times):.1f}s")
    print(f"Total time: {sum(times):.0f}s")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
