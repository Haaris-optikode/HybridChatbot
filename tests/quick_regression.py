"""Quick regression test — runs 10 key questions to verify no regressions after changes."""
import requests
import json
import time
import sys

BASE = "http://localhost:7860"
QUESTIONS = [
    # Q1: Demographics
    ("What is Robert's full name and date of birth?",
     ["Robert James Whitfield", "June 14, 1963"]),
    # Q6: Advance directive
    ("What is Robert's advance directive status?",
     ["Full Code", "Linda Whitfield"]),
    # Q11: Weight gain
    ("How many pounds did the patient gain before admission?",
     ["18"]),
    # Q20: Admission meds
    ("What medications was the patient taking at home before admission?",
     ["lisinopril", "carvedilol", "metformin", "furosemide"]),
    # Q25: Lab abnormals
    ("What was the BNP level on admission?",
     ["1,842", "1842"]),
    # Q35: Discharge prescriptions
    ("What are the discharge prescriptions for Robert?",
     ["Sacubitril", "Carvedilol", "Furosemide", "Spironolactone", "Empagliflozin"]),
    # Q40: Procedures
    ("What procedures were performed during the hospitalization?",
     ["echocardiogram", "thoracentesis", "catheterization"]),
    # Q45: Follow-up
    ("What follow-up appointments are scheduled after discharge?",
     ["cardiology", "nephrology"]),
    # Q50: Weight loss
    ("How much total weight did the patient lose during hospitalization?",
     ["40.5", "18.4"]),
    # Q60: EF improvement
    ("What was the ejection fraction improvement during hospital stay?",
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
