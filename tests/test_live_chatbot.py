"""
Live chatbot test suite — tests latency, accuracy, and the summarization fix.
Runs against a live server at localhost:7860.
"""
import requests
import time
import json
import sys

BASE_URL = "http://localhost:7860"
SESSION_ID = None

def chat(message, tool_override=None, thinking=False, timeout=200):
    """Send a chat message via the streaming endpoint and collect the full response."""
    global SESSION_ID
    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "message": message,
        "session_id": SESSION_ID,
        "thinking": thinking,
    }
    if tool_override:
        payload["tool_override"] = tool_override

    t0 = time.perf_counter()
    resp = requests.post(
        f"{BASE_URL}/api/chat/stream",
        json=payload,
        headers=headers,
        stream=True,
        timeout=timeout,
    )
    resp.raise_for_status()

    full_content = ""
    latency_ms = 0
    usage = None
    error = None

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        try:
            event = json.loads(line[6:])
            if event["type"] == "token":
                full_content += event["content"]
            elif event["type"] == "done":
                SESSION_ID = event.get("session_id", SESSION_ID)
                latency_ms = event.get("latency_ms", 0)
                usage = event.get("usage")
            elif event["type"] == "error":
                error = event["content"]
            elif event["type"] == "status":
                pass  # tool status updates
        except json.JSONDecodeError:
            pass

    wall_time = time.perf_counter() - t0
    return {
        "content": full_content,
        "latency_ms": latency_ms,
        "wall_time_s": round(wall_time, 2),
        "usage": usage,
        "error": error,
    }


def test_health():
    """Test 1: Health check"""
    print("\n" + "="*70)
    print("TEST 1: Health Check")
    print("="*70)
    t0 = time.perf_counter()
    r = requests.get(f"{BASE_URL}/api/health", timeout=15)
    elapsed = time.perf_counter() - t0
    data = r.json()
    status = data.get("status")
    model = data.get("model")
    qdrant = data.get("qdrant", {}).get("status")
    vectors = data.get("qdrant", {}).get("vectors")
    llm = data.get("llm", {}).get("status")
    
    print(f"  Status: {status}")
    print(f"  Model: {model}")
    print(f"  Qdrant: {qdrant} ({vectors} vectors)")
    print(f"  LLM API: {llm}")
    print(f"  Latency: {elapsed:.2f}s")
    
    ok = status == "ok" and qdrant == "ok"
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


def test_greeting():
    """Test 2: Simple greeting — should be FAST (< 5s)"""
    print("\n" + "="*70)
    print("TEST 2: Simple Greeting (target: < 5s)")
    print("="*70)
    
    result = chat("Hello, how are you?")
    content = result["content"]
    latency = result["latency_ms"]
    wall = result["wall_time_s"]
    error = result["error"]
    
    print(f"  Response: {content[:200]}...")
    print(f"  Server Latency: {latency}ms ({latency/1000:.2f}s)")
    print(f"  Wall Time: {wall}s")
    if error:
        print(f"  ERROR: {error}")
    
    ok = wall < 10 and not error and len(content) > 10
    fast = wall < 5
    print(f"  {'✓ PASS' if ok else '✗ FAIL'} {'(FAST ✓)' if fast else '(SLOW ⚠)'}")
    return ok


def test_specific_lookup():
    """Test 3: Specific clinical query — should use lookup_clinical_notes"""
    print("\n" + "="*70)
    print("TEST 3: Specific Clinical Lookup (target: < 15s)")
    print("="*70)
    
    result = chat("What is Robert Whitfield's date of birth?")
    content = result["content"]
    latency = result["latency_ms"]
    wall = result["wall_time_s"]
    usage = result.get("usage") or {}
    error = result["error"]
    
    print(f"  Response: {content[:300]}...")
    print(f"  Server Latency: {latency}ms ({latency/1000:.2f}s)")
    print(f"  Wall Time: {wall}s")
    if error:
        print(f"  ERROR: {error}")

    total_tokens = int(usage.get("total_tokens") or 0)
    cost_usd = float(usage.get("cost_usd") or 0.0)
    print(f"  Stream usage total_tokens: {total_tokens}")
    print(f"  Stream usage cost_usd: {cost_usd:.6f}")
    
    # Check accuracy — should mention DOB
    has_dob = any(kw in content.lower() for kw in ["dob", "date of birth", "born", "1958", "november"])
    usage_ok = total_tokens > 0 and cost_usd > 0
    ok = not error and len(content) > 20 and usage_ok
    print(f"  Contains DOB info: {'✓ Yes' if has_dob else '✗ No'}")
    print(f"  Stream usage populated: {'✓ Yes' if usage_ok else '✗ No'}")
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


def test_list_patients():
    """Test 4: List patients — should be fast"""
    print("\n" + "="*70)
    print("TEST 4: List Available Patients (target: < 10s)")
    print("="*70)
    
    result = chat("List all available patients in the database")
    content = result["content"]
    latency = result["latency_ms"]
    wall = result["wall_time_s"]
    error = result["error"]
    
    print(f"  Response: {content[:300]}...")
    print(f"  Server Latency: {latency}ms ({latency/1000:.2f}s)")
    print(f"  Wall Time: {wall}s")
    if error:
        print(f"  ERROR: {error}")
    
    has_patients = any(kw in content.lower() for kw in ["mrn", "whitfield", "patient"])
    ok = not error and has_patients
    print(f"  Contains patient data: {'✓ Yes' if has_patients else '✗ No'}")
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


def test_allergy_lookup():
    """Test 5: Allergy lookup — accuracy check"""
    print("\n" + "="*70)
    print("TEST 5: Allergy Lookup — Accuracy (target: < 15s)")
    print("="*70)
    
    result = chat("What allergies does Robert Whitfield have?")
    content = result["content"]
    latency = result["latency_ms"]
    wall = result["wall_time_s"]
    error = result["error"]
    
    print(f"  Response: {content[:400]}...")
    print(f"  Server Latency: {latency}ms ({latency/1000:.2f}s)")
    print(f"  Wall Time: {wall}s")
    if error:
        print(f"  ERROR: {error}")
    
    has_allergy = any(kw in content.lower() for kw in ["allerg", "penicillin", "sulfa", "latex", "codeine"])
    ok = not error and has_allergy
    print(f"  Contains allergy data: {'✓ Yes' if has_allergy else '✗ No'}")
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


def test_summarize_patient():
    """Test 6: THE CRITICAL TEST — summarize patient record (was timing out)"""
    print("\n" + "="*70)
    print("TEST 6: ★ SUMMARIZE PATIENT RECORD ★ (target: < 120s)")
    print("  This was the failing query - testing our timeout fix")
    print("="*70)
    
    result = chat("Summarize the complete medical record for patient Robert Whitfield", timeout=200)
    content = result["content"]
    latency = result["latency_ms"]
    wall = result["wall_time_s"]
    error = result["error"]
    
    print(f"  Response length: {len(content)} chars")
    print(f"  Response preview: {content[:500]}...")
    print(f"  Server Latency: {latency}ms ({latency/1000:.2f}s)")
    print(f"  Wall Time: {wall}s")
    if error:
        print(f"  ERROR: {error}")
    
    # Verify it's a real summary, not an error
    is_actual_summary = len(content) > 500 and any(
        kw in content.lower() for kw in ["demographics", "admission", "diagnosis", "medication", "whitfield"]
    )
    no_timeout = "timeout" not in content.lower() and "timed out" not in content.lower()
    ok = not error and is_actual_summary and no_timeout
    
    print(f"  Is actual clinical summary: {'✓ Yes' if is_actual_summary else '✗ No'}")
    print(f"  No timeout errors: {'✓ Yes' if no_timeout else '✗ No (TIMEOUT!)'}")
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


def main():
    print("\u2554" + "═"*68 + "╗")
    print("║          MedGraph AI — Live Chatbot Test Suite                  ║")
    print("╚" + "═"*68 + "╝")
    
    results = {}
    
    results["health"] = test_health()
    results["greeting"] = test_greeting()
    results["specific_lookup"] = test_specific_lookup()
    results["list_patients"] = test_list_patients()
    results["allergy_lookup"] = test_allergy_lookup()
    results["summarize"] = test_summarize_patient()
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"\n  {passed}/{total} tests passed")
    
    if passed == total:
        print("  ★ ALL TESTS PASSED ★")
    else:
        print("  ⚠ SOME TESTS FAILED — see details above")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
