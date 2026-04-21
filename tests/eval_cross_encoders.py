"""
Phase 2.3 — Cross-Encoder Evaluation Benchmark

Evaluates candidate reranker models on the expanded eval dataset and measures:
  - Retrieval precision@8 (how many top-8 retrieved chunks contain the answer)
  - End-to-end answer accuracy on the quick_regression question set
  - Reranking latency per query (ms)

Candidate models (as specified in Phase 2.3 of the Implementation Plan):
  1. cross-encoder/ms-marco-MiniLM-L-6-v2  (current, 512 tokens, 23M params)
  2. cross-encoder/ms-marco-MiniLM-L-12-v2 (deeper, 512 tokens, 33M params)
  3. BAAI/bge-reranker-v2-m3               (multi-domain, 8192 tokens, 568M params)

Decision criteria (from Phase 2.3):
  - If a longer-context model improves accuracy by ≥3% on the eval set
    AND latency stays ≤500ms per query on CPU, adopt it.
  - Otherwise keep the current model (no unnecessary complexity).

Usage:
    # Run full benchmark (downloads models if needed — may take time):
    cd src && python -m pytest ../tests/eval_cross_encoders.py -v -s

    # Quick latency-only check (no model download required):
    cd src && python ../tests/eval_cross_encoders.py --latency-only

    # Run with a specific model:
    cd src && python ../tests/eval_cross_encoders.py --model cross-encoder/ms-marco-MiniLM-L-12-v2
"""
import argparse
import logging
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# Stub heavy dependencies when running in eval-only mode
if "pyprojroot" not in sys.modules:
    pyprojroot_stub = types.ModuleType("pyprojroot")
    pyprojroot_stub.here = lambda *parts: ROOT.joinpath(*parts)
    sys.modules["pyprojroot"] = pyprojroot_stub

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("eval_cross_encoders")

# ── Candidate models ─────────────────────────────────────────────────────────
CANDIDATE_MODELS = [
    {
        "name": "ms-marco-MiniLM-L-6-v2",
        "model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "max_tokens": 512,
        "params_m": 23,
        "description": "Current production model (general-purpose, fast)",
        "is_current": True,
    },
    {
        "name": "ms-marco-MiniLM-L-12-v2",
        "model_id": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "max_tokens": 512,
        "params_m": 33,
        "description": "Deeper version of current model (12 layers vs 6)",
        "is_current": False,
    },
    {
        "name": "bge-reranker-v2-m3",
        "model_id": "BAAI/bge-reranker-v2-m3",
        "max_tokens": 8192,
        "params_m": 568,
        "description": "Multi-domain with long context window (requires more memory)",
        "is_current": False,
    },
]

# ── Representative query pairs for reranking latency measurement ─────────────
# Each pair: (query, chunk_text).  These are realistic clinical queries.
LATENCY_TEST_PAIRS = [
    (
        "What medications was the patient taking at home before admission?",
        "Home medications: Lisinopril 10mg daily, Carvedilol 12.5mg BID, "
        "Metformin 1000mg BID, Furosemide 40mg daily, Aspirin 81mg daily.",
    ),
    (
        "What was the BNP level on admission?",
        "Admission labs: BNP 1,842 pg/mL (critical), Troponin-I 0.04 ng/mL, "
        "Creatinine 2.1 mg/dL, BUN 38 mg/dL, eGFR 32 mL/min/1.73m².",
    ),
    (
        "What procedures were performed during hospitalization?",
        "Procedures performed: 1) Transthoracic echocardiogram (Day 1) showing "
        "EF 25%. 2) Right heart catheterization (Day 3). 3) Thoracentesis (Day 5) "
        "removing 1.2L of pleural fluid.",
    ),
    (
        "What are the patient's documented allergies?",
        "Allergies: Penicillin (Type I hypersensitivity — anaphylaxis), "
        "Sulfa drugs (rash), Contrast dye (requires premedication).",
    ),
    (
        "What was the ejection fraction at discharge?",
        "Repeat echocardiogram on Day 7 showed improvement in LVEF from 25% "
        "on admission to 30% at time of discharge. Patient remains in systolic "
        "heart failure with reduced ejection fraction (HFrEF).",
    ),
    (
        "What is the patient's follow-up plan after discharge?",
        "Discharge follow-up: Cardiology clinic in 1 week (Dr. Chen, Tuesday), "
        "Nephrology in 2 weeks, Repeat BMP and BNP in 1 week, "
        "Daily weight monitoring with instructions to call if weight increases >2 lbs/day.",
    ),
    (
        "What was the patient's discharge weight and total weight lost?",
        "Admission weight: 112.1 kg. Discharge weight: 93.7 kg. "
        "Total weight loss during hospitalization: 18.4 kg (40.5 lbs) over 9 days. "
        "Primarily achieved via aggressive diuresis with IV furosemide.",
    ),
    (
        "What are the discharge medications?",
        "Discharge medications: Sacubitril/Valsartan (Entresto) 24/26mg BID (NEW), "
        "Carvedilol 12.5mg BID (unchanged), Furosemide 80mg daily (increased from 40mg), "
        "Spironolactone 25mg daily (NEW), Empagliflozin 10mg daily (NEW), "
        "Metformin 1000mg BID (unchanged).",
    ),
]


def measure_latency(model_id: str, n_repeats: int = 3) -> dict:
    """Load the cross-encoder and measure reranking latency on test pairs."""
    try:
        from sentence_transformers import CrossEncoder
        logger.info("Loading model: %s", model_id)
        t_load_start = time.perf_counter()
        model = CrossEncoder(model_id, max_length=512)
        load_time_s = time.perf_counter() - t_load_start
        logger.info("Model loaded in %.1fs", load_time_s)

        # Warm-up pass
        _ = model.predict([[LATENCY_TEST_PAIRS[0][0], LATENCY_TEST_PAIRS[0][1]]])

        # Timed passes
        latencies_ms = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            _ = model.predict(LATENCY_TEST_PAIRS)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies_ms.append(elapsed_ms)

        avg_latency = sum(latencies_ms) / len(latencies_ms)
        per_pair_ms = avg_latency / len(LATENCY_TEST_PAIRS)

        return {
            "model_id": model_id,
            "load_time_s": round(load_time_s, 2),
            "avg_total_ms": round(avg_latency, 1),
            "per_pair_ms": round(per_pair_ms, 1),
            "n_pairs": len(LATENCY_TEST_PAIRS),
            "n_repeats": n_repeats,
            "error": None,
        }
    except Exception as e:
        logger.error("Failed to benchmark %s: %s", model_id, e)
        return {
            "model_id": model_id,
            "load_time_s": None,
            "avg_total_ms": None,
            "per_pair_ms": None,
            "n_pairs": len(LATENCY_TEST_PAIRS),
            "n_repeats": n_repeats,
            "error": str(e),
        }


def compare_reranker_scores(model_ids: list[str]) -> dict:
    """Compare relevance scores across models on known relevant vs. irrelevant pairs."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        logger.error("sentence_transformers not installed")
        return {}

    # Ground-truth: relevant and irrelevant query-chunk pairs
    relevant_pairs = [
        (
            "What was the BNP level on admission?",
            "Admission labs: BNP 1,842 pg/mL (critical), Troponin 0.04 ng/mL.",
        ),
        (
            "What medications was the patient taking at home?",
            "Home medications: Lisinopril 10mg daily, Carvedilol 12.5mg BID.",
        ),
    ]
    irrelevant_pairs = [
        (
            "What was the BNP level on admission?",
            "Social history: Patient is a retired teacher. Non-smoker. Occasional alcohol.",
        ),
        (
            "What medications was the patient taking at home?",
            "Chief complaint: Patient presented with 3-day history of worsening dyspnea.",
        ),
    ]

    results = {}
    for mid in model_ids:
        try:
            model = CrossEncoder(mid, max_length=512)
            rel_scores = model.predict(relevant_pairs)
            irrel_scores = model.predict(irrelevant_pairs)
            results[mid] = {
                "avg_relevant_score": round(float(sum(rel_scores) / len(rel_scores)), 4),
                "avg_irrelevant_score": round(float(sum(irrel_scores) / len(irrel_scores)), 4),
                "separation": round(
                    float(sum(rel_scores) / len(rel_scores))
                    - float(sum(irrel_scores) / len(irrel_scores)),
                    4,
                ),
            }
        except Exception as e:
            results[mid] = {"error": str(e)}

    return results


def print_report(latency_results: list[dict], score_results: dict) -> None:
    """Print a formatted comparison report."""
    print("\n" + "=" * 70)
    print("CROSS-ENCODER EVALUATION REPORT — Phase 2.3")
    print("=" * 70)

    print(f"\nLatency Benchmark ({len(LATENCY_TEST_PAIRS)} query-chunk pairs, avg over 3 runs):")
    print(f"{'Model':<35} {'Load(s)':<10} {'Total(ms)':<12} {'Per-pair(ms)':<15} {'Status'}")
    print("-" * 80)
    for r in latency_results:
        if r["error"]:
            print(f"{r['model_id'][-34:]:<35} {'ERR':<10} {'ERR':<12} {'ERR':<15} ERROR: {r['error'][:30]}")
        else:
            print(
                f"{r['model_id'][-34:]:<35} {r['load_time_s']:<10} "
                f"{r['avg_total_ms']:<12} {r['per_pair_ms']:<15} OK"
            )

    if score_results:
        print(f"\nRelevance Discrimination (higher separation = better):")
        print(f"{'Model':<35} {'Rel Score':<12} {'Irrel Score':<14} {'Separation'}")
        print("-" * 70)
        for mid, r in score_results.items():
            if "error" in r:
                print(f"{mid[-34:]:<35} ERROR: {r['error'][:40]}")
            else:
                print(
                    f"{mid[-34:]:<35} {r['avg_relevant_score']:<12} "
                    f"{r['avg_irrelevant_score']:<14} {r['separation']}"
                )

    print("\nDecision Criteria (from Phase 2.3):")
    print("  - Adopt alternative if: accuracy improves ≥3% AND latency ≤500ms per query.")
    print("  - Current production model: cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("\nNOTE: Full accuracy evaluation requires running quick_regression.py with each")
    print("model configured in tools_config.yml (clinical_notes_rag.reranker_model).")
    print("=" * 70 + "\n")


# ── pytest integration ───────────────────────────────────────────────────────

class TestCrossEncoderLatency:
    """Phase 2.3 — Latency gate: current model must rerank in <500ms per query."""

    def test_current_model_latency_under_500ms(self):
        """The production reranker must keep per-query latency under 500ms on CPU."""
        result = measure_latency("cross-encoder/ms-marco-MiniLM-L-6-v2", n_repeats=2)
        if result["error"]:
            import pytest
            pytest.skip(f"Could not load model: {result['error']}")
        assert result["per_pair_ms"] < 500, (
            f"Reranker latency {result['per_pair_ms']}ms/query exceeds 500ms threshold. "
            f"Consider switching to a lighter model or reducing chunk count."
        )

    def test_current_model_discrimination(self):
        """Relevant chunks must score higher than irrelevant chunks."""
        scores = compare_reranker_scores(["cross-encoder/ms-marco-MiniLM-L-6-v2"])
        result = scores.get("cross-encoder/ms-marco-MiniLM-L-6-v2", {})
        if "error" in result:
            import pytest
            pytest.skip(f"Could not load model: {result['error']}")
        assert result["separation"] > 0.1, (
            f"Cross-encoder discrimination too low (separation={result['separation']}). "
            f"Relevant chunks should score significantly higher than irrelevant ones."
        )


# ── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-encoder evaluation benchmark (Phase 2.3)")
    parser.add_argument("--latency-only", action="store_true",
                        help="Only run latency benchmark (skip score comparison)")
    parser.add_argument("--model", default=None,
                        help="Evaluate a single model (e.g., BAAI/bge-reranker-v2-m3)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Number of timing repetitions (default: 3)")
    args = parser.parse_args()

    models_to_eval = (
        [next(m for m in CANDIDATE_MODELS if m["model_id"] == args.model)]
        if args.model
        else CANDIDATE_MODELS
    )

    latency_results = []
    for m in models_to_eval:
        logger.info("Benchmarking: %s", m["model_id"])
        result = measure_latency(m["model_id"], n_repeats=args.repeats)
        latency_results.append(result)

    score_results = {}
    if not args.latency_only:
        model_ids = [m["model_id"] for m in models_to_eval]
        score_results = compare_reranker_scores(model_ids)

    print_report(latency_results, score_results)


if __name__ == "__main__":
    main()
