import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


if "pyprojroot" not in sys.modules:
    pyprojroot_stub = types.ModuleType("pyprojroot")
    pyprojroot_stub.here = lambda *parts: ROOT.joinpath(*parts)
    sys.modules["pyprojroot"] = pyprojroot_stub


from observability.token_cost import PRICING_TABLE, _compute_cost_usd, _extract_usage_from_response, _price_for_model


def test_gpt41_pricing_per_million_tokens():
    cost = _compute_cost_usd("gpt-4.1", 1_000_000, 1_000_000, PRICING_TABLE)
    assert cost == 10.0


def test_gpt41_cached_input_pricing_applies_discount():
    cost = _compute_cost_usd(
        "gpt-4.1",
        1_000_000,
        1_000_000,
        PRICING_TABLE,
        cached_input_tokens=500_000,
    )
    # 500k uncached * $2.00 + 500k cached * $0.50 + 1M output * $8.00
    assert cost == 9.25


def test_gpt41_mini_pricing_per_million_tokens():
    cost = _compute_cost_usd("gpt-4.1-mini", 1_000_000, 1_000_000, PRICING_TABLE)
    assert cost == 2.0


def test_gemini_25_flash_pricing_per_million_tokens():
    cost = _compute_cost_usd("gemini-2.5-flash", 1_000_000, 1_000_000, PRICING_TABLE)
    assert cost == 2.8


def test_gemini_25_pro_pricing_per_million_tokens():
    cost = _compute_cost_usd("gemini-2.5-pro", 1_000_000, 1_000_000, PRICING_TABLE)
    assert cost == 11.25


def test_longest_prefix_match_prefers_gpt41_mini_over_gpt41():
    price = _price_for_model("gpt-4.1-mini-2026-04-14", PRICING_TABLE)
    assert float(price.get("input_per_1m", -1)) == 0.40
    assert float(price.get("output_per_1m", -1)) == 1.60


def test_extract_usage_reads_cached_tokens_from_openai_shape():
    class _Resp:
        llm_output = {
            "model_name": "gpt-4.1",
            "token_usage": {
                "prompt_tokens": 1200,
                "completion_tokens": 300,
                "total_tokens": 1500,
                "prompt_tokens_details": {"cached_tokens": 500},
            },
        }

    usage = _extract_usage_from_response(_Resp())
    assert usage is not None
    assert usage["input_tokens"] == 1200
    assert usage["output_tokens"] == 300
    assert usage["total_tokens"] == 1500
    assert usage["cached_input_tokens"] == 500
