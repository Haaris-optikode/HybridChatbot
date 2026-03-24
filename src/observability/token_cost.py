import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from pyprojroot import here

logger = logging.getLogger(__name__)


@dataclass
class UsageRecord:
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    provider: str
    timestamp_utc: str


def _load_pricing_table() -> Dict[str, Dict[str, float]]:
    """
    Model pricing in USD per 1M tokens.
    Override with MEDGRAPH_MODEL_PRICING_JSON env var containing:
    {
      "gemini-flash-latest": {"input_per_1m": 0.35, "output_per_1m": 1.05},
      ...
    }
    """
    default_prices: Dict[str, Dict[str, float]] = {
        "gemini-flash-latest": {"input_per_1m": 0.35, "output_per_1m": 1.05},
        "gemini-2.5-pro": {"input_per_1m": 3.50, "output_per_1m": 10.50},
        "gpt-4o-mini": {"input_per_1m": 0.15, "output_per_1m": 0.60},
        "gpt-4o": {"input_per_1m": 5.00, "output_per_1m": 15.00},
        "gpt-5": {"input_per_1m": 5.00, "output_per_1m": 15.00},
    }
    raw = os.getenv("MEDGRAPH_MODEL_PRICING_JSON", "").strip()
    if not raw:
        return default_prices
    try:
        loaded = json.loads(raw)
        if isinstance(loaded, dict):
            return loaded
        logger.warning("MEDGRAPH_MODEL_PRICING_JSON must be a JSON object; using defaults")
    except Exception as exc:
        logger.warning("Invalid MEDGRAPH_MODEL_PRICING_JSON; using defaults (%s)", str(exc)[:200])
    return default_prices


def _provider_from_model(model_name: str) -> str:
    m = (model_name or "").lower()
    if "gemini" in m:
        return "google"
    if "gpt" in m or "o1" in m or "o3" in m or "o4" in m:
        return "openai"
    return "unknown"


def _price_for_model(model_name: str, pricing: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    if model_name in pricing:
        return pricing[model_name]
    model_l = model_name.lower()
    for key, value in pricing.items():
        if model_l.startswith(key.lower()):
            return value

    # Common Gemini naming variants:
    # - "gemini-flash-latest" vs "gemini-1.5-flash" vs "gemini-flash"
    # - "gemini-2.5-pro" vs "gemini-2.5-pro-latest"
    if "flash" in model_l:
        for key, value in pricing.items():
            if "flash" in key.lower():
                return value
    if "2.5" in model_l and "pro" in model_l:
        for key, value in pricing.items():
            if "2.5" in key.lower() and "pro" in key.lower():
                return value
    if "2_5" in model_l and "pro" in model_l:
        for key, value in pricing.items():
            if ("2.5" in key.lower() or "2_5" in key.lower()) and "pro" in key.lower():
                return value
    return {"input_per_1m": 0.0, "output_per_1m": 0.0}


def _compute_cost_usd(model_name: str, input_tokens: int, output_tokens: int, pricing: Dict[str, Dict[str, float]]) -> float:
    p = _price_for_model(model_name, pricing)
    in_cost = (input_tokens / 1_000_000.0) * float(p.get("input_per_1m", 0.0))
    out_cost = (output_tokens / 1_000_000.0) * float(p.get("output_per_1m", 0.0))
    return round(in_cost + out_cost, 8)


def _extract_usage_from_response(response: Any) -> Optional[Dict[str, Any]]:
    def _as_int(v: Any) -> int:
        try:
            return int(v)
        except Exception:
            return 0

    def _read_token_counts(d: Dict[str, Any]) -> Dict[str, int]:
        # OpenAI-style keys
        prompt_tokens = _as_int(
            d.get("prompt_tokens")
            or d.get("input_tokens")
            or d.get("prompt_token_count")
        )
        completion_tokens = _as_int(
            d.get("completion_tokens")
            or d.get("output_tokens")
            or d.get("candidates_token_count")
        )
        total_tokens = _as_int(
            d.get("total_tokens")
            or d.get("total_token_count")
        )

        if total_tokens == 0 and (prompt_tokens or completion_tokens):
            total_tokens = prompt_tokens + completion_tokens
        return {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    # Try response.llm_output first (common for OpenAI wrappers)
    llm_output = getattr(response, "llm_output", None) or {}
    token_usage = llm_output.get("token_usage") if isinstance(llm_output, dict) else None

    if token_usage:
        token_counts = _read_token_counts(token_usage if isinstance(token_usage, dict) else {})
        model = (
            llm_output.get("model_name")
            or llm_output.get("model")
            or getattr(response, "model", "")
            or getattr(response, "model_name", "")
            or ""
        )
        return {
            "model": model,
            "input_tokens": token_counts["input_tokens"],
            "output_tokens": token_counts["output_tokens"],
            "total_tokens": token_counts["total_tokens"],
        }

    # Try AIMessage metadata for Google/OpenAI chat models
    generations = getattr(response, "generations", None) or []
    if generations and generations[0]:
        gen = generations[0][0]
        message = getattr(gen, "message", None)
        if message is None:
            return None

        usage_meta = getattr(message, "usage_metadata", None) or {}
        resp_meta = getattr(message, "response_metadata", None) or {}

        model = (
            resp_meta.get("model_name")
            or resp_meta.get("model")
            or getattr(response, "model", "")
            or getattr(response, "model_name", "")
            or llm_output.get("model_name", "")
            or ""
        )

        # Gemini often exposes:
        # - prompt_token_count
        # - candidates_token_count
        # - total_token_count
        token_counts = _read_token_counts(
            {
                **usage_meta,
                **resp_meta,
            }
        )
        return {
            "model": model,
            "input_tokens": token_counts["input_tokens"],
            "output_tokens": token_counts["output_tokens"],
            "total_tokens": token_counts["total_tokens"],
        }

    return None


class RequestUsageCallback(BaseCallbackHandler):
    """Captures LLM usage for a single API request."""

    def __init__(self, pricing_table: Dict[str, Dict[str, float]]) -> None:
        super().__init__()
        self._pricing = pricing_table
        self._records: List[UsageRecord] = []

    @property
    def records(self) -> List[UsageRecord]:
        return list(self._records)

    def on_llm_end(self, response, **kwargs) -> Any:
        usage = _extract_usage_from_response(response)
        if not usage:
            return
        model = usage.get("model", "") or "unknown"
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens))
        self._records.append(
            UsageRecord(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=_compute_cost_usd(model, input_tokens, output_tokens, self._pricing),
                provider=_provider_from_model(model),
                timestamp_utc=datetime.utcnow().isoformat() + "Z",
            )
        )


class TokenCostLedger:
    """Thread-safe aggregate ledger persisted to disk."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._lock = threading.Lock()
        self._path = path or Path(here("logs")) / "token_usage_ledger.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _empty(self) -> Dict[str, Any]:
        return {
            "global": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "requests": 0,
                "updated_at": None,
            },
            "sessions": {},
        }

    def _load(self) -> Dict[str, Any]:
        if not self._path.exists():
            return self._empty()
        try:
            with self._path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    return loaded
        except Exception as exc:
            logger.warning("Failed loading token ledger; starting fresh (%s)", str(exc)[:200])
        return self._empty()

    def _save(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)
        tmp.replace(self._path)

    def record_request(self, session_id: str, request_summary: Dict[str, Any]) -> None:
        with self._lock:
            g = self._data["global"]
            g["input_tokens"] += int(request_summary.get("input_tokens", 0))
            g["output_tokens"] += int(request_summary.get("output_tokens", 0))
            g["total_tokens"] += int(request_summary.get("total_tokens", 0))
            g["cost_usd"] = round(float(g.get("cost_usd", 0.0)) + float(request_summary.get("cost_usd", 0.0)), 8)
            g["requests"] += 1
            g["updated_at"] = datetime.utcnow().isoformat() + "Z"

            sessions = self._data["sessions"]
            s = sessions.setdefault(
                session_id,
                {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "requests": 0,
                    "updated_at": None,
                },
            )
            s["input_tokens"] += int(request_summary.get("input_tokens", 0))
            s["output_tokens"] += int(request_summary.get("output_tokens", 0))
            s["total_tokens"] += int(request_summary.get("total_tokens", 0))
            s["cost_usd"] = round(float(s.get("cost_usd", 0.0)) + float(request_summary.get("cost_usd", 0.0)), 8)
            s["requests"] += 1
            s["updated_at"] = datetime.utcnow().isoformat() + "Z"
            self._save()

    def get_global(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._data.get("global", {}))

    def get_session(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            sessions = self._data.get("sessions", {})
            return dict(
                sessions.get(
                    session_id,
                    {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "cost_usd": 0.0,
                        "requests": 0,
                        "updated_at": None,
                    },
                )
            )


def summarize_request(records: List[UsageRecord]) -> Dict[str, Any]:
    input_tokens = sum(r.input_tokens for r in records)
    output_tokens = sum(r.output_tokens for r in records)
    total_tokens = sum(r.total_tokens for r in records)
    cost_usd = round(sum(r.cost_usd for r in records), 8)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
        "calls": len(records),
        "models": [r.model for r in records],
        "providers": sorted(set(r.provider for r in records)),
    }


PRICING_TABLE = _load_pricing_table()
LEDGER = TokenCostLedger()
