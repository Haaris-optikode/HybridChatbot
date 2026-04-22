"""
Phase 11 — HIPAA Audit Enrichment: Unit Tests

Tests for:
  - _classify_data_sources()
  - _extract_audit_metadata()
  - grounding_result field in State TypedDict
  - _audit_log JSON structure with enriched fields

Run with:
    cd src && python -m pytest ../tests/test_phase11_audit.py -v
"""
import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Path setup ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# Stub pyprojroot before any app import
if "pyprojroot" not in sys.modules:
    _pyprojroot_stub = types.ModuleType("pyprojroot")
    _pyprojroot_stub.here = lambda *parts: ROOT.joinpath(*parts)
    sys.modules["pyprojroot"] = _pyprojroot_stub

# Stub langchain_tavily with a proper BaseTool so bind_tools() doesn't crash
if "langchain_tavily" not in sys.modules:
    from langchain_core.tools import BaseTool

    class _FakeTavilySearch(BaseTool):
        name: str = "tavily_search_results_json"
        description: str = "Web search (stubbed for testing)"
        max_results: int = 5

        def _run(self, query: str, **kwargs) -> str:  # type: ignore[override]
            return ""

        async def _arun(self, query: str, **kwargs) -> str:  # type: ignore[override]
            return ""

    _tavily_stub = types.ModuleType("langchain_tavily")
    _tavily_stub.TavilySearch = _FakeTavilySearch
    sys.modules["langchain_tavily"] = _tavily_stub

# Patch env before any app imports
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")
os.environ.setdefault("GOOGLE_API_KEY", "test-key-not-real")
os.environ.setdefault("GOOGLE_AI_API_KEY", "test-key-not-real")
os.environ.setdefault("TAVILY_API_KEY", "test-key-not-real")
os.environ.setdefault("MEDGRAPH_ENV", "development")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-unit-tests-only")


# ── Helpers for building mock LangGraph event states ─────────────────
def _make_tool_msg(name: str, content: str):
    """Create a mock ToolMessage that passes the __class__.__name__ == 'ToolMessage' check."""
    from langchain_core.messages import ToolMessage
    return ToolMessage(content=content, name=name, tool_call_id="tc-mock")


def _make_state(messages=None, active_patient_mrn="", grounding_result=None):
    """Build a minimal LangGraph state dict."""
    return {
        "messages": messages or [],
        "active_patient_mrn": active_patient_mrn,
        "grounding_result": grounding_result if grounding_result is not None else {},
    }


# ══════════════════════════════════════════════════════════════════════
# Tests: _classify_data_sources
# ══════════════════════════════════════════════════════════════════════

class TestClassifyDataSources:
    @pytest.fixture(autouse=True)
    def _import(self):
        from api import _classify_data_sources
        self.classify = _classify_data_sources

    def test_empty_tools_returns_none(self):
        assert self.classify([]) == "none"

    def test_qdrant_lookup_clinical_notes(self):
        assert self.classify(["lookup_clinical_notes"]) == "qdrant"

    def test_qdrant_lookup_patient_orders(self):
        assert self.classify(["lookup_patient_orders"]) == "qdrant"

    def test_qdrant_summarize_patient_record(self):
        assert self.classify(["summarize_patient_record"]) == "qdrant"

    def test_qdrant_summarize_uploaded_document(self):
        assert self.classify(["summarize_uploaded_document"]) == "qdrant"

    def test_qdrant_list_available_patients(self):
        assert self.classify(["list_available_patients"]) == "qdrant"

    def test_qdrant_list_uploaded_documents(self):
        assert self.classify(["list_uploaded_documents"]) == "qdrant"

    def test_qdrant_cohort_patient_search(self):
        assert self.classify(["cohort_patient_search"]) == "qdrant"

    def test_web_only_tavily(self):
        assert self.classify(["tavily_search_results_json"]) == "web"

    def test_qdrant_and_web(self):
        result = self.classify(["lookup_clinical_notes", "tavily_search_results_json"])
        assert result == "qdrant+web"

    def test_unknown_tool_returns_tool(self):
        assert self.classify(["some_custom_tool"]) == "tool"

    def test_multiple_qdrant_tools(self):
        assert self.classify(["lookup_clinical_notes", "lookup_patient_orders"]) == "qdrant"


# ══════════════════════════════════════════════════════════════════════
# Tests: _extract_audit_metadata
# ══════════════════════════════════════════════════════════════════════

class TestExtractAuditMetadata:
    @pytest.fixture(autouse=True)
    def _import(self):
        from api import _extract_audit_metadata
        self.extract = _extract_audit_metadata

    def test_empty_events_returns_data_sources_none(self):
        result = self.extract([])
        assert result == {"data_sources": "none"}

    def test_extracts_patient_mrn_from_state(self):
        tm = _make_tool_msg("lookup_clinical_notes", "[Source 1 | MRN-001 | notes.pdf | 2026-01-01]")
        state = _make_state(messages=[tm], active_patient_mrn="MRN-001")
        result = self.extract([state])
        assert result.get("patient_mrn") == "MRN-001"

    def test_no_mrn_when_state_empty_string(self):
        tm = _make_tool_msg("lookup_clinical_notes", "[Source 1 | MRN-001 | notes.pdf]")
        state = _make_state(messages=[tm], active_patient_mrn="")
        result = self.extract([state])
        assert "patient_mrn" not in result

    def test_extracts_tools_invoked_ordered(self):
        tm1 = _make_tool_msg("lookup_clinical_notes", "content1")
        tm2 = _make_tool_msg("lookup_patient_orders", "content2")
        state = _make_state(messages=[tm1, tm2], active_patient_mrn="MRN-001")
        result = self.extract([state])
        assert result["tools_invoked"] == ["lookup_clinical_notes", "lookup_patient_orders"]

    def test_tools_invoked_deduplication(self):
        tm1 = _make_tool_msg("lookup_clinical_notes", "first call content")
        tm2 = _make_tool_msg("lookup_clinical_notes", "second call content")
        state = _make_state(messages=[tm1, tm2], active_patient_mrn="MRN-001")
        result = self.extract([state])
        assert result["tools_invoked"] == ["lookup_clinical_notes"]

    def test_no_tools_invoked_key_when_no_tools(self):
        from langchain_core.messages import AIMessage
        ai_msg = AIMessage(content="Hello!")
        state = _make_state(messages=[ai_msg], active_patient_mrn="")
        result = self.extract([state])
        assert "tools_invoked" not in result

    def test_data_sources_qdrant_with_clinical_tool(self):
        tm = _make_tool_msg("lookup_clinical_notes", "[Source 1 | ...]")
        state = _make_state(messages=[tm], active_patient_mrn="MRN-001")
        result = self.extract([state])
        assert result["data_sources"] == "qdrant"

    def test_data_sources_web_with_tavily(self):
        tm = _make_tool_msg("tavily_search_results_json", "some web result")
        state = _make_state(messages=[tm])
        result = self.extract([state])
        assert result["data_sources"] == "web"

    def test_data_sources_none_for_fast_path_no_tools(self):
        from langchain_core.messages import AIMessage
        ai_msg = AIMessage(content="Hello!")
        state = _make_state(messages=[ai_msg])
        result = self.extract([state])
        assert result["data_sources"] == "none"

    def test_grounding_score_extracted_when_present(self):
        tm = _make_tool_msg("lookup_clinical_notes", "[Source 1 | MRN-001 | f.pdf]")
        grounding = {"is_grounded": True, "citation_coverage": 0.85, "issues": [], "citation_count": 2}
        state = _make_state(messages=[tm], active_patient_mrn="MRN-001", grounding_result=grounding)
        result = self.extract([state])
        assert result.get("grounding_score") == pytest.approx(0.85)

    def test_grounding_score_absent_when_empty_result(self):
        tm = _make_tool_msg("lookup_clinical_notes", "[Source 1 | MRN-001 | f.pdf]")
        state = _make_state(messages=[tm], active_patient_mrn="MRN-001")
        result = self.extract([state])
        assert "grounding_score" not in result

    def test_grounding_score_zero_coverage_is_included(self):
        """citation_coverage of 0.0 must still appear in audit (it's a real value)."""
        tm = _make_tool_msg("lookup_clinical_notes", "[Source 1 | MRN-001 | f.pdf]")
        grounding = {"is_grounded": False, "citation_coverage": 0.0,
                     "issues": ["No citations"], "citation_count": 0}
        state = _make_state(messages=[tm], active_patient_mrn="MRN-001", grounding_result=grounding)
        result = self.extract([state])
        assert "grounding_score" in result
        assert result["grounding_score"] == pytest.approx(0.0)

    def test_uses_last_event_state_for_mrn(self):
        """events[-1] must be used, not an earlier state snapshot."""
        tm = _make_tool_msg("lookup_clinical_notes", "early")
        state_early = _make_state(messages=[tm], active_patient_mrn="MRN-EARLY")
        tm2 = _make_tool_msg("lookup_patient_orders", "final")
        # final state has both messages accumulated
        state_final = _make_state(messages=[tm, tm2], active_patient_mrn="MRN-FINAL",
                                  grounding_result={"citation_coverage": 0.5, "issues": []})
        result = self.extract([state_early, state_final])
        assert result.get("patient_mrn") == "MRN-FINAL"
        assert "lookup_clinical_notes" in result["tools_invoked"]
        assert "lookup_patient_orders" in result["tools_invoked"]


# ══════════════════════════════════════════════════════════════════════
# Tests: State TypedDict — grounding_result field
# ══════════════════════════════════════════════════════════════════════

class TestStateTypedDict:
    def test_grounding_result_in_state_annotations(self):
        """State TypedDict must declare grounding_result (Phase 11 addition)."""
        from agent_graph.agent_backend import State
        annotations = State.__annotations__
        assert "grounding_result" in annotations, (
            "State TypedDict must have 'grounding_result' field for Phase 11 audit enrichment"
        )

    def test_state_has_active_patient_mrn(self):
        from agent_graph.agent_backend import State
        assert "active_patient_mrn" in State.__annotations__

    def test_state_has_messages(self):
        from agent_graph.agent_backend import State
        assert "messages" in State.__annotations__


# ══════════════════════════════════════════════════════════════════════
# Tests: _audit_log JSON structure
# ══════════════════════════════════════════════════════════════════════

class TestAuditLogFormat:
    @pytest.fixture(autouse=True)
    def _import(self):
        from api import _audit_log
        import api as _api
        self.audit_log = _audit_log
        self.api = _api

    def _capture_audit(self, *args, **kwargs):
        captured = []
        with patch.object(self.api.audit_logger, "info",
                          side_effect=lambda msg: captured.append(msg)):
            self.audit_log(*args, **kwargs)
        assert len(captured) == 1
        return json.loads(captured[0])

    def test_audit_log_has_required_base_fields(self):
        user = {"sub": "user-123", "role": "clinician"}
        entry = self._capture_audit("chat_response", user, session_id="s1", latency_ms=150)
        assert entry["event"] == "chat_response"
        assert entry["user_id"] == "user-123"
        assert entry["role"] == "clinician"
        assert "timestamp" in entry
        assert entry["session_id"] == "s1"
        assert entry["latency_ms"] == 150

    def test_audit_log_with_phase11_mrn_field(self):
        user = {"sub": "user-456", "role": "admin"}
        entry = self._capture_audit(
            "chat_response", user,
            session_id="s2",
            patient_mrn="MRN-2026-001",
            tools_invoked=["lookup_clinical_notes"],
            data_sources="qdrant",
            grounding_score=0.75,
            latency_ms=200,
        )
        assert entry["patient_mrn"] == "MRN-2026-001"
        assert entry["tools_invoked"] == ["lookup_clinical_notes"]
        assert entry["data_sources"] == "qdrant"
        assert entry["grounding_score"] == pytest.approx(0.75)

    def test_patient_mrn_absent_from_fast_path_audit(self):
        """Fast-path greetings must NOT include patient_mrn."""
        user = {"sub": "user-789", "role": "clinician"}
        entry = self._capture_audit(
            "chat_response", user,
            session_id="s3",
            data_sources="none",
            fast_path=True,
        )
        assert "patient_mrn" not in entry

    def test_audit_log_produces_valid_json(self):
        user = {"sub": "u", "role": "r"}
        captured = []
        with patch.object(self.api.audit_logger, "info",
                          side_effect=lambda msg: captured.append(msg)):
            self.audit_log("test_event", user, key1="val1", key2=42)
        # Must be valid JSON
        entry = json.loads(captured[0])
        assert isinstance(entry, dict)

    def test_unknown_user_fallback(self):
        """Empty user dict must not crash and must use 'unknown' fallback."""
        entry = self._capture_audit("session_cleared", {}, session_id="s4")
        assert entry["user_id"] == "unknown"
        assert entry["role"] == "unknown"


# ══════════════════════════════════════════════════════════════════════
# Tests: Full round-trip _extract_audit_metadata → _audit_log
# ══════════════════════════════════════════════════════════════════════

class TestAuditRoundTrip:
    @pytest.fixture(autouse=True)
    def _import(self):
        from api import _extract_audit_metadata, _audit_log
        import api as _api
        self.extract = _extract_audit_metadata
        self.audit_log = _audit_log
        self.api = _api

    def _capture_audit(self, *args, **kwargs):
        captured = []
        with patch.object(self.api.audit_logger, "info",
                          side_effect=lambda msg: captured.append(msg)):
            self.audit_log(*args, **kwargs)
        return json.loads(captured[0])

    def test_patient_query_audit_includes_mrn_and_tools(self):
        tm = _make_tool_msg("lookup_clinical_notes", "[Source 1 | MRN-2026-001 | notes.pdf | 2026-01-01]")
        grounding = {"citation_coverage": 0.8, "is_grounded": True, "issues": []}
        state = _make_state(messages=[tm], active_patient_mrn="MRN-2026-001", grounding_result=grounding)

        meta = self.extract([state])
        user = {"sub": "dr-jones", "role": "clinician"}
        entry = self._capture_audit(
            "chat_response", user,
            session_id="round-trip-1",
            latency_ms=120,
            tokens_total=500,
            cost_usd=0.001,
            **meta,
        )

        assert entry["patient_mrn"] == "MRN-2026-001"
        assert entry["tools_invoked"] == ["lookup_clinical_notes"]
        assert entry["data_sources"] == "qdrant"
        assert entry["grounding_score"] == pytest.approx(0.8)
        assert entry["event"] == "chat_response"
        assert entry["user_id"] == "dr-jones"

    def test_web_search_audit_has_no_mrn(self):
        tm = _make_tool_msg("tavily_search_results_json", "diabetes treatment guidelines...")
        state = _make_state(messages=[tm], active_patient_mrn="")

        meta = self.extract([state])
        assert "patient_mrn" not in meta
        assert meta["data_sources"] == "web"

        user = {"sub": "dr-jones", "role": "clinician"}
        entry = self._capture_audit("chat_response", user, session_id="web-1", latency_ms=90, **meta)
        assert "patient_mrn" not in entry
        assert entry["data_sources"] == "web"
