# MedGraph AI — Production Readiness Audit

**Date:** February 19, 2026  
**Scope:** Full codebase audit against 2026 healthcare RAG production best practices  
**Status:** Audit complete — awaiting implementation approval

---

## Executive Summary

The codebase is a **well-architected prototype** with strong foundations: PubMedBERT embeddings (HIPAA-safe local inference), two-stage retrieval with cross-encoder reranking, LangGraph agent orchestration, two-layer summary caching (in-memory + disk), and a modern FastAPI + SSE + custom SPA frontend.

However, there are **critical gaps** in latency, security, observability, and resilience that must be addressed before production deployment in a healthcare setting where HIPAA compliance and patient safety are non-negotiable.

### Scoring

| Category | Score | Production Threshold |
|----------|-------|---------------------|
| Latency | 2/10 | 7+ |
| Security & HIPAA | 1/10 | 9+ |
| Error Handling & Resilience | 3/10 | 7+ |
| Scalability | 3/10 | 6+ |
| Monitoring & Observability | 2/10 | 7+ |
| RAG Quality | 6/10 | 7+ |
| Code Hygiene | 5/10 | 6+ |

---

## 1. CRITICAL — 120+ Second Summary Latency

### Root Cause

The summarization pipeline feeds ~84,000 characters (~21K tokens) of raw clinical text to `gpt-5-mini`, a **reasoning model**. Reasoning models spend significant tokens on internal chain-of-thought (8–10K reasoning tokens) before producing visible output. With `max_completion_tokens=16384`, the model must generate ~14K total tokens (reasoning + visible), taking **120+ seconds**.

### Why gpt-5-mini Is Wrong for Summarization

Summarization is a well-structured, deterministic task — exactly the type that OpenAI's latency optimization guide says to use a **smaller, faster model** for. Reasoning models excel at multi-step logic, math, and ambiguous decision-making — not at extracting and reformatting clinical data already present in the context.

**Reference:** OpenAI Latency Optimization Guide — *"Process tokens faster: smaller models usually run faster (and cheaper), and when used correctly can even outperform larger models."*

### Evidence

- **Tested:** Summary generation for patient MRN-2026-004782 took **123.6 seconds** on first call
- **Token budget:** ~21K input + ~14K completion tokens (includes ~8K reasoning tokens that produce zero visible output)
- **File:** `src/agent_graph/tool_clinical_notes_rag.py` — `_get_summarization_llm()` function (lines 440–458)
- **Config:** `configs/tools_config.yml` — `summarization_llm: gpt-5-mini`

### Fixes (Priority Order)

| # | Fix | Effort | Expected Latency Impact |
|---|-----|--------|------------------------|
| 1 | **Switch summarization_llm to `gpt-4.1-mini`** (non-reasoning, temp=0) | 1 line in config | **120s → ~15-20s** |
| 2 | **Pre-compute summaries at ingest time** (`prepare_vector_db.py`) | ~50 lines | **First request: 0s** (pre-built) |
| 3 | **Map-reduce summarization** (parallel chunk-group summaries → merge) | ~100 lines | **120s → ~25-30s** (parallel) |
| 4 | **Stream summary token-by-token** to SSE (not as single blob) | ~30 lines | **Perceived: instant start** |

### Recommendation

Change `summarization_llm` in `configs/tools_config.yml` to `gpt-4.1-mini`. Keep `gpt-5-mini` as the **primary agent** (where reasoning about tool selection is genuinely valuable). This single change removes the #1 production blocker.

---

## 2. CRITICAL — Security & HIPAA Compliance

This system handles **PHI (Protected Health Information)**. Current state is **not HIPAA-compliant**.

| # | Gap | Risk Level | Evidence |
|---|-----|-----------|----------|
| 1 | **No authentication** | CRITICAL | `src/api.py` — no auth middleware on any endpoint. Anyone on the network can query patient data. |
| 2 | **No authorization / RBAC** | CRITICAL | No role-based access control. No distinction between nurse, doctor, admin roles. |
| 3 | **CORS allows `*`** | HIGH | `src/api.py`, line 64: `allow_origins=["*"]`. Any website can make cross-origin requests to the API. |
| 4 | **No HTTPS enforcement** | HIGH | Uvicorn runs plain HTTP (`uvicorn.run(app, host="0.0.0.0", port=7860)`). PHI transmitted in cleartext. |
| 5 | **No HIPAA audit trail** | CRITICAL | HIPAA requires logging WHO accessed WHAT patient data WHEN. No access logging exists. |
| 6 | **PHI in console logs** | MEDIUM | `print()` calls in `tool_clinical_notes_rag.py` expose MRN and patient names to stdout (20+ instances). |
| 7 | **No input sanitization** | HIGH | User input goes directly to the LLM with no length limits, PII filtering, or prompt injection protections. |
| 8 | **Sessions stored in plaintext memory** | MEDIUM | `src/api.py`, line 117: `sessions: dict = {}`. Patient data in unencrypted in-memory dict. |
| 9 | **API keys in .env only** | LOW | No secrets vault integration (HashiCorp Vault, AWS Secrets Manager, etc.). |

### Fixes

| # | Fix | Effort |
|---|-----|--------|
| 1 | Add JWT bearer token authentication middleware | ~80 lines |
| 2 | Add role-based access decorator (e.g., `@require_role("clinician")`) | ~40 lines |
| 3 | Lock CORS to specific allowed origins | 1 line |
| 4 | Deploy behind HTTPS reverse proxy (nginx/Caddy) or add SSL to uvicorn | Config change |
| 5 | Add structured audit logging (who, what patient, when, action) | ~60 lines |
| 6 | Replace `print()` with `logging` module, redact PHI from logs | ~30 lines |
| 7 | Add input length limits + prompt injection detection middleware | ~50 lines |

---

## 3. HIGH — Error Handling & Resilience

| # | Gap | Impact | Evidence |
|---|-----|--------|----------|
| 1 | **No retry logic on LLM calls** | Single OpenAI API hiccup = user-facing error | `tool_clinical_notes_rag.py` — `llm.invoke()` called without retry wrapper |
| 2 | **No timeout on LLM calls** | A stuck call blocks the thread indefinitely | All `ChatOpenAI()` instances created without `request_timeout` parameter |
| 3 | **Bare `except Exception` everywhere** | Swallows real errors, makes debugging impossible | **19 instances** across `api.py`, `tool_clinical_notes_rag.py`, `agent_backend.py` |
| 4 | **No circuit breaker** | If OpenAI is down, every request hangs until timeout (which doesn't exist) | No health-check or fallback pattern |
| 5 | **No model fallback** | If gpt-5-mini is unavailable, the entire system is dead | No secondary model configured |
| 6 | **Thread safety on sessions dict** | `sessions` dict mutated from both async handlers and background threads without a lock | `src/api.py` — `sessions[sid] = history` from multiple concurrent contexts |
| 7 | **No request-level timeout** | Long-running requests consume server resources unboundedly | FastAPI has no timeout middleware configured |

### Fixes

| # | Fix | Effort |
|---|-----|--------|
| 1 | Add `tenacity` retry with exponential backoff on all `llm.invoke()` calls | ~20 lines |
| 2 | Add `request_timeout=60` to all `ChatOpenAI()` constructors | ~5 lines |
| 3 | Replace bare `except Exception` with specific exception types + logging | ~40 lines |
| 4 | Add circuit breaker pattern (e.g., `pybreaker`) | ~30 lines |
| 5 | Configure fallback model in `tools_config.yml` | ~20 lines |
| 6 | Add `threading.Lock` around session mutations | ~15 lines |

---

## 4. HIGH — Scalability

| # | Gap | Impact | Evidence |
|---|-----|--------|----------|
| 1 | **In-memory sessions** | Lost on restart, can't scale horizontally | `src/api.py`, line 117: `sessions: dict = {}` |
| 2 | **Qdrant embedded mode** | Single-process lock — can't run multiple uvicorn workers | `configs/tools_config.yml`: `qdrant_url: ""` (empty = embedded) |
| 3 | **Unbounded conversation history** | Every request sends ALL prior messages to the LLM — token count grows linearly, latency increases per turn | `src/api.py`, lines 182–188: full history appended every call |
| 4 | **Single uvicorn worker** | One blocking LLM call blocks all other requests | `src/api.py`, line 540: `uvicorn.run()` with no `workers` param |
| 5 | **No connection pooling** | Each tool call creates fresh HTTP connections to OpenAI | `ChatOpenAI()` instantiated per-call in `_get_summarization_llm()` |
| 6 | **Memory leak: unbounded caches** | `_memory_cache` has no max-size cap; `sessions` dict grows forever | `_memory_cache` only TTL-deletes on access, no eviction. `sessions` never cleaned up for idle sessions. |

### Fixes

| # | Fix | Effort |
|---|-----|--------|
| 1 | Move sessions to Redis (persistence + horizontal scaling) | ~60 lines |
| 2 | Switch to Qdrant server mode (Docker container) | Config + docker-compose |
| 3 | Trim conversation history to last 3-4 exchanges before sending to LLM | ~15 lines |
| 4 | Run uvicorn with `--workers 4` (requires Qdrant server mode) | 1 line |
| 5 | Create LLM singleton or use LangChain's built-in connection reuse | ~10 lines |
| 6 | Add `max_size` cap on `_memory_cache` (LRU eviction) + idle session cleanup | ~30 lines |

---

## 5. MEDIUM — Monitoring & Observability

| # | Gap | Impact | Evidence |
|---|-----|--------|----------|
| 1 | **LangSmith disabled** | Zero tracing on agent decisions, tool calls, token usage | `load_tools_config.py`: `os.environ["LANGCHAIN_TRACING_V2"] = "false"` (due to langsmith 0.7.3 RunTree/Pydantic bug) |
| 2 | **Only `print()` for logging** | No log levels, no structured JSON logs, can't aggregate or search | **20+ `print()` calls** across source files; zero `logging` module usage |
| 3 | **No metrics collection** | No latency percentiles, error rates, cache hit ratios, token usage tracked | No Prometheus/StatsD/OpenTelemetry integration |
| 4 | **Shallow health check** | `/api/health` returns static `{"status": "ok"}` without verifying Qdrant or OpenAI connectivity | `src/api.py`, lines 348-350 |
| 5 | **No alerting** | No way to detect degradation until users complain | No integration with PagerDuty/Slack/email alerts |

### Fixes

| # | Fix | Effort |
|---|-----|--------|
| 1 | Upgrade `langsmith` package to fix RunTree bug, re-enable tracing | ~5 min |
| 2 | Replace all `print()` with Python `logging` module (structured JSON format) | ~30 lines |
| 3 | Add Prometheus metrics endpoint (`/metrics`) with latency histograms, error counters, cache hit rates | ~50 lines |
| 4 | Deep health check: ping Qdrant collection + OpenAI models API | ~20 lines |
| 5 | Add Slack/webhook alerting on error rate thresholds | ~30 lines |

---

## 6. MEDIUM — RAG Quality

| # | Gap | 2026 Best Practice |
|---|-----|--------------------|
| 1 | **Generic reranker** (`ms-marco-MiniLM-L-6-v2`) | Production clinical RAG uses domain-specific rerankers fine-tuned on medical text (e.g., PubMedBERT cross-encoder) |
| 2 | **No query rewriting** | LLM should rewrite user's natural language query into an optimal retrieval query before vector search |
| 3 | **No relevance threshold** | Returns results even when all similarity scores are very low — feeds irrelevant context to LLM, causing hallucination |
| 4 | **No evaluation framework** | No systematic measurement of retrieval accuracy (RAGAS, BERTScore, human evaluation, regression tests) |
| 5 | **No multi-query retrieval** | Single query can miss information phrased differently in the source docs |
| 6 | **Chunk size untested** | 1000 chars / 200 overlap — no evidence this is optimal for clinical notes (sections vary enormously in length) |
| 7 | **No hybrid search** | Dense-only retrieval; no sparse (BM25) component for exact keyword matching (medication names, lab codes) |

### Fixes

| # | Fix | Effort |
|---|-----|--------|
| 1 | Fine-tune a cross-encoder on clinical passage-ranking data | Research project (~days) |
| 2 | Add LLM-based query rewriting step before vector search | ~30 lines |
| 3 | Add minimum similarity score threshold; return "no relevant data" below it | ~10 lines |
| 4 | Integrate RAGAS evaluation framework with ground-truth test set | ~100 lines + test data |
| 5 | Implement multi-query retrieval (LLM generates 3 query variants) | ~40 lines |
| 6 | A/B test chunk sizes (500, 1000, 1500, 2000) with retrieval eval | Research task |
| 7 | Add BM25 sparse retrieval + reciprocal rank fusion with dense results | ~60 lines |

---

## 7. LOW — Code Hygiene

| # | Issue | Evidence |
|---|-------|----------|
| 1 | **Dead Gradio code** | `src/app.py` (89 lines) and `src/chatbot/chatbot_backend.py` — never imported or used by the FastAPI backend |
| 2 | **Deprecated Tavily import** | `src/agent_graph/tool_tavily_search.py`: uses `langchain_community.tools.tavily_search` which is deprecated in favor of `langchain_tavily` |
| 3 | **No type checking** | No mypy configuration, no type stubs, no CI type-checking step |
| 4 | **No real tests** | `test.py` only checks imports (`import langchain; print("ready!")`). No unit tests, no integration tests, no RAG quality tests. |
| 5 | **Feedback truncation** | `src/api.py`, line 531: `req.bot_response[:500]` — bot responses can be 18K+ chars, but feedback only captures first 500 characters |
| 6 | **`plot_agent_schema` calls IPython** | `src/agent_graph/agent_backend.py`, line 101: calls `display(Image(...))` — fails silently on every server startup (no IPython kernel) |
| 7 | **No `.gitignore` patterns for data/** | Vector DB data, cache files, and memory CSVs may end up in version control |

### Fixes

| # | Fix | Effort |
|---|-----|--------|
| 1 | Remove `src/app.py`, `src/chatbot/chatbot_backend.py`, `src/utils/ui_settings.py` | Deletion |
| 2 | Switch to `langchain_tavily` package | ~5 lines |
| 3 | Add `mypy.ini` and type annotations | Ongoing |
| 4 | Write pytest test suite (API endpoints, RAG retrieval quality, cache behavior) | ~200 lines |
| 5 | Store full response hash in feedback, or save complete response separately | ~10 lines |
| 6 | Guard `plot_agent_schema` with environment check | ~3 lines |

---

## Prioritized Implementation Plan

### P0 — Must Fix Before Any Production Use

These are blocking issues. No healthcare deployment should proceed without addressing them.

| # | Task | Category | Effort | Impact |
|---|------|----------|--------|--------|
| 1 | **Change `summarization_llm` to `gpt-4.1-mini`** | Latency | 1 line | 120s → ~15s |
| 2 | **Add JWT authentication middleware** | Security | ~80 lines | Blocks unauthorized PHI access |
| 3 | **Add HIPAA audit logging** (structured logs: who, what patient, when) | Security | ~60 lines | HIPAA compliance |
| 4 | **Lock CORS to specific allowed origins** | Security | 1 line | Prevents cross-site PHI leakage |
| 5 | **Add `request_timeout=60`** to all `ChatOpenAI()` calls | Resilience | ~5 lines | Prevents infinite hangs |
| 6 | **Add retry with exponential backoff** (`tenacity`) on LLM calls | Resilience | ~20 lines | Survives transient OpenAI failures |
| 7 | **Add request rate limiting** (e.g., `slowapi`) | Security | ~20 lines | Prevents abuse / DoS |

### P1 — Required for Reliable Production

These are needed for a stable, maintainable production service.

| # | Task | Category | Effort | Impact |
|---|------|----------|--------|--------|
| 8 | **Move sessions to Redis** | Scalability | ~60 lines | Persistence + horizontal scaling |
| 9 | **Trim conversation history** to last 3-4 exchanges | Scalability | ~15 lines | Prevents token budget overflow + reduces latency |
| 10 | **Replace `print()` with `logging` module** (structured JSON) | Observability | ~30 lines | Proper log levels, aggregation, PHI redaction |
| 11 | **Add input validation** (length limits, prompt injection guardrails) | Security | ~50 lines | Prevents injection attacks |
| 12 | **Switch to Qdrant server mode** (Docker) | Scalability | Config + docker-compose | Multi-worker support |
| 13 | **Pre-compute summaries at ingest time** | Latency | ~50 lines | Zero first-request latency |
| 14 | **Add threading lock on sessions dict** | Resilience | ~15 lines | Prevents race conditions |
| 15 | **Remove dead Gradio code** | Hygiene | Deletion | Cleaner codebase |

### P2 — Recommended for Production Excellence

These distinguish a good system from a great one.

| # | Task | Category | Effort | Impact |
|---|------|----------|--------|--------|
| 16 | **Enable LangSmith** (upgrade package to fix RunTree bug) | Observability | ~5 min | Full agent tracing |
| 17 | **Add RAGAS evaluation framework** with regression test suite | RAG Quality | ~100 lines + data | Systematic accuracy measurement |
| 18 | **Clinical reranker** (fine-tune cross-encoder on medical text) | RAG Quality | Research project | Better retrieval precision |
| 19 | **LLM query rewriting** before vector search | RAG Quality | ~30 lines | Better recall |
| 20 | **Stream summaries token-by-token** via SSE | Latency (UX) | ~30 lines | Perceived instant response |
| 21 | **Deep health check** (`/api/health` verifies Qdrant + OpenAI) | Observability | ~20 lines | Real system status |
| 22 | **Add Prometheus metrics endpoint** | Observability | ~50 lines | Latency/error/cache dashboards |
| 23 | **Write pytest test suite** (API, RAG quality, cache) | Hygiene | ~200 lines | Regression prevention |
| 24 | **Add hybrid search** (BM25 + dense + RRF) | RAG Quality | ~60 lines | Better keyword matching |
| 25 | **Add max-size cap on memory cache** (LRU eviction) | Scalability | ~20 lines | Prevents OOM |

---

## Architecture Diagram (Current vs Target)

### Current Architecture

```
User → HTML SPA → FastAPI (single worker, no auth)
                    ↓
              LangGraph Agent (gpt-5-mini reasoning model)
                    ↓
        ┌───────────┼───────────┬──────────────┐
        ↓           ↓           ↓              ↓
   Lookup      Summarize    List Patients   Tavily Web
   (vector     (gpt-5-mini   (Qdrant        Search
    search)     120s!!)       scroll)
        ↓           ↓
   Qdrant       Qdrant → gpt-5-mini → cache (memory+disk)
   (embedded    (embedded, single-process)
    mode)

Sessions: in-memory dict (lost on restart)
Logging: print() to stdout
Auth: none
Monitoring: none
```

### Target Architecture (Production)

```
User → HTML SPA → Nginx (HTTPS + rate limit)
                    ↓
              FastAPI (multi-worker, JWT auth, audit log)
                    ↓
              LangGraph Agent (gpt-5-mini — tool routing only)
                    ↓
        ┌───────────┼───────────┬──────────────┐
        ↓           ↓           ↓              ↓
   Lookup      Summarize    List Patients   Tavily Web
   (rewritten  (gpt-4.1-mini  (Qdrant       Search
    query)      ~15s, or       scroll)
        ↓       pre-computed)
   Qdrant       ↓
   (server   Cache (Redis or memory+disk)
    mode,
    Docker)      

Sessions: Redis (persistent, horizontal scaling)
Logging: structured JSON via Python logging
Auth: JWT + RBAC
Monitoring: LangSmith + Prometheus + alerting
```

---

## Verification Notes

All claims in this audit were verified against the actual codebase on February 19, 2026:

- **19 bare `except Exception` instances** confirmed across `api.py`, `tool_clinical_notes_rag.py`, `agent_backend.py`
- **20+ `print()` calls**, zero `logging` module usage confirmed
- **Zero auth/JWT/bearer/token patterns** found in any source file
- **CORS `allow_origins=["*"]`** confirmed at `src/api.py` line 64
- **Zero `request_timeout`/retry/tenacity/backoff** patterns found
- **Zero rate limiting** patterns found
- **`sessions` dict** mutated without thread lock from multiple contexts confirmed
- **Health check** returns static `{"status": "ok"}` without dependency checks confirmed
- **`_memory_cache`** has no max-size eviction — only TTL delete-on-access confirmed
- **Feedback truncates** `bot_response` to 500 chars confirmed at `src/api.py` line 531
- **Single uvicorn worker** — no `--workers` flag confirmed at `src/api.py` line 540
- **SSE streams entire summary as one blob** — not token-by-token confirmed
- **`test.py`** contains only import checks — no actual tests confirmed
- **Gradio code** in `app.py`/`chatbot_backend.py` — dead code, not used by FastAPI confirmed
