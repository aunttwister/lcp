# Feature: Logging Enhancements

**Created:** 2026-08-06
**Status:** open
**Phase:** post-Phase 7

## North Star

The current Logs page is a good start — filterable table with pagination, status badges, error tooltips. But debugging production issues requires deeper inspection: full request/response bodies, search across payloads, timing waterfall, and a live tail of incoming traffic.

---

## Current State (What Exists)

### Logging Output (`logging_config.py`)
- structlog JSON to stdout — consumed by Docker logs
- Log levels configurable via `LOG_LEVEL` env var (default: INFO)
- Key log events: `request_complete`, `request_failed`, `circuit_breaker_tripped`, `alert_fired`, `webhook_sent/failed`, `tool_blocked`

### Logs Page (`pages/logs.html`)
- Full-page table view: #, Time, Profile, Model, Provider, Status, Prompt tokens, Completion tokens, CacheHit, CacheMiss, Cost, Saved, Duration, Error
- Filters: profile dropdown, provider dropdown, status segmented control (All/Success/Errors)
- Pagination: 100 rows per page
- Error tooltip: hover over error badge shows full error text
- Green "saved" column for cache savings
- **No search box** — can't search by request body text or error message
- **No date range** — always shows all time, newest first
- **No request body viewer** — can't see what was actually sent to the LLM
- **No timing breakdown** — only total duration, no DNS/connect/TTFB/streaming splits

### Request Table (`models.py`)
- `Request` model has: timestamp, profile, model, provider, prompt_tokens, completion_tokens, cache_hit_tokens, cache_miss_tokens, cost, latency_ms, success, error_type, error_detail, tools_blocked
- **Missing:** request_body, response_body (the actual payloads for debugging)

### `/errors` Endpoint
- Returns recent errors (last 50): timestamp, profile, error_type, error_detail, model, provider

### Server Handler Logging (`handler.py`)
- Each request logged via structlog to stdout only, not to DB in real-time
- Request details (profile, model, provider, tokens, cost, latency) written to DB AFTER response
- No mid-request logging visible in UI during streaming

---

## Plan

### Phase 1: Request/Response Body Storage

| Step | What |
|---|---|
| 1.1 | Add `request_body` and `response_body` columns to `Request` table (TEXT, nullable — bodies can be large) |
| 1.2 | Alembic migration |
| 1.3 | In `handler.py`, after reading the request body, store it in the DB entry |
| 1.4 | After receiving the LLM response, store the first ~10KB of response body (configurable truncation) |
| 1.5 | Config gate: `logging.store_bodies: true/false` in `gateway.yaml` (default: false for privacy) |

### Phase 2: Full Request/Response Viewer

| Step | What |
|---|---|
| 2.1 | Click any log row → expand inline JSON viewer below the row |
| 2.2 | Syntax-highlighted JSON (lightweight, no library — `<pre>` with manual span coloring) |
| 2.3 | Two tabs: "Request" / "Response" |
| 2.4 | Show: full messages array, tools, model params, and response content |
| 2.5 | Copy-to-clipboard button per tab |
| 2.6 | Collapse/expand toggle for long bodies |

### Phase 3: Search & Advanced Filters

| Step | What |
|---|---|
| 3.1 | Search input in toolbar: searches across request_body + response_body + error_detail |
| 3.2 | Client-side filtering: JS filters the already-loaded page (or triggers server search API for larger datasets) |
| 3.3 | `GET /api/logs/search?q=...&profile=&provider=&status=` — server-side search |
| 3.4 | Date range picker: "From" / "To" datetime inputs with presets (Last hour, 24h, 7d, 30d, all) |
| 3.5 | Filter chips: show active filters as removable pills (e.g., "Profile: l2 ✕", "Status: error ✕") |

### Phase 4: Timing Waterfall

| Step | What |
|---|---|
| 4.1 | In `request_pipeline.py`, capture timing breakdown per request: DNS lookup, TCP connect, TLS handshake, TTFB (time to first byte), streaming duration, total |
| 4.2 | Store in new `request_timing` table or as JSON column on `Request` |
| 4.3 | Waterfall visualization: horizontal stacked bar per request row showing each phase |
| 4.4 | Hover tooltip: exact ms for each phase |
| 4.5 | Aggregate stats: p50/p95/p99 per phase, shown as summary row above table |

### Phase 5: Live Tail

| Step | What |
|---|---|
| 5.1 | `GET /api/logs/live` — SSE endpoint that pushes each new request as it completes |
| 5.2 | Auto-scroll toggle in UI (on by default) |
| 5.3 | Live counter: "X requests in last 60s" |
| 5.4 | Pause/resume button — freezes the stream for inspection |
| 5.5 | Click a live row to expand body viewer (works while tailing) |
| 5.6 | Color flash animation on new rows (fades after 1s) |

### Phase 6: Export & Retention

| Step | What |
|---|---|
| 6.1 | Existing `/export?limit=N` returns CSV — extend with date range filter |
| 6.2 | Retention config: auto-delete requests older than N days (configurable, default: never) |
| 6.3 | `POST /api/logs/purge?before=YYYY-MM-DD` — manual purge |
| 6.4 | Storage stats on logs page: "X requests stored, Y MB, oldest: date" |

---

## Data Model Changes

```python
# Add to existing Request table:
request_body = Column(Text, nullable=True)
response_body = Column(Text, nullable=True)

# New table:
class RequestTiming(Base):
    __tablename__ = "request_timings"
    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(Integer, ForeignKey("requests.id"), nullable=False)
    dns_ms = Column(Integer, default=0)
    connect_ms = Column(Integer, default=0)
    tls_ms = Column(Integer, default=0)
    ttfb_ms = Column(Integer, default=0)
    streaming_ms = Column(Integer, default=0)
    total_ms = Column(Integer, default=0)
```

---

## Config (`gateway.yaml` additions)

```yaml
logging:
  store_bodies: false          # store request/response bodies in DB
  body_max_chars: 10240        # truncate bodies to 10KB
  retention_days: 0            # 0 = keep forever
```

---

## API Endpoints

| Method | Path | Status | Notes |
|---|---|---|---|
| GET | `/api/logs/search` | **NEW** | Full-text search across bodies + errors |
| GET | `/api/logs/live` | **NEW** | SSE stream of new requests |
| GET | `/api/logs/{id}` | **NEW** | Single request detail with bodies |
| POST | `/api/logs/purge` | **NEW** | Delete old logs |
| GET | `/api/logs/stats` | **NEW** | Storage stats |
| GET | `/export` | exists | Extend with `?from=&to=` params |
| GET | `/errors` | exists | Extend with search + date filters |

---

## UI Changes

| What | Where |
|---|---|
| Date range picker | `logs.html` toolbar |
| Search box | `logs.html` toolbar |
| Active filter chips | `logs.html` below toolbar |
| Expandable row (body viewer) | `logs.html` table rows |
| Timing waterfall bars | `logs.html` new column or inline visual |
| Live tail toggle | `logs.html` toolbar |
| Storage stats footer | `logs.html` bottom |
| Pause/resume live | `logs.html` toolbar |

---

## Files Changed

- `src/api/models.py` — add `request_body`, `response_body`, `RequestTiming`
- `src/server/handler.py` — store request/response bodies, timing capture
- `src/api/request_pipeline.py` — timing breakdown capture
- `src/server/endpoints.py` — search, live SSE, detail, purge, stats endpoints
- `src/ui/pages.py` — render extended logs page
- `src/ui/templates/jinja/pages/logs.html` — major redesign
- `config/gateway.yaml` — logging section
- `alembic/versions/006_add_logging_enhancements.py` — **new**
