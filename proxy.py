#!/usr/bin/env python3
"""
LLM Gateway Router — URL-path-based profile routing, tool stripping, fallback, cost tracking.

Zero dependencies beyond Python stdlib + sqlite3.

Routes:
  /{profile}/chat/completions  — profile-scoped (l2, l1, career, cron)
  /{profile}/v1/chat/completions — same with /v1 prefix
  /chat/completions, /v1/chat/completions — default (l2)
  /health — health check
  /v1/models — model list (OpenAI client compat)
  /errors — recent proxy errors (profile-scoped: /l2/errors)
"""

import json
import os
import re
import sqlite3
import time
import traceback
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timezone

# ── Config ──────────────────────────────────────────────────────────────────
DB_PATH = os.environ.get("COST_DB", "/app/data/costs.db")
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "8734"))
OPENCODE_KEY = os.environ["OPENCODE_API_KEY"]
DEEPSEEK_KEY = os.environ["DEEPSEEK_API_KEY"]

# Pricing per 1M tokens (DeepSeek official, June 2026; OpenCode uses same as lower bound)
PRICING = {
    ("deepseek", "deepseek-v4-pro"):   {"cache_hit": 0.003625, "cache_miss": 0.435, "output": 0.87},
    ("deepseek", "deepseek-v4-flash"):  {"cache_hit": 0.0028,   "cache_miss": 0.14,  "output": 0.28},
    ("opencode", "deepseek-v4-pro"):   {"cache_hit": 0.003625, "cache_miss": 0.435, "output": 0.87},
    ("opencode", "deepseek-v4-flash"):  {"cache_hit": 0.0028,   "cache_miss": 0.14,  "output": 0.28},
}

# Profile definitions: forbidden tools + provider chain
PROFILES = {
    "l2": {
        "forbidden": ["write_file", "patch", "cronjob"],
        "chain": [
            ("opencode", "deepseek-v4-pro", "https://opencode.ai/zen/go/v1", OPENCODE_KEY),
            ("deepseek", "deepseek-v4-pro", "https://api.deepseek.com/v1", DEEPSEEK_KEY),
        ],
    },
    "l1": {
        "forbidden": ["write_file", "patch", "terminal", "execute_code", "cronjob",
                       "process", "delegate_task", "memory", "send_message", "vision_analyze"],
        "chain": [
            ("opencode", "deepseek-v4-flash", "https://opencode.ai/zen/go/v1", OPENCODE_KEY),
            ("deepseek", "deepseek-v4-flash", "https://api.deepseek.com/v1", DEEPSEEK_KEY),
        ],
    },
    "career": {
        "forbidden": ["write_file", "patch", "terminal", "execute_code", "cronjob",
                       "process", "delegate_task", "memory", "vision_analyze", "read_file",
                       "search_files", "skill_manage", "todo"],
        "chain": [
            ("deepseek", "deepseek-v4-flash", "https://api.deepseek.com/v1", DEEPSEEK_KEY),
        ],
    },
    "cron": {
        "forbidden": None,  # ALL tools forbidden
        "chain": [
            ("deepseek", "deepseek-v4-flash", "https://api.deepseek.com/v1", DEEPSEEK_KEY),
        ],
    },
}

DEFAULT_PROFILE = "l2"

# ── Circuit Breaker ──────────────────────────────────────────────────────────
# Provider health tracking: per (provider, base_url) key
PROVIDER_HEALTH = {}  # mutable global — populated as providers are used

CB_FAILURES_DEGRADED = 3   # consecutive failures → degraded
CB_FAILURES_DEAD = 6       # consecutive failures → dead
CB_DEGRADED_COOLDOWN = 30  # seconds before retrying degraded provider
CB_DEAD_COOLDOWN = 120     # seconds before retrying dead provider


def _provider_key(provider, base_url, profile):
    return (provider, base_url, profile)


def _get_health(provider, base_url, profile):
    """Get or init health state for a provider+profile combination."""
    key = _provider_key(provider, base_url, profile)
    if key not in PROVIDER_HEALTH:
        PROVIDER_HEALTH[key] = {
            "consecutive_failures": 0,
            "last_failure": None,
            "last_success": None,
            "status": "healthy",
            "tripped_until": None,
        }
    return PROVIDER_HEALTH[key]


def is_provider_available(provider, base_url, profile):
    """Check if provider is currently available for this profile (not circuit-broken)."""
    h = _get_health(provider, base_url, profile)
    if h["status"] == "healthy":
        return True
    if h["tripped_until"] and time.time() >= h["tripped_until"]:
        return True
    return False


def record_provider_success(provider, base_url, profile):
    """Record a successful call — resets circuit breaker for this profile."""
    h = _get_health(provider, base_url, profile)
    h["status"] = "healthy"
    h["consecutive_failures"] = 0
    h["last_success"] = datetime.now(timezone.utc).isoformat()
    h["tripped_until"] = None


def record_provider_failure(provider, base_url, profile):
    """Record a failed call — may trip circuit breaker for this profile."""
    h = _get_health(provider, base_url, profile)
    h["consecutive_failures"] += 1
    h["last_failure"] = datetime.now(timezone.utc).isoformat()
    n = h["consecutive_failures"]
    if n >= CB_FAILURES_DEAD:
        h["status"] = "dead"
        h["tripped_until"] = time.time() + CB_DEAD_COOLDOWN
    elif n >= CB_FAILURES_DEGRADED:
        h["status"] = "degraded"
        h["tripped_until"] = time.time() + CB_DEGRADED_COOLDOWN


# ── Database ─────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")

    # Create tables (if not exists)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            profile TEXT NOT NULL DEFAULT 'unknown',
            model TEXT NOT NULL,
            provider TEXT NOT NULL,
            prompt_tokens INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            prompt_cache_hit_tokens INTEGER DEFAULT 0,
            prompt_cache_miss_tokens INTEGER DEFAULT 0,
            estimated_cost_usd REAL DEFAULT 0.0,
            duration_ms INTEGER DEFAULT 0,
            status TEXT DEFAULT 'success',
            fallback_used INTEGER DEFAULT 0
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_summary (
            date TEXT NOT NULL,
            profile TEXT NOT NULL DEFAULT 'unknown',
            model TEXT NOT NULL,
            provider TEXT NOT NULL,
            total_prompt_tokens INTEGER DEFAULT 0,
            total_completion_tokens INTEGER DEFAULT 0,
            total_cache_hit_tokens INTEGER DEFAULT 0,
            total_cache_miss_tokens INTEGER DEFAULT 0,
            total_cost_usd REAL DEFAULT 0.0,
            request_count INTEGER DEFAULT 0,
            fallback_count INTEGER DEFAULT 0,
            PRIMARY KEY (date, profile, model, provider)
        )""")
    # NEW: persistent error log table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS proxy_errors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            profile TEXT NOT NULL,
            provider TEXT DEFAULT '',
            status_code TEXT DEFAULT '',
            error_type TEXT DEFAULT '',
            error_message TEXT DEFAULT '',
            error_body TEXT DEFAULT '',
            ray_id TEXT DEFAULT ''
        )""")

    # Migration: add columns introduced after initial deploy
    for table, col_def in [
        ("requests", "profile TEXT NOT NULL DEFAULT 'unknown'"),
        ("requests", "fallback_used INTEGER DEFAULT 0"),
    ]:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")
        except sqlite3.OperationalError:
            pass

    # daily_summary primary key changed — drop/recreate if old schema
    try:
        conn.execute("SELECT profile FROM daily_summary LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("DROP TABLE IF EXISTS daily_summary")
        conn.execute("""
            CREATE TABLE daily_summary (
                date TEXT NOT NULL,
                profile TEXT NOT NULL DEFAULT 'unknown',
                model TEXT NOT NULL,
                provider TEXT NOT NULL,
                total_prompt_tokens INTEGER DEFAULT 0,
                total_completion_tokens INTEGER DEFAULT 0,
                total_cache_hit_tokens INTEGER DEFAULT 0,
                total_cache_miss_tokens INTEGER DEFAULT 0,
                total_cost_usd REAL DEFAULT 0.0,
                request_count INTEGER DEFAULT 0,
                fallback_count INTEGER DEFAULT 0,
                PRIMARY KEY (date, profile, model, provider)
            )""")

    # Create indexes (after migration, so columns exist)
    for idx_sql in [
        "CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_requests_profile ON requests(profile, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(model, provider)",
        "CREATE INDEX IF NOT EXISTS idx_errors_ts ON proxy_errors(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_errors_profile ON proxy_errors(profile, timestamp)",
    ]:
        try:
            conn.execute(idx_sql)
        except sqlite3.OperationalError:
            pass

    conn.commit()
    return conn


def extract_sse_usage(raw_sse):
    """Extract usage from SSE stream — scans all chunks for the one with usage data."""
    usage = {}
    for line in raw_sse.split('\n'):
        s = line.strip()
        if s.startswith('data:') and 'data: [DONE]' not in s:
            try:
                chunk = json.loads(s[5:].strip())
                if 'usage' in chunk:
                    usage = chunk['usage']
            except (json.JSONDecodeError, KeyError):
                continue
    return usage


def compute_cost(provider, model, usage):
    key = (provider, model)
    p = PRICING.get(key)
    if not p:
        return 0.0
    cache_hit = usage.get("prompt_cache_hit_tokens", 0)
    cache_miss = usage.get("prompt_cache_miss_tokens", 0)
    output = usage.get("completion_tokens", 0)
    if cache_hit == 0 and cache_miss == 0:
        cache_miss = usage.get("prompt_tokens", 0)
    cost = (cache_hit / 1_000_000) * p["cache_hit"] \
         + (cache_miss / 1_000_000) * p["cache_miss"] \
         + (output / 1_000_000) * p["output"]
    return round(cost, 8)


def log_request(db, profile, model, provider, usage, duration_ms, status="success", fallback_used=0):
    cost = compute_cost(provider, model, usage)
    db.execute(
        """INSERT INTO requests
           (timestamp, profile, model, provider, prompt_tokens, completion_tokens,
            prompt_cache_hit_tokens, prompt_cache_miss_tokens,
            estimated_cost_usd, duration_ms, status, fallback_used)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (datetime.now(timezone.utc).isoformat(), profile, model, provider,
         usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
         usage.get("prompt_cache_hit_tokens", 0), usage.get("prompt_cache_miss_tokens", 0),
         cost, duration_ms, status, fallback_used))
    # Only successful requests count toward daily summaries
    if status == "success":
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        db.execute(
            """INSERT INTO daily_summary (date, profile, model, provider,
               total_prompt_tokens, total_completion_tokens, total_cache_hit_tokens,
               total_cache_miss_tokens, total_cost_usd, request_count, fallback_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
               ON CONFLICT(date, profile, model, provider) DO UPDATE SET
                 total_prompt_tokens = total_prompt_tokens + excluded.total_prompt_tokens,
                 total_completion_tokens = total_completion_tokens + excluded.total_completion_tokens,
                 total_cache_hit_tokens = total_cache_hit_tokens + excluded.total_cache_hit_tokens,
                 total_cache_miss_tokens = total_cache_miss_tokens + excluded.total_cache_miss_tokens,
                 total_cost_usd = total_cost_usd + excluded.total_cost_usd,
                 request_count = request_count + 1,
                 fallback_count = fallback_count + excluded.fallback_count""",
            (today, profile, model, provider,
             usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
             usage.get("prompt_cache_hit_tokens", 0), usage.get("prompt_cache_miss_tokens", 0),
             cost, fallback_used))
    db.commit()


# ── SSE accumulation ─────────────────────────────────────────────────────────
def _accumulate_sse(raw):
    """Accumulate SSE chunks into a single JSON response.
    Handles deepseek's streaming format with reasoning_content and tool_calls."""
    lines = raw.strip().split('\n')
    content_parts = []
    reasoning_parts = []
    tool_calls_delta = {}
    finish_reason = None
    final_usage = {}
    model = None
    id_val = None
    created_ts = None
    sfp = None

    for line in lines:
        line = line.strip()
        if not line.startswith('data: ') or line == 'data: [DONE]':
            continue
        try:
            chunk = json.loads(line[6:])  # strip "data: "
        except (json.JSONDecodeError, IndexError):
            continue

        choices = chunk.get('choices', [])
        # Capture top-level metadata from first chunk
        if id_val is None:
            id_val = chunk.get('id')
            created_ts = chunk.get('created')
            model = chunk.get('model')
            sfp = chunk.get('system_fingerprint')

        if chunk.get('usage'):
            final_usage = chunk['usage']

        for choice in choices:
            delta = choice.get('delta', {})
            if delta.get('content'):
                content_parts.append(delta['content'])
            if delta.get('reasoning_content'):
                reasoning_parts.append(delta['reasoning_content'])
            if delta.get('tool_calls'):
                if not tool_calls_delta:
                    tool_calls_delta = delta['tool_calls']
            if choice.get('finish_reason'):
                finish_reason = choice['finish_reason']

    content = ''.join(content_parts) if content_parts else ''
    reasoning = ''.join(reasoning_parts) if reasoning_parts else ''

    message = {"role": "assistant"}
    if content:
        message["content"] = content
    elif reasoning:
        # reasoning model put everything in reasoning_content — promote to content
        message["content"] = reasoning
        message["reasoning_content"] = reasoning
    else:
        message["content"] = ""
    if reasoning and not content:
        pass  # already handled above — reasoning promoted to content
    elif reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls_delta:
        message["tool_calls"] = tool_calls_delta

    response = {
        "id": id_val or "",
        "object": "chat.completion",
        "created": created_ts or int(time.time()),
        "model": model or "",
        "choices": [{
            "index": 0,
            "message": message,
            "logprobs": None,
            "finish_reason": finish_reason or "stop"
        }],
    }
    if final_usage:
        response["usage"] = final_usage
    if sfp:
        response["system_fingerprint"] = sfp

    return json.dumps(response).encode('utf-8')


# ── Persistent error logging ────────────────────────────────────────────────
def _extract_ray_id(body_text):
    """Extract Cloudflare Ray ID from error body."""
    m = re.search(r'ray_id["\']?\s*[:=]\s*["\']?([a-z0-9]+)', body_text, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r'cf-ray:\s*([a-z0-9-]+)', body_text, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r'[?&]ray=([a-z0-9-]+)', body_text, re.IGNORECASE)
    if m:
        return m.group(1)
    return ''


def log_proxy_error(db, profile, provider, status_code, error_type, error_message, error_body=''):
    """Record a proxy-level error in the persistent proxy_errors table."""
    now = datetime.now(timezone.utc).isoformat()
    ray_id = _extract_ray_id(error_body)
    db.execute(
        """INSERT INTO proxy_errors
           (timestamp, profile, provider, status_code, error_type, error_message, error_body, ray_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (now, profile, provider, str(status_code), error_type, error_message[:500],
         error_body[:2000], ray_id))
    db.commit()
    return ray_id


# ── Tool stripping ──────────────────────────────────────────────────────────
def strip_tools(request_data, profile_name):
    """Remove forbidden tools from the request. Returns count of tools stripped."""
    profile = PROFILES.get(profile_name, PROFILES[DEFAULT_PROFILE])
    forbidden = profile["forbidden"]

    if forbidden is None:
        # ALL tools forbidden — strip everything
        tools = request_data.get("tools", [])
        stripped = len(tools)
        if tools:
            request_data["tools"] = []
        if "tool_choice" in request_data:
            request_data["tool_choice"] = "none"
        return stripped

    tools = request_data.get("tools", [])
    if not tools:
        return 0

    allowed = []
    stripped = 0
    for tool in tools:
        fn = tool.get("function", {})
        name = fn.get("name", "")
        if name in forbidden:
            stripped += 1
        else:
            allowed.append(tool)

    request_data["tools"] = allowed

    # If tool_choice points to a forbidden tool, reset to auto
    tc = request_data.get("tool_choice", {})
    if isinstance(tc, dict) and tc.get("type") == "function":
        fn_name = tc.get("function", {}).get("name", "")
        if fn_name in forbidden:
            request_data["tool_choice"] = "auto"
    elif tc == "none" and not allowed:
        pass  # already none
    elif tc == "none":
        pass
    elif not allowed:
        request_data["tool_choice"] = "none"

    return stripped


# ── Provider routing with fallback ──────────────────────────────────────────
RETRYABLE_CODES = {400, 429, 500, 502, 503, 504}  # 401/403 handled specially for opencode

def try_chain(chain, request_data, profile_name):
    """Try each provider in chain. On retryable error, fall through to next.
    Returns (response_body, provider_name, model_name, fallback_used)."""
    last_error = None
    last_body = None
    last_provider = None

    for idx, (provider, model, base_url, api_key) in enumerate(chain):
        # Circuit breaker: skip unhealthy providers (per-profile)
        if not is_provider_available(provider, base_url, profile_name):
            h = _get_health(provider, base_url, profile_name)
            print(f"[{profile_name}] CIRCUIT BREAKER: {provider} is {h['status']} ({h['consecutive_failures']} failures) — skipping",
                  flush=True)
            continue

        api_model = model  # Use the model name as-is for upstream
        url = f"{base_url.rstrip('/')}/chat/completions"
        upstream_data = dict(request_data)
        upstream_data["model"] = api_model

        # deepseek-v4-pro is a reasoning model — non-streaming leaves content blank.
        # Keep streaming enabled so deepseek sends SSE chunks with content.
        upstream_data.pop("stream_options", None)  # strip stream_options (deepseek rejects it)

        # Retry same provider on transient errors (400/429/5xx)
        max_provider_retries = 3
        for attempt in range(max_provider_retries):
            body = json.dumps(upstream_data).encode("utf-8")

            # Use streaming for the upstream call
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {api_key}",
                         "Accept": "text/event-stream"},
                method="POST")

            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    response_body = resp.read()
                    # Return raw SSE if that's what deepseek sent — gateway expects streaming
                    raw = response_body.decode('utf-8', errors='replace')
                    if raw.strip().startswith('data:') and 'data: [DONE]' in raw:
                        # Reassemble into a proper SSE response: make sure each chunk
                        # is properly separated and ends with the [DONE] marker.
                        if not raw.rstrip().endswith('data: [DONE]'):
                            raw = raw.rstrip() + '\n\ndata: [DONE]\n\n'
                        response_body = raw.encode('utf-8')
                    else:
                        # Plain JSON — just validate the content
                        response_data = json.loads(response_body)
                        # Promote reasoning to content if content is empty
                        msg = response_data['choices'][0]['message']
                        if not msg.get('content') and msg.get('reasoning_content'):
                            msg['content'] = msg['reasoning_content']
                        response_body = json.dumps(response_data).encode('utf-8')
                    if attempt > 0:
                        print(f"[{profile_name}] RETRY OK: {provider}/{api_model} (attempt {attempt+1})",
                              flush=True)
                    record_provider_success(provider, base_url, profile_name)
                    return response_body, provider, api_model, idx > 0, last_provider, getattr(last_error, 'code', None) if last_error else None
            except urllib.error.HTTPError as e:
                error_body = e.read()
                last_error = e
                last_body = error_body
                last_provider = provider
                # OpenCode-specific: 401 (insufficient balance) or 403 (Cloudflare) → fall through
                if e.code in (401, 403) and provider == "opencode":
                    record_provider_failure(provider, base_url, profile_name)
                    break  # break provider retry loop → next provider in chain
                # Retryable codes — retry same provider
                if e.code in RETRYABLE_CODES:
                    if attempt < max_provider_retries - 1:
                        print(f"[{profile_name}] RETRYING: {provider}/{api_model} (attempt {attempt+1}, HTTP {e.code})",
                              flush=True)
                        continue
                    break  # exhausted retries, try next provider
                # All other codes — stop immediately
                raise
            except Exception as e:
                last_error = e
                last_body = None
                last_provider = provider
                if attempt < max_provider_retries - 1:
                    print(f"[{profile_name}] RETRYING: {provider}/{api_model} ({type(e).__name__})",
                          flush=True)
                    continue
                record_provider_failure(provider, base_url, profile_name)
                break  # exhausted retries, try next provider

    # All providers failed
    if last_error:
        raise last_error
    raise Exception(f"All providers in chain exhausted for {profile_name}")


# ── Cost Dashboard ───────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Gateway — Cost Dashboard</title>
<style>
/* shadcn/ui — Dark Theme */
:root {
  --background: 222.2 84% 4.9%;
  --foreground: 210 40% 98%;
  --card: 222.2 84% 4.9%;
  --card-border: 217.2 32.6% 17.5%;
  --primary: 210 40% 98%;
  --primary-foreground: 222.2 47.4% 11.2%;
  --secondary: 217.2 32.6% 17.5%;
  --secondary-foreground: 210 40% 98%;
  --muted: 217.2 32.6% 17.5%;
  --muted-foreground: 215 20.2% 65.1%;
  --destructive: 0 62.8% 30.6%;
  --destructive-foreground: 210 40% 98%;
  --border: 217.2 32.6% 17.5%;
  --radius: 0.5rem;
  /* semantic */
  --green-bg: 142.1 76.2% 6%;
  --green-fg: 142.1 70.6% 45.3%;
  --amber-bg: 26 83.3% 7%;
  --amber-fg: 43.3 96.4% 56.3%;
  --red-bg: 0 74.2% 7%;
  --red-fg: 0 83.2% 60.2%;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: hsl(var(--background));
  color: hsl(var(--foreground));
  padding: 2rem;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}
h1 { font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.25rem; }
.subtitle { color: hsl(var(--muted-foreground)); font-size: 0.8125rem; margin-bottom: 2rem; }

/* Cards */
.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 0.75rem;
}
.card {
  background: hsl(var(--card));
  border: 1px solid hsl(var(--card-border));
  border-radius: var(--radius);
  padding: 1.25rem;
  transition: border-color 0.15s;
}
.card:hover { border-color: hsl(var(--border) / 0.5); }
.card .label {
  font-size: 0.6875rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: hsl(var(--muted-foreground));
}
.card .value {
  font-size: 1.75rem;
  font-weight: 700;
  margin-top: 0.375rem;
  font-variant-numeric: tabular-nums;
}
.card .sub {
  font-size: 0.75rem;
  color: hsl(var(--muted-foreground));
  margin-top: 0.25rem;
}

/* Details / Collapsible Sections */
details.dashboard-section {
  margin-bottom: 1.5rem;
  border: 1px solid hsl(var(--card-border));
  border-radius: var(--radius);
  overflow: hidden;
  background: hsl(var(--card));
}
details.dashboard-section > summary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.75rem 1.25rem;
  font-size: 0.8125rem;
  font-weight: 600;
  cursor: pointer;
  user-select: none;
  color: hsl(var(--foreground));
  list-style: none;
}
details.dashboard-section > summary::-webkit-details-marker { display: none; }
details.dashboard-section[open] > summary {
  border-bottom: 1px solid hsl(var(--card-border));
  background: hsl(var(--secondary));
}
details.dashboard-section > summary .chevron {
  font-size: 0.6875rem;
  transition: transform 0.2s;
  color: hsl(var(--muted-foreground));
}
details.dashboard-section[open] > summary .chevron {
  transform: rotate(180deg);
}
details.dashboard-section .section-content {
  padding: 1rem 1.25rem;
}

/* Provider Health */
.provider-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 0.75rem;
}
.provider-card {
  background: hsl(var(--card));
  border: 1px solid hsl(var(--card-border));
  border-radius: var(--radius);
  padding: 1rem;
}
.provider-card .name {
  font-size: 0.9375rem;
  font-weight: 600;
  font-family: 'SF Mono', 'Fira Code', monospace;
}
.status-pill {
  display: inline-block;
  padding: 0.125rem 0.625rem;
  border-radius: 9999px;
  font-size: 0.6875rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  margin-left: 0.5rem;
}
.pill-healthy { background: hsl(var(--green-bg)); color: hsl(var(--green-fg)); }
.pill-degraded { background: hsl(var(--amber-bg)); color: hsl(var(--amber-fg)); }
.pill-dead { background: hsl(var(--red-bg)); color: hsl(var(--red-fg)); }
.provider-card .detail {
  font-size: 0.75rem;
  color: hsl(var(--muted-foreground));
  margin-top: 0.375rem;
  line-height: 1.5;
}
.provider-card .detail .mono {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 0.6875rem;
}

/* Tables */
.table-wrap {
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
}
table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  font-size: 0.8125rem;
}
thead th {
  text-align: left;
  padding: 0.5rem 0.75rem;
  color: hsl(var(--muted-foreground));
  font-size: 0.6875rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  border-bottom: 1px solid hsl(var(--card-border));
}
tbody td {
  padding: 0.5rem 0.75rem;
  border-bottom: 1px solid hsl(var(--card-border) / 0.35);
}
tbody tr:hover td { background: hsl(var(--secondary) / 0.4); }
.cost { text-align: right; }
.mono { font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace; font-size: 0.75rem; }

/* Badges */
.badge {
  display: inline-block;
  padding: 0.0625rem 0.5rem;
  border-radius: 9999px;
  font-size: 0.6875rem;
  font-weight: 600;
  letter-spacing: 0.01em;
}
.badge-fallback { background: hsl(var(--amber-bg)); color: hsl(var(--amber-fg)); }
.badge-error { background: hsl(var(--red-bg)); color: hsl(var(--red-fg)); }
.badge-success { background: hsl(var(--green-bg)); color: hsl(var(--green-fg)); }

/* Footer */
.refresh {
  font-size: 0.75rem;
  color: hsl(var(--muted-foreground));
  text-align: center;
  margin-top: 2rem;
  padding-top: 1rem;
  border-top: 1px solid hsl(var(--card-border));
}

/* Empty state */
.empty { text-align: center; color: hsl(var(--muted-foreground)); padding: 1.5rem; font-size: 0.8125rem; }

/* Status colors for summary values */
.green { color: hsl(var(--green-fg)); }
.amber { color: hsl(var(--amber-fg)); }
.red   { color: hsl(var(--red-fg)); }
</style>
</head>
<body>
<h1>LLM Gateway Dashboard</h1>
<p class="subtitle">Cost tracking · Provider health · Request history</p>
{profile_select}

<details class="dashboard-section" open>
<summary><span>Provider Health</span><span class="chevron">▾</span></summary>
<div class="section-content provider-grid">
{provider_cards}
</div>
</details>

<details class="dashboard-section" open>
<summary><span>Summary</span><span class="chevron">▾</span></summary>
<div class="section-content cards">
{summary_cards}
</div>
</details>

<details class="dashboard-section">
<summary><span>Daily Costs</span><span class="chevron">▾</span></summary>
<div class="section-content">
<div class="table-wrap">
<table>
<thead><tr><th>Date</th><th>Profile</th><th>Model</th><th>Provider</th><th class="cost">Reqs</th><th class="cost">Fb</th><th class="cost">Cache Hit</th><th class="cost">Cache Miss</th><th class="cost">Output</th><th class="cost">Cost</th></tr></thead>
<tbody>
{daily_rows}
</tbody>
</table>
</div>
</div>
</details>

<details class="dashboard-section">
<summary><span>Recent Requests</span><span class="chevron">▾</span></summary>
<div class="section-content">
<div class="table-wrap">
<table>
<thead><tr><th>Time</th><th>Profile</th><th>Model</th><th>Provider</th><th class="cost">Status</th><th class="cost">Dur</th><th class="cost">Cost</th></tr></thead>
<tbody>
{recent_rows}
</tbody>
</table>
</div>
</div>
</details>

<details class="dashboard-section">
<summary><span>Recent Errors</span><span class="chevron">▾</span></summary>
<div class="section-content">
<div class="table-wrap">
<table>
<thead><tr><th>Time</th><th>Profile</th><th>Provider</th><th>Status</th><th>Type</th><th>Message</th><th>Ray ID</th></tr></thead>
<tbody>
{error_rows}
</tbody>
</table>
</div>
</div>
</details>

<p class="refresh">Generated {generated_at} · Refresh page to update</p>
</body>
</html>"""


def _fmt_tokens(n):
    if n is None or n == 0:
        return "0"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def _fmt_cost(n):
    if n is None:
        return "$0.000000"
    if n == 0:
        return "$0.000000"
    # Show exact value — never round to 0
    if n < 0.000001:
        return f"${n:.10f}".rstrip('0')
    if n < 0.0001:
        return f"${n:.8f}".rstrip('0')
    if n < 0.01:
        return f"${n:.6f}".rstrip('0')
    return f"${n:.6f}".rstrip('0')


def _fmt_duration(ms):
    if ms is None or ms == 0:
        return "-"
    if ms < 1000:
        return f"{ms}ms"
    return f"{ms/1000:.1f}s"


def _status_badge(status):
    if not status:
        return '<span class="badge badge-success">ok</span>'
    s = str(status)
    if s == "success":
        return '<span class="badge badge-success">ok</span>'
    if "error" in s.lower():
        return f'<span class="badge badge-error">{s[:20]}</span>'
    return s[:20]


def build_dashboard_html(db, profile_filter=None):
    from datetime import datetime, timezone

    now_dt = datetime.now(timezone.utc)
    now = now_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    today = now_dt.strftime("%Y-%m-%d")

    # Profile selector
    profile_select = '<div style="margin-bottom:1.5rem;display:flex;gap:0.5rem;flex-wrap:wrap">'
    for p in ["all", "l2", "l1", "career", "cron"]:
        active = (p == "all" and not profile_filter) or (p == profile_filter)
        href = f'/{p}/dashboard' if p != "all" else '/dashboard'
        style = 'style="padding:0.25rem 0.75rem;border-radius:9999px;font-size:0.75rem;font-weight:600;text-decoration:none;'
        if active:
            style += 'background:hsl(var(--primary));color:hsl(var(--primary-foreground))"'
        else:
            style += 'background:hsl(var(--secondary));color:hsl(var(--muted-foreground))"'
        profile_select += f'<a href="{href}" {style}>{p.upper()}</a> '
    profile_select += '</div>'
    if profile_filter:
        profile_select += f'<p class="subtitle">Showing profile: <strong>{profile_filter.upper()}</strong></p>'

    # Provider health cards — grouped by profile
    provider_cards = ""
    if PROVIDER_HEALTH:
        # Group by profile
        by_profile = {}
        for key, h in PROVIDER_HEALTH.items():
            prov_name, base_url, prof = key
            if profile_filter and prof != profile_filter:
                continue
            by_profile.setdefault(prof, []).append((prov_name, base_url, h))
        
        for prof, entries in sorted(by_profile.items()):
            provider_cards += f'<div style="grid-column:1/-1;margin-top:0.5rem;font-size:0.75rem;font-weight:700;color:hsl(var(--muted-foreground));text-transform:uppercase;letter-spacing:0.05em">{prof}</div>'
            for prov_name, base_url, h in entries:
                status_class = f"pill-{h['status']}"
                tripped_str = ""
                if h["tripped_until"]:
                    tripped_dt = datetime.fromtimestamp(h["tripped_until"], tz=timezone.utc)
                    tripped_str = f' · Tripped until {tripped_dt.strftime("%H:%M:%S")}'
                provider_cards += f"""<div class="provider-card">
  <span class="name">{prov_name}</span><span class="status-pill {status_class}">{h['status'].upper()}</span>
  <div class="detail" style="margin-top:8px">{base_url}</div>
  <div class="detail">Failures: {h['consecutive_failures']}{tripped_str}</div>
  <div class="detail">Last success: {h['last_success'] or 'never'}</div>
  <div class="detail">Last failure: {h['last_failure'] or 'never'}</div>
</div>"""
    if not provider_cards:
        provider_cards = '<div class="card"><div class="label">No provider data yet</div></div>'

    # Summary cards
    summary_where = "WHERE profile = ?" if profile_filter else ""
    summary_params = (profile_filter,) if profile_filter else ()
    totals = db.execute(f"""
        SELECT 
            COUNT(DISTINCT date) as active_days,
            SUM(request_count) as total_requests,
            SUM(fallback_count) as total_fallbacks,
            SUM(total_cache_hit_tokens) as total_cache_hit,
            SUM(total_cache_miss_tokens) as total_cache_miss,
            SUM(total_completion_tokens) as total_output,
            ROUND(SUM(total_cost_usd), 6) as total_cost
        FROM daily_summary
        {summary_where}
    """, summary_params).fetchone()

    active_days = totals[0] or 0
    total_requests = totals[1] or 0
    total_fallbacks = totals[2] or 0
    total_cache_hit = totals[3] or 0
    total_cache_miss = totals[4] or 0
    total_output = totals[5] or 0
    total_cost = totals[6] or 0

    total_prompt = total_cache_hit + total_cache_miss
    cache_pct = f"{(total_cache_hit / total_prompt * 100):.1f}%" if total_prompt > 0 else "n/a"
    fb_pct = f"{(total_fallbacks / total_requests * 100):.1f}%" if total_requests > 0 else "n/a"

    # Per-profile costs
    profile_costs = db.execute(f"""
        SELECT profile, ROUND(SUM(total_cost_usd), 6) as cost, SUM(request_count) as reqs
        FROM daily_summary {summary_where} GROUP BY profile ORDER BY cost DESC
    """, summary_params).fetchall()

    profile_lines = ""
    for prof, cost, reqs in profile_costs:
        profile_lines += f'<div class="card"><div class="label">{prof} · {reqs} reqs</div><div class="value">{_fmt_cost(cost)}</div></div>\n'

    summary_cards = f"""<div class="card">
  <div class="label">Total Cost</div>
  <div class="value">{_fmt_cost(total_cost)}</div>
  <div class="sub">{active_days} active days</div>
</div>
<div class="card">
  <div class="label">Total Requests</div>
  <div class="value">{total_requests:,}</div>
  <div class="sub">{total_fallbacks:,} fallbacks ({fb_pct})</div>
</div>
<div class="card">
  <div class="label">Cache Hit Ratio</div>
  <div class="value good">{cache_pct}</div>
  <div class="sub">{_fmt_tokens(total_cache_hit)} hit / {_fmt_tokens(total_cache_miss)} miss</div>
</div>
<div class="card">
  <div class="label">Output Tokens</div>
  <div class="value">{_fmt_tokens(total_output)}</div>
  <div class="sub">prompt: {_fmt_tokens(total_prompt)}</div>
</div>
{profile_lines}"""

    # Daily costs table
    daily = db.execute(f"""
        SELECT date, profile, model, provider, request_count, fallback_count,
               total_cache_hit_tokens, total_cache_miss_tokens, total_completion_tokens,
               ROUND(total_cost_usd, 6)
        FROM daily_summary
        {summary_where}
        ORDER BY date DESC, profile, total_cost_usd DESC
        LIMIT 100
    """, summary_params).fetchall()

    daily_rows = ""
    for row in daily:
        date, prof, model, prov, reqs, fbs, ch, cm, out, cost = row
        daily_rows += f"""<tr>
  <td class="mono">{date}</td><td>{prof}</td><td class="mono">{model}</td><td>{prov}</td>
  <td class="cost">{reqs}</td><td class="cost">{fbs}</td>
  <td class="cost">{_fmt_tokens(ch)}</td><td class="cost">{_fmt_tokens(cm)}</td>
  <td class="cost">{_fmt_tokens(out)}</td><td class="cost mono">{_fmt_cost(cost)}</td>
</tr>\n"""

    if not daily_rows:
        daily_rows = '<tr><td colspan="10" style="text-align:center;color:#64748b;padding:24px">No data yet</td></tr>'

    # Recent requests table
    recent_where = "WHERE profile = ?" if profile_filter else ""
    recent = db.execute(f"""
        SELECT timestamp, profile, model, provider, status, duration_ms,
               estimated_cost_usd, fallback_used
        FROM requests
        {recent_where}
        ORDER BY id DESC LIMIT 50
    """, (profile_filter,) if profile_filter else ()).fetchall()

    recent_rows = ""
    for row in recent:
        ts, prof, model, prov, status, dur, cost, fb = row
        ts_short = ts
        if ts and "T" in ts:
            dt_part, tm_part = ts.split("T", 1)
            if dt_part == now.split(" ")[0]:
                ts_short = tm_part.split(".")[0][:8]
            else:
                ts_short = ts[:16].replace("T", " ")
        status_html = _status_badge(status)
        fb_badge = ' <span class="badge badge-fallback">FB</span>' if fb else ""
        recent_rows += f"""<tr>
  <td class="mono">{ts_short}</td><td>{prof}</td><td class="mono">{model}</td><td>{prov}</td>
  <td class="cost">{status_html}{fb_badge}</td>
  <td class="cost">{_fmt_duration(dur)}</td><td class="cost mono">{_fmt_cost(cost)}</td>
</tr>\n"""

    if not recent_rows:
        recent_rows = '<tr><td colspan="7" style="text-align:center;color:#64748b;padding:24px">No requests yet</td></tr>'

    # Error log — from proxy_errors table with full detail
    err_where = "WHERE profile = ?" if profile_filter else ""
    errors = db.execute(f"""
        SELECT timestamp, profile, provider, status_code, error_type,
               substr(error_message, 1, 200) as error_message, ray_id
        FROM proxy_errors
        {err_where}
        ORDER BY id DESC LIMIT 30
    """, (profile_filter,) if profile_filter else ()).fetchall()

    error_rows = ""
    for row in errors:
        ts, prof, prov, sc, etype, emsg, rid = row
        ts_short = ts
        if ts and "T" in ts:
            dt_part, tm_part = ts.split("T", 1)
            if dt_part == now.split(" ")[0]:
                ts_short = tm_part.split(".")[0][:8]
            else:
                ts_short = ts[:16].replace("T", " ")
        sc_display = sc if sc else "-"
        etype_display = etype if etype else "-"
        emsg_display = emsg[:80] if emsg else "-"
        rid_display = rid[:12] if rid else "-"
        error_rows += f"""<tr>
  <td class="mono">{ts_short}</td><td>{prof}</td><td>{prov}</td>
  <td class="cost">{_status_badge(sc_display)}</td>
  <td class="mono" style="font-size:0.6875rem">{etype_display}</td>
  <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="{emsg or ''}">{emsg_display}</td>
  <td class="mono" style="font-size:0.625rem">{rid_display}</td>
</tr>\n"""

    if not error_rows:
        error_rows = '<tr><td colspan="7" style="text-align:center;color:#64748b;padding:24px">No errors</td></tr>'

    return (DASHBOARD_HTML
        .replace('{profile_select}', profile_select)
        .replace('{provider_cards}', provider_cards)
        .replace('{summary_cards}', summary_cards)
        .replace('{daily_rows}', daily_rows)
        .replace('{recent_rows}', recent_rows)
        .replace('{error_rows}', error_rows)
        .replace('{generated_at}', now))


# ── HTTP Handler ────────────────────────────────────────────────────────────
class GatewayHandler(BaseHTTPRequestHandler):
    db = None

    def _parse_profile(self):
        """Extract profile name from URL path. Returns (profile_name, is_chat)."""
        path = self.path.split("?")[0]
        # /l2/chat/completions  or  /l2/v1/chat/completions
        m = re.match(r"^/(l2|l1|career|cron)(?:/v1)?/chat/completions$", path)
        if m:
            return m.group(1)
        # /chat/completions or /v1/chat/completions → default
        if path in ("/chat/completions", "/v1/chat/completions"):
            return DEFAULT_PROFILE
        return None

    def do_POST(self):
        profile_name = self._parse_profile()
        if profile_name is None:
            self.send_error(404, f"Unknown path: {self.path}")
            return

        t0 = time.time()
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            request_data = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        # Strip forbidden tools
        stripped = strip_tools(request_data, profile_name)
        if stripped:
            print(f"[{profile_name}] stripped {stripped} forbidden tools", flush=True)

        profile = PROFILES.get(profile_name, PROFILES[DEFAULT_PROFILE])
        chain = profile["chain"]
        client_model = request_data.get("model", "unknown")

        try:
            response_body, provider, api_model, fallback_used, failed_provider, failed_code = try_chain(
                chain, request_data, profile_name)
            # Check if response is SSE or JSON
            raw_rb = response_body.decode('utf-8', errors='replace')
            is_sse_response = raw_rb.strip().startswith('data:')
            
            if is_sse_response:
                # SSE response — forward raw to gateway without parsing
                duration_ms = int((time.time() - t0) * 1000)
                log_request(self.db, profile_name, api_model, provider, extract_sse_usage(raw_rb),
                           duration_ms, "success", 1 if fallback_used else 0)
                
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)
                
                if fallback_used:
                    print(f"[{profile_name}] FALLBACK: {provider}/{api_model} (primary failed)",
                          flush=True)
                return
            
            # JSON response — parse and log normally
            response_data = json.loads(response_body)
            usage = response_data.get("usage", {})
            duration_ms = int((time.time() - t0) * 1000)

            log_request(self.db, profile_name, api_model, provider, usage,
                       duration_ms, "success", 1 if fallback_used else 0)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

            if fallback_used:
                print(f"[{profile_name}] FALLBACK: {provider}/{api_model} (primary failed)",
                      flush=True)
                # Log the failed provider for error history
                if failed_provider:
                    err_status = f"error_{failed_code}" if failed_code else "error_upstream"
                    log_request(self.db, profile_name, api_model, failed_provider, {},
                               duration_ms, err_status)

        except urllib.error.HTTPError as e:
            error_body = e.read()
            duration_ms = int((time.time() - t0) * 1000)
            log_request(self.db, profile_name, client_model, "error", {},
                       duration_ms, f"error_{e.code}")
            # Persist the error
            error_text = error_body.decode('utf-8', errors='replace')
            ray_id = log_proxy_error(self.db, profile_name, getattr(e, 'provider', ''),
                                     e.code, type(e).__name__, str(e), error_text)
            print(f"[{profile_name}] UPSTREAM ERROR {e.code} (ray: {ray_id or 'N/A'})", flush=True)
            print(f"[{profile_name}] ERROR BODY: {error_text[:800]}", flush=True)
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(error_body)

        except Exception as e:
            duration_ms = int((time.time() - t0) * 1000)
            log_request(self.db, profile_name, client_model, "error", {},
                       duration_ms, f"error_{type(e).__name__}")

            # Extract upstream response info if available
            upstream_provider = ""
            upstream_body = ""
            try:
                upstream_provider = provider
                upstream_body = response_body.decode('utf-8', errors='replace')[:2000]
            except (UnboundLocalError, NameError, AttributeError):
                upstream_body = "<response body not available from try_chain>"

            ray_id = log_proxy_error(self.db, profile_name, upstream_provider,
                                     '502', type(e).__name__, str(e), upstream_body)

            # Print detailed error to stdout (visible in docker logs)
            error_details = f"[{profile_name}] ERROR: {type(e).__name__}: {e}"
            if upstream_provider:
                error_details += f" [upstream: {upstream_provider}"
                if upstream_body:
                    error_details += f", body: {upstream_body[:500]}"
                error_details += "]"
            error_details += f" (ray: {ray_id or 'N/A'})"
            print(error_details, flush=True)

            self.send_error(502, f"Gateway error: {e}")

    def do_GET(self):
        path = self.path.split("?")[0]

        # /health
        if path == "/health" or path.startswith("/health"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            
            # Build per-profile provider health summary
            providers_health = {}
            for key, h in PROVIDER_HEALTH.items():
                prov_name, base_url, prof = key
                if prof not in providers_health:
                    providers_health[prof] = {}
                tripped_iso = None
                if h["tripped_until"]:
                    tripped_iso = datetime.fromtimestamp(h["tripped_until"], tz=timezone.utc).isoformat()
                providers_health[prof][prov_name] = {
                    "base_url": base_url,
                    "status": h["status"],
                    "consecutive_failures": h["consecutive_failures"],
                    "last_failure": h["last_failure"],
                    "last_success": h["last_success"],
                    "tripped_until": tripped_iso,
                }
            
            health_data = {
                "status": "ok",
                "profiles": ["l2", "l1", "career", "cron"],
                "providers": providers_health
            }
            self.wfile.write(json.dumps(health_data, indent=2).encode() + b'\n')
            return

        # /v1/models or /models
        if path in ("/v1/models", "/models"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "object": "list",
                "data": [
                    {"id": "deepseek-v4-pro", "object": "model"},
                    {"id": "deepseek-v4-flash", "object": "model"},
                ]
            }).encode())
            return

        # /errors — all errors
        # /l2/errors — profile-scoped errors
        m = re.match(r"^/(l2|l1|career|cron)?/?errors?$", path)
        if m:
            profile_filter = m.group(1)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            try:
                if profile_filter:
                    rows = self.db.execute(
                        """SELECT timestamp, provider, status_code, error_type, error_message,
                                  substr(error_body, 1, 500) as error_body_preview, ray_id
                           FROM proxy_errors WHERE profile = ?
                           ORDER BY id DESC LIMIT 50""",
                        (profile_filter,)).fetchall()
                else:
                    rows = self.db.execute(
                        """SELECT timestamp, profile, provider, status_code, error_type, error_message,
                                  substr(error_body, 1, 500) as error_body_preview, ray_id
                           FROM proxy_errors ORDER BY id DESC LIMIT 50""").fetchall()
                cols = [d[0] for d in self.db.execute("PRAGMA table_info(proxy_errors)").fetchall()]
                errors = []
                for row in rows:
                    entry = dict(zip(cols, row))
                    # Only keep relevant keys
                    errors.append({k: entry.get(k, '') for k in entry if k != 'id'})
                self.wfile.write(json.dumps({"count": len(errors), "errors": errors}, indent=2).encode())
            except Exception as e:
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return

        # /errors/count — error count summary
        m = re.match(r"^/(l2|l1|career|cron)?/?errors?/count$", path)
        if m:
            profile_filter = m.group(1)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            try:
                if profile_filter:
                    row = self.db.execute(
                        "SELECT COUNT(*) as count, MAX(timestamp) as latest FROM proxy_errors WHERE profile = ?",
                        (profile_filter,)).fetchone()
                else:
                    row = self.db.execute(
                        "SELECT COUNT(*) as count, MAX(timestamp) as latest FROM proxy_errors").fetchone()
                self.wfile.write(json.dumps({
                    "total_errors": row[0],
                    "latest_error": row[1] or "none",
                }).encode())
            except Exception as e:
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return

        # /dashboard — cost tracking dashboard (also at /)
        # /l2/dashboard, /l1/dashboard, /career/dashboard, /cron/dashboard — profile-scoped
        m = re.match(r"^/(l2|l1|career|cron)?/?dashboard$", path)
        if m or path == "/":
            profile_filter = m.group(1) if m else None
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            try:
                html = build_dashboard_html(self.db, profile_filter)
                self.wfile.write(html.encode('utf-8'))
            except Exception as e:
                self.wfile.write(f"<h1>Dashboard Error</h1><pre>{e}</pre>".encode())
            return

        self.send_error(404)

    def log_message(self, format, *args):
        pass  # suppress default access log


def main():
    db = init_db()
    GatewayHandler.db = db
    server = HTTPServer(("0.0.0.0", LISTEN_PORT), GatewayHandler)
    print(f"LLM Gateway Router listening on :{LISTEN_PORT}", flush=True)
    print(f"Profiles: {list(PROFILES.keys())}  Default: {DEFAULT_PROFILE}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        db.close()


if __name__ == "__main__":
    main()
