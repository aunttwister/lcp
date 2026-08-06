#!/usr/bin/env python3
"""Seed the LCP database with realistic demo data."""

import os
import sys
import random
import hashlib
from datetime import datetime, timezone, timedelta

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.api.models import (
    Base,
    Request,
    Team,
    User,
    AuditLog,
    ApiKey,
    Budget,
    Alert,
    get_engine,
    get_session,
)

DB_PATH = os.environ.get("COST_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "costs.db"))


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def seed():
    print(f"Seeding database: {DB_PATH}")

    # Ensure parent dir exists
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)

    engine = get_engine(DB_PATH)
    Base.metadata.create_all(engine)

    session = get_session(engine)

    # ── Clean existing data ──
    for table in [Alert, Budget, ApiKey, AuditLog, User, Team, Request]:
        session.query(table).delete()
    session.commit()

    # ── Teams ──
    teams = [
        Team(name="Engineering", monthly_budget=500.0,
             created_at=(datetime.now(timezone.utc) - timedelta(days=60)).isoformat()),
        Team(name="Marketing", monthly_budget=200.0,
             created_at=(datetime.now(timezone.utc) - timedelta(days=45)).isoformat()),
        Team(name="Research", monthly_budget=1000.0,
             created_at=(datetime.now(timezone.utc) - timedelta(days=30)).isoformat()),
    ]
    session.add_all(teams)
    session.commit()
    print(f"  Added {len(teams)} teams.")

    # ── Users ──
    users = [
        User(team_id=teams[0].id, username="alice", api_key_hash=hash_key("sk-alice-key"),
             credit_limit=250.0, is_admin=1,
             created_at=(datetime.now(timezone.utc) - timedelta(days=55)).isoformat()),
        User(team_id=teams[0].id, username="bob", api_key_hash=hash_key("sk-bob-key"),
             credit_limit=150.0, is_admin=0,
             created_at=(datetime.now(timezone.utc) - timedelta(days=40)).isoformat()),
        User(team_id=teams[1].id, username="carol", api_key_hash=hash_key("sk-carol-key"),
             credit_limit=100.0, is_admin=0,
             created_at=(datetime.now(timezone.utc) - timedelta(days=35)).isoformat()),
        User(team_id=teams[2].id, username="dave", api_key_hash=hash_key("sk-dave-key"),
             credit_limit=500.0, is_admin=0,
             created_at=(datetime.now(timezone.utc) - timedelta(days=20)).isoformat()),
    ]
    session.add_all(users)
    session.commit()
    print(f"  Added {len(users)} users.")

    # ── Requests (14 days of realistic data) ──
    profiles = ["l2", "l1", "career", "cron"]
    providers = ["opencode", "deepseek"]
    models = ["deepseek-v4-pro", "deepseek-v4-flash"]
    error_types = [None, None, None, None, None, "rate_limit", "timeout", "auth_error"]

    # Base daily request counts (vary by profile)
    base_rates = {"l2": 80, "l1": 40, "career": 15, "cron": 8}

    now = datetime.now(timezone.utc)
    request_count = 0

    for day_offset in range(13, -1, -1):  # last 14 days
        day = now - timedelta(days=day_offset)
        # Weekend dip
        weekend_factor = 0.5 if day.weekday() >= 5 else 1.0
        # Random daily variation
        noise = random.uniform(0.7, 1.3)

        for profile in profiles:
            num_reqs = max(1, int(base_rates[profile] * weekend_factor * noise))
            for _ in range(num_reqs):
                # Pick provider and model based on profile
                if profile == "cron":
                    provider = "deepseek"
                    model = "deepseek-v4-flash"
                elif profile == "career":
                    provider = "deepseek"
                    model = random.choices(models, weights=[0.3, 0.7])[0]
                else:
                    provider = random.choices(providers, weights=[0.6, 0.4])[0]
                    model = random.choices(models, weights=[0.4, 0.6])[0]

                # Random tokens
                prompt_tokens = random.randint(200, 8000)
                completion_tokens = random.randint(50, 2000)
                cache_hit_tokens = random.randint(0, prompt_tokens)
                cache_miss_tokens = prompt_tokens - cache_hit_tokens

                # Cost calculation (approximate pricing)
                pricing = {
                    ("deepseek", "deepseek-v4-pro"): {"cache_hit": 0.003625, "cache_miss": 0.435, "output": 0.87},
                    ("deepseek", "deepseek-v4-flash"): {"cache_hit": 0.0028, "cache_miss": 0.14, "output": 0.28},
                    ("opencode", "deepseek-v4-pro"): {"cache_hit": 0.003625, "cache_miss": 0.435, "output": 0.87},
                    ("opencode", "deepseek-v4-flash"): {"cache_hit": 0.0028, "cache_miss": 0.14, "output": 0.28},
                }
                p = pricing.get((provider, model), pricing[("deepseek", "deepseek-v4-flash")])
                cost = (cache_hit_tokens / 1_000_000) * p["cache_hit"] + \
                       (cache_miss_tokens / 1_000_000) * p["cache_miss"] + \
                       (completion_tokens / 1_000_000) * p["output"]

                latency_ms = random.randint(300, 8000)
                success = 1 if random.random() > 0.05 else 0
                error_type = random.choice(error_types) if not success else None

                # Random timestamp within the day
                ts = day + timedelta(
                    hours=random.randint(6, 23),
                    minutes=random.randint(0, 59),
                    seconds=random.randint(0, 59),
                )

                # For l2 profile, occasionally block tools
                tools_blocked = None
                if profile == "l2" and success and random.random() < 0.08:
                    tools_blocked = random.choice(["write_file", "patch", "cronjob"])

                req = Request(
                    timestamp=ts.isoformat(),
                    profile=profile,
                    model=model,
                    provider=provider,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cache_hit_tokens=cache_hit_tokens,
                    cache_miss_tokens=cache_miss_tokens,
                    cost=round(cost, 6),
                    latency_ms=latency_ms,
                    success=success,
                    error_type=error_type,
                    tools_blocked=tools_blocked,
                )
                session.add(req)
                request_count += 1

    session.commit()
    print(f"  Added {request_count} requests.")

    # ── Audit Logs ──
    actions = ["request", "auth_fail", "credit_exhausted"]
    audit_count = 0
    for day_offset in range(13, -1, -1):
        day = now - timedelta(days=day_offset)
        for _ in range(random.randint(2, 8)):
            action = random.choices(actions, weights=[0.8, 0.15, 0.05])[0]
            ts = day + timedelta(
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
            )
            log = AuditLog(
                timestamp=ts.isoformat(),
                user_id=random.choice(users).id,
                team_id=random.choice(teams).id,
                action=action,
                detail=f'{{"action": "{action}", "profile": "{random.choice(profiles)}"}}',
                ip_address=f"{random.randint(10, 200)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}",
            )
            session.add(log)
            audit_count += 1
    session.commit()
    print(f"  Added {audit_count} audit log entries.")

    # ── API Keys ──
    api_keys = [
        ApiKey(key_hash=hash_key("sk-lcp-demo-l2"),
               key_prefix="sk-lcp-d", name="Demo L2 Key",
               allowed_profiles="l2,l1", spend_limit=100.0, total_spend=23.45,
               status="active", last_used_at=(now - timedelta(hours=2)).isoformat()),
        ApiKey(key_hash=hash_key("sk-lcp-demo-admin"),
               key_prefix="sk-lcp-a", name="Admin Full Access",
               allowed_profiles=None, spend_limit=0.0, total_spend=156.78,
               status="active", last_used_at=(now - timedelta(hours=1)).isoformat()),
        ApiKey(key_hash=hash_key("sk-lcp-demo-cron"),
               key_prefix="sk-lcp-c", name="Cron Only",
               allowed_profiles="cron", spend_limit=50.0, total_spend=48.92,
               status="active", last_used_at=(now - timedelta(hours=6)).isoformat()),
        ApiKey(key_hash=hash_key("sk-lcp-revoked"),
               key_prefix="sk-lcp-r", name="Old Dev Key (revoked)",
               allowed_profiles="l2", spend_limit=25.0, total_spend=25.0,
               status="revoked", revoked_at=(now - timedelta(days=3)).isoformat()),
    ]
    session.add_all(api_keys)
    session.commit()
    print(f"  Added {len(api_keys)} API keys.")

    # ── Budgets ──
    budgets = [
        Budget(name="L2 Monthly Cap", key_id=None, profile="l2",
               amount=200.0, current_spend=156.78, period="monthly",
               threshold_pct="50,80,90", action="block", status="active"),
        Budget(name="Cron Hard Limit", key_id=None, profile="cron",
               amount=50.0, current_spend=48.92, period="monthly",
               threshold_pct="80,95", action="block", status="active",
               last_alert_at=(now - timedelta(hours=4)).isoformat()),
        Budget(name="Demo L2 Key Limit", key_id=api_keys[0].id, profile=None,
               amount=100.0, current_spend=23.45, period="total",
               threshold_pct="80", action="log", status="active"),
        Budget(name="Global Monthly", key_id=None, profile=None,
               amount=500.0, current_spend=234.56, period="monthly",
               threshold_pct="50,80,90", action="log", status="active"),
        Budget(name="Career Free Tier", key_id=None, profile="career",
               amount=25.0, current_spend=25.0, period="monthly",
               threshold_pct="80", action="block", status="exceeded",
               last_alert_at=(now - timedelta(hours=1)).isoformat()),
    ]
    session.add_all(budgets)
    session.commit()
    print(f"  Added {len(budgets)} budgets.")

    # ── Alert History ──
    import json
    alerts_data = [
        {"dedup_key": "budget:cron:t95", "rule": "budget_breach", "severity": "critical",
         "title": "Budget 'Cron Hard Limit' at 97.8%", "status": "firing",
         "message": "Cron profile budget has reached 97.8% of its $50.00 monthly limit.",
         "metadata_json": json.dumps({"budget_id": budgets[1].id, "threshold": 95, "spend_pct": 97.8})},
        {"dedup_key": "provider:opencode:l2:status", "rule": "provider_degraded", "severity": "warning",
         "title": "Provider opencode/l2: healthy → degraded", "status": "resolved",
         "message": "OpenCode provider for profile l2 was degraded due to 3 consecutive timeouts.",
         "resolved_at": (now - timedelta(hours=12)).isoformat(),
         "metadata_json": json.dumps({"provider": "opencode", "profile": "l2"})},
        {"dedup_key": "budget:0:t80", "rule": "budget_breach", "severity": "warning",
         "title": "Budget 'L2 Monthly Cap' at 78.4%", "status": "firing",
         "message": "L2 profile monthly budget has reached 78.4% of its $200.00 limit.",
         "metadata_json": json.dumps({"budget_id": budgets[0].id, "threshold": 80, "spend_pct": 78.4})},
        {"dedup_key": "error_spike", "rule": "error_spike", "severity": "warning",
         "title": "Error spike: 12 errors in 5min", "status": "resolved",
         "message": "Detected 12 errors in the last 5 minutes (threshold: 10).",
         "resolved_at": (now - timedelta(days=1)).isoformat(),
         "metadata_json": json.dumps({"error_count": 12, "window_minutes": 5})},
        {"dedup_key": "circuit:deepseek:l1:trip", "rule": "circuit_breaker_trip", "severity": "warning",
         "title": "Circuit breaker: deepseek/l1 degraded", "status": "resolved",
         "message": "Circuit breaker tripped for deepseek provider on l1 profile after 3 failures.",
         "resolved_at": (now - timedelta(days=2)).isoformat(),
         "metadata_json": json.dumps({"provider": "deepseek", "profile": "l1"})},
        {"dedup_key": "circuit:deepseek:l1:recovery", "rule": "circuit_breaker_recovery", "severity": "info",
         "title": "Circuit breaker: deepseek/l1 recovered",
         "message": "Circuit breaker recovered for deepseek provider on l1 profile.",
         "status": "resolved", "resolved_at": (now - timedelta(days=2, hours=1)).isoformat(),
         "metadata_json": json.dumps({"provider": "deepseek", "profile": "l1"})},
        {"dedup_key": "budget:career:t80", "rule": "budget_breach", "severity": "critical",
         "title": "Budget 'Career Free Tier' at 100% — EXCEEDED",
         "message": "Career profile monthly budget has been fully consumed. Further requests blocked.",
         "status": "firing",
         "metadata_json": json.dumps({"budget_id": budgets[4].id, "threshold": 100, "spend_pct": 100.0})},
    ]

    for a in alerts_data:
        resolved = a.pop("resolved_at", None)
        alert = Alert(
            timestamp=(now - timedelta(hours=random.randint(1, 72))).isoformat(),
            acknowledged=1 if random.random() < 0.5 else 0,
            **a,
        )
        if resolved:
            alert.resolved_at = resolved
        if alert.acknowledged:
            alert.acknowledged_at = alert.timestamp
        session.add(alert)
    session.commit()
    print(f"  Added {len(alerts_data)} alerts.")

    session.close()
    print("✅ Seeding complete!")


if __name__ == "__main__":
    seed()
