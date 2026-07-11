#!/usr/bin/env python3
"""Seed the LLM Control Plane database with realistic demo data."""

import os
import sys
import random
import hashlib
from datetime import datetime, timezone, timedelta

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import (
    Base,
    Request,
    Team,
    User,
    AuditLog,
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
    for table in [AuditLog, User, Team, Request]:
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

    session.close()
    print("✅ Seeding complete!")


if __name__ == "__main__":
    seed()
