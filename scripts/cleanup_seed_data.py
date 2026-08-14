"""One-time cleanup: remove old bulk-seeded capability scores.

The capability matrix used to be bulk-seeded from public datasets
(``source="livebench"`` hand-typed/HF snapshots and ``source="arena"`` Elo).
Capability scores are now produced only by LiveBench benchmarks you run
(``source="lcp_benchmark"``). This script deletes the obsolete seeded rows and
keeps your benchmark grades (plus any ``gateway_yaml`` admin overrides).

Usage:
    .venv/bin/python scripts/cleanup_seed_data.py --dry-run        # preview
    .venv/bin/python scripts/cleanup_seed_data.py                  # delete
    .venv/bin/python scripts/cleanup_seed_data.py --db-path /path/to/costs.db
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter


# Sources that came from bulk seeding and are now obsolete.
OBSOLETE_SOURCES = ("livebench", "arena")

# Sources we keep: user-run benchmarks and admin config overrides.
KEEP_SOURCES = ("lcp_benchmark", "gateway_yaml")


def resolve_db_path(explicit: str | None) -> str:
    """Return the SQLite path from --db-path, COST_DB, or sensible defaults."""
    if explicit:
        return explicit
    env = os.environ.get("COST_DB", "").strip()
    if env:
        return env

    # Local dev DB takes precedence over the Docker-path in gateway.yaml.
    if os.path.isfile("data/costs.db"):
        return "data/costs.db"

    # Fall back to gateway.yaml's database.path (production container).
    try:
        import yaml
        with open("config/gateway.yaml") as f:
            raw = yaml.safe_load(f) or {}
        path = (raw.get("database") or {}).get("path")
        if path:
            return path
    except Exception:
        pass
    return "data/costs.db"


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove obsolete bulk-seeded capability scores")
    parser.add_argument("--db-path", default=None, help="SQLite DB path (default: COST_DB or config)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be deleted without deleting")
    args = parser.parse_args()

    sys.path.insert(0, os.getcwd())

    from src.api.models import get_engine, get_session, ModelCapability

    db_path = resolve_db_path(args.db_path)
    engine = get_engine(db_path)

    with get_session(engine) as session:
        rows = session.query(ModelCapability).all()
        by_source = Counter(r.source for r in rows)
        print(f"DB: {db_path}")
        print(f"Current rows by source: {dict(by_source) or '(empty)'}")

        obsolete = [r for r in rows if r.source in OBSOLETE_SOURCES]
        kept = [r for r in rows if r.source not in OBSOLETE_SOURCES]

        print(f"\nWill {'delete' if not args.dry_run else 'delete (dry-run)'}: "
              f"{len(obsolete)} rows (source in {OBSOLETE_SOURCES})")
        print(f"Will keep:               {len(kept)} rows (sources {KEEP_SOURCES} + any other)")

        if obsolete:
            obsolete_by_source = Counter(r.source for r in obsolete)
            print("\nObsolete rows by source:")
            for src, n in sorted(obsolete_by_source.items()):
                print(f"  {src}: {n}")

        if args.dry_run:
            print("\nDry run — no changes made. Re-run without --dry-run to delete.")
            return

        for r in obsolete:
            session.delete(r)
        session.commit()

        remaining = session.query(ModelCapability).all()
        print(f"\nDeleted {len(obsolete)} rows.")
        print(f"Remaining rows by source: {dict(Counter(r.source for r in remaining)) or '(empty)'}")


if __name__ == "__main__":
    main()
