"""One-time cleanup: remove old bulk-seeded capability scores.

The capability matrix used to be bulk-seeded from public datasets
(``source="livebench"`` hand-typed/HF snapshots and ``source="arena"`` Elo).
Capability scores are now produced only by LiveBench benchmarks you run
(``source="lcp_benchmark"``). This script deletes the obsolete seeded rows and
keeps your benchmark grades (plus any ``gateway_yaml`` admin overrides).

This script uses ONLY the Python standard library (``sqlite3``) so it runs with
a plain ``python3`` — no venv, no ``sqlalchemy``, no app imports. Run it on the
host that has the SQLite file, or inside the LCP container:

    # local dev (DB at data/costs.db)
    python3 scripts/cleanup_seed_data.py --dry-run

    # production (DB inside the Docker volume, container name 'lcp')
    docker cp scripts/cleanup_seed_data.py lcp:/tmp/cleanup_seed_data.py
    docker exec lcp python3 /tmp/cleanup_seed_data.py --dry-run
    docker exec lcp python3 /tmp/cleanup_seed_data.py   # actually delete

Usage:
    python3 scripts/cleanup_seed_data.py --dry-run            # preview
    python3 scripts/cleanup_seed_data.py                      # delete
    python3 scripts/cleanup_seed_data.py --db-path /path/to/costs.db
"""

from __future__ import annotations

import argparse
import os
import sqlite3
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

    # Local dev DB takes precedence over the seed default (config is DB-backed
    # now; gateway.yaml no longer exists).
    if os.path.isfile("data/costs.db"):
        return "data/costs.db"

    # Seed default (same as src/api/config.py SEED_CONFIG database.path).
    return "/app/data/costs.db"


def counts_by_source(conn: sqlite3.Connection) -> Counter:
    cur = conn.execute("SELECT source, COUNT(*) FROM model_capabilities GROUP BY source")
    return Counter({src: n for src, n in cur.fetchall()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove obsolete bulk-seeded capability scores")
    parser.add_argument("--db-path", default=None, help="SQLite DB path (default: COST_DB or config)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be deleted without deleting")
    args = parser.parse_args()

    db_path = resolve_db_path(args.db_path)
    if not os.path.isfile(db_path):
        print(f"ERROR: database file not found: {db_path}")
        print("       pass --db-path, set COST_DB, or run inside the LCP container.")
        sys.exit(1)

    conn = sqlite3.connect(db_path)

    before = counts_by_source(conn)
    print(f"DB: {db_path}")
    print(f"Current rows by source: {dict(before) or '(empty)'}")

    placeholders = ",".join("?" for _ in OBSOLETE_SOURCES)
    cur = conn.execute(
        f"SELECT source, COUNT(*) FROM model_capabilities WHERE source IN ({placeholders}) GROUP BY source",
        OBSOLETE_SOURCES,
    )
    obsolete_counts = Counter({src: n for src, n in cur.fetchall()})
    total_obsolete = sum(obsolete_counts.values())

    print(f"\nWill {'delete' if not args.dry_run else 'delete (dry-run)'}: "
          f"{total_obsolete} rows (source in {OBSOLETE_SOURCES})")
    if obsolete_counts:
        print("Obsolete rows by source:")
        for src, n in sorted(obsolete_counts.items()):
            print(f"  {src}: {n}")

    keep_total = sum(n for s, n in before.items() if s not in OBSOLETE_SOURCES)
    print(f"Will keep: {keep_total} rows (sources {KEEP_SOURCES} + any other)")

    if args.dry_run:
        print("\nDry run — no changes made. Re-run without --dry-run to delete.")
        conn.close()
        return

    if total_obsolete == 0:
        print("\nNothing to delete.")
        conn.close()
        return

    conn.execute(
        f"DELETE FROM model_capabilities WHERE source IN ({placeholders})",
        OBSOLETE_SOURCES,
    )
    conn.commit()

    after = counts_by_source(conn)
    print(f"\nDeleted {total_obsolete} rows.")
    print(f"Remaining rows by source: {dict(after) or '(empty)'}")
    conn.close()


if __name__ == "__main__":
    main()
