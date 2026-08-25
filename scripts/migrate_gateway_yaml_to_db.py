#!/usr/bin/env python3
"""One-time migration: import a legacy ``config/gateway.yaml`` into the DB.

The gateway config is now DB-backed (``gateway_config:<section>`` JSON blobs in
the ``settings`` table, seeded from ``src/api/config.py`` ``SEED_CONFIG``).
This script reads an existing legacy YAML and writes its sections into the
target DB, so a production deployment's custom config (profiles, providers,
pricing, …) is preserved when the YAML is obsoleted.

Usage:
    python scripts/migrate_gateway_yaml_to_db.py [--yaml PATH] [--db PATH]
        [--overwrite] [--if-absent] [--dry-run]

    # local dev (YAML at ./config/gateway.yaml, DB at data/costs.db)
    python scripts/migrate_gateway_yaml_to_db.py --dry-run

    # production (inside the LCP container)
    docker cp scripts/migrate_gateway_yaml_to_db.py lcp:/tmp/migrate.py
    docker exec lcp python3 /tmp/migrate.py --overwrite --dry-run
    docker exec lcp python3 /tmp/migrate.py --overwrite

YAML path:  ``--yaml``, else ``./config/gateway.yaml``, else
            ``/app/config/gateway.yaml``.
DB path:    ``--db``, else ``$COST_DB``, else ``./data/costs.db`` if it
            exists, else the seed default (``/app/data/costs.db``).

By default the YAML values WIN (overwrite) — that's the migration intent: the
prod YAML is the source of truth being moved into the DB. Pass ``--if-absent``
to only write sections the DB doesn't already have (non-destructive merge).
``--dry-run`` prints what would be written without touching the DB.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Optional

# Allow running as a plain script from anywhere in the repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402

from src.api.config import ALL_SECTIONS, REQUIRED_SECTIONS, SEED_CONFIG, _validate  # noqa: E402
from src.api.exceptions import ConfigError  # noqa: E402


def resolve_yaml_path(explicit: Optional[str]) -> str:
    """Return the YAML path from --yaml, a local file, or the container path."""
    if explicit:
        return explicit
    for candidate in ("./config/gateway.yaml", "/app/config/gateway.yaml"):
        if os.path.isfile(candidate):
            return candidate
    return "./config/gateway.yaml"


def resolve_db_path(explicit: Optional[str]) -> str:
    """Return the SQLite path from --db, $COST_DB, a local dev DB, or seed."""
    if explicit:
        return explicit
    env = os.environ.get("COST_DB", "").strip()
    if env:
        return env
    if os.path.isfile("data/costs.db"):
        return "data/costs.db"
    return SEED_CONFIG["database"]["path"]


def load_yaml(path: str) -> dict:
    """Load + basic-validate a legacy gateway.yaml. Raises ConfigError on bad input."""
    if not os.path.isfile(path):
        raise ConfigError(f"YAML config file not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"YAML config must be a mapping: {path}")
    missing = [s for s in REQUIRED_SECTIONS if s not in raw]
    if missing:
        raise ConfigError(
            f"YAML config is missing required section(s): {', '.join(missing)}"
        )
    for section in REQUIRED_SECTIONS:
        try:
            _validate(section, raw.get(section))
        except ConfigError as exc:
            raise ConfigError(f"YAML section '{section}' is invalid: {exc}")
    return raw


def migrate_gateway_yaml(
    yaml_path: str,
    db_path: str,
    overwrite: bool = True,
    dry_run: bool = False,
) -> dict[str, str]:
    """Import a legacy gateway.yaml into the settings DB.

    Returns a ``{section: status}`` summary where status is one of:
    ``written``, ``would_write`` (dry-run), ``skipped_exists`` (--if-absent),
    or ``absent`` (not present in the YAML).
    """
    from src.api.cost_cache import SettingsStore
    from src.api.models import Base, get_engine

    raw = load_yaml(yaml_path)
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    store = SettingsStore(engine)

    summary: dict[str, str] = {}
    for section in ALL_SECTIONS:
        if section not in raw or raw[section] is None:
            summary[section] = "absent"
            continue
        value = raw[section]
        existing = store.get_config_section(section, None)
        if existing is not None and not overwrite:
            summary[section] = "skipped_exists"
            continue
        if dry_run:
            summary[section] = "would_write"
            continue
        store.set_config_section(section, value)
        summary[section] = "written"
    return summary


def _print_summary(yaml_path: str, db_path: str, summary: dict[str, str],
                   overwrite: bool, dry_run: bool) -> None:
    action = "Would write" if dry_run else "Wrote"
    mode = "overwrite" if overwrite else "if-absent"
    print(f"YAML : {yaml_path}")
    print(f"DB   : {db_path}")
    print(f"Mode : {mode} ({'dry-run' if dry_run else 'live'})")
    print("-" * 56)
    for section in ALL_SECTIONS:
        status = summary.get(section, "absent")
        if status == "written":
            print(f"  [{'would write' if dry_run else 'written':<12}] {section}")
        elif status == "would_write":
            print(f"  [would write   ] {section}")
        elif status == "skipped_exists":
            print(f"  [skipped (exists)] {section}")
        elif status == "absent":
            print(f"  [absent        ] {section}")
    written = [s for s, st in summary.items() if st in ("written", "would_write")]
    print("-" * 56)
    print(f"{action} {len(written)}/{len(ALL_SECTIONS)} sections "
          f"({len([s for s, st in summary.items() if st == 'skipped_exists'])} skipped).")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", dest="yaml_path", default=None,
                        help="Path to the legacy gateway.yaml (default: ./config/gateway.yaml)")
    parser.add_argument("--db", dest="db_path", default=None,
                        help="Path to the SQLite DB (default: $COST_DB, ./data/costs.db, or seed)")
    parser.add_argument("--overwrite", action="store_true",
                        help="YAML values win over existing DB sections (default)")
    parser.add_argument("--if-absent", dest="if_absent", action="store_true",
                        help="Only write sections the DB doesn't already have")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be written without touching the DB")
    args = parser.parse_args(argv)

    yaml_path = resolve_yaml_path(args.yaml_path)
    db_path = resolve_db_path(args.db_path)
    # Explicitly default to overwrite (migration intent); --if-absent opts out.
    overwrite = not args.if_absent

    try:
        summary = migrate_gateway_yaml(
            yaml_path, db_path, overwrite=overwrite, dry_run=args.dry_run,
        )
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    _print_summary(yaml_path, db_path, summary, overwrite, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
