#!/usr/bin/env python3
"""One-time migration: import a legacy ``config/gateway.yaml`` into the DB.

The gateway config is now DB-backed (``gateway_config:<section>`` JSON blobs in
the ``settings`` table, seeded from ``src/api/config.py`` ``SEED_CONFIG``).
This script reads an existing legacy YAML and writes its sections into the
target DB, so a production deployment's custom config (profiles, providers,
pricing, ...) is preserved when the YAML is obsoleted.

This script is SELF-CONTAINED: it uses only the Python standard library plus
``yaml`` (no app imports), so it runs with a plain ``python3`` — no venv, no
``src`` package on the path. Run it on the host that has the SQLite file, or
inside the LCP container:

    # local dev (YAML at ./config/gateway.yaml, DB at data/costs.db)
    python3 scripts/migrate_gateway_yaml_to_db.py --dry-run
    python3 scripts/migrate_gateway_yaml_to_db.py

    # production (inside the LCP container, from the repo mount)
    docker exec lcp python3 /app/scripts/migrate_gateway_yaml_to_db.py --dry-run
    docker exec lcp python3 /app/scripts/migrate_gateway_yaml_to_db.py

    # ...or copy it in and run from /tmp
    docker cp scripts/migrate_gateway_yaml_to_db.py lcp:/tmp/migrate.py
    docker exec lcp python3 /tmp/migrate.py --dry-run
    docker exec lcp python3 /tmp/migrate.py

Usage:
    python3 scripts/migrate_gateway_yaml_to_db.py [--yaml PATH] [--db PATH]
        [--overwrite] [--if-absent] [--dry-run]

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
import json
import os
import sqlite3
import sys
from typing import Any, Optional

import yaml


# ── Section inventory (mirrors src/api/config.py) ──────────────────────────
# Sections stored in the DB. ``plugins`` is optional; the rest are required.
REQUIRED_SECTIONS = ("server", "profiles", "providers", "pricing",
                     "circuit_breaker", "database")
ALL_SECTIONS = ("server", "profiles", "providers", "pricing",
                "circuit_breaker", "retry", "database",
                "dynamic_routing", "model_limits", "plugins")

# Seed default DB path (mirrors src/api/config.py SEED_CONFIG database.path).
SEED_DB_PATH = "/app/data/costs.db"


class ConfigError(Exception):
    """Raised for invalid YAML / missing sections."""


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
    return SEED_DB_PATH


def _validate(section: str, data: Any) -> None:
    """Validate a loaded section; raise ConfigError on structural problems."""
    if section == "server":
        if not isinstance(data, dict) or "port" not in data:
            raise ConfigError("Missing 'server.port'")
        if "default_profile" not in data:
            raise ConfigError("Missing 'server.default_profile'")
    elif section == "profiles":
        if not isinstance(data, dict):
            raise ConfigError("'profiles' must be a dict")
        for name, prof in data.items():
            if not isinstance(prof, dict) or "chain" not in prof:
                raise ConfigError(f"Profile '{name}' missing 'chain'")
            if not prof["chain"]:
                raise ConfigError(f"Profile '{name}' has empty 'chain'")
    elif section == "pricing":
        if not isinstance(data, list):
            raise ConfigError("'pricing' must be a list")
    elif section in ("providers", "circuit_breaker", "database"):
        if not isinstance(data, dict):
            raise ConfigError(f"'{section}' must be a dict")
    # dynamic_routing / retry / model_limits / plugins are optional; handled by
    # the accessors.


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


def _read_section(conn: sqlite3.Connection, section: str) -> Optional[Any]:
    """Return a stored gateway_config section (parsed), or None if absent."""
    row = conn.execute(
        "SELECT value FROM settings WHERE key = ?",
        (f"gateway_config:{section}",),
    ).fetchone()
    if row is None:
        return None
    try:
        return json.loads(row[0])
    except (TypeError, ValueError):
        return None


def _write_section(conn: sqlite3.Connection, section: str, value: Any) -> None:
    """Upsert a gateway_config section (JSON blob) into the settings table."""
    conn.execute(
        "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
        "updated_at = excluded.updated_at",
        (f"gateway_config:{section}", json.dumps(value), _now()),
    )


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


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
    raw = load_yaml(yaml_path)
    conn = sqlite3.connect(db_path)
    try:
        # Ensure the settings table exists (same schema as src/api/models.py).
        conn.execute(
            "CREATE TABLE IF NOT EXISTS settings ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "key TEXT UNIQUE NOT NULL, "
            "value TEXT NOT NULL, "
            "updated_at TEXT NOT NULL)"
        )
        summary: dict[str, str] = {}
        for section in ALL_SECTIONS:
            if section not in raw or raw[section] is None:
                summary[section] = "absent"
                continue
            value = raw[section]
            existing = _read_section(conn, section)
            if existing is not None and not overwrite:
                summary[section] = "skipped_exists"
                continue
            if dry_run:
                summary[section] = "would_write"
                continue
            _write_section(conn, section, value)
            summary[section] = "written"
        conn.commit()
    finally:
        conn.close()
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
