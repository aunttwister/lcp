# QA case generator 4: Cron CRUD/ops + Work sources.
import json

CASES = []
G = "cron-sources"

# ── Cron API surface ──
CASES += [
    {"id": "cr-001", "group": G, "title": "cron view counts sane",
     "cmd": "curl -s '{{BASE}}/api/work/cron' | python3 -c \"import sys,json; d=json.load(sys.stdin); assert d['available'] is True; assert d['counts']['total']>0\""},
    {"id": "cr-002", "group": G, "title": "cron ops list empty-pending",
     "cmd": "curl -s '{{BASE}}/api/work/cron/ops' | python3 -c \"import sys,json; d=json.load(sys.stdin); assert 'pending' in d and 'done' in d\""},
    {"id": "cr-003", "group": G, "title": "ops spool dir writable (POST surface)",
     "cmd": "[ -d /your/data/app/lcp/cron-ops/ops ] && [ -d /your/data/app/lcp/cron-ops/done ]"},
]

# ── Ops validation (reject malformed without side effects) ──
BAD_OPS = [
    ("cr-010", "unknown action", {"action": "explode", "profile": "backups"}),
    ("cr-011", "unknown profile", {"action": "remove", "profile": "nope-profile", "job_id": "a1b2c3d4e5f6"}),
    ("cr-012", "create without prompt/script", {"action": "create", "profile": "backups", "name": "x", "schedule": "30m"}),
    ("cr-013", "script with slash", {"action": "create", "profile": "backups", "name": "x", "schedule": "30m", "script": "../evil.sh"}),
    ("cr-014", "edit empty fields", {"action": "edit", "profile": "backups", "job_id": "a1b2c3d4e5f6", "fields": {}}),
    ("cr-015", "bad deliver", {"action": "create", "profile": "backups", "name": "x", "schedule": "30m", "prompt": "p", "deliver": "nowhere"}),
    ("cr-016", "workdir outside /your/data", {"action": "create", "profile": "backups", "name": "x", "schedule": "30m", "prompt": "p", "workdir": "/etc"}),
]
import json as _json
for cid, title, payload in BAD_OPS:
    CASES.append({
        "id": cid, "group": G, "title": "reject: %s" % title,
        "cmd": ("[ \"$(curl -s -o /dev/null -w '{{HTTP}}' -X POST '{{BASE}}/api/work/cron/ops' "
                "-H 'Content-Type: application/json' -d '%s')\" = \"400\" ]"
                % _json.dumps(payload).replace('"', '\\"')),
    })

# ── Ops lifecycle on the backups profile (harmless, self-cleaning) ──
CASES += [
    {"id": "cr-020", "group": G, "title": "create op accepted (backups)",
     "cmd": "op=$(curl -s -X POST '{{BASE}}/api/work/cron/ops' -H 'Content-Type: application/json' -d '{\"action\":\"create\",\"profile\":\"backups\",\"name\":\"qa-suite-probe\",\"schedule\":\"2033-01-01T00:00:00\",\"prompt\":\"qa probe - safe to delete\",\"deliver\":\"local\"}'); echo \"$op\" | python3 -c 'import sys,json; d=json.load(sys.stdin); assert d.get(\"status\")==\"pending\", d'"},
]

with open("/your/data/docker-apps/lcp/qa/cases.cron.json", "w") as fh:
    json.dump(CASES, fh, indent=1)
print("cases.cron:", len(CASES))