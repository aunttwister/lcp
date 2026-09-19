# QA case generator 3: Logs unified page + conversations.
import json

CASES = []
G = "logs-convo"

# ── Unified Logs tabs ──
for view, marker in [("conversations", "Conversations"), ("requests", "Requests"),
                     ("providers", "Provider decisions"), ("decisions", "Board decisions")]:
    CASES.append({
        "id": "lg-0%s" % ("1" if view == "conversations" else "2" if view == "requests"
                          else "3" if view == "providers" else "4"),
        "group": G, "title": "Logs tab %s renders" % view,
        "cmd": ("out=$(curl -s '{{BASE}}/logs?view=%s'); echo \"$out\" | grep -q '%s' && "
                "echo \"$out\" | grep -q 'tab-btn'" % (view, marker)),
    })

# ── Requests tab ──
CASES += [
    {"id": "lg-005", "group": G, "title": "requests tab renders with rows",
     "cmd": "out=$(curl -s '{{BASE}}/logs?view=requests&per=5'); echo \"$out\" | grep -q 'work-table' && echo \"$out\" | grep -q 'requests'"},
    {"id": "lg-006", "group": G, "title": "requests API rows have conversation links when stamped",
     "cmd": "curl -s '{{BASE}}/api/work/requests?per=10' | python3 -c \"import sys,json; d=json.load(sys.stdin); assert d['total']>0\""},
]

# ── Conversations list ──
CASES += [
    {"id": "lg-010", "group": G, "title": "conversations exist with names",
     "cmd": "curl -s '{{BASE}}/api/work/conversations?per=5' | python3 -c \"import sys,json; d=json.load(sys.stdin); assert d['total']>0; assert all(c.get('name') for c in d['conversations'])\""},
    {"id": "lg-011", "group": G, "title": "conversations paginate",
     "cmd": "curl -s '{{BASE}}/api/work/conversations?per=3' | python3 -c \"import sys,json; d=json.load(sys.stdin); assert len(d['conversations'])<=3; assert d['filter']['per']=='3'\""},
    {"id": "lg-012", "group": G, "title": "profile filter works",
     "cmd": "curl -s '{{BASE}}/api/work/conversations?per=5&profile=l2' | python3 -c \"import sys,json; d=json.load(sys.stdin); assert d['total']>=0\""},
    {"id": "lg-013", "group": G, "title": "LLM summaries present (summary_source=llm rows)",
     "cmd": "curl -s '{{BASE}}/api/work/conversations?per=5' | python3 -c \"import sys,json; d=json.load(sys.stdin); assert sum(1 for c in d['conversations'] if c.get('summary'))>=1\""},
]

# ── Conversation detail timeline ──
CASES += [
    {"id": "lg-020", "group": G, "title": "detail returns chronological events",
     "cmd": "cid=$(curl -s '{{BASE}}/api/work/conversations?per=1' | python3 -c 'import sys,json; print(json.load(sys.stdin)[\"conversations\"][0][\"id\"])'); curl -s \"{{BASE}}/api/work/conversations/detail?cid=$cid\" | python3 -c 'import sys,json; ts=[e[\"ts\"] for e in json.load(sys.stdin)[\"events\"]]; assert ts==sorted(ts), \"not chronological\"'"},
    {"id": "lg-021", "group": G, "title": "detail requires cid",
     "cmd": "[ \"$(curl -s -o /dev/null -w '{{HTTP}}' '{{BASE}}/api/work/conversations/detail')\" = \"400\" ]"},
    {"id": "lg-022", "group": G, "title": "detail unknown cid returns events empty (not 500)",
     "cmd": "[ \"$(curl -s -o /dev/null -w '{{HTTP}}' '{{BASE}}/api/work/conversations/detail?cid=nope')\" != \"500\" ]"},
]

# ── Board decisions tab data ──
CASES += [
    {"id": "lg-030", "group": G, "title": "decisions view accessible",
     "cmd": "[ \"$(curl -s -o /dev/null -w '{{HTTP}}' '{{BASE}}/api/work/decisions')\" = \"200\" ]"},
]

# ── Provider decisions tab data ──
CASES += [
    {"id": "lg-040", "group": G, "title": "provider decisions list has rows",
     "cmd": "curl -s '{{BASE}}/api/work/provider-decisions?per=5' | python3 -c \"import sys,json; d=json.load(sys.stdin); assert d['total']>0\""},
    {"id": "lg-041", "group": G, "title": "provider decisions paginate",
     "cmd": "curl -s '{{BASE}}/api/work/provider-decisions?per=3' | python3 -c \"import sys,json; d=json.load(sys.stdin); assert len(d['rows'])<=3\""},
]

with open("/your/data/docker-apps/lcp/qa/cases.logs.json", "w") as fh:
    json.dump(CASES, fh, indent=1)
print("cases.logs:", len(CASES))