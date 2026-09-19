#!/usr/bin/env python3
"""LCP staging QA runner — executes suite_cases.json against a base URL.

Case schema:
  {id, group, title, cmd, expect: {exit: 0, contains: [..], not_contains: [..]},
   sleep_secs, timeout}

Every case is a bash command (curl | python3 -c asserts carry their own
pass/fail via exit code); {{BASE}} is substituted with --base. Results:
results.json + summary printed. A skipped/missing case is a FAIL.
"""
import argparse
import json
import subprocess
import sys
import time

def main() -> int:
    ap = argparse.ArgumentParser(description="LCP QA runner")
    ap.add_argument("--base", default="http://192.168.1.198:8735")
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None, help="comma ids to run (optional)")
    args = ap.parse_args()

    suite = json.load(open(args.suite, encoding="utf-8"))
    if args.only:
        only = set(x.strip() for x in args.only.split(",") if x.strip())
        cases = [c for c in suite["cases"] if c.get("id") in only]
    else:
        cases = suite["cases"]

    results = {"base": args.base, "suite": suite.get("name"),
               "cases": [], "pass": 0, "fail": 0, "error": 0,
               "groups": {}}
    for c in cases:
        cid = c.get("id") or "?"
        group = c.get("group") or "misc"
        cmd = c["cmd"].replace("{{BASE}}", args.base)
        if c.get("sleep_secs"):
            time.sleep(c["sleep_secs"])
        expect = c.get("expect") or {}
        row = {"id": cid, "group": group, "title": c.get("title"), "ok": False}
        try:
            proc = subprocess.run(["bash", "-c", cmd], capture_output=True,
                                  text=True, timeout=c.get("timeout", 45))
            out = (proc.stdout or "") + (proc.stderr or "")
            code = proc.returncode
        except subprocess.TimeoutExpired:
            results["cases"].append({**row, "ok": False, "note": "timeout"})
            results["fail"] += 1
            results["groups"].setdefault(group, [0, 0])[1] += 1
            continue
        except Exception as exc:  # noqa: BLE001
            results["cases"].append({**row, "ok": False, "note": "runner error: %s" % exc})
            results["error"] += 1
            continue

        ok = code == int(expect.get("exit", 0))
        notes = []
        if ok:
            for s in expect.get("contains", []):
                if s not in out:
                    ok = False
                    notes.append("missing %r" % s)
                    break
        if ok:
            for s in expect.get("not_contains", []):
                if s in out:
                    ok = False
                    notes.append("unexpected %r" % s)
                    break
        row["ok"] = ok
        if not ok:
            row["exit"] = code
            row["tail"] = "\n".join(out.splitlines()[-4:])
            if notes:
                row["note"] = "; ".join(notes)
        results["cases"].append(row)
        if ok:
            results["pass"] += 1
        else:
            results["fail"] += 1
        g = results["groups"].setdefault(group, [0, 0])
        g[0] += 1 if ok else 0
        g[1] += 1

    out_path = args.out or (args.suite + ".results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print("QA suite: %d passed, %d failed, %d errors (of %d)" % (
        results["pass"], results["fail"], results["error"], len(cases)))
    for g, (p, n) in sorted(results["groups"].items()):
        print("  %-24s %3d / %3d" % (g[:24], p, n))
    return 0 if results["fail"] == 0 and results["error"] == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())