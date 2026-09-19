#!/usr/bin/env python3
"""Merge per-group case fragments into suite_cases.json + run generator files."""
import glob
import json
import os
import subprocess
import sys

QA = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    # 1) generate fragments
    count = 0
    for gen in sorted(glob.glob(os.path.join(QA, "gen_*.py"))):
        subprocess.run([sys.executable, gen], check=True)
    # 2) merge
    all_cases = []
    for frag in sorted(glob.glob(os.path.join(QA, "cases.*.json"))):
        frag_cases = json.load(open(frag, encoding="utf-8"))
        for c in frag_cases:
            c["cmd"] = c["cmd"].replace("{{HTTP}}", "%{http_code}")
        all_cases.extend(frag_cases)
    # dedupe by id
    seen = set()
    uniq = []
    for c in all_cases:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        uniq.append(c)
    suite = {"name": "LCP staging QA sweep", "cases": uniq}
    out = os.path.join(QA, "suite_cases.json")
    json.dump(suite, open(out, "w", encoding="utf-8"), indent=1)
    print("TOTAL CASES:", len(uniq))