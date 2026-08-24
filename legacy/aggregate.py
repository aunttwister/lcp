#!/usr/bin/env python3
"""Daily LLM cost aggregation. Reads from SQLite, outputs summary."""

import sqlite3
import sys
from datetime import datetime, timezone, timedelta

DB_PATH = "/app/data/costs.db"

def daily_report(db_path=DB_PATH, days=1):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    yesterday = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    
    rows = conn.execute(
        """SELECT model, provider,
                  total_prompt_tokens, total_completion_tokens,
                  total_cache_hit_tokens, total_cache_miss_tokens,
                  total_cost_usd, request_count
           FROM daily_summary
           WHERE date = ?
           ORDER BY total_cost_usd DESC""",
        (yesterday,),
    ).fetchall()
    
    if not rows:
        print("No usage data for", yesterday)
        conn.close()
        return
    
    total_cost = sum(r["total_cost_usd"] for r in rows)
    total_requests = sum(r["request_count"] for r in rows)
    total_tokens = sum(r["total_prompt_tokens"] + r["total_completion_tokens"] for r in rows)
    
    print(f"LLM Cost — {yesterday}")
    print(f"Total: ${total_cost:.6f} | {total_requests} requests | {total_tokens:,} tokens")
    print()
    
    for r in rows:
        cache_pct = ""
        cache_total = r["total_cache_hit_tokens"] + r["total_cache_miss_tokens"]
        if cache_total > 0:
            hit_pct = r["total_cache_hit_tokens"] / cache_total * 100
            cache_pct = f" (cache {hit_pct:.0f}% hit)"
        
        print(f"  {r['model']} ({r['provider']}): ${r['total_cost_usd']:.6f}"
              f" — {r['request_count']} req, {r['total_prompt_tokens']+r['total_completion_tokens']:,} tokens{cache_pct}")
    
    conn.close()
    return total_cost


if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    daily_report(days=days)
