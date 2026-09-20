#!/usr/bin/env bash
# QA sweep watchdog — keeps l1-staging-qa.service alive and visibly honest.
# Runs as a no_agent cron every 15 min. Prints ONLY when it acts or when the
# nightly run ends, so the operator hears about restarts/failures, not OK ticks.
set -u
SVC=l1-staging-qa.service
LOG=/your/data/docker-apps/lcp/qa/watchdog.log
TASK=/root/.hermes/profiles/homelab-expert-l2/work/tasks/in_progress/l1-staging-qa-sweep
ts() { date -u +%FT%TZ; }

# Finished: the deliverable exists → the sweep ended; do NOT restart, just say so once.
if [ -f "$TASK/FINDINGS.md" ]; then
  echo "$(ts) sweep FINISHED (FINDINGS.md present) — watchdog standing down" >> "$LOG"
  echo "QA watchdog: L1 staging QA sweep finished — see $TASK/FINDINGS.md"
  exit 0
fi

state=$(systemctl is-active "$SVC" 2>/dev/null || echo dead)
# Hook-denial trap: alive but blocked on approvals for >2 min.
denials=$(journalctl -u "$SVC" --since "10 minutes ago" --no-pager 2>/dev/null \
          | grep -c "Timeout — denying command" || true)

if [ "$state" = "active" ]; then
  if [ "${denials:-0}" -ge 3 ]; then
    echo "$(ts) ACTIVE but stuck on hook denials ($denials) — restarting" >> "$LOG"
    systemctl restart "$SVC"
    echo "QA watchdog: L1 session stalled on hook approvals; restarted $(ts)"
  fi
  exit 0
fi

# Dead/inactive without FINDINGS.md → crashed; restart and report.
echo "$(ts) DEAD ($state) — restarting " >> "$LOG"
systemctl start "$SVC" 2>/dev/null || true
sleep 8
st2=$(systemctl is-active "$SVC" 2>/dev/null || echo dead)
if [ "$st2" = "active" ]; then
  echo "QA watchdog: L1 QA session restarted at $(ts)"
else
  echo "QA watchdog: FAILED to start L1 QA at $(ts) — state=$st2"
fi
exit 0