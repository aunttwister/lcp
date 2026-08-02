# TODO — Small Fixes & Improvements

**Created:** 2026-08-02

---

1. **Persist OpenCode database across redeployments**

   The OpenCode plugin's SQLite DB must survive container restarts/redeploys.
   Ensure the DB path is bind-mounted correctly in `docker-compose.yml` or
   falls under a persistent volume.

2. **OpenCode DB validation + show rolling/weekly/monthly tokens**

   If the OpenCode DB file doesn't exist but the provider is configured, the
   gateway should surface a clear error/warning. Currently the dashboard shows
   no token data (rolling, weekly, monthly) in this state — need proper
   resolution: either auto-create the DB schema on first use, or fail loudly.

3. **Decouple cost plugin filters (OpenCode vs DeepSeek)**

   When a spending filter is set on DeepSeek in the dashboard, it incorrectly
   applies to the OpenCode Go page too. Each provider plugin needs its own
   isolated filter state. Also, introduce a spending filter on the OpenCode Go
   page (currently missing).

4. **Clarify usage widget green amounts**

   The green currency amounts shown below usage charts are confusing.
   Document what they represent (estimated savings? cache hits? something else?)
   and consider a clearer label or tooltip in the UI.

5. **Redesign the Summary page**

   The current summary view looks broken / chaotic. Needs a cleaner layout
   with meaningful grouped stats, not a raw data dump.

6. **Extract errors/logs into a dedicated "Logs" navbar page**

   - Remove errors section from the Summary page.
   - Create a new "Logs" page accessible from the sidebar/navbar.
   - Show a wider, filterable log view: each individual request with status,
     provider, model, latency, tokens, cost, and any errors.
   - Merge the existing "Recent Requests" into this Logs page — no need for two
     separate views of the same data.

7. **→ This file lives here. Keep it updated.**
