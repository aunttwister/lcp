# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue for security vulnerabilities. Instead,
use **GitHub private vulnerability reporting** (Security → *Report a
vulnerability* on the repository) so the report stays private until it is
resolved.

You should receive an acknowledgement within **48 hours**. Once a fix is
prepared we will coordinate a disclosure date.

## Scope

LCP is a self-hosted gateway that stores **encrypted upstream provider API
keys** and manages **budgets / spend limits**. Vulnerabilities in the
following areas are in scope:

- Bypass of API-key authentication or per-profile access control
- Exposure or decryption of stored provider credentials (the `credential_store`
  / `crypto` modules)
- Budget / spend-limit bypass (hard-stop and threshold enforcement)
- Server-side request handling — SSRF, injection, or unauthenticated state
  mutation via the dashboard or HTTP API
- Secrets leaking into logs, error responses, or the UI

Out of scope (by design):

- **No TLS termination** — LCP is designed to sit behind a reverse proxy
  (e.g. Caddy / Traefik) that terminates HTTPS. Raw HTTP is not a supported
  security boundary.
- **Multi-tenant isolation** — LCP is a single-instance, single-org gateway.
  It is not designed to isolate mutually untrusted tenants.

## Supported versions

| Version | Supported |
|---|---|
| 0.5.x  | ✅ |
| < 0.5  | ❌ |

## Security model notes

- Provider API keys and the OpenCode cookie are encrypted with Fernet using
  `LCP_SECRET_KEY` and stored in SQLite — never in `gateway.yaml` or the repo.
- API keys are stored as SHA-256 hashes; the plaintext is shown only once at
  creation.
- If `LCP_SECRET_KEY` is unset, a random key is generated and persisted under
  `data/.lcp_secret_key` (gitignored). This key must be kept private and
  stable across restarts, otherwise stored credentials become undecryptable.

## Disclosure

We believe in responsible disclosure. Please give us a reasonable window
(default 90 days) to fix and release before public disclosure.
