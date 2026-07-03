# Threat model

What distil trusts, what it protects, and what it deliberately does not defend
against. Scope: the local proxy (`distil proxy` / `wrap`), the async proxy, the
multi-tenant gateway, and the MCP server. Audited 2026-07 (serving-path
security review); update this file when a trust boundary moves.

## Assets

1. **Customer API keys** — pass through every request.
2. **Conversation content** — full prompts and responses transit the process;
   originals of digested blocks are retained for reversibility.
3. **Usage metadata** — per-tenant token/dollar volumes (gateway), savings
   ledger, shadow equivalence samples.

## Trust boundaries

| Component | Trusts | Does NOT trust |
|---|---|---|
| proxy / wrap | the local user (same UID), the operator-configured upstream URL | request bodies (parsed defensively; compression failure fails open to the original), upstream response content |
| gateway | the operator (flags/env), upstream URL | callers: tenant identity derives from the credential hash; `x-distil-tenant` honored only under `--trust-tenant-header`; `/distil/*` requires `--admin-token` off loopback |
| MCP server | the local MCP client (stdio, same UID) | tool arguments (validated) |

## Guarantees (enforced in code, tested)

- **Keys are never persisted or logged.** Forwarded only to the configured
  upstream. Ledger/shadow/learn/telemetry files carry counts, hashes, and
  booleans — never content, never credentials.
- **No auto-redirect on the forward path.** A 3xx from the upstream is relayed
  to the client, never followed — credentials are never re-sent to a host the
  operator didn't configure.
- **TLS verification is stock** (urllib/aiohttp defaults). No verify-off
  escape hatch exists.
- **SSRF/path guards** on every forward (`httpguard.safe_forward_path`):
  userinfo, scheme injection, traversal, and control characters are rejected;
  request bodies are size-capped.
- **Fail open to fidelity, closed on safety.** A compression error forwards
  the original request unchanged; a guard rejection returns an error — content
  is never silently altered by a failure path.
- **Bounded state.** Restore stores and session maps are capped; the MCP
  handle store is FIFO-bounded and chmod 0600.

## Content at rest (deliberate, documented)

- `~/.distil/mcp_store.json` (MCP server only): originals of digested blocks,
  plaintext, owner-only, bounded to 512 entries. This is the price of
  cross-process reversibility on the MCP path; the proxies keep restore state
  in memory only.
- The savings ledger / shadow ledger / learn stats: numbers only.

## Out of scope (explicitly not defended)

- **A hostile local user on the same machine.** The proxies bind loopback by
  default and trust the local UID; distil is not a privilege boundary.
- **A malicious operator-configured upstream.** Whoever controls `--upstream`
  sees the traffic — that is the point of a proxy. Point it only at providers
  you trust.
- **Malicious model output.** Distil relays responses byte-faithfully; agent-
  side prompt-injection defense belongs to the agent harness.
- **Network eavesdropping between distil and the client.** Local loopback
  traffic is unencrypted; bind non-loopback only behind your own TLS/network
  controls (and set `--admin-token`).

## Residual risks (known, accepted, ranked)

1. Gateway per-tenant metadata is visible to anyone holding the admin token —
   scope the token like a credential.
2. `DeltaSession` keys on the first message hash; agents with identical fixed
   first turns share a session's content-free prefix stats (cosmetic only —
   verified no content crosses sessions).
3. Upstream error strings are relayed (truncated to 200 chars) to the local
   client for debuggability.

## Reporting

Security reports: open a GitHub security advisory (preferred) or a private
issue. Do not post exploits in public issues.
