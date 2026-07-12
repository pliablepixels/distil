"""Post-hoc dissection of one wrap session — everything distil knows about it.

``distil dissect`` answers "what exactly happened to my session?": token and
dollar accounting per model, what was digested (and is still recoverable), what
was *not* optimized and why, plus cache-delta, shadow and expand activity.

Data sources (all local; all content-free except restore-blob *existence*):

- ``savings.jsonl`` rows tagged with the session id — token/dollar accounting.
- ``sessions/<sid>.json`` — wrap manifest (tool, argv, flags, billing).
- ``sessions/<sid>.requests.jsonl`` — per-request detail (token breakdown,
  per-block digest signatures, shadow/expand flags).
- ``sessions/<sid>`` / ``.hb`` / ``.exit`` — liveness breadcrumbs.
- ``restore/<handle>`` existence — is a digested block still recoverable?
- ``shadow.jsonl`` rows inside the session's time window (rows are not
  session-tagged; the join is by time and is labelled as such).

Manifest and request detail exist only for sessions wrapped by this version or
newer; older sessions degrade to the ledger-only view with a note.
"""

from __future__ import annotations

import html as _html
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ledger import (
    default_path,
    session_manifest_path,
    session_marker_path,
    session_requests_path,
)


def _state_dir() -> Path:
    return default_path().parent


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Tolerant JSONL reader: missing file -> [], corrupt lines skipped."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


@dataclass
class SessionOverview:
    """One row of the no-argument session picker."""

    sid: str
    tool: str = ""
    started: float = 0.0
    last_ts: float = 0.0
    requests: int = 0
    baseline_tokens: int = 0
    distil_tokens: int = 0
    status: str = ""  # "live" | "exited" | ""


def list_sessions() -> list[SessionOverview]:
    """Every session distil has heard of: ledger rows ∪ session manifests.

    Newest-last-activity first, so the session you just ran is on top.
    """
    by_sid: dict[str, SessionOverview] = {}
    for rec in _read_jsonl(default_path()):
        sid = rec.get("session")
        if not sid or not isinstance(sid, str):
            continue
        ov = by_sid.setdefault(sid, SessionOverview(sid=sid))
        ts = float(rec.get("ts") or 0.0)
        ov.started = min(ov.started or ts, ts)
        ov.last_ts = max(ov.last_ts, ts)
        ov.requests += int(rec.get("turns") or 0)
        ov.baseline_tokens += int(rec.get("baseline_input_tokens") or 0)
        ov.distil_tokens += int(rec.get("distil_input_tokens") or 0)
    sess_dir = _state_dir() / "sessions"
    try:
        manifests = sorted(sess_dir.glob("s*.json"))
    except OSError:
        manifests = []
    for mp in manifests:
        try:
            man = json.loads(mp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        sid = man.get("sid") or mp.stem
        ov = by_sid.setdefault(sid, SessionOverview(sid=sid))
        ov.tool = str(man.get("tool") or "")
        started = float(man.get("started_ts") or 0.0)
        if started:
            ov.started = min(ov.started or started, started)
            ov.last_ts = max(ov.last_ts, started)
    for ov in by_sid.values():
        marker = session_marker_path(ov.sid)
        if marker is not None:
            if marker.with_suffix(".exit").exists():
                ov.status = "exited"
            elif marker.exists():
                ov.status = "live"
    return sorted(by_sid.values(), key=lambda o: o.last_ts, reverse=True)


def resolve_sid(query: str) -> str | None:
    """Resolve ``latest`` or a unique session-id prefix to a full id."""
    sessions = list_sessions()
    if not sessions:
        return None
    if query == "latest":
        return sessions[0].sid
    hits = [o.sid for o in sessions if o.sid == query]
    if hits:
        return hits[0]
    hits = [o.sid for o in sessions if o.sid.startswith(query)]
    return hits[0] if len(hits) == 1 else None


@dataclass
class Dissection:
    """Everything distil knows about one wrap session, joined and totalled."""

    sid: str
    manifest: dict[str, Any] | None
    ledger_rows: list[dict[str, Any]]
    requests: list[dict[str, Any]]
    marker: str | None  # "0" (wrapped, no traffic yet), "1" (traffic seen), None
    heartbeat: str | None
    exit_note: str | None
    shadow_window_rows: int = 0
    shadow_window_agree: int = 0
    # Derived digest inventory: handle -> {"sig", "tokens", "folds", "recoverable"}
    blocks: dict[str, dict[str, Any]] = field(default_factory=dict)

    # ---- ledger-derived totals (available even for pre-manifest sessions) ----
    @property
    def baseline_tokens(self) -> int:
        return sum(int(r.get("baseline_input_tokens") or 0) for r in self.ledger_rows)

    @property
    def distil_tokens(self) -> int:
        return sum(int(r.get("distil_input_tokens") or 0) for r in self.ledger_rows)

    @property
    def dollars_saved(self) -> float:
        return sum(
            float(r.get("baseline_dollars") or 0.0) - float(r.get("distil_dollars") or 0.0)
            for r in self.ledger_rows
        )

    @property
    def pct_saved(self) -> float:
        b = self.baseline_tokens
        return 100.0 * (b - self.distil_tokens) / b if b else 0.0

    @property
    def started(self) -> float:
        cands = [float(r.get("ts") or 0.0) for r in self.ledger_rows]
        cands += [float(r.get("ts") or 0.0) for r in self.requests]
        if self.manifest:
            cands.append(float(self.manifest.get("started_ts") or 0.0))
        cands = [c for c in cands if c]
        return min(cands) if cands else 0.0

    @property
    def ended(self) -> float:
        cands = [float(r.get("ts") or 0.0) for r in self.ledger_rows]
        cands += [float(r.get("ts") or 0.0) for r in self.requests]
        return max(cands) if cands else 0.0

    @property
    def billing(self) -> str:
        """Manifest billing mode, falling back to this machine's current mode
        (labelled detection, not a session fact) for pre-manifest sessions."""
        if self.manifest and self.manifest.get("billing"):
            return str(self.manifest["billing"])
        try:
            from .doctor import subscription_mode

            return "subscription" if subscription_mode() else "metered"
        except Exception:  # noqa: BLE001 — billing detection is cosmetic
            return "unknown"

    def per_model(self) -> list[tuple[str, int, int, int]]:
        """[(model, booked_requests, baseline_tokens, distil_tokens)] biggest first."""
        agg: dict[str, list[int]] = {}
        for r in self.ledger_rows:
            m = agg.setdefault(str(r.get("model") or "unknown"), [0, 0, 0])
            m[0] += int(r.get("turns") or 0)
            m[1] += int(r.get("baseline_input_tokens") or 0)
            m[2] += int(r.get("distil_input_tokens") or 0)
        return sorted(((k, v[0], v[1], v[2]) for k, v in agg.items()), key=lambda t: -t[2])

    # ---- request-detail-derived (needs sessions/<sid>.requests.jsonl) ----
    @property
    def detail_available(self) -> bool:
        return bool(self.requests)

    @property
    def delta_tokens_saved(self) -> int:
        return sum(int(r.get("delta_tokens_saved") or 0) for r in self.requests)

    @property
    def overhead_tokens_avg(self) -> int:
        vals = [int(r.get("overhead_tokens") or 0) for r in self.requests]
        return sum(vals) // len(vals) if vals else 0

    @property
    def verbatim_requests(self) -> int:
        return sum(1 for r in self.requests if r.get("mode") in (None, "", "verbatim"))

    @property
    def unbooked_requests(self) -> int:
        return sum(1 for r in self.requests if not r.get("booked"))

    @property
    def shadow_sampled(self) -> int:
        return sum(1 for r in self.requests if r.get("shadow_sampled"))

    @property
    def expand_resolved(self) -> int:
        return sum(1 for r in self.requests if r.get("expanded"))

    def blocks_by_kind(self) -> list[tuple[str, int, int]]:
        """[(signature, unique_blocks, tokens)] biggest-token first."""
        agg: dict[str, list[int]] = {}
        for info in self.blocks.values():
            m = agg.setdefault(str(info.get("sig") or "?"), [0, 0])
            m[0] += 1
            m[1] += int(info.get("tokens") or 0)
        return sorted(((k, v[0], v[1]) for k, v in agg.items()), key=lambda t: -t[2])

    def top_blocks(self, n: int = 10) -> list[tuple[str, str, int, int, bool]]:
        """[(handle, sig, tokens, folds, recoverable)] biggest first."""
        rows = [
            (
                h,
                str(i.get("sig") or "?"),
                int(i.get("tokens") or 0),
                int(i.get("folds") or 0),
                bool(i.get("recoverable")),
            )
            for h, i in self.blocks.items()
        ]
        return sorted(rows, key=lambda t: -t[2])[:n]


def dissect(sid: str) -> Dissection:
    """Assemble a full Dissection for *sid* from every local source."""
    ledger_rows = [r for r in _read_jsonl(default_path()) if r.get("session") == sid]
    manifest: dict[str, Any] | None = None
    mp = session_manifest_path(sid)
    if mp is not None:
        try:
            loaded = json.loads(mp.read_text(encoding="utf-8"))
            manifest = loaded if isinstance(loaded, dict) else None
        except (OSError, json.JSONDecodeError):
            manifest = None
    rp = session_requests_path(sid)
    requests = _read_jsonl(rp) if rp is not None else []

    marker = heartbeat = exit_note = None
    marker_p = session_marker_path(sid)
    if marker_p is not None:
        for attr, path in (
            ("marker", marker_p),
            ("heartbeat", marker_p.with_suffix(".hb")),
            ("exit_note", marker_p.with_suffix(".exit")),
        ):
            try:
                val = path.read_text(encoding="utf-8").strip()
            except OSError:
                val = None
            if attr == "marker":
                marker = val
            elif attr == "heartbeat":
                heartbeat = val
            else:
                exit_note = val

    d = Dissection(
        sid=sid,
        manifest=manifest,
        ledger_rows=ledger_rows,
        requests=requests,
        marker=marker,
        heartbeat=heartbeat,
        exit_note=exit_note,
    )

    restore_dir = _state_dir() / "restore"
    for rec in requests:
        for blk in rec.get("blocks") or []:
            h = blk.get("h")
            if not isinstance(h, str):
                continue
            info = d.blocks.setdefault(
                h, {"sig": blk.get("sig"), "tokens": int(blk.get("tokens") or 0), "folds": 0}
            )
            info["folds"] += 1
    for h, info in d.blocks.items():
        info["recoverable"] = (restore_dir / h).exists()

    if d.started and d.ended:
        for row in _read_jsonl(_state_dir() / "shadow.jsonl"):
            ts = float(row.get("ts") or 0.0)
            if d.started - 1 <= ts <= d.ended + 300:
                d.shadow_window_rows += 1
                d.shadow_window_agree += 1 if row.get("equivalent") else 0
    return d


# --------------------------------------------------------------------------- render
def _human(n: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:.0f}"


def _when(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "—"


def render_sessions_text(sessions: list[SessionOverview], *, color: bool = True) -> str:
    """The no-argument picker: one line per known session, newest first."""

    def c(code: str, s: str) -> str:
        return f"\x1b[{code}m{s}\x1b[0m" if color else s

    if not sessions:
        return (
            "No wrap sessions recorded yet. Start one with:\n"
            "  distil wrap -- claude   (or codex/gemini/any base-url-honoring tool)"
        )
    out = [c("1", "distil sessions — pick one to dissect"), ""]
    hdr = f"{'session':<22} {'tool':<10} {'started':<17} {'last':<17} {'reqs':>5} {'saved':>7}  status"
    out.append(c("2", hdr))
    for o in sessions:
        pct = 100.0 * (o.baseline_tokens - o.distil_tokens) / o.baseline_tokens if o.baseline_tokens else 0.0
        out.append(
            f"{o.sid:<22} {(o.tool or '?'):<10} {_when(o.started):<17} {_when(o.last_ts):<17} "
            f"{o.requests:>5} {pct:>6.1f}%  {o.status}"
        )
    out += ["", "dissect one:  distil dissect <session>   (a unique prefix or `latest` works)"]
    return "\n".join(out)


def _flags_line(man: dict[str, Any]) -> str:
    flags = man.get("flags") or {}
    on = [k for k in ("expand", "session_delta", "lossless_only", "verbatim") if flags.get(k)]
    if float(flags.get("shadow_rate") or 0.0) > 0:
        on.append(f"shadow={flags['shadow_rate']}")
    if (flags.get("shape_output") or "off") != "off":
        on.append(f"shape_output={flags['shape_output']}")
    return ", ".join(on) or "defaults"


def render_text(d: Dissection, *, color: bool = True) -> str:
    """Terminal report. Sections degrade gracefully when a source is missing."""

    def c(code: str, s: str) -> str:
        return f"\x1b[{code}m{s}\x1b[0m" if color else s

    man = d.manifest or {}
    subscription = d.billing == "subscription"
    out: list[str] = []
    tool = man.get("tool") or "unknown tool"
    out.append(c("1", f"distil dissect — {d.sid} ({tool})"))

    # Session card
    dur = ""
    if d.started and d.ended and d.ended > d.started:
        dur = f"  ({(d.ended - d.started) / 60:.0f} min)"
    out.append(f"  window   {_when(d.started)} → {_when(d.ended)}{dur}")
    if man:
        out.append(f"  wrap     {' '.join(man.get('argv') or [])}  [{_flags_line(man)}]")
        out.append(f"  distil   v{man.get('distil_version', '?')}, billing: {d.billing}")
    else:
        out.append("  wrap     manifest not recorded (session predates dissect logging)")
    if d.exit_note:
        out.append(f"  exit     {d.exit_note}")
    elif d.marker == "0":
        out.append(c("33", "  status   marker=0 — wrapped but NO traffic went through distil (bypass?)"))
    elif d.marker == "1":
        out.append(f"  status   live (heartbeat: {d.heartbeat or '—'})")

    # Savings (ledger)
    out.append("")
    out.append(c("1", "savings (input tokens, booked 2xx only)"))
    if not d.ledger_rows:
        out.append("  no booked requests in the ledger for this session")
    else:
        note = " (notional — flat-rate plan)" if subscription else ""
        out.append(
            f"  {_human(d.baseline_tokens)} → {_human(d.distil_tokens)}"
            f"  ({d.pct_saved:.1f}% saved, ${d.dollars_saved:.2f}{note})"
        )
        for model, reqs, bt, dt in d.per_model():
            pct = 100.0 * (bt - dt) / bt if bt else 0.0
            out.append(f"    {model:<34} {reqs:>4} req  {_human(bt):>8} → {_human(dt):>8}  {pct:>5.1f}%")

    # Request detail
    out.append("")
    out.append(c("1", "request detail"))
    if not d.detail_available:
        out.append("  not recorded — per-request detail needs a wrap from this distil version or newer")
    else:
        n = len(d.requests)
        out.append(
            f"  {n} proxied requests: {n - d.unbooked_requests} booked, {d.unbooked_requests} not booked "
            f"(non-2xx/retry), {d.verbatim_requests} verbatim (nothing worth compressing)"
        )
        out.append(f"  cache-delta: {_human(d.delta_tokens_saved)} tokens referenced instead of resent")
        out.append(
            f"  not optimized by design: ~{_human(d.overhead_tokens_avg)} tokens/request of system prompt "
            "+ tool definitions (sent verbatim every request), plus recent turns kept for fidelity"
        )
        if d.blocks:
            out.append("")
            out.append(c("1", "digested blocks (content-free: kind:size, tokens)"))
            for sig, uniq, toks in d.blocks_by_kind():
                out.append(f"    {sig:<12} {uniq:>4} blocks  {_human(toks):>8} tokens")
            recoverable = sum(1 for i in d.blocks.values() if i.get("recoverable"))
            out.append(f"  recoverable now: {recoverable}/{len(d.blocks)} blocks still in restore/")
            out.append("  largest folds:")
            for h, sig, toks, folds, rec in d.top_blocks(5):
                mark = "✓" if rec else "expired"
                out.append(f"    {h}  {sig:<12} {_human(toks):>8} tokens  ×{folds} requests  [{mark}]")

    # Quality loops
    out.append("")
    out.append(c("1", "quality loops"))
    if d.detail_available:
        out.append(f"  expand: {d.expand_resolved} requests had distil_expand calls resolved in-proxy")
        out.append(f"  shadow: {d.shadow_sampled} requests sampled for decision-equivalence")
    if d.shadow_window_rows:
        out.append(
            f"  shadow verdicts in this session's time window (time-joined, not session-tagged): "
            f"{d.shadow_window_agree}/{d.shadow_window_rows} equivalent"
        )
    elif not d.detail_available:
        out.append("  no session-scoped signal recorded for this session")

    out.append("")
    out.append(
        c("2", "sources: savings.jsonl, sessions/<sid>{.json,.requests.jsonl,.hb,.exit}, restore/, shadow.jsonl")
    )
    out.append(c("2", "retention: session detail follows the sessions/ TTL sweep; restore blobs are pruned separately"))
    return "\n".join(out)


def to_json(d: Dissection) -> dict[str, Any]:
    """Machine-readable dissection (same numbers the text/html reports show)."""
    return {
        "session": d.sid,
        "manifest": d.manifest,
        "window": {"started_ts": d.started, "ended_ts": d.ended},
        "savings": {
            "baseline_input_tokens": d.baseline_tokens,
            "distil_input_tokens": d.distil_tokens,
            "pct_saved": round(d.pct_saved, 2),
            "dollars_saved": round(d.dollars_saved, 4),
            "dollars_notional": d.billing == "subscription",
            "per_model": [
                {"model": m, "requests": r, "baseline_tokens": b, "distil_tokens": t}
                for m, r, b, t in d.per_model()
            ],
        },
        "detail_available": d.detail_available,
        "requests": {
            "total": len(d.requests),
            "unbooked": d.unbooked_requests,
            "verbatim": d.verbatim_requests,
            "delta_tokens_saved": d.delta_tokens_saved,
            "overhead_tokens_avg": d.overhead_tokens_avg,
        },
        "blocks": {
            "by_kind": [
                {"sig": s, "unique": u, "tokens": t} for s, u, t in d.blocks_by_kind()
            ],
            "recoverable": sum(1 for i in d.blocks.values() if i.get("recoverable")),
            "unique": len(d.blocks),
            "top": [
                {"handle": h, "sig": s, "tokens": t, "folds": f, "recoverable": r}
                for h, s, t, f, r in d.top_blocks()
            ],
        },
        "quality": {
            "expand_resolved_requests": d.expand_resolved,
            "shadow_sampled_requests": d.shadow_sampled,
            "shadow_window_rows": d.shadow_window_rows,
            "shadow_window_agree": d.shadow_window_agree,
        },
        "liveness": {"marker": d.marker, "heartbeat": d.heartbeat, "exit": d.exit_note},
    }


def render_html(d: Dissection) -> str:
    """Self-contained dark page in the ledger `render_html` style."""
    man = d.manifest or {}
    subscription = d.billing == "subscription"
    e = _html.escape
    dol_note = " (notional — flat-rate plan)" if subscription else ""

    model_rows = "".join(
        f"<tr><td>{e(m)}</td><td class='r'>{r}</td><td class='r'>{_human(b)}</td>"
        f"<td class='r'>{_human(t)}</td><td class='r'>{100.0 * (b - t) / b if b else 0.0:.1f}%</td></tr>"
        for m, r, b, t in d.per_model()
    ) or "<tr><td class='muted' colspan='5'>no booked requests</td></tr>"

    kind_rows = "".join(
        f"<tr><td>{e(s)}</td><td class='r'>{u}</td><td class='r'>{_human(t)}</td></tr>"
        for s, u, t in d.blocks_by_kind()
    )
    top_rows = "".join(
        f"<tr><td><code>{e(h)}</code></td><td>{e(s)}</td><td class='r'>{_human(t)}</td>"
        f"<td class='r'>×{f}</td><td>{'recoverable' if r else '<span class=muted>expired</span>'}</td></tr>"
        for h, s, t, f, r in d.top_blocks()
    )
    detail_card = (
        f"""<h2>Request detail</h2>
<p>{len(d.requests)} proxied requests — {d.unbooked_requests} not booked (non-2xx/retry),
{d.verbatim_requests} verbatim (nothing worth compressing).
Cache-delta referenced <b>{_human(d.delta_tokens_saved)}</b> tokens instead of resending them.
Not optimized by design: ~{_human(d.overhead_tokens_avg)} tokens/request of system prompt + tool
definitions, plus recent turns kept verbatim for fidelity.</p>
<h2>Digested blocks <span class="muted">(content-free: kind:size)</span></h2>
<table><tr><th>kind</th><th>blocks</th><th>tokens</th></tr>{kind_rows}</table>
<h2>Largest folds</h2>
<table><tr><th>handle</th><th>kind</th><th>tokens</th><th>seen</th><th>restore</th></tr>{top_rows}</table>"""
        if d.detail_available
        else "<h2>Request detail</h2><p class='muted'>Not recorded — per-request detail needs a "
        "wrap from this distil version or newer.</p>"
    )
    quality = (
        f"<p>expand: {d.expand_resolved} requests resolved in-proxy · shadow: {d.shadow_sampled} "
        f"sampled · window verdicts (time-joined): {d.shadow_window_agree}/{d.shadow_window_rows} equivalent</p>"
        if (d.detail_available or d.shadow_window_rows)
        else "<p class='muted'>no session-scoped quality signal recorded</p>"
    )
    exit_line = f"<p class='muted'>exit: {e(d.exit_note)}</p>" if d.exit_note else ""

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<link rel="icon" type="image/svg+xml" href="https://dshakes.github.io/distil/assets/logo.svg"/>
<title>Distil — dissect {e(d.sid)}</title><style>
body{{margin:0;background:#06070b;color:#f2f3f7;font:15px/1.6 Inter,ui-sans-serif,sans-serif;
 -webkit-font-smoothing:antialiased}}
.wrap{{max-width:760px;margin:0 auto;padding:48px 24px}}
h1{{font-size:30px;font-weight:800;letter-spacing:-.02em;margin:0 0 6px}}
h2{{font-size:17px;font-weight:700;margin:26px 0 10px}}
.sub{{color:#9aa1b3;margin:0 0 28px}}
.tot{{display:flex;gap:14px;margin:0 0 28px;flex-wrap:wrap}}
.card{{flex:1;min-width:150px;background:linear-gradient(180deg,#12151f,#0b0d15);
 border:1px solid #252c3e;border-radius:14px;padding:20px}}
.card .l{{color:#9aa1b3;font-size:12px}} .card .v{{font-size:30px;font-weight:800;margin-top:4px}}
.g{{background:linear-gradient(135deg,#8b7bff,#5ad1c9);-webkit-background-clip:text;background-clip:text;color:transparent}}
table{{width:100%;border-collapse:collapse;border:1px solid #1b2030;border-radius:12px;overflow:hidden}}
th,td{{padding:11px 14px;border-bottom:1px solid #1b2030;text-align:left}}
th{{color:#5b6177;font-size:11px;text-transform:uppercase;letter-spacing:.07em}}
td.r{{text-align:right;color:#5ad1c9;font-variant-numeric:tabular-nums}} .muted{{color:#5b6177}}
code{{color:#8b7bff}}
.foot{{color:#5b6177;font-size:12.5px;margin-top:22px}}
</style></head><body><div class="wrap">
<h1>Session <span class="g">dissected</span></h1>
<p class="sub">{e(d.sid)} — {e(man.get("tool") or "unknown tool")},
{e(_when(d.started))} → {e(_when(d.ended))} · wrap flags: {e(_flags_line(man)) if man else "unknown"}
· billing: {e(d.billing)}</p>
<div class="tot">
<div class="card"><div class="l">Input tokens</div><div class="v">{_human(d.baseline_tokens)} → {_human(d.distil_tokens)}</div></div>
<div class="card"><div class="l">Saved</div><div class="v g">{d.pct_saved:.1f}%</div></div>
<div class="card"><div class="l">Dollars{e(dol_note)}</div><div class="v">${d.dollars_saved:.2f}</div></div>
</div>
<h2>Per model</h2>
<table><tr><th>model</th><th>req</th><th>baseline</th><th>distil</th><th>saved</th></tr>{model_rows}</table>
{detail_card}
<h2>Quality loops</h2>
{quality}
{exit_line}
<p class="foot">Local-first: assembled from savings.jsonl, sessions/&lt;sid&gt;*, restore/ and
shadow.jsonl on this machine. Content-free — handles and kind:size signatures only.</p>
</div></body></html>"""
