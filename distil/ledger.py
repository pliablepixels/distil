"""Savings ledger — local-first, privacy-preserving community savings tracking.

Every certified run can append an aggregate record (ids + numbers only, never
context content) to a local JSONL. `summary()` rolls it up so you can see
cumulative tokens and dollars saved across an agent fleet over time.

Community aggregation (a shared leaderboard) is a deliberate OPT-IN: it would
mean network egress of your run metadata, so this module never sends anything.
The `share=` seam is where an explicit, consented uploader would plug in.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_PATH = Path.home() / ".distil" / "savings.jsonl"


@dataclass
class SavingsRecord:
    trajectory_id: str
    model: str
    turns: int
    baseline_dollars: float
    distil_dollars: float
    baseline_input_tokens: int
    distil_input_tokens: int
    tokenizer: str
    ts: float

    @property
    def dollars_saved(self) -> float:
        return self.baseline_dollars - self.distil_dollars

    @property
    def tokens_saved(self) -> int:
        return self.baseline_input_tokens - self.distil_input_tokens

    @property
    def pct_saved(self) -> float:
        return (self.dollars_saved / self.baseline_dollars * 100) if self.baseline_dollars else 0.0


def record(
    *,
    trajectory_id: str,
    model: str,
    turns: int,
    baseline_dollars: float,
    distil_dollars: float,
    baseline_input_tokens: int,
    distil_input_tokens: int,
    tokenizer: str = "heuristic",
    path: Path = DEFAULT_PATH,
    share: bool = False,  # opt-in network egress; intentionally unimplemented
) -> SavingsRecord:
    rec = SavingsRecord(
        trajectory_id,
        model,
        turns,
        baseline_dollars,
        distil_dollars,
        baseline_input_tokens,
        distil_input_tokens,
        tokenizer,
        time.time(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(rec)) + "\n")
    return rec


@dataclass
class LedgerSummary:
    runs: int
    total_dollars_saved: float
    total_tokens_saved: int
    by_trajectory: dict[str, float]  # id -> dollars saved


def summary(path: Path = DEFAULT_PATH) -> LedgerSummary:
    if not path.exists():
        return LedgerSummary(0, 0.0, 0, {})
    runs = 0
    dollars = 0.0
    tokens = 0
    by_traj: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        runs += 1
        saved = d["baseline_dollars"] - d["distil_dollars"]
        dollars += saved
        tokens += d["baseline_input_tokens"] - d["distil_input_tokens"]
        by_traj[d["trajectory_id"]] = by_traj.get(d["trajectory_id"], 0.0) + saved
    return LedgerSummary(runs, dollars, tokens, by_traj)


def render_html(s: LedgerSummary) -> str:
    """Render the ledger as a self-contained dark HTML page — GENUINE savings
    from your own usage (the `live-proxy` source is real proxy traffic)."""
    rows = (
        "".join(
            f'<tr><td>{tid}</td><td class="r">${saved:,.4f}</td></tr>'
            for tid, saved in sorted(s.by_trajectory.items(), key=lambda kv: -kv[1])
        )
        or '<tr><td colspan="2" class="muted">no runs recorded yet</td></tr>'
    )
    live = "live-proxy" in s.by_trajectory
    note = (
        "Includes <b>live-proxy</b> — genuine savings measured on your real traffic."
        if live
        else "Run <code>distil proxy</code> to record genuine savings from real traffic."
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Distil — your savings</title><style>
body{{margin:0;background:#06070b;color:#f2f3f7;font:15px/1.6 Inter,ui-sans-serif,sans-serif;
 -webkit-font-smoothing:antialiased}}
.wrap{{max-width:760px;margin:0 auto;padding:48px 24px}}
h1{{font-size:30px;font-weight:800;letter-spacing:-.02em;margin:0 0 6px}}
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
.foot{{color:#5b6177;font-size:12.5px;margin-top:22px}}
</style></head><body><div class="wrap">
<h1>Your <span class="g">savings</span></h1>
<p class="sub">Genuine, local-first — measured from your own runs and proxy traffic. No content leaves your machine.</p>
<div class="tot">
 <div class="card"><div class="l">Tokens saved</div><div class="v">{s.total_tokens_saved:,}</div></div>
 <div class="card"><div class="l">Dollars saved</div><div class="v g">${s.total_dollars_saved:,.4f}</div></div>
 <div class="card"><div class="l">Runs</div><div class="v">{s.runs:,}</div></div>
</div>
<table><thead><tr><th>source</th><th style="text-align:right">$ saved</th></tr></thead><tbody>{rows}</tbody></table>
<p class="foot">{note} Share verifiably across instances with <code>distil federated-leaderboard</code>.</p>
</div></body></html>"""
