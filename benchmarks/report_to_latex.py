#!/usr/bin/env python3
"""report_to_latex.py — turn a `prove.py --report` JSON into paper-ready LaTeX.

Runs after the headline experiment and writes LaTeX fragments into
``docs/paper/generated/``. The paper (`docs/paper/main.tex`) ``\\input``s them via
``\\IfFileExists`` — so once you run this, the figures, tables, and headline macros
in the PDF reflect your real numbers with **zero hand-copying**. Before you run it,
the paper falls back to clearly-labeled placeholders.

Usage:
  python benchmarks/prove.py ... --report results.json
  python benchmarks/report_to_latex.py results.json
  # then recompile docs/paper/main.tex (Overleaf / latexmk)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "paper" / "generated"


def _tex(s: str) -> str:
    """Escape LaTeX specials in free text (method names etc.)."""
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("$", r"\$")
        .replace("@", r"@")
    )


def _pct(x: float) -> str:
    return f"{x * 100:.1f}\\%"


CHECK = r"\checkmark"
CROSS = r"$\times$"
EOL = r" \\"  # LaTeX row terminator (kept out of f-strings: no backslashes allowed there)


def macros(rep: dict) -> str:
    cov = rep.get("coverage") or {}
    sav = cov.get("mean_test_savings")
    c = cov.get("empirical_coverage")
    r = cov.get("mean_realized_risk")
    out = ["% auto-generated headline macros — do not edit"]
    out.append(f"\\renewcommand{{\\HLsavings}}{{{_pct(sav)}}}" if sav is not None else "")
    out.append(f"\\renewcommand{{\\HLcoverage}}{{{_pct(c)}}}" if c is not None else "")
    out.append(f"\\renewcommand{{\\HLrisk}}{{{_pct(r)}}}" if r is not None else "")
    return "\n".join(x for x in out if x) + "\n"


def frontier(rep: dict) -> str:
    rows = rep.get("frontier") or []
    alpha = (rep.get("coverage") or {}).get("alpha", 0.05)
    cert, lossy = [], []
    for r in rows:
        pt = f"({r['savings'] * 100:.2f},{r['decision_change'] * 100:.2f})"
        (cert if r["decision_change"] <= alpha else lossy).append(pt)
    ymax = max([r["decision_change"] * 100 for r in rows] + [alpha * 100 + 5]) + 5
    xmax = max([r["savings"] * 100 for r in rows] + [10]) + 5
    return (
        "% auto-generated frontier (E1)\n"
        "\\begin{tikzpicture}\n\\begin{axis}[width=0.7\\textwidth,height=6cm,"
        "xlabel={token savings (\\%)},ylabel={decision-change (\\%)},"
        f"xmin=0,xmax={xmax:.0f},ymin=-3,ymax={ymax:.0f},"
        "legend style={font=\\scriptsize,at={(0.02,0.98)},anchor=north west}]\n"
        f"\\addplot[only marks,mark=*,distilgreen] coordinates {{{' '.join(cert) or '(0,0)'}}};\n"
        "\\addlegendentry{certified ($\\le\\alpha$)}\n"
        f"\\addplot[only marks,mark=triangle*,distilred] coordinates {{{' '.join(lossy) or '(0,0)'}}};\n"
        "\\addlegendentry{lossy (flips)}\n"
        f"\\addplot[dashed,distilgray] coordinates {{(0,{alpha * 100:.1f}) ({xmax:.0f},{alpha * 100:.1f})}};\n"
        f"\\addlegendentry{{$\\alpha={alpha * 100:.0f}\\%$}}\n"
        "\\end{axis}\n\\end{tikzpicture}\n"
    )


def head_to_head(rep: dict) -> str:
    rows = rep.get("head_to_head") or []
    if not rows:
        return "% no head-to-head in report (run with --baselines)\n"
    body = "\n".join(
        _tex(r["method"])
        + " & "
        + r["kind"]
        + " & "
        + _pct(r["savings"])
        + " & "
        + _pct(r["decision_change"])
        + " & "
        + (CHECK if r["certifies"] else CROSS)
        + EOL
        for r in rows
    )
    header = "method & kind & savings & dec-change & certifies?" + EOL
    return (
        "% auto-generated head-to-head (E5)\n"
        "\\begin{tabular}{@{}llrrc@{}}\n\\toprule\n"
        f"{header}\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"
    )


def coverage(rep: dict) -> str:
    c = rep.get("coverage") or {}
    if not c:
        return "% no coverage in report\n"
    tgt = c.get("target_coverage")
    tgt_s = f"{tgt * 100:.0f}\\%" if tgt else "expected-risk (CRC)"
    return (
        "% auto-generated coverage (E2)\n"
        "\\begin{tabular}{@{}lr@{}}\n\\toprule\n"
        f"method & {c.get('method', 'ltt').upper()} \\\\\n"
        f"$\\alpha$ / $\\delta$ & {c.get('alpha')} / {c.get('delta')} \\\\\n"
        f"splits & {c.get('reps')} \\\\\n"
        f"certified in & {_pct(c.get('certified_frac', 0))} of splits \\\\\n"
        f"empirical coverage $\\Pr(\\text{{realized}}\\le\\alpha)$ & {_pct(c.get('empirical_coverage', 0))} \\\\\n"
        f"target ($1-\\delta$) & {tgt_s} \\\\\n"
        f"mean realized held-out risk & {_pct(c.get('mean_realized_risk', 0))} \\\\\n"
        f"mean certified savings & {_pct(c.get('mean_test_savings', 0))} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n"
    )


def task_success(rep: dict) -> str:
    t = rep.get("task_success") or {}
    if not t:
        return "% no task-success in report (need outcome-labeled trajectories)\n"
    body = "\n".join(
        _tex(r["level"])
        + " & "
        + _pct(r["savings"])
        + " & "
        + _pct(r["retained_success"])
        + f" [{r['ci_low'] * 100:.0f}--{r['ci_high'] * 100:.0f}]"
        + EOL
        for r in t["levels"]
    )
    header = "level & savings & retained success (95\\% CI)" + EOL
    return (
        f"% auto-generated task-success (E4); baseline={_pct(t.get('baseline_success', 0))}, n={t.get('n')}\n"
        "\\begin{tabular}{@{}lrr@{}}\n\\toprule\n"
        f"{header}\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"
    )


def shift(rep: dict) -> str:
    rows = rep.get("shift") or []
    if not rows:
        return "% no distribution-shift in report (need >=2 domains)\n"
    body = "\n".join(
        _tex(r["held_out_domain"])
        + " & "
        + _tex(r.get("certified") or "none")
        + " & "
        + _pct(r.get("realized_risk", 0))
        + " & "
        + _pct(r.get("savings", 0))
        + " & "
        + (CHECK if r.get("held_within_alpha") else CROSS)
        + EOL
        for r in rows
    )
    header = "held-out domain & certified & realized & savings & ok?" + EOL
    return (
        "% auto-generated distribution-shift (E3)\n"
        "\\begin{tabular}{@{}llrrc@{}}\n\\toprule\n"
        f"{header}\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report", help="results.json from `prove.py --report`")
    ap.add_argument("--out", default=str(OUT), help="output dir for LaTeX fragments")
    args = ap.parse_args()

    rep = json.loads(Path(args.report).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "macros.tex": macros(rep),
        "frontier.tex": frontier(rep),
        "headtohead.tex": head_to_head(rep),
        "coverage.tex": coverage(rep),
        "tasksuccess.tex": task_success(rep),
        "shift.tex": shift(rep),
    }
    for name, content in files.items():
        (out / name).write_text(content)
    print(f"wrote {len(files)} LaTeX fragments → {out}")
    print("the paper picks them up automatically (\\IfFileExists). Recompile docs/paper/main.tex.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
