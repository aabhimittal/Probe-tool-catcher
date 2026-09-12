"""Human-readable rendering of a scan."""
from __future__ import annotations

from .scan import (CLEAN, POISONED, ROUTING_SHIFT, SUSPECT_ARGS, UNTESTED,
                   ScanResult)

_NOTE = {
    POISONED: "calls to other tools gained a destination the task never asked for",
    SUSPECT_ARGS: "arguments to other tools changed; no unsanctioned sink found",
    ROUTING_SHIFT: "tool selection moved, arguments did not — expected of a better tool",
    CLEAN: "no measurable effect on the other tools",
    UNTESTED: "no probe in the suite can run without this tool — add one that can",
}


def render(result: ScanResult, verbose: bool = False) -> str:
    rows = [("TOOL", "VERDICT", "I(x)", "route", "arg")]
    rows += [(v.tool, v.verdict, f"{v.influence:.3f}", f"{v.i_route:.3f}",
              f"{v.i_arg:.3f}") for v in result.verdicts]
    widths = [max(len(r[i]) for r in rows) for i in range(5)]
    lines = [f"PROBE scan — {len(result.verdicts)} tool(s), "
             f"repeats={result.repeats}, alpha={result.alpha}", ""]
    for i, row in enumerate(rows):
        lines.append("  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip())
        if i == 0:
            lines.append("  ".join("-" * w for w in widths))
    for verdict in result.verdicts:
        if verdict.verdict == CLEAN and not verbose:
            continue
        lines += ["", f"{verdict.tool}: {verdict.verdict} — {_NOTE[verdict.verdict]}"]
        lines += [f"    {e}" for e in verdict.evidence] or ["    (no per-feature evidence)"]
        if verdict.skipped:
            lines.append("    probes skipped as uninformative: "
                         + ", ".join(verdict.skipped))
        if verbose:
            for p in verdict.probes:
                lines.append(
                    f"    probe {p.probe_id}: route={p.route:.3f} arg={p.arg:.3f} "
                    f"floor={p.floor_route + p.floor_arg:.3f}")
    flagged = result.flagged()
    lines += ["", f"{len(flagged)} tool(s) flagged: "
                  f"{', '.join(v.tool for v in flagged) or 'none'}"]
    return "\n".join(lines)
