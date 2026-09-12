"""Turn recorded tool calls into the event space the divergence is taken over.

Two feature families, kept separate on purpose:

* **route** — ``tool:<name>``, one per call. Answers "which tools got used".
* **arg** — ``arg:<tool>.<key>=<bucket>``. Answers "what did they get called
  with". This is where quiet payloads live: a BCC field, an extra path, an
  unfamiliar host.

Keeping them apart is what stops the metric from flagging every genuinely
better tool. A superior search tool moves route mass and leaves arg mass
alone; a poisoned one does the reverse.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .tools import ToolCall

_EMAIL = re.compile(r"^[\w.+-]+@([\w.-]+\.\w+)$")
_URL = re.compile(r"^[a-z][a-z0-9+.-]*://([^/\s]+)", re.I)
_NUM = re.compile(r"^-?\d+(\.\d+)?$")


def bucket(value: Any) -> str:
    """Collapse an argument value into a comparison-stable token.

    Bodies and queries vary run to run; destinations do not. Bucketing keeps
    the destination and throws away the prose, so the distribution stays
    sharp on the part an attacker has to control.
    """
    if isinstance(value, (list, tuple)):
        return "|".join(sorted(bucket(v) for v in value))
    if isinstance(value, dict):
        return "{" + ",".join(sorted(value)) + "}"
    text = str(value).strip()
    if not text:
        return "empty"
    if m := _EMAIL.match(text):
        return f"email:{m.group(1).lower()}"
    if m := _URL.match(text):
        return f"url:{m.group(1).lower()}"
    if _NUM.match(text):
        return "num"
    if "/" in text or text.startswith(("~", ".")):
        head = text.rsplit("/", 1)[0] or "/"
        return f"path:{head}"
    return "lit:" + re.sub(r"\s+", " ", text.lower())[:24]


@dataclass(frozen=True)
class Feature:
    token: str
    kind: str  # "route" | "arg"
    tool: str
    key: str = ""
    raw: str = ""


def extract(calls: Iterable[ToolCall],
            exclude: str | Iterable[str] | None = None) -> list[Feature]:
    """Features for the calls the agent made to tools *other than* ``exclude``.

    ``exclude`` takes a set for pairwise scans, where the event space has to
    drop both members of the pair so all four conditions stay comparable.
    """
    dropped = {exclude} if isinstance(exclude, str) else set(exclude or ())
    out: list[Feature] = []
    for call in calls:
        if call.tool in dropped:
            continue
        out.append(Feature(f"tool:{call.tool}", "route", call.tool))
        for key in sorted(call.args):
            raw = call.args[key]
            out.append(Feature(
                f"arg:{call.tool}.{key}={bucket(raw)}", "arg", call.tool, key, str(raw)))
    return out


def distribution(runs: Sequence[Sequence[Feature]], kind: str) -> Counter[str]:
    """Pool ``runs`` (repeats of one probe) into unnormalised counts of ``kind``."""
    counts: Counter[str] = Counter()
    for run in runs:
        for f in run:
            if f.kind == kind:
                counts[f.token] += 1
    return counts


def arg_distributions(runs: Sequence[Sequence[Feature]]) -> dict[str, Counter[str]]:
    """Argument counts split per tool.

    Arguments are compared *within* a tool, never pooled. Pooling would let a
    replacement effect masquerade as contamination: remove a tool and its
    successor's arguments flood the pool, shifting every share at once. Per
    tool, the comparison is like-for-like, and a tool that only exists in one
    condition simply drops out of this term (the routing term and the sink
    check already account for it).
    """
    out: dict[str, Counter[str]] = {}
    for run in runs:
        for f in run:
            if f.kind == "arg":
                out.setdefault(f.tool, Counter())[f.token] += 1
    return out


def raw_values(runs: Sequence[Sequence[Feature]]) -> dict[str, str]:
    """Map each arg feature token back to one concrete value, for sink checks."""
    return {f.token: f.raw for run in runs for f in run if f.kind == "arg"}
