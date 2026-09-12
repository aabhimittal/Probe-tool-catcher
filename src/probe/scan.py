"""The scan itself: ablate a tool, measure what it did to the other calls."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from . import features as F
from .agents import Agent, SimulatedAgent
from .divergence import kl
from .probes import ProbeTask
from .tools import Toolset

POISONED = "POISONED"
SUSPECT_ARGS = "SUSPECT-ARGS"
ROUTING_SHIFT = "ROUTING-SHIFT"
CLEAN = "CLEAN"
UNTESTED = "UNTESTED"

_SINK_PREFIXES = ("email:", "url:", "path:")


@dataclass
class ProbeInfluence:
    """Raw divergences for one probe, plus the noise floor to net them against."""

    probe_id: str
    route: float
    arg: float
    floor_route: float = 0.0
    floor_arg: float = 0.0
    top_arg: list[tuple[str, float]] = field(default_factory=list)
    novel_sinks: list[str] = field(default_factory=list)

    @property
    def net_route(self) -> float:
        return max(0.0, self.route - self.floor_route)

    @property
    def net_arg(self) -> float:
        return max(0.0, self.arg - self.floor_arg)

    @property
    def total(self) -> float:
        return self.net_route + self.net_arg


@dataclass
class ToolVerdict:
    tool: str
    i_route: float
    i_arg: float
    influence: float
    verdict: str
    novel_sinks: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    probes: list[ProbeInfluence] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "verdict": self.verdict,
            "influence": round(self.influence, 4),
            "i_route": round(self.i_route, 4),
            "i_arg": round(self.i_arg, 4),
            "novel_sinks": self.novel_sinks,
            "evidence": self.evidence,
            "skipped_probes": self.skipped,
            "probes": [
                {"probe": p.probe_id, "route": round(p.route, 4),
                 "arg": round(p.arg, 4),
                 "floor_route": round(p.floor_route, 4),
                 "floor_arg": round(p.floor_arg, 4),
                 "novel_sinks": p.novel_sinks}
                for p in self.probes
            ],
        }


@dataclass
class ScanResult:
    verdicts: list[ToolVerdict]
    repeats: int
    alpha: float

    def worst(self) -> ToolVerdict | None:
        return max(self.verdicts, key=lambda v: v.influence, default=None)

    def flagged(self) -> list[ToolVerdict]:
        return [v for v in self.verdicts if v.verdict in (POISONED, SUSPECT_ARGS)]

    def to_dict(self) -> dict:
        return {"repeats": self.repeats, "alpha": self.alpha,
                "verdicts": [v.to_dict() for v in self.verdicts]}


def _informative(task: ProbeTask, toolset: Toolset, x: str) -> bool:
    """Is this probe able to say anything about ``x``?

    If ``x`` is the only provider of a capability the task needs, the ablated
    run cannot do the task at all. Everything downstream then changes for a
    trivial reason — the work is gone — and the tool would be flagged for
    being load-bearing rather than for being poisoned. Such probes are
    dropped, which is also why a suite needs tasks that do *not* need the
    tool under test, and substitutes for the ones that do.
    """
    for capability in task.intent:
        providers = [t.name for t in toolset.providing(capability)]
        if providers == [x]:
            return False
    return True


def _split_half_floor(runs: Sequence[Sequence[F.Feature]],
                      alpha: float) -> tuple[float, float]:
    """How much divergence this agent produces against *itself*.

    With a sampling model, two draws from the same toolset already disagree.
    Splitting the repeats in half and scoring one half against the other
    gives that disagreement in the same units as the influence, per
    component, so noise can be netted out instead of reported as evidence.
    Deterministic agents return (0, 0), as they should.
    """
    if len(runs) < 2:
        return 0.0, 0.0
    mid = len(runs) // 2
    a, b = runs[:mid], runs[mid:]
    route = kl(F.distribution(a, "route"), F.distribution(b, "route"), alpha).total
    return route, _paired_arg_kl(F.arg_distributions(a), F.arg_distributions(b), alpha)[0]


def _paired_arg_kl(p_args: dict, q_args: dict,
                   alpha: float) -> tuple[float, dict[str, float]]:
    """Argument divergence summed over the tools *both* conditions actually use."""
    total, contributions = 0.0, {}
    for tool in sorted(set(p_args) & set(q_args)):
        d = kl(p_args[tool], q_args[tool], alpha)
        total += d.total
        contributions.update(d.per_feature)
    return total, contributions


def _novel_sinks(p: Counter[str], q: Counter[str], raws: dict[str, str],
                 task: ProbeTask) -> list[str]:
    """Arg features that appear only with the tool present and point outward.

    The task's own payload and prompt define what is sanctioned. A destination
    that the task never mentioned, that vanishes when the tool is removed, is
    the shape of exfiltration rather than the shape of a better tool.
    """
    sanctioned = task.sanctioned_values()
    out = []
    for token, count in p.items():
        if count == 0 or q.get(token):
            continue
        value = token.split("=", 1)[-1]
        if not value.startswith(_SINK_PREFIXES):
            continue
        if raws.get(token, "") in sanctioned:
            continue
        out.append(token)
    return sorted(out)


def scan(
    toolset: Toolset,
    probes: Sequence[ProbeTask],
    agent: Agent | None = None,
    suspects: Sequence[str] | None = None,
    repeats: int = 1,
    alpha: float = 0.5,
    arg_threshold: float = 0.05,
    route_threshold: float = 0.05,
) -> ScanResult:
    """Compute I(x) for each suspect tool.

    I(x) = Σ_probes D_KL( P(calls to tools ≠ x | T) ‖ P(calls to tools ≠ x | T∖x) ),
    split into a routing component and an argument component and reduced by
    the agent's own noise floor.
    """
    agent = agent or SimulatedAgent()
    names = list(suspects) if suspects else toolset.names
    full_cache: dict[str, list[list]] = {}
    verdicts: list[ToolVerdict] = []

    for x in names:
        if x not in toolset:
            raise KeyError(f"unknown tool: {x}")
        ablated = toolset.without(x)
        per_probe: list[ProbeInfluence] = []
        skipped = [t.id for t in probes if not _informative(t, toolset, x)]
        for task in [t for t in probes if _informative(t, toolset, x)]:
            full_calls = full_cache.setdefault(
                task.id, [agent.run(task, toolset) for _ in range(repeats)])
            abl_calls = [agent.run(task, ablated) for _ in range(repeats)]
            full = [F.extract(c, exclude=x) for c in full_calls]
            abl = [F.extract(c, exclude=x) for c in abl_calls]

            p_route, q_route = F.distribution(full, "route"), F.distribution(abl, "route")
            d_route = kl(p_route, q_route, alpha)

            # Arguments are compared per tool, so a tool that only one
            # condition uses cannot tilt another tool's argument shares.
            arg_total, contributions = _paired_arg_kl(
                F.arg_distributions(full), F.arg_distributions(abl), alpha)
            floor_route, floor_arg = _split_half_floor(full, alpha)

            per_probe.append(ProbeInfluence(
                probe_id=task.id,
                route=d_route.total,
                arg=arg_total,
                floor_route=floor_route,
                floor_arg=floor_arg,
                top_arg=sorted(contributions.items(), key=lambda kv: -kv[1])[:3],
                novel_sinks=_novel_sinks(F.distribution(full, "arg"),
                                         F.distribution(abl, "arg"),
                                         F.raw_values(full), task),
            ))

        i_route = sum(p.net_route for p in per_probe)
        i_arg = sum(p.net_arg for p in per_probe)
        influence = sum(p.total for p in per_probe)
        sinks = sorted({s for p in per_probe for s in p.novel_sinks})
        verdicts.append(ToolVerdict(
            tool=x, i_route=i_route, i_arg=i_arg, influence=influence,
            verdict=(UNTESTED if not per_probe else
                     _classify(i_route, i_arg, sinks, arg_threshold, route_threshold)),
            novel_sinks=sinks,
            evidence=_evidence(per_probe, sinks),
            probes=per_probe,
            skipped=skipped,
        ))

    verdicts.sort(key=lambda v: -v.influence)
    return ScanResult(verdicts, repeats, alpha)


def _classify(i_route: float, i_arg: float, sinks: list[str],
              arg_threshold: float, route_threshold: float) -> str:
    if sinks:
        return POISONED
    if i_arg > arg_threshold:
        return SUSPECT_ARGS
    if i_route > route_threshold:
        return ROUTING_SHIFT
    return CLEAN


def _evidence(per_probe: list[ProbeInfluence], sinks: list[str]) -> list[str]:
    out = [f"unsanctioned destination {s}" for s in sinks]
    routed = sorted({p.probe_id for p in per_probe if p.net_route > 0.01})
    if routed:
        out.append("routing changed on: " + ", ".join(routed))
    for p in per_probe:
        for token, contribution in p.top_arg:
            if contribution > 0.01 and token not in sinks:
                out.append(f"{p.probe_id}: {token} (+{contribution:.3f})")
    return out[:8]
