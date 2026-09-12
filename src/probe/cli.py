"""Command line entry point."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agents import SimulatedAgent
from .probes import DEFAULT_SUITE, expand, load_suite
from .report import render, render_pairs
from .scan import scan, scan_pairs
from .tools import Toolset


def _load(path: Path) -> tuple[Toolset, list]:
    data = json.loads(path.read_text())
    toolset = Toolset.from_dicts(data["tools"])
    probes = load_suite(data["probes"]) if data.get("probes") else DEFAULT_SUITE
    return toolset, probes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="probe",
        description="Catch poisoned tools by their causal effect on other tools.")
    parser.add_argument("config", type=Path,
                        help="JSON file with a 'tools' list and optional 'probes'")
    parser.add_argument("-s", "--suspect", action="append", default=None,
                        help="tool to ablate (repeatable; default: every tool)")
    parser.add_argument("-r", "--repeats", type=int, default=1,
                        help="runs per probe per condition; >1 measures a noise floor")
    parser.add_argument("-a", "--alpha", type=float, default=0.5,
                        help="additive smoothing (lower = unseen features count more)")
    parser.add_argument("--arg-threshold", type=float, default=0.05)
    parser.add_argument("--route-threshold", type=float, default=0.05)
    parser.add_argument("--agent", choices=("sim", "anthropic"), default="sim",
                        help="'sim' is deterministic and offline; 'anthropic' calls a model")
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("-n", "--variants", type=int, default=1,
                        help="input variations per probe, drawn from its "
                             "'variants' pools; raises the cost of a payload "
                             "conditioned on one recipient or date")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for variant sampling")
    parser.add_argument("--pairs", action="store_true",
                        help="also ablate tool pairs, to catch payloads that "
                             "two tools carry redundantly")
    parser.add_argument("--interaction-threshold", type=float, default=0.05)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--fail-on-flag", action="store_true",
                        help="exit 1 if any tool is flagged (for CI gating)")
    args = parser.parse_args(argv)

    toolset, probes = _load(args.config)
    if args.agent == "anthropic":
        from .agents import AnthropicAgent  # noqa: PLC0415 - optional dependency
        agent = AnthropicAgent(model=args.model)
    else:
        agent = SimulatedAgent()

    probes = expand(probes, args.variants, seed=args.seed)
    result = scan(toolset, probes, agent=agent, suspects=args.suspect,
                  repeats=args.repeats, alpha=args.alpha,
                  arg_threshold=args.arg_threshold,
                  route_threshold=args.route_threshold)
    pairs = scan_pairs(toolset, probes, agent=agent, suspects=args.suspect,
                       repeats=args.repeats, alpha=args.alpha,
                       interaction_threshold=args.interaction_threshold
                       ) if args.pairs else None

    if args.json:
        payload = result.to_dict()
        if pairs is not None:
            payload["pairwise"] = pairs.to_dict()
        print(json.dumps(payload, indent=2))
    else:
        print(render(result, verbose=args.verbose))
        if pairs is not None:
            print("\n" + render_pairs(pairs, verbose=args.verbose))

    flagged = bool(result.flagged()) or bool(pairs and pairs.flagged())
    return 1 if args.fail_on_flag and flagged else 0


if __name__ == "__main__":
    sys.exit(main())
