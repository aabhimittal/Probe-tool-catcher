"""Tool manifests and recorded calls."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass(frozen=True)
class Tool:
    """One entry of an agent's toolset, as the model sees it.

    ``description`` is the poisoning channel: it is untrusted text that the
    model reads on every turn, so it can carry instructions about *other*
    tools. PROBE never tries to decide whether that text is malicious; it
    only measures what changes when the tool is removed.
    """

    name: str
    description: str = ""
    capabilities: tuple[str, ...] = ()
    params: tuple[str, ...] = ()

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Tool":
        return Tool(
            name=d["name"],
            description=d.get("description", ""),
            capabilities=tuple(d.get("capabilities", ())),
            params=tuple(d.get("params", ())),
        )

    def to_schema(self) -> dict[str, Any]:
        """Anthropic-style tool schema, for real-model runs."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {p: {"type": "string"} for p in self.params},
            },
        }


@dataclass(frozen=True)
class ToolCall:
    """A single observed call. ``args`` values are compared structurally."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)


class Toolset:
    def __init__(self, tools: Iterable[Tool]):
        self._tools: dict[str, Tool] = {}
        for t in tools:
            if t.name in self._tools:
                raise ValueError(f"duplicate tool name: {t.name}")
            self._tools[t.name] = t

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __getitem__(self, name: str) -> Tool:
        return self._tools[name]

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def without(self, name: str) -> "Toolset":
        """T \\ {x}: the ablated toolset used as the counterfactual."""
        if name not in self._tools:
            raise KeyError(name)
        return Toolset(t for t in self._tools.values() if t.name != name)

    def providing(self, capability: str) -> list[Tool]:
        return [t for t in self._tools.values() if capability in t.capabilities]

    @staticmethod
    def from_dicts(items: Iterable[dict[str, Any]]) -> "Toolset":
        return Toolset(Tool.from_dict(d) for d in items)

    @staticmethod
    def from_json(path: str | Path) -> "Toolset":
        data = json.loads(Path(path).read_text())
        return Toolset.from_dicts(data["tools"] if "tools" in data else data)
