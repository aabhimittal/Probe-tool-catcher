"""Smoothed KL divergence between two empirical call distributions."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Divergence:
    total: float
    per_feature: dict[str, float]

    def top(self, n: int = 5) -> list[tuple[str, float]]:
        return sorted(self.per_feature.items(), key=lambda kv: -abs(kv[1]))[:n]


def kl(p: Counter[str], q: Counter[str], alpha: float = 0.5) -> Divergence:
    """D_KL(P ‖ Q) with additive smoothing over the union of the supports.

    Smoothing is not cosmetic here. A poisoned tool's signature feature has
    zero mass under the ablation, which is exactly the unsmoothed infinity
    case; ``alpha`` turns that into a large finite number that stays
    comparable across probes. Smaller ``alpha`` means "a never-seen feature
    is more surprising", so it raises the score of one-off payloads.
    """
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    support = sorted(set(p) | set(q))
    if not support:
        return Divergence(0.0, {})
    k = len(support)
    np_, nq = sum(p.values()), sum(q.values())
    dp, dq = np_ + alpha * k, nq + alpha * k
    per: dict[str, float] = {}
    for token in support:
        pi = (p[token] + alpha) / dp
        qi = (q[token] + alpha) / dq
        per[token] = pi * math.log(pi / qi)
    return Divergence(sum(per.values()), per)


def jsd(p: Counter[str], q: Counter[str], alpha: float = 0.5) -> float:
    """Symmetric, bounded alternative — useful when reporting to humans."""
    m: Counter[str] = Counter()
    for token in set(p) | set(q):
        m[token] = p[token] + q[token]
    return 0.5 * kl(p, m, alpha).total + 0.5 * kl(q, m, alpha).total
