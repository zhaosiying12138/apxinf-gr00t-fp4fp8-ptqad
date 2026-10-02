"""Conditional paired inference for the four fixed W4A4 paper contrasts.

No model loading, result selection, or changes to experiment artifacts. The
caller must supply the complete, identity-verified five-arm comparison.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

PLAN = Path(__file__).with_name("analysis_plan_w4a4.json")


def holm_adjust(pvalues):
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted, previous = {}, 0.0
    for index, name in enumerate(ordered):
        p = pvalues[name]
        if type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("p-values must be finite probabilities")
        previous = max(previous, min(1.0, (len(ordered) - index) * p))
        adjusted[name] = previous
    return adjusted


def quantile(values, p):
    rank = (len(values) - 1) * p
    left = math.floor(rank)
    right = math.ceil(rank)
    return values[left] + (values[right] - values[left]) * (rank - left)


def paired_effect(left, right, *, replicates=20000, seed=20261002, level=.95):
    """Bootstrap paired differences within each fixed task, then task-average.

    Resampling {-1,0,+1} counts is exactly equivalent to resampling paired
    episode records for a success-rate difference. Arms are never resampled
    separately, and tasks are not treated as draws from an unseen-task set.
    """
    if not left or len(left) != len(right):
        raise ValueError("paired episode lists must have equal positive length")
    if type(replicates) is not int or replicates < 2 or not 0 < level < 1:
        raise ValueError("invalid bootstrap settings")
    differences, identities = {}, set()
    gains = losses = 0
    for a, b in zip(left, right):
        if type(a.get("success")) is not bool or type(b.get("success")) is not bool:
            raise ValueError("outcomes must be Boolean")
        if {k: v for k, v in a.items() if k != "success"} != {
                k: v for k, v in b.items() if k != "success"}:
            raise ValueError("episode identities do not pair")
        key = (a.get("task"), a.get("episode_index"))
        if not isinstance(key[0], str) or type(key[1]) is not int or key in identities:
            raise ValueError("missing or duplicate task/episode identity")
        identities.add(key)
        difference = int(b["success"]) - int(a["success"])
        differences.setdefault(key[0], []).append(difference)
        gains += difference == 1
        losses += difference == -1
    groups = [differences[task] for task in sorted(differences)]
    # The protocol balances tasks. Otherwise pooled McNemar and a task-macro
    # difference would test different weightings of the task effects.
    if len({len(group) for group in groups}) != 1:
        raise ValueError("fixed-task inference requires balanced episode counts")
    observed = sum(sum(group) / len(group) for group in groups) / len(groups)
    strata = [(len(group), Counter(group)) for group in groups]
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        total = 0.0
        for count, frequencies in strata:
            total += sum(rng.choices((-1, 0, 1), weights=[frequencies[v] for v in (-1, 0, 1)],
                                     k=count)) / count
        draws.append(100 * total / len(strata))
    draws.sort()
    tail = (1 - level) / 2
    discordant = gains + losses
    p = min(1.0, 2 * sum(math.comb(discordant, i) for i in range(min(gains, losses) + 1))
            / (2 ** discordant)) if discordant else 1.0
    return {
        "difference_pp": 100 * observed,
        "pointwise_ci_pp": [quantile(draws, tail), quantile(draws, 1 - tail)],
        "paired_episodes": len(left), "tasks": len(groups),
        "only_treatment_success": gains, "only_baseline_success": losses,
        "discordant_episodes": discordant, "exact_mcnemar_two_sided_p": p,
        "zero_discordance_warning": discordant == 0,
    }


def analyze(comparison):
    for key in ("environment_pairing_verified", "protocol_consistency_verified", "source_accounting_verified"):
        if comparison.get(key) is not True:
            raise ValueError("complete verified comparison required: " + key)
    plan_bytes = PLAN.read_bytes()
    plan = json.loads(plan_bytes)
    settings = plan["confidence_interval"]
    reports = {}
    for name, (baseline, treatment) in plan["contrasts"].items():
        reports[name] = {"baseline": baseline, "treatment": treatment, **paired_effect(
            comparison["arms"][baseline]["episodes"], comparison["arms"][treatment]["episodes"],
            replicates=settings["replicates"], seed=settings["seed"], level=settings["level"])}
    adjusted = holm_adjust({name: row["exact_mcnemar_two_sided_p"] for name, row in reports.items()})
    for name, row in reports.items():
        row["holm_adjusted_p"] = adjusted[name]
    return {"format": "w4a4_paired_uncertainty_v1",
            "analysis_plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "confidence_interval": settings, "tests": plan["tests"],
            "contrasts": reports, "interpretation": plan["interpretation"]}
