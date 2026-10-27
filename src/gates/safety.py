"""Hard safety constraints applied after the learned acceptance gate."""

from __future__ import annotations

from src.gates.features import topology_compatibility


ACTION_THRESHOLDS = {"remove": 0.90, "relabel": 0.75, "add": 0.80}


def accept_action(
    action: str,
    candidate: dict,
    conclusion: dict,
    accept_probability: float,
    json_valid: bool = True,
    tau_reg: float = 0.5,
) -> bool:
    if action not in ACTION_THRESHOLDS or not json_valid:
        return False
    if float(candidate.get("q_reg", 1.0)) < tau_reg:
        return False
    if accept_probability < ACTION_THRESHOLDS[action]:
        return False
    if action == "remove":
        return candidate["source"] in {"DET", "BOTH"}
    if action == "relabel":
        return candidate["source"] in {"DET", "BOTH"} and topology_compatibility(conclusion) >= 1.0
    return candidate["source"] == "DIFF"
