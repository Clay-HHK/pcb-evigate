"""Map independent MLLM conclusions to system-side candidate actions."""

from __future__ import annotations


CLASS_NAMES = ["open", "short", "mousebite", "spur", "pinhole", "spurious_copper"]
CLASS_TO_TOPOLOGY = {
    "open": {"disconnection", "more_endpoints"},
    "short": {"new_connection"},
    "mousebite": {"none"},
    "spur": {"extra_branch"},
    "pinhole": {"hole_change"},
    "spurious_copper": {"new_component"},
}


def _supports_relabel(detector_class: str, predicted_class: str, reported_patterns: list[str]) -> bool:
    patterns = set(reported_patterns) - {"uncertain"}
    predicted_consistent = bool(patterns & CLASS_TO_TOPOLOGY[predicted_class])
    detector_contradicted = not bool(patterns & CLASS_TO_TOPOLOGY[detector_class])
    return predicted_consistent and detector_contradicted


def map_action(candidate: dict, conclusion: dict) -> str:
    source = candidate["source"]
    predicted_class = conclusion["predicted_class"]
    present = conclusion["defect_present"]
    if source in {"DET", "BOTH"}:
        if present is False or predicted_class == "normal":
            return "remove"
        if present is True and predicted_class in CLASS_NAMES:
            detector_class = CLASS_NAMES[int(candidate["class"])]
            if detector_class == predicted_class:
                return "keep"
            return "relabel" if _supports_relabel(detector_class, predicted_class, conclusion["topology_pattern"]) else "keep"
        return "defer"
    if present is True and predicted_class in CLASS_NAMES:
        return "add"
    if present is False or predicted_class == "normal":
        return "reject"
    return "defer"
