"""Candidate-oracle upper bound on a fixed candidate set."""

from __future__ import annotations

from collections import Counter

from src.evaluation.apply import apply_actions
from src.evaluation.utility import candidate_detection_index, prediction_gt_matches, spatial_gt_match
from src.mllm.action_map import CLASS_NAMES


def candidate_oracle(
    prediction: dict, candidates: list[dict], ground_truth: list[dict]
) -> tuple[dict, dict[str, int]]:
    """Use GT to choose the best available action for every candidate."""
    record = {"prediction": prediction, "ground_truth": ground_truth}
    tp_matches = prediction_gt_matches(prediction, ground_truth)
    covered_gt = {match["gt_index"] for match in tp_matches.values()}
    decisions = []
    action_counts: Counter[str] = Counter()

    for candidate in candidates:
        candidate_record = {**record, "candidate": candidate}
        source = candidate["source"]
        if source in {"DET", "BOTH"}:
            detection_index = candidate_detection_index(candidate_record)
            match = spatial_gt_match(
                candidate["det_box"], int(prediction["classes"][detection_index]), ground_truth
            )
            if detection_index in tp_matches:
                action = "keep"
                predicted_class = CLASS_NAMES[int(prediction["classes"][detection_index])]
            elif match is not None and match["gt_index"] not in covered_gt:
                action = "relabel"
                predicted_class = CLASS_NAMES[int(ground_truth[match["gt_index"]]["class"])]
                covered_gt.add(match["gt_index"])
            else:
                action = "remove"
                predicted_class = CLASS_NAMES[int(prediction["classes"][detection_index])]
        else:
            match = spatial_gt_match(candidate["diff_box"], 0, ground_truth)
            if match is not None and match["gt_index"] not in covered_gt:
                action = "add"
                predicted_class = CLASS_NAMES[int(ground_truth[match["gt_index"]]["class"])]
                covered_gt.add(match["gt_index"])
            else:
                action = "reject"
                predicted_class = "normal"
        action_counts[action] += 1
        decisions.append(
            {
                "action": action,
                "conclusion": {"predicted_class": predicted_class},
                "accept_prob": 1.0,
                "accepted": action in {"remove", "relabel", "add"},
            }
        )

    corrected, _ = apply_actions(prediction, candidates, decisions)
    return corrected, dict(action_counts)
