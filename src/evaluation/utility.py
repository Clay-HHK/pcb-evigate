"""Ground-truth matching and candidate-action utility labels."""

from __future__ import annotations

from src.detector.calibrate import box_iou
from src.mllm.action_map import CLASS_NAMES


def prediction_gt_matches(prediction: dict, ground_truth: list[dict]) -> dict[int, dict]:
    """Return the class-aware one-to-one TP matches used by detection F1."""
    matched_gt: set[int] = set()
    matches: dict[int, dict] = {}
    ranked = sorted(range(len(prediction["boxes"])), key=lambda index: -float(prediction["scores"][index]))
    for prediction_index in ranked:
        candidates = [
            (gt_index, box_iou(prediction["boxes"][prediction_index], gt["box"]))
            for gt_index, gt in enumerate(ground_truth)
            if gt_index not in matched_gt
            and int(prediction["classes"][prediction_index]) == int(gt["class"])
        ]
        gt_index, iou = max(candidates, key=lambda item: item[1]) if candidates else (-1, 0.0)
        if iou >= 0.5:
            matched_gt.add(gt_index)
            matches[prediction_index] = {
                "gt_index": gt_index,
                "iou": float(iou),
                "class_correct": True,
            }
    return matches


def spatial_gt_match(box: list[float], class_id: int, ground_truth: list[dict]) -> dict | None:
    """Return the best IoU>=0.5 GT match, retaining class correctness separately."""
    candidates = [(gt_index, box_iou(box, gt["box"])) for gt_index, gt in enumerate(ground_truth)]
    gt_index, iou = max(candidates, key=lambda item: item[1]) if candidates else (-1, 0.0)
    if iou < 0.5:
        return None
    return {
        "gt_index": gt_index,
        "iou": float(iou),
        "class_correct": int(class_id) == int(ground_truth[gt_index]["class"]),
    }


def candidate_detection_index(record: dict) -> int:
    candidate = record["candidate"]
    target = candidate["det_box"]
    return max(
        range(len(record["prediction"]["boxes"])),
        key=lambda index: box_iou(record["prediction"]["boxes"][index], target),
    )


def candidate_action_utility(record: dict, action: str, conclusion: dict) -> tuple[bool, dict | None, dict | None]:
    """Label an action from GT and return its before/after candidate matches."""
    candidate = record["candidate"]
    ground_truth = record["ground_truth"]
    prediction = record["prediction"]
    tp_matches = prediction_gt_matches(prediction, ground_truth)
    covered_gt = {match["gt_index"] for match in tp_matches.values()}

    before = None
    detection_index = None
    if candidate["source"] in {"DET", "BOTH"}:
        detection_index = candidate_detection_index(record)
        before = spatial_gt_match(
            candidate["det_box"],
            int(prediction["classes"][detection_index]),
            ground_truth,
        )

    after = before
    if action == "remove":
        after = None
    elif action in {"relabel", "add"}:
        action_box = candidate["diff_box"] if action == "add" else candidate["action_box"]
        class_id = CLASS_NAMES.index(conclusion["predicted_class"])
        after = spatial_gt_match(action_box, class_id, ground_truth)

    if action == "keep":
        utility = detection_index in tp_matches
    elif action == "remove":
        utility = detection_index not in tp_matches
    elif action == "relabel":
        utility = after is not None and bool(after["class_correct"])
    elif action == "add":
        utility = after is not None and bool(after["class_correct"]) and after["gt_index"] not in covered_gt
    else:
        utility = False
    return bool(utility), before, after
