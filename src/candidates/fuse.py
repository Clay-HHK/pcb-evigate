"""Fuse detector and difference candidates according to specification §3.6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def iou(box_a: list[float], box_b: list[float]) -> float:
    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    return intersection / (area_a + area_b - intersection)


def center_in_expanded(det_box: list[float], diff_box: list[float], expand: int) -> bool:
    cx = (det_box[0] + det_box[2]) / 2.0
    cy = (det_box[1] + det_box[3]) / 2.0
    return diff_box[0] - expand <= cx <= diff_box[2] + expand and diff_box[1] - expand <= cy <= diff_box[3] + expand


def union_box(box_a: list[float], box_b: list[float]) -> list[float]:
    return [min(box_a[0], box_b[0]), min(box_a[1], box_b[1]), max(box_a[2], box_b[2]), max(box_a[3], box_b[3])]


def fuse_candidates(pair_id: str, detections: list[dict], differences: list[dict], expand: int = 8) -> list[dict]:
    used_diff = set()
    output = []
    for det_index, det in enumerate(detections):
        matches = []
        for diff_index, diff in enumerate(differences):
            current_iou = iou(det["box"], diff["box"])
            if diff_index not in used_diff and (current_iou >= 0.3 or center_in_expanded(det["box"], diff["box"], expand)):
                if current_iou >= 0.1:
                    matches.append((diff_index, diff, current_iou))
        if matches:
            diff_index, diff, current_iou = max(matches, key=lambda item: (item[2], item[1]["score"]))
            used_diff.add(diff_index)
            output.append(
                {
                    "candidate_id": f"{pair_id}_b{len(output):02d}",
                    "source": "BOTH",
                    "box": union_box(det["box"], diff["box"]),
                    "action_box": det["box"],
                    "det_box": det["box"],
                    "diff_box": diff["box"],
                    "class": int(det["class"]),
                    "score": float(det["score"]),
                    "diff_score": float(diff["score"]),
                    "iou": float(current_iou),
                    "appearance_residual": diff["appearance_residual"],
                    "topology_vector": diff["topology_vector"],
                }
            )
        else:
            output.append(
                {
                    "candidate_id": f"{pair_id}_d{det_index:02d}",
                    "source": "DET",
                    "box": det["box"],
                    "action_box": det["box"],
                    "det_box": det["box"],
                    "class": int(det["class"]),
                    "score": float(det["score"]),
                }
            )
    for diff_index, diff in enumerate(differences):
        if diff_index not in used_diff:
            output.append(
                {
                    "candidate_id": f"{pair_id}_x{diff_index:02d}",
                    "source": "DIFF",
                    "box": diff["box"],
                    "action_box": diff["box"],
                    "diff_box": diff["box"],
                    "class": -1,
                    "score": float(diff["score"]),
                    "appearance_residual": diff["appearance_residual"],
                    "topology_vector": diff["topology_vector"],
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--differences", type=Path, default=None)
    parser.add_argument("--expand", type=int, default=8)
    args = parser.parse_args()
    prediction_path = args.predictions or Path(f"cache/detector_predictions/fold{args.fold}.jsonl")
    difference_path = args.differences or Path(f"cache/difference_candidates/fold{args.fold}.jsonl")
    detector_records = {item["pair_id"]: item for item in map(json.loads, prediction_path.read_text(encoding="utf-8").splitlines())}
    output_path = Path("cache/candidates")
    output_path.mkdir(parents=True, exist_ok=True)
    lines = []
    for line in difference_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        detections = [
            {"box": box, "class": class_id, "score": score}
            for box, class_id, score in zip(detector_records[item["pair_id"]]["boxes"], detector_records[item["pair_id"]]["classes"], detector_records[item["pair_id"]]["scores"])
        ]
        lines.append({"pair_id": item["pair_id"], "candidates": fuse_candidates(item["pair_id"], detections, item["candidates"], args.expand)})
    path = output_path / f"fold{args.fold}.jsonl"
    path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in lines) + "\n", encoding="utf-8")
    print(f"{path}: {sum(len(item['candidates']) for item in lines)}")


if __name__ == "__main__":
    main()
