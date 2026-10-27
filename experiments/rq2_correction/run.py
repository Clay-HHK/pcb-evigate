"""RQ2 detector thresholds, full correction, and oracle upper bounds."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.common import card_paths, prepare_smoke_candidates, write_json
from src.detector.calibrate import read_gt
from src.detector.dataset import rows_for_role
from src.evaluation.apply import apply_actions, write_outputs
from src.evaluation.metrics import correction_metrics, detection_metrics
from src.evaluation.oracle import candidate_oracle
from src.evaluation.utility import candidate_action_utility
from src.gates.acceptance_gate import AcceptanceGate
from src.gates.safety import accept_action
from src.mllm.action_map import map_action
from src.mllm.client import review_candidates


def _load_predictions(path: Path, pair_ids: set[str]) -> list[dict]:
    return [item for item in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if item["pair_id"] in pair_ids]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    candidate_records = prepare_smoke_candidates(200 if args.smoke else 800, ["A6"], complete_images=True)
    candidate_pair_ids = {record["row"]["pair_id"] for record in candidate_records}
    image_rows = [row for row in rows_for_role(0, "test") if row["pair_id"] in candidate_pair_ids]
    image_count = len(image_rows)
    pair_ids = {row["pair_id"] for row in image_rows}
    ground_truth = {
        row["pair_id"]: [{"box": box, "class": class_id} for box, class_id in read_gt(Path(row["anno_path"]))]
        for row in image_rows
    }
    b0 = _load_predictions(Path("cache/detector_predictions/fold0_lowconf.jsonl"), pair_ids)
    b1 = _load_predictions(Path("cache/detector_predictions/fold0.jsonl"), pair_ids)
    responses = review_candidates(
        [record["candidate"] for record in candidate_records],
        card_paths(candidate_records, "A6"),
        "A6",
        Path("configs/mllm/stub.yaml"),
    )
    gate_samples = []
    mapped = []
    for record, response in zip(candidate_records, responses):
        conclusion = response["result"]
        action = map_action(record["candidate"], conclusion)
        utility, gt_match_before, gt_match_after = candidate_action_utility(record, action, conclusion)
        sample = {
            "action": action,
            "candidate": record["candidate"],
            "conclusion": conclusion,
            "positive_utility": utility,
            "gt_match_before": gt_match_before,
            "gt_match_after": gt_match_after,
        }
        gate_samples.append(sample)
        mapped.append((record, response, sample))
    gate = AcceptanceGate(minimum_samples=30).fit(gate_samples)
    print(f"接受门训练样本数: {gate.training_counts}")
    print(f"接受门各动作正负样本数: {gate.label_counts}")
    utility_counts = {
        "true": sum(int(sample["positive_utility"]) for sample in gate_samples),
        "false": sum(int(not sample["positive_utility"]) for sample in gate_samples),
    }
    print(f"positive_utility 数量: {utility_counts}")
    grouped_candidates: dict[str, list[dict]] = defaultdict(list)
    grouped_b7: dict[str, list[dict]] = defaultdict(list)
    grouped_full: dict[str, list[dict]] = defaultdict(list)
    for record, response, sample in mapped:
        candidate = record["candidate"]
        action = sample["action"]
        probability = gate.predict_probability(action, candidate, sample["conclusion"]) if action in {"remove", "relabel", "add"} else 0.0
        base = {
            "action": action,
            "conclusion": sample["conclusion"],
            "accept_prob": probability,
            "positive_utility": sample["positive_utility"],
            "gt_match_before": sample["gt_match_before"],
            "gt_match_after": sample["gt_match_after"],
            "mllm_called": True,
            "latency_seconds": response["latency_seconds"],
        }
        grouped_candidates[candidate["pair_id"]].append(candidate)
        accepted = (
            accept_action(action, candidate, sample["conclusion"], probability)
            if action in {"remove", "relabel", "add"}
            else False
        )
        grouped_b7[candidate["pair_id"]].append({**base, "accepted": accepted})
        grouped_full[candidate["pair_id"]].append({**base, "accepted": accepted})
    b7, full, audits, oracle_predictions = [], [], [], []
    oracle_action_counts: defaultdict[str, int] = defaultdict(int)
    for prediction in b1:
        candidates = grouped_candidates[prediction["pair_id"]]
        b7_record, _ = apply_actions(prediction, candidates, grouped_b7[prediction["pair_id"]])
        full_record, full_audit = apply_actions(prediction, candidates, grouped_full[prediction["pair_id"]])
        b7.append(b7_record)
        full.append(full_record)
        audits.extend(full_audit)
        oracle_record, pair_oracle_counts = candidate_oracle(
            prediction, candidates, ground_truth[prediction["pair_id"]]
        )
        oracle_predictions.append(oracle_record)
        for action, count in pair_oracle_counts.items():
            oracle_action_counts[action] += count
    original_counts = {"TP": 258, "错分类": 12, "FP": 796, "漏检": 1587}
    b1_metrics = detection_metrics(b1, ground_truth)
    b7_metrics = detection_metrics(b7, ground_truth)
    oracle_metrics = detection_metrics(oracle_predictions, ground_truth)
    action_names = ("keep", "remove", "relabel", "add", "reject", "defer")
    action_counter = Counter(sample["action"] for _, _, sample in mapped)
    action_counts = {name: action_counter[name] for name in action_names}
    presence_counter = Counter(
        "true" if sample["conclusion"]["defect_present"] is True
        else "false" if sample["conclusion"]["defect_present"] is False
        else "uncertain"
        for _, _, sample in mapped
    )
    source_counter = Counter(record["candidate"]["source"] for record in candidate_records)
    unreachable_action_reasons = {}
    if source_counter["DIFF"] == 0:
        unreachable_action_reasons = {
            "add": "当前 smoke 候选集中没有 DIFF 来源候选，无法触发有缺陷 DIFF 的 add 映射",
            "reject": "当前 smoke 候选集中没有 DIFF 来源候选，无法触发无缺陷 DIFF 的 reject 映射",
        }
    accepted_audits = [item for item in audits if item["accepted"]]
    accepted_positive = sum(int(item["positive_utility"]) for item in accepted_audits)
    error_transfer = {}
    for selected_action in ("remove", "relabel", "add"):
        action_predictions = []
        selected_audits = []
        for prediction in b1:
            pair_id = prediction["pair_id"]
            decisions = [
                {**decision, "accepted": decision["accepted"] and decision["action"] == selected_action}
                for decision in grouped_b7[pair_id]
            ]
            corrected, pair_audits = apply_actions(prediction, grouped_candidates[pair_id], decisions)
            action_predictions.append(corrected)
            selected_audits.extend(pair_audits)
        action_f1 = detection_metrics(action_predictions, ground_truth)["f1"]
        selected_accepted_audits = [item for item in selected_audits if item["accepted"]]
        error_transfer[selected_action] = {
            "accepted": len(selected_accepted_audits),
            "positive": sum(int(item["positive_utility"]) for item in selected_accepted_audits),
            "harmful": sum(int(not item["positive_utility"]) for item in selected_accepted_audits),
            "f1": action_f1,
            "f1_delta_vs_B1": action_f1 - b1_metrics["f1"],
        }
    result = {
        "passed": (
            len(b0) == image_count
            and len(b1) == image_count
            and presence_counter["true"] > 0
            and presence_counter["false"] > 0
            and len(accepted_audits) > 0
            and all(action_counts[name] > 0 or name in unreachable_action_reasons for name in action_names)
        ),
        "images": image_count,
        "candidates": len(candidate_records),
        "candidate_sources": dict(source_counter),
        "defect_present_counts": dict(presence_counter),
        "action_counts": action_counts,
        "unreachable_action_reasons": unreachable_action_reasons,
        "accepted_actions": len(accepted_audits),
        "accepted_positive_utility": accepted_positive,
        "acceptance_training_samples": gate.training_counts,
        "acceptance_training_labels": gate.label_counts,
        "positive_utility_counts": utility_counts,
        "B0": detection_metrics(b0, ground_truth),
        "B1": b1_metrics,
        "B7": b7_metrics,
        "Full": detection_metrics(full, ground_truth),
        "candidate_oracle": oracle_metrics,
        "candidate_oracle_actions": dict(oracle_action_counts),
        "acceptance_upper_bound": sum(int(sample["positive_utility"]) for _, _, sample in mapped),
        "error_transfer": error_transfer,
        "correction": correction_metrics(audits, original_counts),
    }
    print(f"候选上界 F1: B1={b1_metrics['f1']:.6f}, Candidate Oracle={oracle_metrics['f1']:.6f}")
    print(f"B7 对比 B1: B1={b1_metrics['f1']:.6f}, B7={b7_metrics['f1']:.6f}")
    print(f"defect_present 分布: {dict(presence_counter)}")
    print(f"六类提议动作数量: {action_counts}")
    print(f"接受动作数: {len(accepted_audits)}，其中正效用: {accepted_positive}")
    for reason in unreachable_action_reasons.values():
        print(f"不可达说明: {reason}")
    if b7_metrics["f1"] < b1_metrics["f1"]:
        print(f"B7 错误转移分析: {error_transfer}")
    write_outputs(full, audits, Path("experiments/rq2_correction/smoke_outputs"))
    output = write_json(Path("experiments/rq2_correction/smoke_results.json"), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"产出文件: {output}")


if __name__ == "__main__":
    main()
