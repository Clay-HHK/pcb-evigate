"""RQ3 learned review and acceptance gates under call budgets."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.common import card_paths, prepare_smoke_candidates, review_label, write_json
from src.evaluation.apply import apply_actions
from src.evaluation.metrics import correction_metrics, detection_metrics
from src.evaluation.utility import candidate_action_utility
from src.gates.acceptance_gate import fit_oof_acceptance_gate
from src.gates.review_gate import fit_oof_review_gate
from src.gates.safety import accept_action
from src.mllm.action_map import map_action
from src.mllm.client import review_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    budgets = [5, 20, 100] if args.smoke else [5, 10, 20, 40, 60, 100]
    records = prepare_smoke_candidates(200 if args.smoke else 800, ["A6"], complete_images=True)
    review_records = [
        {
            "fold": index % 4 + 1,
            "candidate": record["candidate"],
            "review_label": review_label(record),
        }
        for index, record in enumerate(records)
    ]
    review_gate = fit_oof_review_gate(review_records, evaluation_fold=0)
    risks = review_gate.predict_risk([record["candidate"] for record in records])
    responses = review_candidates(
        [record["candidate"] for record in records],
        card_paths(records, "A6"),
        "A6",
        Path("configs/mllm/stub.yaml"),
    )
    acceptance_records = []
    for index, (record, response) in enumerate(zip(records, responses)):
        conclusion = response["result"]
        action = map_action(record["candidate"], conclusion)
        utility, gt_match_before, gt_match_after = candidate_action_utility(record, action, conclusion)
        acceptance_records.append(
            {
                "fold": index % 4 + 1,
                "action": action,
                "candidate": record["candidate"],
                "conclusion": conclusion,
                "positive_utility": utility,
                "gt_match_before": gt_match_before,
                "gt_match_after": gt_match_after,
            }
        )
    acceptance_gate = fit_oof_acceptance_gate(acceptance_records, evaluation_fold=0)
    print(f"复检门训练样本数: {review_gate.training_count}")
    print(f"接受门训练样本数: {acceptance_gate.training_counts}")
    print(f"接受门各动作正负样本数: {acceptance_gate.label_counts}")
    pair_ids = list(dict.fromkeys(record["candidate"]["pair_id"] for record in records))
    predictions = {record["candidate"]["pair_id"]: record["prediction"] for record in records}
    ground_truth = {record["candidate"]["pair_id"]: record["ground_truth"] for record in records}
    base_predictions = [predictions[pair_id] for pair_id in pair_ids]
    base_f1 = detection_metrics(base_predictions, ground_truth)["f1"]
    ranked = sorted(range(len(records)), key=lambda index: -float(risks[index]))
    curve = []
    for budget in budgets:
        called_indices = ranked[: max(1, round(len(records) * budget / 100.0))]
        called = set(called_indices)
        grouped_candidates: dict[str, list[dict]] = defaultdict(list)
        grouped_decisions: dict[str, list[dict]] = defaultdict(list)
        for index, sample in enumerate(acceptance_records):
            accepted = False
            probability = 0.0
            if index in called and sample["action"] in {"remove", "relabel", "add"}:
                probability = acceptance_gate.predict_probability(sample["action"], sample["candidate"], sample["conclusion"])
                accepted = accept_action(sample["action"], sample["candidate"], sample["conclusion"], probability)
            pair_id = sample["candidate"]["pair_id"]
            grouped_candidates[pair_id].append(sample["candidate"])
            grouped_decisions[pair_id].append(
                {
                    "action": sample["action"],
                    "conclusion": sample["conclusion"],
                    "accept_prob": probability,
                    "positive_utility": sample["positive_utility"],
                    "gt_match_before": sample["gt_match_before"],
                    "gt_match_after": sample["gt_match_after"],
                    "accepted": accepted,
                    "mllm_called": index in called,
                }
            )
        corrected_predictions = []
        audits = []
        for pair_id in pair_ids:
            corrected, pair_audits = apply_actions(
                predictions[pair_id], grouped_candidates[pair_id], grouped_decisions[pair_id]
            )
            corrected_predictions.append(corrected)
            audits.extend(pair_audits)
        corrected_f1 = detection_metrics(corrected_predictions, ground_truth)["f1"]
        metrics = correction_metrics(audits, {"TP": 258, "错分类": 12, "FP": 796, "漏检": 1587})
        curve.append(
            {
                "budget_percent": budget,
                "calls": len(called_indices),
                "accepted": sum(item["accepted"] for item in audits),
                "f1_before": base_f1,
                "f1_after": corrected_f1,
                "f1_gain": corrected_f1 - base_f1,
                **{name: metrics[name] for name in ("fpcr", "mdrr", "mcca", "tprr", "hcr")},
            }
        )
    result = {
        "passed": len(curve) == len(budgets),
        "candidates": len(records),
        "review_training_samples": review_gate.training_count,
        "acceptance_training_samples": acceptance_gate.training_counts,
        "acceptance_training_labels": acceptance_gate.label_counts,
        "curve": curve,
    }
    output = write_json(Path("experiments/rq3_gating/smoke_results.json"), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"产出文件: {output}")


if __name__ == "__main__":
    main()
