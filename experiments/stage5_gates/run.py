"""Train five-fold OOF review and action-specific acceptance gates."""

from __future__ import annotations

import csv
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/tmp/zyp-matplotlib"

import matplotlib.pyplot as plt
import numpy as np
import yaml
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.rq1_evidence.run import align_for_pilot, detector_items, enrich_candidate, load_configs, prediction_map
from src.candidates.fuse import fuse_candidates
from src.candidates.propose import propose_candidates
from src.detector.calibrate import box_iou, read_gt
from src.detector.dataset import rows_for_role
from src.detector.error_decompose import decompose_record
from src.evidence.appearance import compute_appearance
from src.evidence.topology import compute_topology
from src.gates.features import REVIEW_FEATURE_NAMES, acceptance_features, review_features, topology_compatibility
from src.mllm.action_map import map_action
from src.mllm.client import _stub
from src.registration.align import load_gray


SEED = 20260811
OUTPUT_ROOT = Path("experiments/stage5_gates")
CACHE_ROOT = Path("cache/gate_oof")
ACTIONS = ("remove", "relabel", "add")
CLASS_NAMES = ("open", "short", "mousebite", "spur", "pinhole", "spurious_copper")
ACCEPT_FEATURE_NAMES = REVIEW_FEATURE_NAMES + [
    "accept_q_reg", *[f"mllm_class_{name}" for name in (*CLASS_NAMES, "normal", "uncertain")],
    "presence_true", "presence_false", "presence_uncertain", "evidence_strength",
    "reference_reliability", "topology_compatibility", "appearance_compatibility", "json_valid",
]


def box_key(box: list[float]) -> tuple[float, ...]:
    return tuple(round(float(value), 4) for value in box)


def prepare_fold(fold: int) -> tuple[int, int]:
    cache_path = CACHE_ROOT / f"fold{fold}.jsonl"
    if cache_path.exists():
        return fold, len(cache_path.read_text(encoding="utf-8").splitlines())
    registration_config, appearance_config, topology_config, candidate_config, _ = load_configs()
    stub_config = yaml.safe_load(Path("configs/mllm/stub.yaml").read_text(encoding="utf-8"))
    predictions = prediction_map(fold)
    records = []
    for row in rows_for_role(fold, "test"):
        prediction = predictions[row["pair_id"]]
        current = load_gray(Path(row["test_path"]))
        template = load_gray(Path(row["temp_path"]))
        aligned = align_for_pilot(current, template, registration_config)
        appearance = compute_appearance(aligned["normalized_test"], aligned["normalized_template"], appearance_config)
        normalized_current = ((aligned["normalized_test"] * 32.0) + 128.0).clip(0, 255).astype("uint8")
        normalized_template = ((aligned["normalized_template"] * 32.0) + 128.0).clip(0, 255).astype("uint8")
        topology = compute_topology(normalized_current, normalized_template, topology_config)
        detections = detector_items(prediction)
        differences = propose_candidates(
            appearance, topology, current, aligned["aligned_template"], detections, candidate_config, fold
        )
        candidates = fuse_candidates(
            row["pair_id"], detections, differences["candidates"], int(candidate_config["center_expand_px"])
        )
        ground_truth = [{"box": box, "class": int(class_id)} for box, class_id in read_gt(Path(row["anno_path"]))]
        errors = decompose_record(prediction, row)
        error_by_box = {box_key(json.loads(item["pred_box"])): item for item in errors if item["pred_box"] != "[]"}
        missed = [item for item in errors if item["error_type"] == "漏检"]
        for candidate in candidates:
            candidate["pair_id"] = row["pair_id"]
            enrich_candidate(candidate, aligned, appearance, topology, differences["threshold"])
            if candidate["source"] in {"DET", "BOTH"}:
                error = error_by_box[box_key(candidate["det_box"])]
                review_label = int(error["error_type"] in {"FP", "错分类"})
                error_type = error["error_type"]
                gt_class = int(error["gt_class"])
                gt_iou = float(error["iou"])
            else:
                matches = [
                    (item, box_iou(candidate["action_box"], json.loads(item["gt_box"]))) for item in missed
                ]
                matched, gt_iou = max(matches, key=lambda item: item[1]) if matches else ({"gt_class": -1}, 0.0)
                review_label = int(gt_iou >= 0.3)
                error_type = "漏检" if review_label else "差异噪声"
                gt_class = int(matched["gt_class"]) if review_label else -1
            conclusion = _stub(candidate, stub_config)
            action = map_action(candidate, conclusion)
            if action == "remove":
                positive_utility = error_type == "FP"
            elif action == "relabel":
                positive_utility = error_type == "错分类" and CLASS_NAMES.index(conclusion["predicted_class"]) == gt_class
            elif action == "add":
                positive_utility = error_type == "漏检" and CLASS_NAMES.index(conclusion["predicted_class"]) == gt_class
            else:
                positive_utility = False
            records.append(
                {
                    "fold": fold,
                    "pair_id": row["pair_id"],
                    "candidate": candidate,
                    "review_label": review_label,
                    "error_type": error_type,
                    "gt_class": gt_class,
                    "gt_iou": gt_iou,
                    "conclusion": conclusion,
                    "action": action,
                    "positive_utility": bool(positive_utility),
                }
            )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n", encoding="utf-8")
    return fold, len(records)


def load_records() -> list[dict]:
    with ProcessPoolExecutor(max_workers=4) as executor:
        counts = list(executor.map(prepare_fold, range(5)))
    print("OOF 候选数:", dict(counts), flush=True)
    return [
        json.loads(line)
        for fold in range(5)
        for line in (CACHE_ROOT / f"fold{fold}.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def fixed_acceptance_score(sample: dict) -> float:
    conclusion = sample["conclusion"]
    reliability = {None: 0.0, "low": 0.0, "medium": 0.5, "high": 1.0}[conclusion["reference_reliability"]]
    score = 0.35 * float(conclusion["evidence_strength"]) / 3.0 + 0.30 * reliability + 0.35 * float(sample["candidate"]["q_reg"])
    if sample["action"] == "relabel":
        score *= topology_compatibility(conclusion)
    return score


def fit_predict_review(records: list[dict]) -> tuple[dict[str, np.ndarray], list[dict], list[dict]]:
    labels = np.asarray([item["review_label"] for item in records], dtype=int)
    logistic_scores = np.zeros(len(records), dtype=float)
    lightgbm_scores = np.zeros(len(records), dtype=float)
    weights = []
    samples = []
    for evaluation_fold in range(5):
        train_indices = [index for index, item in enumerate(records) if item["fold"] != evaluation_fold]
        test_indices = [index for index, item in enumerate(records) if item["fold"] == evaluation_fold]
        x_train = np.vstack([review_features(records[index]["candidate"]) for index in train_indices])
        y_train = labels[train_indices]
        x_test = np.vstack([review_features(records[index]["candidate"]) for index in test_indices])
        logistic = make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, l1_ratio=0.0, max_iter=1000, random_state=SEED)
        )
        logistic.fit(x_train, y_train)
        logistic_scores[test_indices] = logistic.predict_proba(x_test)[:, 1]
        lightgbm = LGBMClassifier(n_estimators=200, reg_lambda=1.0, random_state=SEED, verbosity=-1)
        lightgbm.fit(x_train, y_train)
        lightgbm_scores[test_indices] = lightgbm.predict_proba(x_test)[:, 1]
        coefficient = logistic.named_steps["logisticregression"].coef_[0]
        weights.extend(
            {"evaluation_fold": evaluation_fold, "feature": name, "weight": float(value)}
            for name, value in zip(REVIEW_FEATURE_NAMES, coefficient)
        )
        samples.append(
            {
                "gate": "review", "evaluation_fold": evaluation_fold, "action": "all",
                "training_samples": len(train_indices), "positive": int(y_train.sum()),
                "negative": int(len(y_train) - y_train.sum()),
            }
        )
    return {"logistic": logistic_scores, "lightgbm": lightgbm_scores, "labels": labels}, weights, samples


def fit_predict_acceptance(records: list[dict]) -> tuple[dict[str, dict[str, np.ndarray]], list[dict], list[dict], list[str]]:
    scores: dict[str, dict[str, np.ndarray]] = {}
    weights = []
    samples = []
    warnings_text = []
    for action in ACTIONS:
        subset = [item for item in records if item["action"] == action]
        labels = np.asarray([int(item["positive_utility"]) for item in subset], dtype=int)
        logistic_scores = np.zeros(len(subset), dtype=float)
        lightgbm_scores = np.zeros(len(subset), dtype=float)
        for evaluation_fold in range(5):
            train_indices = [index for index, item in enumerate(subset) if item["fold"] != evaluation_fold]
            test_indices = [index for index, item in enumerate(subset) if item["fold"] == evaluation_fold]
            y_train = labels[train_indices]
            positives = int(y_train.sum())
            negatives = int(len(y_train) - positives)
            samples.append(
                {
                    "gate": "acceptance", "evaluation_fold": evaluation_fold, "action": action,
                    "training_samples": len(train_indices), "positive": positives, "negative": negatives,
                }
            )
            if len(train_indices) < 30 or len(set(y_train.tolist())) < 2:
                message = f"警告：评价 fold {evaluation_fold} 的 {action} 接受门训练样本 {len(train_indices)}，正样本 {positives}，退化为固定阈值加安全规则"
                warnings.warn(message, stacklevel=2)
                warnings_text.append(message)
                for index in test_indices:
                    logistic_scores[index] = fixed_acceptance_score(subset[index])
                    lightgbm_scores[index] = logistic_scores[index]
                continue
            x_train = np.vstack([
                acceptance_features(subset[index]["candidate"], subset[index]["conclusion"]) for index in train_indices
            ])
            x_test = np.vstack([
                acceptance_features(subset[index]["candidate"], subset[index]["conclusion"]) for index in test_indices
            ])
            logistic = make_pipeline(
                StandardScaler(), LogisticRegression(C=1.0, l1_ratio=0.0, max_iter=1000, random_state=SEED)
            )
            logistic.fit(x_train, y_train)
            logistic_scores[test_indices] = logistic.predict_proba(x_test)[:, 1]
            lightgbm = LGBMClassifier(n_estimators=200, reg_lambda=1.0, random_state=SEED, verbosity=-1)
            lightgbm.fit(x_train, y_train)
            lightgbm_scores[test_indices] = lightgbm.predict_proba(x_test)[:, 1]
            coefficient = logistic.named_steps["logisticregression"].coef_[0]
            weights.extend(
                {"evaluation_fold": evaluation_fold, "action": action, "feature": name, "weight": float(value)}
                for name, value in zip(ACCEPT_FEATURE_NAMES, coefficient)
            )
        scores[action] = {"logistic": logistic_scores, "lightgbm": lightgbm_scores, "labels": labels}
    return scores, weights, samples, warnings_text


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_review(scores: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    labels = scores["labels"]
    summary = {}
    plt.figure(figsize=(6.4, 5.0))
    for model in ("logistic", "lightgbm"):
        precision, recall, _ = precision_recall_curve(labels, scores[model])
        ap = average_precision_score(labels, scores[model])
        summary[model] = {"average_precision": float(ap), "roc_auc": float(roc_auc_score(labels, scores[model]))}
        plt.plot(recall, precision, label=f"{model} AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.tight_layout()
    plt.savefig(OUTPUT_ROOT / "review_pr_curve.png", dpi=180); plt.close()
    plt.figure(figsize=(6.4, 5.0))
    for model in ("logistic", "lightgbm"):
        fpr, tpr, _ = roc_curve(labels, scores[model])
        plt.plot(fpr, tpr, label=f"{model} AUC={summary[model]['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle=":", color="gray")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend(); plt.tight_layout()
    plt.savefig(OUTPUT_ROOT / "review_roc_curve.png", dpi=180); plt.close()
    plt.figure(figsize=(6.4, 5.0))
    for model in ("logistic", "lightgbm"):
        observed, predicted = calibration_curve(labels, scores[model], n_bins=10, strategy="quantile")
        plt.plot(predicted, observed, marker="o", label=model)
    plt.plot([0, 1], [0, 1], linestyle=":", color="gray")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency"); plt.legend(); plt.tight_layout()
    plt.savefig(OUTPUT_ROOT / "review_calibration.png", dpi=180); plt.close()
    return summary


def plot_acceptance(scores: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, dict[str, float] | None]]:
    summary = {}
    plt.figure(figsize=(7.2, 5.4))
    for action in ACTIONS:
        labels = scores[action]["labels"]
        summary[action] = {}
        for model, linestyle in (("logistic", "-"), ("lightgbm", "--")):
            if len(set(labels.tolist())) < 2:
                summary[action][model] = None
                continue
            precision, recall, _ = precision_recall_curve(labels, scores[action][model])
            ap = average_precision_score(labels, scores[action][model])
            summary[action][model] = {"average_precision": float(ap)}
            plt.plot(recall, precision, linestyle=linestyle, label=f"{action} {model} AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(OUTPUT_ROOT / "acceptance_pr_curve.png", dpi=180); plt.close()
    plt.figure(figsize=(7.2, 5.4))
    for action in ACTIONS:
        labels = scores[action]["labels"]
        if len(set(labels.tolist())) < 2:
            continue
        observed, predicted = calibration_curve(labels, scores[action]["logistic"], n_bins=8, strategy="quantile")
        plt.plot(predicted, observed, marker="o", label=action)
    plt.plot([0, 1], [0, 1], linestyle=":", color="gray")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency"); plt.legend(); plt.tight_layout()
    plt.savefig(OUTPUT_ROOT / "acceptance_calibration.png", dpi=180); plt.close()
    return summary


def build_report(
    records: list[dict], sample_rows: list[dict], review_summary: dict, acceptance_summary: dict, warnings_text: list[str]
) -> None:
    review_rows = [item for item in sample_rows if item["gate"] == "review"]
    acceptance_rows = [item for item in sample_rows if item["gate"] == "acceptance"]
    lines = [
        "# 阶段 C：五折 OOF 门控训练报告", "", "## 数据与防泄漏", "",
        f"门控候选来自五折 test 的 out-of-fold 检测预测，共 {len(records)} 个候选。评价第 i 折时只用其他四折训练，随机种子为 {SEED}。复检门主模型为 L2 Logistic Regression，对照为 LightGBM。",
        "", "接受门按 remove、relabel、add 分开训练。动作提议由冻结的确定性证据代理生成，不产生大模型调用；正效用标签由 OOF 预测和 GT 按动作定义计算。A2 是后续大模型视觉输入，8 维拓扑向量只作为两个门控的结构化特征。",
        "", "## 复检门样本", "", "| 评价折 | 训练样本 | 正样本 | 负样本 | 正样本比例 |", "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in review_rows:
        lines.append(f"| {item['evaluation_fold']} | {item['training_samples']} | {item['positive']} | {item['negative']} | {item['positive'] / item['training_samples']:.4f} |")
    lines.extend(["", "## 复检门性能", "", "| 模型 | PR AP | ROC AUC |", "| --- | ---: | ---: |"])
    for model, values in review_summary.items():
        lines.append(f"| {model} | {values['average_precision']:.4f} | {values['roc_auc']:.4f} |")
    lines.extend(["", "## 接受门样本", "", "| 动作 | 评价折 | 训练样本 | 正样本 | 负样本 | 正样本比例 |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for item in acceptance_rows:
        ratio = item["positive"] / item["training_samples"] if item["training_samples"] else 0.0
        lines.append(f"| {item['action']} | {item['evaluation_fold']} | {item['training_samples']} | {item['positive']} | {item['negative']} | {ratio:.4f} |")
    lines.extend(["", "## 接受门 PR AP", "", "| 动作 | Logistic | LightGBM |", "| --- | ---: | ---: |"])
    for action, values in acceptance_summary.items():
        logistic = "未定义" if values["logistic"] is None else f"{values['logistic']['average_precision']:.4f}"
        lightgbm = "未定义" if values["lightgbm"] is None else f"{values['lightgbm']['average_precision']:.4f}"
        lines.append(f"| {action} | {logistic} | {lightgbm} |")
    lines.extend(["", "## 退化警告", ""])
    lines.extend([f"* {item}" for item in warnings_text] or ["无。所有动作在各评价折均达到 30 个样本且标签包含正负两类。"])
    lines.extend([
        "", "## 产物", "",
        "复检门产出 PR 曲线、ROC、校准曲线和标准化特征权重表。接受门产出分动作 PR 曲线、校准曲线和标准化特征权重表。逐折训练样本数与正负比例保存在 `training_samples.csv`。",
    ])
    (OUTPUT_ROOT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    records = load_records()
    review_scores, review_weights, review_samples = fit_predict_review(records)
    acceptance_scores, acceptance_weights, acceptance_samples, warnings_text = fit_predict_acceptance(records)
    write_csv(OUTPUT_ROOT / "review_feature_weights.csv", review_weights)
    if acceptance_weights:
        write_csv(OUTPUT_ROOT / "acceptance_feature_weights.csv", acceptance_weights)
    else:
        (OUTPUT_ROOT / "acceptance_feature_weights.csv").write_text("evaluation_fold,action,feature,weight\n", encoding="utf-8")
    sample_rows = review_samples + acceptance_samples
    write_csv(OUTPUT_ROOT / "training_samples.csv", sample_rows)
    review_summary = plot_review(review_scores)
    acceptance_summary = plot_acceptance(acceptance_scores)
    summary = {"review": review_summary, "acceptance": acceptance_summary, "warnings": warnings_text}
    (OUTPUT_ROOT / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    build_report(records, sample_rows, review_summary, acceptance_summary, warnings_text)
    print(json.dumps({"candidates": len(records), **summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
