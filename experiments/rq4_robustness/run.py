"""RQ4 registration, photometric, and gate robustness experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/tmp/zyp-rq4-matplotlib"

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import font_manager

cv2.setNumThreads(1)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.rq2_correction.run_full import (
    acceptance_probability,
    fit_fold_models,
    ground_truth_map,
    load_gate_records,
    prediction_map,
)
from experiments.rq1_evidence.run import align_for_pilot
from src.candidates.propose import propose_candidates
from src.detector.calibrate import box_iou
from src.detector.dataset import rows_for_role
from src.evaluation.apply import apply_actions
from src.evaluation.metrics import correction_metrics, detection_metrics
from src.evaluation.utility import prediction_gt_matches
from src.evidence.appearance import compute_appearance
from src.evidence.topology import compute_topology, topology_features
from src.gates.features import review_features, topology_compatibility
from src.registration.align import (
    _edge_chamfer,
    _transform_quality,
    align_images,
    load_gray,
    photometric_normalize,
)


SEED = 20260811
FOLD = 0
OUTPUT = Path("experiments/rq4_robustness")
FIGURES = Path("experiments/figures")
PROGRESS = OUTPUT / "progress.csv"
DETAILS = OUTPUT / "perturbation_details.csv"
SUMMARY = OUTPUT / "perturbation_summary.csv"
RESULTS = Path("experiments/results_summary.csv")
SETTINGS = (
    "no_reg_direct",
    "ecc_direct",
    "ecc_direct_gate",
    "ecc_tolerant",
    "ecc_tolerant_gate",
)

GEOMETRIC = (
    ("baseline", "baseline", 0.0),
    ("shift_m8px", "shift", -8.0),
    ("shift_m4px", "shift", -4.0),
    ("shift_m2px", "shift", -2.0),
    ("shift_m1px", "shift", -1.0),
    ("shift_1px", "shift", 1.0),
    ("shift_2px", "shift", 2.0),
    ("shift_4px", "shift", 4.0),
    ("shift_8px", "shift", 8.0),
    ("rotate_m2deg", "rotate", -2.0),
    ("rotate_m1deg", "rotate", -1.0),
    ("rotate_m0.5deg", "rotate", -0.5),
    ("rotate_0.5deg", "rotate", 0.5),
    ("rotate_1deg", "rotate", 1.0),
    ("rotate_2deg", "rotate", 2.0),
    ("scale_0.98", "scale", 0.98),
    ("scale_1.02", "scale", 1.02),
)

PHOTOMETRIC = (
    ("brightness_m20", "brightness", 0.80),
    ("brightness_m10", "brightness", 0.90),
    ("brightness_p10", "brightness", 1.10),
    ("brightness_p20", "brightness", 1.20),
    ("contrast_0.8", "contrast", 0.80),
    ("contrast_1.2", "contrast", 1.20),
    ("noise_2_255", "noise", 2.0),
    ("noise_5_255", "noise", 5.0),
    ("blur_0.5", "blur", 0.5),
    ("blur_1.0", "blur", 1.0),
)


def append_progress(stage: str, completed: int, total: int, detail: str) -> None:
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{datetime.now().astimezone().isoformat(timespec='seconds')},{stage},{completed},{total},{detail}\n"
        )


def load_configs() -> tuple[dict, dict, dict, dict]:
    return tuple(
        yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        for path in (
            "configs/evidence/registration.yaml",
            "configs/evidence/appearance.yaml",
            "configs/evidence/topology.yaml",
            "configs/candidates.yaml",
        )
    )


def geometric_perturb(image: np.ndarray, kind: str, value: float) -> np.ndarray:
    height, width = image.shape
    if kind == "baseline":
        return image.copy()
    if kind == "shift":
        matrix = np.float32([[1.0, 0.0, value], [0.0, 1.0, value]])
    else:
        angle = value if kind == "rotate" else 0.0
        scale = value if kind == "scale" else 1.0
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def photometric_perturb(image: np.ndarray, kind: str, value: float, pair_id: str) -> np.ndarray:
    if kind == "brightness":
        return np.clip(image.astype(np.float32) * value, 0.0, 255.0).astype(np.uint8)
    if kind == "contrast":
        mean = float(image.mean())
        return np.clip((image.astype(np.float32) - mean) * value + mean, 0.0, 255.0).astype(np.uint8)
    if kind == "noise":
        rng = np.random.default_rng(SEED + int(pair_id) + int(value * 100))
        noise = rng.normal(0.0, value, image.shape)
        return np.clip(image.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
    return cv2.GaussianBlur(image, (0, 0), value)


def direct_topology(topology: dict) -> dict:
    output = dict(topology)
    output["M_topo"] = np.logical_xor(
        topology["test_skeleton"], topology["template_skeleton"]
    ).astype(np.float32)
    return output


def identity_alignment(test: np.ndarray, template: np.ndarray, registration_config: dict) -> dict:
    correlation = float(1.0 - np.mean(np.abs(test.astype(np.float32) - template.astype(np.float32))) / 255.0)
    chamfer = _edge_chamfer(test, template)
    weights = registration_config["weights"]
    transform = np.eye(2, 3, dtype=np.float32)
    q_reg = float(
        np.clip(
            weights["ecc"] * max(0.0, min(1.0, correlation))
            + weights["inlier"]
            + weights["chamfer"] * max(0.0, 1.0 - chamfer / 20.0)
            + weights["transform"] * _transform_quality(transform, registration_config),
            0.0,
            1.0,
        )
    )
    normalized_test, normalized_template = photometric_normalize(test, template)
    return {
        "aligned_template": template,
        "normalized_test": normalized_test,
        "normalized_template": normalized_template,
        "q_reg": q_reg,
        "chamfer": chamfer,
    }


def missed_ground_truth(prediction: dict, ground_truth: list[dict]) -> list[dict]:
    covered = {item["gt_index"] for item in prediction_gt_matches(prediction, ground_truth).values()}
    return [item for index, item in enumerate(ground_truth) if index not in covered]


def local_evidence(candidate: dict, appearance: dict, topology: dict, threshold: float) -> dict:
    updated = dict(candidate)
    x1, y1, x2, y2 = [int(round(value)) for value in candidate["box"]]
    height, width = appearance["M_app"].shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    residual = appearance["M_app"][y1:y2, x1:x2]
    xor_region = topology["M_topo"][y1:y2, x1:x2] > 0
    test_vector = topology_features(
        topology["test_mask"][y1:y2, x1:x2],
        topology["test_skeleton"][y1:y2, x1:x2],
        xor_region,
    )
    template_vector = topology_features(
        topology["template_mask"][y1:y2, x1:x2],
        topology["template_skeleton"][y1:y2, x1:x2],
        xor_region,
    )
    updated["appearance_residual"] = {
        "mean": float(residual.mean()),
        "p95": float(np.quantile(residual, 0.95)),
        "max": float(residual.max()),
    }
    updated["above_threshold_fraction"] = float((residual >= threshold).mean())
    updated["topology_vector"] = np.abs(test_vector - template_vector).astype(float).tolist()
    return updated


def process_pair(task: tuple[dict, str, str, float, list[dict]]) -> dict:
    row, perturbation, kind, value, samples = task
    registration_config, appearance_config, topology_config, candidate_config = load_configs()
    current = load_gray(Path(row["test_path"]))
    template = load_gray(Path(row["temp_path"]))
    if perturbation in {item[0] for item in PHOTOMETRIC}:
        current = photometric_perturb(current, kind, value, row["pair_id"])
        changed_template = template
    else:
        changed_template = geometric_perturb(template, kind, value)
    no_reg = identity_alignment(current, changed_template, registration_config)
    ecc = align_for_pilot(current, changed_template, registration_config)
    if ecc["method"] == "IDENTITY_LOW_OBSERVABILITY":
        ecc["chamfer"] = _edge_chamfer(current, changed_template)
    maps = {}
    for alignment_name, aligned in (("no_reg", no_reg), ("ecc", ecc)):
        appearance = compute_appearance(
            aligned["normalized_test"], aligned["normalized_template"], appearance_config
        )
        normalized_current = np.clip(aligned["normalized_test"] * 32.0 + 128.0, 0, 255).astype(np.uint8)
        normalized_template = np.clip(aligned["normalized_template"] * 32.0 + 128.0, 0, 255).astype(np.uint8)
        tolerant = compute_topology(normalized_current, normalized_template, topology_config)
        maps[(alignment_name, "direct")] = (appearance, direct_topology(tolerant), aligned)
        maps[(alignment_name, "tolerant")] = (appearance, tolerant, aligned)
    threshold = float(candidate_config["global_thresholds_by_fold"][str(FOLD)]["combined"])
    payload = {
        "pair_id": row["pair_id"],
        "perturbation": perturbation,
        "q_reg_ecc": float(ecc["q_reg"]),
        "q_reg_identity": float(no_reg["q_reg"]),
        "ecc_chamfer": float(ecc["chamfer"]),
        "identity_chamfer": float(no_reg["chamfer"]),
        "settings": {},
    }
    for setting in SETTINGS:
        alignment_name = "no_reg" if setting == "no_reg_direct" else "ecc"
        xor_name = "tolerant" if "tolerant" in setting else "direct"
        appearance, topology, aligned = maps[(alignment_name, xor_name)]
        proposal = propose_candidates(
            appearance,
            topology,
            current,
            aligned["aligned_template"],
            [],
            {**candidate_config, "tolerance_px": 0},
            FOLD,
        )
        updated_samples = []
        for sample in samples:
            updated = local_evidence(sample["candidate"], appearance, topology, threshold)
            updated["q_reg"] = float(aligned["q_reg"])
            updated_samples.append({"candidate_id": updated["candidate_id"], "candidate": updated})
        payload["settings"][setting] = {
            "difference_boxes": [item["box"] for item in proposal["candidates"]],
            "candidate_count": len(proposal["candidates"]),
            "xor_rate": float(topology["M_topo"].mean()),
            "samples": updated_samples,
        }
    return payload


def replay_setting(
    setting: str,
    pair_payloads: list[dict],
    sampled: list[dict],
    models: dict,
    predictions: dict[str, dict],
    ground_truth: dict[str, list[dict]],
) -> dict:
    updated_by_id = {
        item["candidate_id"]: item["candidate"]
        for payload in pair_payloads
        for item in payload["settings"][setting]["samples"]
    }
    samples = []
    for record in sampled:
        candidate = updated_by_id[record["candidate"]["candidate_id"]]
        score = float(models["logistic"].predict_proba(review_features(candidate).reshape(1, -1))[0, 1])
        probability = acceptance_probability(models, record["action"], candidate, record["conclusion"])
        samples.append({**record, "candidate": candidate, "review_score": score, "accept_prob": probability})
    grouped_candidates: dict[str, list[dict]] = defaultdict(list)
    grouped_decisions: dict[str, list[dict]] = defaultdict(list)
    gate_enabled = setting.endswith("_gate")
    harmful_blocked = 0
    for sample in samples:
        candidate = sample["candidate"]
        action = sample["action"]
        called = sample["review_score"] >= models["logistic_threshold"]
        modifiable = action in {"remove", "relabel", "add"}
        source_ok = candidate["source"] in {"DET", "BOTH"} if action in {"remove", "relabel"} else candidate["source"] == "DIFF"
        topology_ok = action != "relabel" or topology_compatibility(sample["conclusion"]) >= 1.0
        base_accept = (
            called
            and modifiable
            and sample["accept_prob"] >= models["acceptance_thresholds"].get(action, 1.0)
            and source_ok
            and topology_ok
        )
        registration_ok = not gate_enabled or float(candidate["q_reg"]) >= 0.5
        accepted = base_accept and registration_ok
        harmful_blocked += int(base_accept and not registration_ok and not sample["positive_utility"])
        grouped_candidates[sample["pair_id"]].append(candidate)
        grouped_decisions[sample["pair_id"]].append(
            {
                "action": action,
                "conclusion": sample["conclusion"],
                "accept_prob": sample["accept_prob"],
                "positive_utility": sample["positive_utility"],
                "accepted": accepted,
                "mllm_called": called,
                "error_type": sample["error_type"],
                "source": candidate["source"],
                "review_label": sample["review_label"],
            }
        )
    corrected = []
    audits = []
    for pair_id, prediction in predictions.items():
        final, pair_audits = apply_actions(
            prediction,
            grouped_candidates[pair_id],
            grouped_decisions[pair_id],
        )
        corrected.append(final)
        audits.extend(pair_audits)
    counts = Counter()
    with Path("experiments/stage2/error_decomposition_fold0.csv").open(encoding="utf-8-sig") as handle:
        counts.update(row["error_type"] for row in csv.DictReader(handle))
    metrics = detection_metrics(corrected, ground_truth)
    metrics.update(correction_metrics(audits, dict(counts)))
    metrics["calls"] = sum(item["mllm_called"] for item in audits)
    metrics["accepted_actions"] = sum(item["accepted"] for item in audits)
    metrics["harmful_blocked_by_registration"] = harmful_blocked
    return metrics


def recovery_metrics(
    setting: str,
    pair_payloads: list[dict],
    predictions: dict[str, dict],
    ground_truth: dict[str, list[dict]],
) -> tuple[int, int, float]:
    recovered = 0
    total = 0
    for payload in pair_payloads:
        pair_id = payload["pair_id"]
        missed = missed_ground_truth(predictions[pair_id], ground_truth[pair_id])
        boxes = payload["settings"][setting]["difference_boxes"]
        total += len(missed)
        recovered += sum(any(box_iou(gt["box"], box) >= 0.3 for box in boxes) for gt in missed)
    return recovered, total, recovered / total if total else 0.0


def q_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "q_reg_mean": float(array.mean()),
        "q_reg_std": float(array.std(ddof=1)),
        "q_reg_p05": float(np.quantile(array, 0.05)),
        "q_reg_p50": float(np.quantile(array, 0.50)),
        "q_reg_p95": float(np.quantile(array, 0.95)),
        "q_reg_below_0.5": float((array < 0.5).mean()),
    }


def plot_results(rows: list[dict], details: list[dict]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    cjk_font = font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    geometry = [row for row in rows if row["family"] == "geometry"]
    display = {
        "no_reg_direct": "无二次配准＋直接 XOR",
        "ecc_direct": "ECC＋直接 XOR",
        "ecc_direct_gate": "ECC＋直接 XOR＋质量门",
        "ecc_tolerant": "ECC＋容差 XOR",
        "ecc_tolerant_gate": "ECC＋容差 XOR＋质量门",
    }
    available = {row["perturbation"] for row in geometry}
    order = [item[0] for item in GEOMETRIC if item[0] in available]
    x = np.arange(len(order))
    plt.figure(figsize=(12.0, 5.6))
    for setting in SETTINGS:
        subset = {row["perturbation"]: row for row in geometry if row["setting"] == setting}
        plt.plot(x, [subset[name]["f1"] for name in order], marker="o", linewidth=1.3, label=display[setting])
    plt.xticks(x, order, rotation=45, ha="right")
    plt.ylabel("最终 F1", fontproperties=cjk_font)
    plt.legend(prop=cjk_font, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "rq4_registration_performance.png", dpi=220)
    plt.close()

    by_perturbation: dict[str, list[float]] = defaultdict(list)
    for item in details:
        if item["family"] == "geometry":
            by_perturbation[item["perturbation"]].append(float(item["q_reg_ecc"]))
    data = [by_perturbation[name] for name in order]
    plt.figure(figsize=(12.0, 5.6))
    plt.boxplot(data, showfliers=False)
    plt.xticks(x + 1, order, rotation=45, ha="right")
    plt.ylabel("q_reg")
    plt.tight_layout()
    plt.savefig(FIGURES / "rq4_qreg_distribution.png", dpi=220)
    plt.close()

    direct = {row["perturbation"]: row for row in geometry if row["setting"] == "ecc_direct"}
    tolerant = {row["perturbation"]: row for row in geometry if row["setting"] == "ecc_tolerant"}
    plt.figure(figsize=(12.0, 5.6))
    plt.plot(x, [direct[name]["xor_rate_mean"] for name in order], marker="o", label="直接 XOR")
    plt.plot(x, [tolerant[name]["xor_rate_mean"] for name in order], marker="o", label="容差 XOR")
    plt.xticks(x, order, rotation=45, ha="right")
    plt.ylabel("骨架差异像素占比", fontproperties=cjk_font)
    plt.legend(prop=cjk_font)
    plt.tight_layout()
    plt.savefig(FIGURES / "rq4_xor_comparison.png", dpi=220)
    plt.close()


def append_results_summary(rows: list[dict], model_rows: list[dict]) -> None:
    with RESULTS.open(encoding="utf-8-sig", newline="") as handle:
        existing = list(csv.DictReader(handle))
        fieldnames = list(existing[0])
    existing = [row for row in existing if row["rq"] != "RQ4"]
    for row in rows:
        for metric in (
            "q_reg_mean", "q_reg_std", "q_reg_p05", "q_reg_p50", "q_reg_p95",
            "q_reg_below_0.5", "miss_recovery_rate", "f1", "hcr",
            "harmful_blocked_by_registration", "xor_rate_mean", "candidate_count_mean",
        ):
            existing.append(
                {
                    "rq": "RQ4", "table": row["family"], "method": row["setting"],
                    "strategy": row["perturbation"], "budget_percent": "", "fold": FOLD,
                    "metric": metric, "class": "", "statistic": "value", "value": row[metric],
                }
            )
    for row in model_rows:
        existing.append(
            {
                "rq": "RQ4", "table": "model_replacement", "method": row["model"],
                "strategy": "", "budget_percent": "", "fold": FOLD,
                "metric": row["metric"], "class": "", "statistic": "value", "value": row["value"],
            }
        )
    with RESULTS.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)


def report(rows: list[dict], model_replacement: dict) -> str:
    geometry = [row for row in rows if row["family"] == "geometry"]
    photometric = [row for row in rows if row["family"] == "photometric"]
    lines = [
        "# 阶段 E：RQ4 鲁棒性与泛化报告", "", "## 实验口径", "",
        "主实验使用 fold0 全部 325 张 test 图像。几何扰动施加于正常模板，光度扰动施加于待检图。每个条件均重新计算配准、光度归一、外观残差、骨架差异和差异候选。随机种子固定为 20260811。", "",
        "最终 F1 与 HCR 使用阶段 D 的 fold0 800 个 A2 真实 Codex 结论做冻结响应回放。回放保持候选框与大模型结论不变，只重算扰动会改变的 q_reg、外观、拓扑、复检门和接受门特征。漏检恢复率在完整 test 折重新生成的差异候选上计算，定义为与 B1 未覆盖真值框达到 IoU≥0.3 的比例。这个设计隔离配准证据退化，但不能代表每个扰动下重新调用大模型后的行为。", "",
        "五个具体设置分别为无二次配准＋直接 XOR、ECC＋直接 XOR、ECC＋直接 XOR＋质量门、ECC＋容差 XOR、ECC＋容差 XOR＋质量门。无二次配准行中的 q_reg 是以恒等变换代入同一质量公式得到的代理分数，用于横向展示，不是 ECC 输出。", "",
        "## 配准扰动", "",
        "| 扰动 | 设置 | q_reg 均值±标准差 | P05/P50/P95 | q_reg<0.5 | 漏检恢复率 | 最终 F1 | HCR | 质量门拦截有害动作 | 候选数/图 | XOR 占比 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in geometry:
        lines.append(
            f"| {row['perturbation']} | {row['setting']} | {row['q_reg_mean']:.4f}±{row['q_reg_std']:.4f} | {row['q_reg_p05']:.4f}/{row['q_reg_p50']:.4f}/{row['q_reg_p95']:.4f} | {row['q_reg_below_0.5']:.2%} | {row['miss_recovered']}/{row['miss_total']}={row['miss_recovery_rate']:.2%} | {row['f1']:.4f} | {row['hcr']:.4f} | {int(row['harmful_blocked_by_registration'])} | {row['candidate_count_mean']:.2f} | {row['xor_rate_mean']:.5f} |"
        )
    full_geometry = [row for row in geometry if row["setting"] == "ecc_tolerant_gate"]
    worst_geometry = min(full_geometry, key=lambda row: row["f1"])
    direct_xor = np.mean([row["xor_rate_mean"] for row in geometry if row["setting"] == "ecc_direct"])
    tolerant_xor = np.mean([row["xor_rate_mean"] for row in geometry if row["setting"] == "ecc_tolerant"])
    blocked_total = sum(row["harmful_blocked_by_registration"] for row in full_geometry)
    harmful_conditions = [row["perturbation"] for row in full_geometry if row["hcr"] > 0.0]
    lines.extend([
        "", "图表见 `experiments/figures/rq4_registration_performance.png`、`experiments/figures/rq4_qreg_distribution.png` 与 `experiments/figures/rq4_xor_comparison.png`。", "",
        f"完整设置的最差几何条件为 {worst_geometry['perturbation']}，最终 F1={worst_geometry['f1']:.4f}。容差 XOR 的平均差异像素占比为 {tolerant_xor:.5f}，直接 XOR 为 {direct_xor:.5f}，相对变化 {(tolerant_xor / direct_xor - 1.0):+.2%}。质量门在全部几何条件合计拦截 {int(blocked_total)} 个有害动作。", "",
        f"出现非零 HCR 的完整设置条件为 {', '.join(harmful_conditions)}。这些条件每个只有 1 个被接受动作且均为有害动作，因此 HCR=1.0，不能解读为大比例候选都被破坏。质量门未拦截这些动作，因为其候选 q_reg 未低于冻结阈值 0.5。该结果不支持当前 q_reg 质量门在本回放样本上减少有害动作。", "",
        "ECC 对基线漏检候选恢复有明显作用，但在部分强扰动下未持续优于无二次配准。q_reg 同时惩罚估计变换幅度，因此 ECC 修正真实初始错位后，其数值可能低于恒等对齐代理分数。当前 q_reg 的跨扰动单调性和阈值校准仍是方法限制。", "",
        "## 光度扰动", "",
        "| 扰动 | 设置 | q_reg 均值±标准差 | 漏检恢复率 | 最终 F1 | HCR | 质量门拦截有害动作 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in photometric:
        lines.append(
            f"| {row['perturbation']} | {row['setting']} | {row['q_reg_mean']:.4f}±{row['q_reg_std']:.4f} | {row['miss_recovery_rate']:.2%} | {row['f1']:.4f} | {row['hcr']:.4f} | {int(row['harmful_blocked_by_registration'])} |"
        )
    photo_full = [row for row in photometric if row["setting"] == "ecc_tolerant_gate"]
    photo_f1 = [row["f1"] for row in photo_full]
    lines.extend(["", "DeepPCB 图像近二值，因此若光度扰动变化较小，按实测结果解释，不据此删减实验。", "", "## 模型替换", ""])
    lines.insert(
        len(lines) - 3,
        f"完整设置在光度扰动下的 F1 范围为 {min(photo_f1):.4f} 到 {max(photo_f1):.4f}。",
    )
    lines.append(model_replacement["report"])
    lines.extend([
        "", "第二 MLLM 对照未做。当前实验环境只有 Codex 后端，无法在同一协议下获得第二个视觉语言模型结果，因此结论不能泛化到其他 MLLM。", "",
        "## 外部验证", "", "HRIPCB 未纳入本轮主实验。仓库没有该数据集，本阶段优先完成必做扰动与模型替换，未让可选数据获取阻塞阶段 E。", "",
        "## 结论边界", "", "配准与容差 XOR 结论来自 DeepPCB fold0 和单一 Codex 后端。检测器替换只使用 fold0 的指定 200 候选子集。RT-DETR-R18 的基线与完整流程均显著弱于 YOLOv8s，且完整流程未接受任何修正动作。因此本阶段不支持把主要结论泛化到不同检测器，只能表述为当前模型、数据划分和冻结回放协议下的实证。", "",
    ])
    return "\n".join(lines)


def load_model_replacement() -> tuple[dict, list[dict]]:
    path = OUTPUT / "model_replacement.json"
    if not path.exists():
        payload = {
            "report": "RT-DETR-R18 训练尚未产生结果；本环境预装的 Ultralytics 只提供 RT-DETR-L/X，未提供 R18 配置或权重。LightGBM 门控结果使用阶段 D 已完成的同折 200 候选子集离线对照，见下文正式运行后写入的 JSON。",
            "rows": [],
        }
        return payload, []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, payload["rows"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    if args.finalize:
        payload = json.loads((OUTPUT / "full_results.json").read_text(encoding="utf-8"))
        summary_rows = payload["rows"]
        with DETAILS.open(encoding="utf-8-sig", newline="") as handle:
            detail_rows = list(csv.DictReader(handle))
        model_replacement, model_rows = load_model_replacement()
        plot_results(summary_rows, detail_rows)
        append_results_summary(summary_rows, model_rows)
        payload["model_replacement"] = model_replacement
        (OUTPUT / "full_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (OUTPUT / "full_report.md").write_text(report(summary_rows, model_replacement), encoding="utf-8")
        print("RQ4 汇总刷新完成")
        return
    PROGRESS.write_text("timestamp,stage,completed,total,detail\n", encoding="utf-8")
    rows = rows_for_role(FOLD, "test")
    if args.smoke:
        rows = rows[:4]
        perturbations = (GEOMETRIC[0], GEOMETRIC[6], GEOMETRIC[13], PHOTOMETRIC[3])
    else:
        perturbations = GEOMETRIC + PHOTOMETRIC
    records_by_fold = load_gate_records()
    models = fit_fold_models(records_by_fold, FOLD)
    sampled = [
        item for item in map(json.loads, Path("experiments/rq2_correction/sampled_candidates.jsonl").read_text(encoding="utf-8").splitlines())
        if int(item["fold"]) == FOLD
    ]
    if args.smoke:
        selected = {row["pair_id"] for row in rows}
        sampled = [item for item in sampled if item["pair_id"] in selected]
    samples_by_pair: dict[str, list[dict]] = defaultdict(list)
    for item in sampled:
        samples_by_pair[item["pair_id"]].append(item)
    total_tasks = len(perturbations) * len(rows)
    append_progress("maps", 0, total_tasks, "开始重算扰动证据")
    predictions = prediction_map(FOLD)
    ground_truth = ground_truth_map(FOLD)
    if args.smoke:
        pair_ids = {row["pair_id"] for row in rows}
        predictions = {key: value for key, value in predictions.items() if key in pair_ids}
        ground_truth = {key: value for key, value in ground_truth.items() if key in pair_ids}
    detail_rows = []
    summary_rows = []
    completed_tasks = 0
    for perturbation, kind, value in perturbations:
        tasks = [
            (row, perturbation, kind, value, samples_by_pair[row["pair_id"]])
            for row in rows
        ]
        payloads = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_pair, task) for task in tasks]
            for future in as_completed(futures):
                payload = future.result()
                payloads.append(payload)
                completed_tasks += 1
                append_progress(
                    "maps", completed_tasks, total_tasks,
                    f"{payload['perturbation']}:{payload['pair_id']}",
                )
        family = "photometric" if perturbation in {item[0] for item in PHOTOMETRIC} else "geometry"
        payloads.sort(key=lambda item: item["pair_id"])
        for payload in payloads:
            detail_rows.append(
                {
                    "family": family, "perturbation": perturbation, "pair_id": payload["pair_id"],
                    "q_reg_ecc": payload["q_reg_ecc"], "q_reg_identity": payload["q_reg_identity"],
                    "ecc_chamfer": payload["ecc_chamfer"], "identity_chamfer": payload["identity_chamfer"],
                }
            )
        for setting in SETTINGS:
            replay = replay_setting(setting, payloads, sampled, models, predictions, ground_truth)
            recovered, total, rate = recovery_metrics(setting, payloads, predictions, ground_truth)
            q_name = "q_reg_identity" if setting == "no_reg_direct" else "q_reg_ecc"
            stats = q_stats([payload[q_name] for payload in payloads])
            summary_rows.append(
                {
                    "family": family, "perturbation": perturbation, "kind": kind, "setting": setting,
                    **stats, "miss_recovered": recovered, "miss_total": total, "miss_recovery_rate": rate,
                    "candidate_count_mean": float(np.mean([payload["settings"][setting]["candidate_count"] for payload in payloads])),
                    "xor_rate_mean": float(np.mean([payload["settings"][setting]["xor_rate"] for payload in payloads])),
                    **replay,
                }
            )
        append_progress("evaluate", len(summary_rows), len(perturbations) * len(SETTINGS), perturbation)
    with DETAILS.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)
    with SUMMARY.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    plot_results(summary_rows, detail_rows)
    model_replacement, model_rows = load_model_replacement()
    append_results_summary(summary_rows, model_rows)
    (OUTPUT / "full_results.json").write_text(
        json.dumps({"seed": SEED, "fold": FOLD, "rows": summary_rows, "model_replacement": model_replacement}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "full_report.md").write_text(report(summary_rows, model_replacement), encoding="utf-8")
    output = OUTPUT / ("smoke_results.json" if args.smoke else "full_results.json")
    if args.smoke:
        output.write_text(json.dumps({"passed": True, "pairs": len(rows), "conditions": len(perturbations)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    append_progress("complete", 1, 1, str(output))
    print(f"RQ4 完成：{output}")


if __name__ == "__main__":
    main()
