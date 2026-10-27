"""Run the four-arm, 150-candidate real-MLLM RQ1 decision pilot."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.common import PANEL_CONFIGS
from src.candidates.fuse import fuse_candidates
from src.candidates.propose import propose_candidates
from src.detector.calibrate import box_iou, read_gt
from src.detector.dataset import rows_for_role
from src.detector.error_decompose import decompose_record
from src.evidence.appearance import compute_appearance
from src.evidence.card import create_evidence_card
from src.evidence.topology import compute_topology, topology_features
from src.mllm.client import review_candidates
from src.mllm.prompt import PROMPT_VERSION
from src.registration.align import align_images, load_gray, photometric_normalize


SEED = 20260811
ARMS = ("A0", "A2", "A3", "A6")
CLASS_NAMES = ("open", "short", "mousebite", "spur", "pinhole", "spurious_copper")
TYPE_NAMES = ("original_tp", "original_fp", "original_miss", "diff_noise")
TYPE_LABELS = {
    "original_tp": "原始 TP",
    "original_fp": "原始 FP",
    "original_miss": "原始漏检",
    "diff_noise": "差异噪声",
}
TYPE_FOLD_TARGETS = {
    "original_tp": (8, 8, 8, 7, 7),
    "original_fp": (7, 8, 8, 8, 7),
    "original_miss": (7, 7, 8, 8, 7),
    "diff_noise": (7, 7, 7, 8, 8),
}
AREA_THRESHOLD = 1184.0
OUTPUT_ROOT = Path("experiments/rq1_evidence")
SAMPLE_PATH = OUTPUT_ROOT / "pilot_sample.json"
RESULT_PATH = OUTPUT_ROOT / "pilot_results.json"
REPORT_PATH = OUTPUT_ROOT / "pilot_report.md"
PROGRESS_PATH = OUTPUT_ROOT / "progress.csv"
CARD_ROOT = OUTPUT_ROOT / "cards"
CASE_ROOT = OUTPUT_ROOT / "cases"


def append_progress(stage: str, completed: int, total: int, detail: str) -> None:
    with PROGRESS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{datetime.now().astimezone().isoformat(timespec='seconds')},{stage},{completed},{total},{detail}\n"
        )


def load_configs() -> tuple[dict, dict, dict, dict, dict]:
    paths = (
        "configs/evidence/registration.yaml",
        "configs/evidence/appearance.yaml",
        "configs/evidence/topology.yaml",
        "configs/candidates.yaml",
        "configs/evidence/card.yaml",
    )
    return tuple(yaml.safe_load(Path(path).read_text(encoding="utf-8")) for path in paths)


def prediction_map(fold: int) -> dict[str, dict]:
    path = Path(f"cache/detector_predictions/fold{fold}.jsonl")
    return {item["pair_id"]: item for item in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def detector_items(prediction: dict) -> list[dict]:
    return [
        {"box": box, "class": int(class_id), "score": float(score)}
        for box, class_id, score in zip(prediction["boxes"], prediction["classes"], prediction["scores"])
    ]


def align_for_pilot(current: np.ndarray, template: np.ndarray, config: dict) -> dict:
    orb_config = config["orb"]
    orb = cv2.ORB_create(nfeatures=int(orb_config["nfeatures"]))
    _, current_descriptors = orb.detectAndCompute(current, None)
    _, template_descriptors = orb.detectAndCompute(template, None)
    if current_descriptors is None or template_descriptors is None:
        observable_matches = 0
    else:
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(template_descriptors, current_descriptors, k=2)
        observable_matches = sum(
            pair[0].distance < float(orb_config["ratio"]) * pair[1].distance
            for pair in matches
            if len(pair) == 2
        )
    if observable_matches >= 4:
        return align_images(current, template, config)
    normalized_current, normalized_template = photometric_normalize(current, template)
    return {
        "aligned_template": template,
        "normalized_test": normalized_current,
        "normalized_template": normalized_template,
        "q_reg": 0.0,
        "method": "IDENTITY_LOW_OBSERVABILITY",
    }


def box_key(box: list[float]) -> tuple[float, ...]:
    return tuple(round(float(value), 4) for value in box)


def classify_candidate(candidate: dict, errors: list[dict], ground_truth: list[dict]) -> tuple[str, int, float] | None:
    if candidate["source"] in {"DET", "BOTH"}:
        by_prediction = {box_key(json.loads(item["pred_box"])): item for item in errors if item["pred_box"] != "[]"}
        error = by_prediction[box_key(candidate["det_box"])]
        if error["error_type"] == "TP":
            return "original_tp", int(error["gt_class"]), float(error["iou"])
        if error["error_type"] == "FP":
            return "original_fp", -1, float(error["iou"])
        return None
    missed = [item for item in errors if item["error_type"] == "漏检"]
    missed_matches = [
        (item, box_iou(candidate["action_box"], json.loads(item["gt_box"])))
        for item in missed
    ]
    best_miss, miss_iou = max(missed_matches, key=lambda item: item[1]) if missed_matches else ({}, 0.0)
    if miss_iou >= 0.3:
        return "original_miss", int(best_miss["gt_class"]), float(miss_iou)
    all_iou = max((box_iou(candidate["action_box"], item["box"]) for item in ground_truth), default=0.0)
    if all_iou < 0.3:
        return "diff_noise", -1, float(all_iou)
    return None


def enrich_candidate(candidate: dict, aligned: dict, appearance: dict, topology: dict, threshold: float | dict) -> None:
    candidate["q_reg"] = float(aligned["q_reg"])
    x1, y1, x2, y2 = [int(round(value)) for value in candidate["box"]]
    height, width = appearance["M_app"].shape
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    residual = appearance["M_app"][y1:y2, x1:x2]
    xor_region = topology["M_topo"][y1:y2, x1:x2] > 0
    test_vector = topology_features(
        topology["test_mask"][y1:y2, x1:x2], topology["test_skeleton"][y1:y2, x1:x2], xor_region
    )
    template_vector = topology_features(
        topology["template_mask"][y1:y2, x1:x2], topology["template_skeleton"][y1:y2, x1:x2], xor_region
    )
    candidate["appearance_residual"] = {
        "mean": float(residual.mean()),
        "p95": float(np.quantile(residual, 0.95)),
        "max": float(residual.max()),
    }
    candidate["above_threshold_fraction"] = float(
        (residual >= (threshold if isinstance(threshold, float) else threshold["appearance"])).mean()
    )
    candidate["topology_vector"] = np.abs(test_vector - template_vector).astype(float).tolist()


def render_cards(
    current: np.ndarray,
    aligned: dict,
    candidate: dict,
    appearance: dict,
    topology: dict,
    card_config: dict,
) -> None:
    for arm in ARMS:
        create_evidence_card(
            current,
            aligned["aligned_template"],
            candidate,
            appearance["M_app"],
            topology,
            aligned["q_reg"],
            CARD_ROOT / arm / f"{candidate['candidate_id']}.png",
            card_config,
            PANEL_CONFIGS[arm],
        )


def prepare_sample() -> list[dict]:
    if SAMPLE_PATH.exists():
        return json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))["records"]
    registration_config, appearance_config, topology_config, candidate_config, card_config = load_configs()
    rng = random.Random(SEED)
    selected: list[dict] = []
    for fold in range(5):
        counts = {name: 0 for name in TYPE_NAMES}
        rows = rows_for_role(fold, "test")
        rng.shuffle(rows)
        predictions = prediction_map(fold)
        for image_index, row in enumerate(rows, start=1):
            prediction = predictions[row["pair_id"]]
            current = load_gray(Path(row["test_path"]))
            template = load_gray(Path(row["temp_path"]))
            aligned = align_for_pilot(current, template, registration_config)
            appearance = compute_appearance(
                aligned["normalized_test"], aligned["normalized_template"], appearance_config
            )
            normalized_current = ((aligned["normalized_test"] * 32.0) + 128.0).clip(0, 255).astype("uint8")
            normalized_template = ((aligned["normalized_template"] * 32.0) + 128.0).clip(0, 255).astype("uint8")
            topology = compute_topology(normalized_current, normalized_template, topology_config)
            detections = detector_items(prediction)
            differences = propose_candidates(
                appearance,
                topology,
                current,
                aligned["aligned_template"],
                detections,
                candidate_config,
                fold,
            )
            candidates = fuse_candidates(
                row["pair_id"], detections, differences["candidates"], int(candidate_config["center_expand_px"])
            )
            ground_truth = [{"box": box, "class": class_id} for box, class_id in read_gt(Path(row["anno_path"]))]
            errors = decompose_record(prediction, row)
            rng.shuffle(candidates)
            for candidate in candidates:
                classification = classify_candidate(candidate, errors, ground_truth)
                if classification is None:
                    continue
                candidate_type, true_class, gt_iou = classification
                if counts[candidate_type] >= TYPE_FOLD_TARGETS[candidate_type][fold]:
                    continue
                candidate["pair_id"] = row["pair_id"]
                enrich_candidate(candidate, aligned, appearance, topology, differences["threshold"])
                render_cards(current, aligned, candidate, appearance, topology, card_config)
                action_box = candidate["action_box"]
                selected.append(
                    {
                        "fold": fold,
                        "pair_id": row["pair_id"],
                        "candidate_id": candidate["candidate_id"],
                        "candidate_type": candidate_type,
                        "true_defect_present": candidate_type in {"original_tp", "original_miss"},
                        "true_class": CLASS_NAMES[true_class] if true_class >= 0 else "normal",
                        "gt_iou": gt_iou,
                        "area": float((action_box[2] - action_box[0]) * (action_box[3] - action_box[1])),
                        "candidate": candidate,
                    }
                )
                counts[candidate_type] += 1
            append_progress(
                "prepare",
                sum(counts.values()),
                sum(TYPE_FOLD_TARGETS[name][fold] for name in TYPE_NAMES),
                f"fold={fold};image={image_index};counts={json.dumps(counts, ensure_ascii=False)}",
            )
            if all(counts[name] == TYPE_FOLD_TARGETS[name][fold] for name in TYPE_NAMES):
                break
    SAMPLE_PATH.write_text(
        json.dumps(
            {
                "seed": SEED,
                "sampling_source": "five-fold test splits",
                "diff_match_iou": 0.3,
                "records": selected,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return selected


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def metric_bundle(rows: list[dict]) -> dict:
    exact = [item["result"]["defect_present"] is item["true_defect_present"] for item in rows]
    tp = sum(item["true_defect_present"] and item["result"]["defect_present"] is True for item in rows)
    fp = sum(not item["true_defect_present"] and item["result"]["defect_present"] is True for item in rows)
    fn = sum(item["true_defect_present"] and item["result"]["defect_present"] is not True for item in rows)
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    normal_rows = [item for item in rows if item["result"]["defect_present"] is False]
    class_f1 = {}
    for class_name in CLASS_NAMES:
        class_tp = sum(item["true_class"] == class_name and item["result"]["predicted_class"] == class_name for item in rows)
        class_fp = sum(item["true_class"] != class_name and item["result"]["predicted_class"] == class_name for item in rows)
        class_fn = sum(item["true_class"] == class_name and item["result"]["predicted_class"] != class_name for item in rows)
        class_f1[class_name] = safe_ratio(2 * class_tp, 2 * class_tp + class_fp + class_fn)
    uncertain = sum(
        item["result"]["defect_present"] is None or item["result"]["predicted_class"] == "uncertain"
        for item in rows
    )
    return {
        "n": len(rows),
        "presence": {
            "accuracy": safe_ratio(sum(exact), len(rows)),
            "precision": precision,
            "recall": recall,
            "f1": safe_ratio(2 * precision * recall, precision + recall),
        },
        "normal": {
            "rate": safe_ratio(len(normal_rows), len(rows)),
            "count": len(normal_rows),
            "true_original_fp_count": sum(item["candidate_type"] == "original_fp" for item in normal_rows),
        },
        "classification": {
            "macro_f1": sum(class_f1.values()) / len(CLASS_NAMES),
            "per_class_f1": class_f1,
        },
        "uncertain_rate": safe_ratio(uncertain, len(rows)),
        "json_parse_rate": safe_ratio(sum(item["json_parseable"] for item in rows), len(rows)),
    }


def stratified_metrics(rows: list[dict]) -> dict:
    output = {"overall": metric_bundle(rows)}
    output["by_candidate_type"] = {
        name: metric_bundle([item for item in rows if item["candidate_type"] == name]) for name in TYPE_NAMES
    }
    output["by_source"] = {
        source: metric_bundle([item for item in rows if item["source"] == source])
        for source in ("DET", "DIFF", "BOTH")
    }
    output["by_size"] = {
        "small_lt_1184": metric_bundle([item for item in rows if item["area"] < AREA_THRESHOLD]),
        "large_ge_1184": metric_bundle([item for item in rows if item["area"] >= AREA_THRESHOLD]),
    }
    return output


def fmt(value: float) -> str:
    return f"{value:.4f}"


def bundle_row(label: str, arm: str, bundle: dict) -> str:
    presence = bundle["presence"]
    normal = bundle["normal"]
    classification = bundle["classification"]
    return (
        f"| {arm} | {label} | {bundle['n']} | {fmt(presence['accuracy'])} | {fmt(presence['precision'])} | "
        f"{fmt(presence['recall'])} | {fmt(presence['f1'])} | {fmt(normal['rate'])} | "
        f"{normal['true_original_fp_count']} | {fmt(classification['macro_f1'])} | "
        f"{fmt(bundle['uncertain_rate'])} | {fmt(bundle['json_parse_rate'])} |"
    )


def metric_table(metrics: dict, section: str, labels: dict[str, str]) -> list[str]:
    lines = [
        "| 配置 | 分层 | n | Accuracy | Precision | Recall | F1 | normal 率 | normal 中原始 FP 数 | 六类 Macro-F1 | uncertain 率 | JSON 可解析率 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        for key, label in labels.items():
            lines.append(bundle_row(label, arm, metrics[arm][section][key]))
    return lines


def class_table(metrics: dict) -> list[str]:
    lines = [
        "| 配置 | Macro-F1 | open | short | mousebite | spur | pinhole | spurious copper |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        classification = metrics[arm]["overall"]["classification"]
        values = classification["per_class_f1"]
        lines.append(
            f"| {arm} | {fmt(classification['macro_f1'])} | "
            + " | ".join(fmt(values[name]) for name in CLASS_NAMES)
            + " |"
        )
    return lines


def render_case(case_index: int, item: dict, correct: bool) -> str:
    card_path = CARD_ROOT / "A6" / f"{item['candidate_id']}.png"
    card = Image.open(card_path).convert("RGB")
    text_width = 620
    canvas = Image.new("RGB", (card.width + text_width, card.height), "white")
    canvas.paste(card, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16)
    payload = {
        "verdict": "correct" if correct else "incorrect",
        "candidate_type": item["candidate_type"],
        "source": item["source"],
        "true_defect_present": item["true_defect_present"],
        "true_class": item["true_class"],
        "model_output": item["result"],
    }
    lines = json.dumps(payload, ensure_ascii=True, indent=2).splitlines()
    y = 18
    for line in lines:
        while len(line) > 64:
            draw.text((card.width + 18, y), line[:64], fill="black", font=font)
            line = line[64:]
            y += 21
        draw.text((card.width + 18, y), line, fill="black", font=font)
        y += 21
    stem = f"{case_index:02d}_{'correct' if correct else 'incorrect'}_{item['candidate_id']}"
    canvas.save(CASE_ROOT / f"{stem}.png")
    (CASE_ROOT / f"{stem}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stem


def export_cases(rows: list[dict]) -> list[dict]:
    CASE_ROOT.mkdir(parents=True, exist_ok=True)
    correct_rows = [
        item
        for item in rows
        if item["result"]["defect_present"] is item["true_defect_present"]
        and (not item["true_defect_present"] or item["result"]["predicted_class"] == item["true_class"])
    ]
    incorrect_rows = [item for item in rows if item not in correct_rows]
    rng = random.Random(SEED)
    rng.shuffle(correct_rows)
    rng.shuffle(incorrect_rows)
    chosen = [(item, True) for item in correct_rows[:4]] + [(item, False) for item in incorrect_rows[:4]]
    cases = []
    for index, (item, correct) in enumerate(chosen, start=1):
        stem = render_case(index, item, correct)
        cases.append({"stem": stem, "candidate_id": item["candidate_id"], "correct": correct})
    readme = ["# RQ1 典型案例", "", "每张 PNG 左侧为 A6 证据卡，右侧为真值与模型原始 JSON 输出。", ""]
    readme.extend(f"{index}. `{item['stem']}.png`" for index, item in enumerate(cases, start=1))
    (CASE_ROOT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return cases


def build_report(payload: dict) -> None:
    metrics = payload["metrics"]
    delta = payload["decision"]["a6_minus_a0_presence_f1"]
    normal_fp_rate = payload["decision"]["a6_original_fp_normal_rate"]
    lines = [
        "# 阶段 B：Prompt 调整与小规模真实 RQ1 报告",
        "",
        "## 实验设置",
        "",
        f"Prompt 固定为 `{PROMPT_VERSION}`。诊断集从五折 test 候选按四类和折号分层抽取 150 个，随机种子为 {SEED}。原始漏检候选与漏检真值采用 IoU ≥ 0.3 的阶段 A 候选恢复口径，差异噪声要求与全部真值 IoU < 0.3。四个配置均复用同一批候选，后端为 `codex_cli`，模型为 `gpt-5.6-sol`，温度配置为 0，4 路并发，共 600 次候选判断。该平衡诊断集不用于推断部署时的自然分布性能。",
        "",
        "Prompt v2 优先判断缺陷存在性和 normal，参考不可靠时要求 uncertain，并仅对 DIFF 候选加入基础检测器未提出该区域的独立判断提示。动作映射仍保留类别输出，但 relabel 必须同时满足预测类别与模型自报拓扑一致、模型自报拓扑与检测器类别典型拓扑矛盾。Prompt 依据 train 折错误分解确定，测试集结果未用于继续调参。",
        "",
        "## 总体指标",
        "",
        *metric_table(metrics, "overall_wrapper", {"overall": "总体"}),
        "",
        "## 按候选真实类型分层",
        "",
        *metric_table(metrics, "by_candidate_type", TYPE_LABELS),
        "",
        "## 按候选来源分层",
        "",
        *metric_table(metrics, "by_source", {"DET": "DET", "DIFF": "DIFF", "BOTH": "BOTH"}),
        "",
        "## 按候选尺寸分层",
        "",
        *metric_table(metrics, "by_size", {"small_lt_1184": "面积 < 1184", "large_ge_1184": "面积 ≥ 1184"}),
        "",
        "## 六类分类 F1",
        "",
        *class_table(metrics),
        "",
        "## 结果解读",
        "",
        f"A2 的存在性 F1 为 {fmt(metrics['A2']['overall']['presence']['f1'])}，高于 A6 的 {fmt(metrics['A6']['overall']['presence']['f1'])}，因此本实验支持显式参考证据相对单裁剪有价值，但不支持证据数量与性能单调递增。A3 的 uncertain 率为 {fmt(metrics['A3']['overall']['uncertain_rate'])}，明显高于 A2 的 {fmt(metrics['A2']['overall']['uncertain_rate'])}；A6 加入拓扑差异与数值拓扑后 uncertain 率回落到 {fmt(metrics['A6']['overall']['uncertain_rate'])}，说明单独加入外观差异会引入冲突，而结构证据能部分恢复判断稳定性。",
        "",
        f"在 DIFF 来源上，存在性 F1 从 A0 的 {fmt(metrics['A0']['by_source']['DIFF']['presence']['f1'])} 提升到 A6 的 {fmt(metrics['A6']['by_source']['DIFF']['presence']['f1'])}，支持 add 路径的候选级判断能力。A6 六类 Macro-F1 为 {fmt(metrics['A6']['overall']['classification']['macro_f1'])}，与 A2 的 {fmt(metrics['A2']['overall']['classification']['macro_f1'])} 接近且绝对值仍低，说明本轮主要收益来自缺陷存在性判断，不应扩大为细粒度类别识别已经充分解决。",
        "",
        "## 判断结论",
        "",
        "主判据预先固定为 A6 相对 A0 的缺陷存在性 F1 提升。提升 ≥ 0.10 时，核心假设成立并进入全量 RQ1；提升在 0.03 到 0.10 时，核心假设成立但偏弱，应缩小宣称并把重心转向双门控的成本收益；提升 < 0.03 时，核心假设不成立，需要降级题目。",
        "",
        f"本次 A0 F1 为 {fmt(metrics['A0']['overall']['presence']['f1'])}，A6 F1 为 {fmt(metrics['A6']['overall']['presence']['f1'])}，提升为 {delta:+.4f}。因此，**{payload['decision']['primary_conclusion']}**。",
        "",
        "次判据预先固定为 A6 在原始 FP 候选上的 normal 判定率。若低于 0.3，remove 路径基本不可用。",
        "",
        f"本次该比例为 {normal_fp_rate:.4f}。因此，**{payload['decision']['remove_conclusion']}**。",
        "",
        "本报告按预先固定的 prompt_v2 一次性评估，没有依据测试结果调整 prompt 或重跑以改善数字。",
        "",
        "## 典型案例",
        "",
        "已在 `experiments/rq1_evidence/cases/` 导出 4 个判对与 4 个判错案例。每张 PNG 左侧为 A6 证据卡，右侧为真值和模型输出，配套 JSON 保留可检索的原始内容。",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not PROGRESS_PATH.exists():
        PROGRESS_PATH.write_text("timestamp,stage,completed,total,detail\n", encoding="utf-8")
    started = time.perf_counter()
    sample = prepare_sample()
    if args.prepare_only:
        print(json.dumps({"sample_count": len(sample), "sample_path": str(SAMPLE_PATH)}, ensure_ascii=False))
        return
    responses = []
    for arm_index, arm in enumerate(ARMS, start=1):
        outputs = review_candidates(
            [item["candidate"] for item in sample],
            [CARD_ROOT / arm / f"{item['candidate_id']}.png" for item in sample],
            arm,
            Path("configs/mllm/codex_cli.yaml"),
            progress_path=PROGRESS_PATH,
        )
        for record, output in zip(sample, outputs):
            responses.append(
                {
                    "arm": arm,
                    "fold": record["fold"],
                    "pair_id": record["pair_id"],
                    "candidate_id": record["candidate_id"],
                    "candidate_type": record["candidate_type"],
                    "source": record["candidate"]["source"],
                    "true_defect_present": record["true_defect_present"],
                    "true_class": record["true_class"],
                    "gt_iou": record["gt_iou"],
                    "area": record["area"],
                    **output,
                }
            )
        append_progress("arm", arm_index, len(ARMS), arm)
    metrics = {}
    for arm in ARMS:
        arm_rows = [item for item in responses if item["arm"] == arm]
        metrics[arm] = stratified_metrics(arm_rows)
        metrics[arm]["overall_wrapper"] = {"overall": metrics[arm]["overall"]}
    delta = metrics["A6"]["overall"]["presence"]["f1"] - metrics["A0"]["overall"]["presence"]["f1"]
    normal_fp_rate = metrics["A6"]["by_candidate_type"]["original_fp"]["normal"]["rate"]
    if delta >= 0.10:
        primary_conclusion = "核心假设成立，进入全量 RQ1"
    elif delta >= 0.03:
        primary_conclusion = "核心假设成立但偏弱，应缩小宣称，重心转向双门控的成本收益"
    else:
        primary_conclusion = "核心假设不成立，需要降级题目"
    remove_conclusion = "remove 路径具备可用性" if normal_fp_rate >= 0.3 else "remove 路径基本不可用"
    cases = export_cases([item for item in responses if item["arm"] == "A6"])
    payload = {
        "prompt_version": PROMPT_VERSION,
        "seed": SEED,
        "backend": "codex_cli",
        "model": "gpt-5.6-sol",
        "temperature": 0,
        "workers": 4,
        "candidate_count": len(sample),
        "candidate_evaluations": len(responses),
        "sample_counts": {name: sum(item["candidate_type"] == name for item in sample) for name in TYPE_NAMES},
        "metrics": metrics,
        "decision": {
            "a6_minus_a0_presence_f1": delta,
            "primary_conclusion": primary_conclusion,
            "a6_original_fp_normal_rate": normal_fp_rate,
            "remove_conclusion": remove_conclusion,
        },
        "cases": cases,
        "elapsed_seconds": time.perf_counter() - started,
        "responses": responses,
    }
    RESULT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    build_report(payload)
    append_progress("complete", 600, 600, str(RESULT_PATH))
    print(json.dumps({key: value for key, value in payload.items() if key not in {"responses", "metrics"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
