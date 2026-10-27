"""Run the 800-candidate, fourteen-arm RQ1 ablation study."""

from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.common import PANEL_CONFIGS
from experiments.rq1_evidence.run import (
    CLASS_NAMES,
    TYPE_LABELS,
    TYPE_NAMES,
    align_for_pilot,
    classify_candidate,
    detector_items,
    enrich_candidate,
    load_configs,
    metric_bundle,
    prediction_map,
)
from src.candidates.fuse import fuse_candidates
from src.candidates.propose import propose_candidates
from src.detector.calibrate import read_gt
from src.detector.dataset import rows_for_role
from src.detector.error_decompose import decompose_record
from src.evidence.appearance import compute_appearance
from src.evidence.card import create_evidence_card
from src.evidence.topology import compute_topology
from src.mllm.action_map import map_action
from src.mllm.client import _run_one, cache_key
from src.mllm.prompt import (
    CLOSED_DEFECT_KNOWLEDGE_V2,
    PANEL_DESCRIPTIONS,
    SYSTEM_PROMPT_V2,
    build_prompt,
    build_prompt_v1,
    write_schema,
)
from src.registration.align import load_gray


SEED = 20260811
# A2N/A2T/A6F 已在阶段 B2 用 450 次专门调用得出确定结论（均低于 A2），不再重跑
ARMS = (
    "A0", "A1", "A2", "A3", "A4", "A5", "A6",
    "A6_label_aware", "A6_no_reg_warning", "A6_free_text", "A6_prompt_v1",
)
TARGET_PER_TYPE_FOLD = 10
OUTPUT_ROOT = Path("experiments/rq1_evidence")
SAMPLE_PATH = OUTPUT_ROOT / "full_sample.json"
RESULT_PATH = OUTPUT_ROOT / "full_results.json"
REPORT_PATH = OUTPUT_ROOT / "full_report.md"
PROGRESS_PATH = OUTPUT_ROOT / "progress_full.csv"
CARD_ROOT = OUTPUT_ROOT / "cards_full"
CONFIG_PATH = Path("configs/mllm/codex_cli.yaml")
FREE_CACHE_PATH = Path("cache/mllm_outputs/codex_cli_free_text.jsonl")
PANELS = {
    **{name: tuple(sorted(panels)) for name, panels in PANEL_CONFIGS.items()},
    "A2N": ("P1", "P2", "P3", "P6"),
    "A2T": ("P1", "P2", "P3", "P5"),
    "A6F": ("P1", "P2", "P3", "P4", "P5", "P6"),
}


def progress(completed: int, total: int, arm: str, started: float) -> None:
    elapsed = time.perf_counter() - started
    eta = elapsed * (total - completed) / completed if completed else 0.0
    with PROGRESS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{datetime.now().astimezone().isoformat(timespec='seconds')},{completed},"
            f"{total - completed},{arm},{eta:.1f}\n"
        )


def prepare_sample(started: float) -> list[dict]:
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
        for row in rows:
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
            ground_truth = [{"box": box, "class": class_id} for box, class_id in read_gt(Path(row["anno_path"]))]
            errors = decompose_record(prediction, row)
            rng.shuffle(candidates)
            for candidate in candidates:
                classification = classify_candidate(candidate, errors, ground_truth)
                if classification is None:
                    continue
                candidate_type, true_class, gt_iou = classification
                if counts[candidate_type] >= TARGET_PER_TYPE_FOLD:
                    continue
                candidate["pair_id"] = row["pair_id"]
                enrich_candidate(candidate, aligned, appearance, topology, differences["threshold"])
                action_box = candidate["action_box"]
                record = {
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
                render_cards(current, aligned, appearance, topology, card_config, differences["threshold"], record)
                selected.append(record)
                counts[candidate_type] += 1
            if len(selected) % 20 == 0:
                progress(0, len(ARMS) * 200, f"prepare_fold{fold}", started)
            if all(count == TARGET_PER_TYPE_FOLD for count in counts.values()):
                break
        if not all(count == TARGET_PER_TYPE_FOLD for count in counts.values()):
            raise RuntimeError(f"fold {fold} 分层候选不足: {counts}")
    SAMPLE_PATH.write_text(
        json.dumps({"seed": SEED, "sampling_source": "five-fold test", "records": selected}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return selected


def render_cards(
    current: np.ndarray,
    aligned: dict,
    appearance: dict,
    topology: dict,
    card_config: dict,
    threshold: float | dict,
    record: dict,
) -> None:
    candidate = record["candidate"]
    for arm in ("A0", "A1", "A2", "A3", "A4", "A5", "A6"):
        create_evidence_card(
            current, aligned["aligned_template"], candidate, appearance["M_app"], topology, aligned["q_reg"],
            CARD_ROOT / arm / f"{record['candidate_id']}.png", card_config, set(PANELS[arm]),
        )
    no_numeric = {**card_config, "include_appearance_numeric": False}
    create_evidence_card(
        current, aligned["aligned_template"], candidate, appearance["M_app"], topology, aligned["q_reg"],
        CARD_ROOT / "A2N" / f"{record['candidate_id']}.png", no_numeric, set(PANELS["A2N"]),
    )
    create_evidence_card(
        current, aligned["aligned_template"], candidate, appearance["M_app"], topology, aligned["q_reg"],
        CARD_ROOT / "A2T" / f"{record['candidate_id']}.png", card_config, set(PANELS["A2T"]),
    )
    appearance_threshold = threshold if isinstance(threshold, float) else threshold["appearance"]
    binary_config = {**card_config, "appearance_mode": "binary", "appearance_threshold": appearance_threshold}
    create_evidence_card(
        current, aligned["aligned_template"], candidate, appearance["M_app"], topology, aligned["q_reg"],
        CARD_ROOT / "A6F" / f"{record['candidate_id']}.png", binary_config, set(PANELS["A6F"]),
    )


def custom_prompt(candidate: dict, arm: str) -> str:
    if arm == "A6_prompt_v1":
        return build_prompt_v1(candidate, "A6")
    if arm == "A6_label_aware":
        return build_prompt(candidate, "A6", label_aware=True)
    panels = PANELS[arm] if arm in PANELS else PANELS["A6"]
    panel_text = "\n".join(f"{panel}: {PANEL_DESCRIPTIONS[panel]};" for panel in panels)
    source_text = (
        "\nThis candidate marks a region that the base detector did not propose. Independently decide whether a defect exists here."
        if candidate["source"] == "DIFF" else ""
    )
    system = SYSTEM_PROMPT_V2
    knowledge = CLOSED_DEFECT_KNOWLEDGE_V2
    reliability = "The normal reference represents expected structure only when it is reliable.\n"
    if arm == "A6_no_reg_warning":
        system = system.replace(" or the supplied normal reference appears unreliable or misaligned", "")
        knowledge = knowledge.split(" If the reference is unreliable", maxsplit=1)[0]
        reliability = ""
    ending = "Return the required JSON only."
    if arm == "A6_free_text":
        ending = (
            "Return plain text, not JSON, on one line using exactly these fields separated by semicolons: "
            "defect_present=true|false|uncertain; predicted_class=<closed label>; "
            "reference_reliability=high|medium|low|uncertain; evidence_strength=0|1|2|3."
        )
    return (
        f"{system}\n\n{knowledge}\n\nThe evidence card contains only these panels:\n{panel_text}\n\n"
        f"{reliability}First decide defect existence or normality, then the most likely class and evidence strength."
        f"{source_text}\n\n{ending}"
    )


def free_text_task(task: tuple[dict, dict, str, str, str]) -> dict:
    config, candidate, image_path, prompt_text, key = task
    output_path = Path("cache/mllm_outputs/tmp") / f"{key}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    environment = __import__("os").environ.copy()
    environment["https_proxy"] = config["proxy"]
    environment["http_proxy"] = config["proxy"]
    command = [
        config["codex_path"], "exec", "-m", config["model"], "-c",
        f"model_reasoning_effort={config['reasoning_effort']}", "-c",
        f"model_temperature={config['temperature']}", "-i", image_path, "-o", str(output_path),
        "--skip-git-repo-check", prompt_text,
    ]
    started = time.perf_counter()
    subprocess.run(command, env=environment, check=True, stdout=subprocess.DEVNULL)
    text = output_path.read_text(encoding="utf-8").strip()
    output_path.unlink()
    fields = dict(re.findall(r"(defect_present|predicted_class|reference_reliability|evidence_strength)\s*=\s*([^;\s]+)", text))
    present = {"true": True, "false": False, "uncertain": None}[fields["defect_present"]]
    reliability = None if fields["reference_reliability"] == "uncertain" else fields["reference_reliability"]
    result = {
        "defect_present": present,
        "predicted_class": fields["predicted_class"],
        "reference_reliability": reliability,
        "appearance_pattern": ["uncertain"],
        "topology_pattern": ["uncertain"],
        "evidence_strength": int(fields["evidence_strength"]),
        "reason_codes": ["NO_SUPPORT"],
    }
    return {"key": key, "result": result, "json_parseable": False, "raw_text": text, "latency_seconds": time.perf_counter() - started}


def run_arm(sample: list[dict], arm: str, offset: int, started: float) -> list[dict]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    schema_path = write_schema()
    cache_path = FREE_CACHE_PATH if arm == "A6_free_text" else Path(config["cache_file"])
    cached = {item["key"]: item for item in map(json.loads, cache_path.read_text(encoding="utf-8").splitlines())} if cache_path.exists() else {}
    card_arm = arm if arm in PANELS else "A6"
    keys = [cache_key(item["candidate"], arm, config["model"], float(config["temperature"])) for item in sample]
    tasks = []
    for item, key in zip(sample, keys):
        if key in cached:
            continue
        image_path = str(CARD_ROOT / card_arm / f"{item['candidate_id']}.png")
        prompt_text = custom_prompt(item["candidate"], arm) if arm not in PANEL_CONFIGS else build_prompt(item["candidate"], arm)
        if arm == "A6_free_text":
            tasks.append((config, item["candidate"], image_path, prompt_text, key))
        else:
            tasks.append((config, item["candidate"], image_path, str(schema_path), prompt_text, key))
    completed = len(sample) - len(tasks)
    if completed:
        progress(offset + completed, len(sample) * len(ARMS), arm, started)
    if tasks:
        worker = free_text_task if arm == "A6_free_text" else _run_one
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a", encoding="utf-8") as handle:
            with ProcessPoolExecutor(max_workers=int(config["workers"])) as executor:
                futures = [executor.submit(worker, task) for task in tasks]
                for future in as_completed(futures):
                    output = future.result()
                    handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                    handle.flush()
                    cached[output["key"]] = output
                    completed += 1
                    if completed % 20 == 0 or completed == len(sample):
                        progress(offset + completed, len(sample) * len(ARMS), arm, started)
    return [cached[key] for key in keys]


def expected_action(candidate_type: str) -> str:
    return {
        "original_tp": "keep", "original_fp": "remove", "original_miss": "add", "diff_noise": "reject",
    }[candidate_type]


def metrics_for(rows: list[dict]) -> dict:
    bundle = metric_bundle(rows)
    bundle["action_accuracy"] = sum(
        map_action(item["candidate"], item["result"]) == expected_action(item["candidate_type"]) for item in rows
    ) / len(rows)
    return bundle


def stratify(rows: list[dict], key_name: str, assignments: dict[str, str]) -> dict:
    return {
        name: metrics_for([item for item in rows if assignments[item["candidate_id"]] == name])
        for name in dict.fromkeys(assignments.values())
    }


def bootstrap_delta(rows: list[dict], a0_rows: list[dict]) -> list[float]:
    by_arm = {item["candidate_id"]: item for item in rows}
    by_a0 = {item["candidate_id"]: item for item in a0_rows}
    pair_ids = list(dict.fromkeys(item["pair_id"] for item in rows))
    by_pair = defaultdict(list)
    for item in rows:
        by_pair[item["pair_id"]].append(item["candidate_id"])
    rng = np.random.default_rng(SEED)
    deltas = []
    for _ in range(2000):
        sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
        ids = [candidate_id for pair_id in sampled for candidate_id in by_pair[pair_id]]
        deltas.append(metrics_for([by_arm[item] for item in ids])["presence"]["f1"] - metrics_for([by_a0[item] for item in ids])["presence"]["f1"])
    return [float(value) for value in np.quantile(deltas, [0.025, 0.975])]


def summarize(sample: list[dict], responses: list[dict]) -> dict:
    q_values = np.asarray([item["candidate"]["q_reg"] for item in sample])
    area_values = np.asarray([item["area"] for item in sample])
    q1, q2 = np.quantile(q_values, [1 / 3, 2 / 3])
    a1, a2 = np.quantile(area_values, [1 / 3, 2 / 3])
    q_assignment = {item["candidate_id"]: "低" if item["candidate"]["q_reg"] <= q1 else "中" if item["candidate"]["q_reg"] <= q2 else "高" for item in sample}
    size_assignment = {item["candidate_id"]: "小" if item["area"] <= a1 else "中" if item["area"] <= a2 else "大" for item in sample}
    source_assignment = {item["candidate_id"]: item["candidate"]["source"] for item in sample}
    by_arm = {arm: [item for item in responses if item["arm"] == arm] for arm in ARMS}
    output = {}
    for arm in ARMS:
        output[arm] = {
            "overall": metrics_for(by_arm[arm]),
            "by_size": stratify(by_arm[arm], "size", size_assignment),
            "by_q_reg": stratify(by_arm[arm], "q_reg", q_assignment),
            "by_source": stratify(by_arm[arm], "source", source_assignment),
        }
        delta = output[arm]["overall"]["presence"]["f1"] - metrics_for(by_arm["A0"])["presence"]["f1"]
        output[arm]["a0_f1_delta"] = delta
        output[arm]["a0_f1_delta_ci95"] = bootstrap_delta(by_arm[arm], by_arm["A0"])
    return output


def build_report(payload: dict) -> None:
    metrics = payload["metrics"]
    lines = [
        "# 阶段 C：全量 RQ1 消融报告", "", "## 实验口径", "",
        f"五折 test 候选按原始 TP、原始 FP、原始漏检、差异噪声及折号分层采样 800 个候选，随机种子 {SEED}。每个折和每类各 40 个。该候选级诊断集不用于推断部署性能。",
        "", "下游 RQ2、RQ3 固定采用 A2，只向大模型展示当前紧裁剪、正常模板和上下文。外观与拓扑差异证据在候选判断中未产生正增益，8 维拓扑向量重新定位为复检门和接受门的结构化特征，不再作为大模型视觉输入。",
        "", "## 总体结果", "",
        "| 配置 | n | Accuracy | Precision | Recall | F1 | 相对 A0 | 聚类 bootstrap 95% CI | Macro-F1 | 动作准确率 | normal 率 | normal 命中原始 FP | uncertain | JSON 可解析率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        item = metrics[arm]
        overall = item["overall"]
        presence = overall["presence"]
        ci = item["a0_f1_delta_ci95"]
        lines.append(
            f"| {arm} | {overall['n']} | {presence['accuracy']:.4f} | {presence['precision']:.4f} | {presence['recall']:.4f} | {presence['f1']:.4f} | {item['a0_f1_delta']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {overall['classification']['macro_f1']:.4f} | {overall['action_accuracy']:.4f} | {overall['normal']['rate']:.4f} | {overall['normal']['true_original_fp_count']} | {overall['uncertain_rate']:.4f} | {overall['json_parse_rate']:.4f} |"
        )
    lines.extend(["", "## 六类 F1", "", "| 配置 | open | short | mousebite | spur | pinhole | spurious copper |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for arm in ARMS:
        values = metrics[arm]["overall"]["classification"]["per_class_f1"]
        lines.append("| " + arm + " | " + " | ".join(f"{values[name]:.4f}" for name in CLASS_NAMES) + " |")
    for section, title in (("by_size", "缺陷尺寸"), ("by_q_reg", "配准质量 q_reg"), ("by_source", "候选来源")):
        lines.extend(["", f"## 按{title}分层", "", "| 配置 | 分层 | n | Accuracy | Precision | Recall | F1 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
        for arm in ARMS:
            for name, bundle in metrics[arm][section].items():
                presence = bundle["presence"]
                lines.append(f"| {arm} | {name} | {bundle['n']} | {presence['accuracy']:.4f} | {presence['precision']:.4f} | {presence['recall']:.4f} | {presence['f1']:.4f} |")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text("timestamp,completed,remaining,current_group,eta_seconds\n", encoding="utf-8")
    REPORT_PATH.write_text("# 阶段 C：全量 RQ1 消融报告\n\n状态：运行中。实时进度见 `progress_full.csv`。\n", encoding="utf-8")
    started = time.perf_counter()
    progress(0, len(ARMS) * 200, "prepare", started)
    sample = prepare_sample(started)
    responses = []
    for arm_index, arm in enumerate(ARMS):
        outputs = run_arm(sample, arm, arm_index * len(sample), started)
        for record, output in zip(sample, outputs):
            responses.append({"arm": arm, **record, **output})
    metrics = summarize(sample, responses)
    payload = {
        "seed": SEED, "candidate_count": len(sample), "arms": list(ARMS),
        "candidate_evaluations": len(responses), "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started, "responses": responses,
    }
    RESULT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    build_report(payload)
    progress(len(responses), len(responses), "complete", started)


if __name__ == "__main__":
    main()
