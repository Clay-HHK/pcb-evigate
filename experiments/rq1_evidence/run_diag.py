"""Diagnose appearance-map contribution on the frozen 150-candidate RQ1 sample."""

from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.rq1_evidence.run import align_for_pilot, load_configs, metric_bundle
from src.detector.dataset import rows_for_role
from src.evidence.appearance import compute_appearance
from src.evidence.card import create_evidence_card
from src.evidence.topology import compute_topology
from src.mllm.client import _run_one, cache_key
from src.mllm.prompt import CLOSED_DEFECT_KNOWLEDGE_V2, PROMPT_VERSION, SYSTEM_PROMPT_V2, write_schema
from src.registration.align import load_gray


SEED = 20260811
ARMS = ("A2N", "A2T", "A6F")
OUTPUT_ROOT = Path("experiments/rq1_evidence")
SAMPLE_PATH = OUTPUT_ROOT / "pilot_sample.json"
PILOT_RESULT_PATH = OUTPUT_ROOT / "pilot_results.json"
RESULT_PATH = OUTPUT_ROOT / "diagnosis_results.json"
REPORT_PATH = OUTPUT_ROOT / "diagnosis_report.md"
PROGRESS_PATH = OUTPUT_ROOT / "progress_diag.csv"
CARD_ROOT = OUTPUT_ROOT / "cards_diag"
CASE_ROOT = OUTPUT_ROOT / "cases_diag"
CONFIG_PATH = Path("configs/mllm/codex_cli.yaml")

PANELS = {
    "A2N": ("P1", "P2", "P3", "P6"),
    "A2T": ("P1", "P2", "P3", "P5"),
    "A6F": ("P1", "P2", "P3", "P4", "P5", "P6"),
}

PANEL_DESCRIPTIONS = {
    "P1": "current candidate crop",
    "P2": "aligned normal-reference crop",
    "P3": "larger current context with candidate box",
    "P4": "binary appearance-difference evidence; only red pixels above the fixed preselected fold threshold are shown",
    "P5": "topology/skeleton-difference evidence",
    "P6": "numeric registration and topology summary",
}


def append_progress(stage: str, completed: int, total: int, detail: str) -> None:
    with PROGRESS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{datetime.now().astimezone().isoformat(timespec='seconds')},{stage},{completed},{total},{detail}\n"
        )


def presence_correct(row: dict) -> bool:
    return row["result"]["defect_present"] is row["true_defect_present"]


def summarize(rows: list[dict]) -> dict:
    bundle = metric_bundle(rows)
    return {
        "n": bundle["n"],
        "accuracy": bundle["presence"]["accuracy"],
        "precision": bundle["presence"]["precision"],
        "recall": bundle["presence"]["recall"],
        "f1": bundle["presence"]["f1"],
        "uncertain_rate": bundle["uncertain_rate"],
        "json_parse_rate": bundle["json_parse_rate"],
    }


def compare_strata(
    sample: list[dict], response_by_key: dict[tuple[str, str], dict], assignments: dict[str, str]
) -> list[dict]:
    output = []
    for stratum in dict.fromkeys(assignments.values()):
        candidate_ids = [item["candidate_id"] for item in sample if assignments[item["candidate_id"]] == stratum]
        a2 = summarize([response_by_key[("A2", candidate_id)] for candidate_id in candidate_ids])
        a3 = summarize([response_by_key[("A3", candidate_id)] for candidate_id in candidate_ids])
        output.append({"stratum": stratum, "A2": a2, "A3": a3, "a3_minus_a2_f1": a3["f1"] - a2["f1"]})
    return output


def existing_analysis(sample: list[dict], pilot: dict) -> dict:
    response_by_key = {(item["arm"], item["candidate_id"]): item for item in pilot["responses"]}
    q_values = np.array([item["candidate"]["q_reg"] for item in sample], dtype=float)
    area_values = np.array([item["area"] for item in sample], dtype=float)
    q_low, q_high = np.quantile(q_values, [1.0 / 3.0, 2.0 / 3.0]).tolist()
    area_small, area_large = np.quantile(area_values, [1.0 / 3.0, 2.0 / 3.0]).tolist()
    q_assignments = {}
    area_assignments = {}
    source_assignments = {}
    for item in sample:
        candidate_id = item["candidate_id"]
        q_reg = float(item["candidate"]["q_reg"])
        area = float(item["area"])
        q_assignments[candidate_id] = "低" if q_reg <= q_low else "中" if q_reg <= q_high else "高"
        area_assignments[candidate_id] = "小" if area <= area_small else "中" if area <= area_large else "大"
        source_assignments[candidate_id] = item["candidate"]["source"]
    regressions = [
        item["candidate_id"]
        for item in sample
        if presence_correct(response_by_key[("A2", item["candidate_id"])])
        and not presence_correct(response_by_key[("A3", item["candidate_id"])])
    ]
    return {
        "q_reg_cutoffs": [q_low, q_high],
        "area_cutoffs": [area_small, area_large],
        "by_q_reg": compare_strata(sample, response_by_key, q_assignments),
        "by_area": compare_strata(sample, response_by_key, area_assignments),
        "by_source": compare_strata(sample, response_by_key, source_assignments),
        "a2_correct_a3_wrong_count": len(regressions),
        "a2_correct_a3_wrong_ids": regressions,
    }


def render_cards(sample: list[dict]) -> None:
    registration_config, appearance_config, topology_config, candidate_config, card_config = load_configs()
    row_index = {
        (fold, row["pair_id"]): row
        for fold in range(5)
        for row in rows_for_role(fold, "test")
    }
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for item in sample:
        grouped[(item["fold"], item["pair_id"])].append(item)
    for completed, ((fold, pair_id), records) in enumerate(grouped.items(), start=1):
        row = row_index[(fold, pair_id)]
        current = load_gray(Path(row["test_path"]))
        template = load_gray(Path(row["temp_path"]))
        aligned = align_for_pilot(current, template, registration_config)
        appearance = compute_appearance(aligned["normalized_test"], aligned["normalized_template"], appearance_config)
        normalized_current = ((aligned["normalized_test"] * 32.0) + 128.0).clip(0, 255).astype("uint8")
        normalized_template = ((aligned["normalized_template"] * 32.0) + 128.0).clip(0, 255).astype("uint8")
        topology = compute_topology(normalized_current, normalized_template, topology_config)
        appearance_threshold = float(candidate_config["global_thresholds_by_fold"][str(fold)]["appearance"])
        for record in records:
            candidate = record["candidate"]
            a2n_config = {**card_config, "include_appearance_numeric": False}
            create_evidence_card(
                current,
                aligned["aligned_template"],
                candidate,
                appearance["M_app"],
                topology,
                aligned["q_reg"],
                CARD_ROOT / "A2N" / f"{record['candidate_id']}.png",
                a2n_config,
                set(PANELS["A2N"]),
            )
            create_evidence_card(
                current,
                aligned["aligned_template"],
                candidate,
                appearance["M_app"],
                topology,
                aligned["q_reg"],
                CARD_ROOT / "A2T" / f"{record['candidate_id']}.png",
                card_config,
                set(PANELS["A2T"]),
            )
            binary_config = {
                **card_config,
                "appearance_mode": "binary",
                "appearance_threshold": appearance_threshold,
            }
            create_evidence_card(
                current,
                aligned["aligned_template"],
                candidate,
                appearance["M_app"],
                topology,
                aligned["q_reg"],
                CARD_ROOT / "A6F" / f"{record['candidate_id']}.png",
                binary_config,
                set(PANELS["A6F"]),
            )
        append_progress("render", completed, len(grouped), f"fold={fold};pair_id={pair_id}")


def build_diag_prompt(candidate: dict, arm: str) -> str:
    panel_text = "\n".join(f"{panel}: {PANEL_DESCRIPTIONS[panel]};" for panel in PANELS[arm])
    source_instruction = ""
    if candidate["source"] == "DIFF":
        source_instruction = (
            "\nThis candidate marks a region that the base detector did not propose. "
            "Independently decide whether a defect exists here."
        )
    return (
        f"{SYSTEM_PROMPT_V2}\n\n{CLOSED_DEFECT_KNOWLEDGE_V2}\n\n"
        f"The evidence card contains only these panels:\n{panel_text}\n\n"
        "The normal reference represents expected structure only when it is reliable.\n"
        "Determine independently:\n1. whether a supported defect is present;\n2. the most likely class;\n"
        "3. whether the reference is reliable;\n4. which appearance and topology patterns are supported."
        f"{source_instruction}\n\nReturn the required JSON only."
    )


def review_arm(sample: list[dict], arm: str, arm_index: int) -> tuple[list[dict], int]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    schema_path = write_schema()
    cache_path = Path(config["cache_file"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = {
        item["key"]: item
        for item in map(json.loads, cache_path.read_text(encoding="utf-8").splitlines())
    } if cache_path.exists() else {}
    keys = [cache_key(item["candidate"], arm, config["model"], float(config["temperature"])) for item in sample]
    cache_hits = sum(key in cached for key in keys)
    tasks = [
        (
            config,
            item["candidate"],
            str(CARD_ROOT / arm / f"{item['candidate_id']}.png"),
            str(schema_path),
            build_diag_prompt(item["candidate"], arm),
            key,
        )
        for item, key in zip(sample, keys)
        if key not in cached
    ]
    if tasks:
        with cache_path.open("a", encoding="utf-8") as handle:
            with ProcessPoolExecutor(max_workers=int(config["workers"])) as executor:
                futures = [executor.submit(_run_one, task) for task in tasks]
                for completed, future in enumerate(as_completed(futures), start=1):
                    output = future.result()
                    handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                    handle.flush()
                    cached[output["key"]] = output
                    global_completed = arm_index * len(sample) + cache_hits + completed
                    append_progress("mllm", global_completed, len(sample) * len(ARMS), output["key"])
    return [cached[key] for key in keys], cache_hits


def export_cases(sample: list[dict], pilot: dict, regression_ids: list[str]) -> list[dict]:
    CASE_ROOT.mkdir(parents=True, exist_ok=True)
    by_response = {(item["arm"], item["candidate_id"]): item for item in pilot["responses"]}
    by_sample = {item["candidate_id"]: item for item in sample}
    rng = random.Random(SEED)
    by_source: dict[str, list[str]] = defaultdict(list)
    for candidate_id in regression_ids:
        by_source[by_sample[candidate_id]["candidate"]["source"]].append(candidate_id)
    chosen = []
    for source in ("DET", "DIFF", "BOTH"):
        rng.shuffle(by_source[source])
        chosen.extend(by_source[source][:2])
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    exported = []
    for index, candidate_id in enumerate(chosen, start=1):
        a2_card = Image.open(OUTPUT_ROOT / "cards" / "A2" / f"{candidate_id}.png").convert("RGB")
        a3_card = Image.open(OUTPUT_ROOT / "cards" / "A3" / f"{candidate_id}.png").convert("RGB")
        canvas = Image.new("RGB", (a2_card.width + a3_card.width, a2_card.height + 38), "white")
        canvas.paste(a2_card, (0, 38))
        canvas.paste(a3_card, (a2_card.width, 38))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), "A2 correct", fill="black", font=font)
        draw.text((a2_card.width + 12, 8), "A3 wrong", fill="black", font=font)
        stem = f"{index:02d}_{candidate_id}"
        canvas.save(CASE_ROOT / f"{stem}.png")
        record = by_sample[candidate_id]
        payload = {
            "candidate_id": candidate_id,
            "candidate_type": record["candidate_type"],
            "source": record["candidate"]["source"],
            "q_reg": record["candidate"]["q_reg"],
            "area": record["area"],
            "true_defect_present": record["true_defect_present"],
            "A2": by_response[("A2", candidate_id)]["result"],
            "A3": by_response[("A3", candidate_id)]["result"],
        }
        (CASE_ROOT / f"{stem}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        exported.append(payload | {"stem": stem})
    readme = [
        "# A2 判对但 A3 判错的对照案例",
        "",
        "按固定随机种子从 DET、DIFF、BOTH 各选 2 个。每张图左侧为 A2，右侧为 A3。",
        "",
        *[f"{index}. `{item['stem']}.png`" for index, item in enumerate(exported, start=1)],
    ]
    (CASE_ROOT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return exported


def metric_row(arm: str, metrics: dict) -> str:
    return (
        f"| {arm} | {metrics['n']} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | "
        f"{metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['uncertain_rate']:.4f} | "
        f"{metrics['json_parse_rate']:.4f} |"
    )


def stratum_table(rows: list[dict]) -> list[str]:
    lines = [
        "| 分层 | n | A2 F1 | A3 F1 | A3 相对 A2 | A2 uncertain | A3 uncertain |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in rows:
        lines.append(
            f"| {item['stratum']} | {item['A2']['n']} | {item['A2']['f1']:.4f} | "
            f"{item['A3']['f1']:.4f} | {item['a3_minus_a2_f1']:+.4f} | "
            f"{item['A2']['uncertain_rate']:.4f} | {item['A3']['uncertain_rate']:.4f} |"
        )
    return lines


def build_report(payload: dict) -> None:
    analysis = payload["existing_analysis"]
    q_low, q_high = analysis["q_reg_cutoffs"]
    area_small, area_large = analysis["area_cutoffs"]
    metrics = payload["metrics"]
    decision = payload["decision"]
    low_delta = next(item["a3_minus_a2_f1"] for item in analysis["by_q_reg"] if item["stratum"] == "低")
    mid_delta = next(item["a3_minus_a2_f1"] for item in analysis["by_q_reg"] if item["stratum"] == "中")
    high_delta = next(item["a3_minus_a2_f1"] for item in analysis["by_q_reg"] if item["stratum"] == "高")
    lines = [
        "# 阶段 B2：外观差异图负贡献诊断报告",
        "",
        "## 实验口径",
        "",
        f"分析与对照实验固定使用阶段 B 的同一批 150 个候选，随机种子 {SEED}。新调用后端为 `codex_cli`，模型为 `gpt-5.6-sol`，prompt 为 `{PROMPT_VERSION}`，温度 0，4 路并发。A2 与 A6 直接复用已有结果。",
        "",
        "## 第一部分：已有 600 条结果的零成本分析",
        "",
        "### 按 q_reg 分层",
        "",
        f"按候选级 q_reg 三分位数分档。低档为 q_reg ≤ {q_low:.4f}，中档为 {q_low:.4f} < q_reg ≤ {q_high:.4f}，高档为 q_reg > {q_high:.4f}。同一图像候选共享 q_reg，因此边界并列使各档数量不完全相等。",
        "",
        *stratum_table(analysis["by_q_reg"]),
        "",
        f"结论：低档 A3 相对 A2 的 F1 变化为 {low_delta:+.4f}，确实明显下降，但中档下降更大，为 {mid_delta:+.4f}，高档也下降 {high_delta:+.4f}。因此低配准质量会加重问题，但数据不支持把负贡献归因于低 q_reg 配准残差这一单一机制。外观图在三个档位都没有正贡献。",
        "",
        "### 按候选面积分层",
        "",
        f"按候选面积三分位数分档。小档面积 ≤ {area_small:.2f}，中档为 {area_small:.2f} < 面积 ≤ {area_large:.2f}，大档面积 > {area_large:.2f}。",
        "",
        *stratum_table(analysis["by_area"]),
        "",
        "结论：小缺陷基线最难，但 A3 的最大退化出现在中等面积档，而不是小面积档。小缺陷信号被噪声淹没不能解释总体负贡献。",
        "",
        "### 按候选来源分层",
        "",
        *stratum_table(analysis["by_source"]),
        "",
        "结论：DET、DIFF、BOTH 三类来源的 A3 F1 均低于 A2。DET 退化最大，BOTH 退化最小，负贡献不是某一种候选生成路径独有。",
        "",
        "### 错误案例",
        "",
        f"A2 缺陷存在性判对而 A3 判错共有 {analysis['a2_correct_a3_wrong_count']} 个，占全部候选的 {analysis['a2_correct_a3_wrong_count'] / 150:.1%}。固定种子从 DET、DIFF、BOTH 各选 2 个，共 6 个，已导出到 `experiments/rq1_evidence/cases_diag/`。每个案例包含 A2 与 A3 并排证据卡和两者原始结构化输出。",
        "",
        "## 第二部分：渲染实现核实",
        "",
        "外观差异图没有按每张图的 min/max 自动归一化。`compute_appearance` 先把 M_app 截断到 `[0,1]`，`create_evidence_card` 再直接执行 `clip(M_app × 255, 0, 255)` 并使用 TURBO 色图。因此所有样本共享固定的 0 到 1 数值范围，当前实现符合固定色标要求。连续色图没有把每张样本自身的最小值和最大值分别拉到色标两端。",
        "",
        "拓扑差异图也没有自动拉伸。它将模板骨架、当前骨架和容差 XOR 分别映射为固定的蓝色、绿色和红色，背景固定为黑色。M_topo 本身为二值图。",
        "",
        "因此，本阶段不能把 A3 退化解释为规格书所警告的逐样本自动拉伸错误。按预先规则，A6F 不做固定色标修复，而采用一次性二值显示：每折使用阶段 A 已固定的 appearance 全局阈值，只把超过阈值的像素显示为红色，其余为黑色。没有搜索或调整新阈值。",
        "",
        "## 第三部分：450 次新调用对照实验",
        "",
        "A2N 只显示当前裁剪、正常模板、上下文以及数值注册与拓扑摘要，不显示任何热图，也不在摘要中显示外观残差均值。A2T 只增加固定颜色的拓扑差异图。A6F 保留完整证据卡，但把外观差异图改为上述固定阈值二值图。",
        "",
        "| 配置 | n | Accuracy | Precision | Recall | F1 | uncertain 率 | JSON 可解析率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *[metric_row(arm, metrics[arm]) for arm in ("A2", "A2N", "A2T", "A6F", "A6")],
        "",
        f"新配置共完成 {payload['candidate_evaluations']} 次候选判断。缓存命中 {payload['cache_hits']} 次，实际新增执行 {payload['candidate_evaluations'] - payload['cache_hits']} 次。三组均使用完全相同的 150 个候选。",
        "",
        "## 明确决策",
        "",
        decision["conclusion"],
        "",
        f"判断依据：A2 的 F1 为 {metrics['A2']['f1']:.4f}。三个新配置中的最高值为 {decision['best_new_arm']} 的 {decision['best_new_f1']:.4f}，相对 A2 为 {decision['best_new_f1'] - metrics['A2']['f1']:+.4f}。本轮没有继续调色标、阈值或 prompt。",
        "",
        f"必要限定：A2N 的 F1 也只有 {metrics['A2N']['f1']:.4f}，因此数值拓扑摘要本身同样没有证明能提高 MLLM 候选判断。‘模板参照 + 数值拓扑证据’只能作为方法层面的简化表述，数值拓扑宜保留为结构化辅助输出，不应宣称带来增量性能。若按本诊断集的候选判断 F1 选择证据卡，实证最优仍是 A2。",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not PROGRESS_PATH.exists():
        PROGRESS_PATH.write_text("timestamp,stage,completed,total,detail\n", encoding="utf-8")
    started = time.perf_counter()
    sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))["records"]
    pilot = json.loads(PILOT_RESULT_PATH.read_text(encoding="utf-8"))
    analysis = existing_analysis(sample, pilot)
    append_progress("analysis", 600, 600, "existing_results_complete")
    cases = export_cases(sample, pilot, analysis["a2_correct_a3_wrong_ids"])
    append_progress("cases", len(cases), 6, str(CASE_ROOT))
    render_cards(sample)
    responses = []
    cache_hits = 0
    for arm_index, arm in enumerate(ARMS):
        outputs, arm_cache_hits = review_arm(sample, arm, arm_index)
        cache_hits += arm_cache_hits
        for record, output in zip(sample, outputs):
            responses.append(
                {
                    "arm": arm,
                    "fold": record["fold"],
                    "pair_id": record["pair_id"],
                    "candidate_id": record["candidate_id"],
                    "candidate_type": record["candidate_type"],
                    "source": record["candidate"]["source"],
                    "q_reg": record["candidate"]["q_reg"],
                    "true_defect_present": record["true_defect_present"],
                    "true_class": record["true_class"],
                    "gt_iou": record["gt_iou"],
                    "area": record["area"],
                    **output,
                }
            )
        append_progress("arm", arm_index + 1, len(ARMS), f"{arm};cache_hits={arm_cache_hits}")
    pilot_by_arm = {
        arm: [item for item in pilot["responses"] if item["arm"] == arm]
        for arm in ("A2", "A6")
    }
    metrics = {
        "A2": summarize(pilot_by_arm["A2"]),
        "A6": summarize(pilot_by_arm["A6"]),
        **{arm: summarize([item for item in responses if item["arm"] == arm]) for arm in ARMS},
    }
    best_new_arm = max(ARMS, key=lambda arm: metrics[arm]["f1"])
    best_new_f1 = metrics[best_new_arm]["f1"]
    if best_new_f1 > metrics["A2"]["f1"]:
        conclusion = (
            f"**选择第一种结论：外观与拓扑证据的问题可以修复。将 {best_new_arm} 定义为全量 RQ1 的 A6。"
            f"该配置修复的是 {('连续外观图造成的视觉干扰' if best_new_arm == 'A6F' else '证据形态组合不当')}。"
        )
    else:
        conclusion = (
            "**选择第二种结论：三个新配置均未超过 A2，外观热图对本任务无效。建议把方法简化为‘模板参照 + 数值拓扑证据’，并在论文中如实报告该消融负面发现。**"
        )
    payload = {
        "prompt_version": PROMPT_VERSION,
        "seed": SEED,
        "backend": "codex_cli",
        "model": "gpt-5.6-sol",
        "temperature": 0,
        "workers": 4,
        "candidate_count": len(sample),
        "candidate_evaluations": len(responses),
        "cache_hits": cache_hits,
        "existing_analysis": analysis,
        "rendering_audit": {
            "appearance_auto_normalized": False,
            "appearance_scale": "fixed_0_to_1",
            "topology_auto_normalized": False,
            "a6f_mode": "binary_above_preselected_fold_appearance_threshold",
        },
        "metrics": metrics,
        "decision": {
            "best_new_arm": best_new_arm,
            "best_new_f1": best_new_f1,
            "a2_f1": metrics["A2"]["f1"],
            "new_arm_exceeds_a2": best_new_f1 > metrics["A2"]["f1"],
            "conclusion": conclusion,
        },
        "cases": cases,
        "elapsed_seconds": time.perf_counter() - started,
        "responses": responses,
    }
    RESULT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    build_report(payload)
    append_progress("complete", len(responses), 450, str(RESULT_PATH))
    print(json.dumps({key: value for key, value in payload.items() if key != "responses"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
