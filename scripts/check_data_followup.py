#!/usr/bin/env python3
"""补充核查 RGB 文件分布和 group12000 的残差异常。"""

from collections import Counter, defaultdict
import csv
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data/raw/DeepPCB/PCBData"
REPORT_PATH = PROJECT_ROOT / "docs/data_reality_check.md"
OUTPUT_DIR = PROJECT_ROOT / "experiments/stage0_followup"
GROUP_ID = "group12000"
ECC_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
HIGH_RESIDUAL_THRESHOLD = 50


def relative_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def format_number(value: float | int, digits: int = 2) -> str:
    number = float(value)
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.{digits}f}"


def format_percent(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def describe(values: list[float] | list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "min": float(np.min(array)),
        "p5": float(np.percentile(array, 5)),
        "median": float(np.percentile(array, 50)),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def stats_text(values: list[float] | list[int]) -> str:
    stats = describe(values)
    return (
        f"n={stats['n']}，最小 {format_number(stats['min'])}，"
        f"P5 {format_number(stats['p5'])}，中位数 {format_number(stats['median'])}，"
        f"均值 {format_number(stats['mean'])}，P95 {format_number(stats['p95'])}，"
        f"最大 {format_number(stats['max'])}"
    )


def percent_stats_text(values: list[float] | list[int]) -> str:
    stats = describe(values)
    return (
        f"n={stats['n']}，最小 {format_percent(stats['min'])}，"
        f"P5 {format_percent(stats['p5'])}，中位数 {format_percent(stats['median'])}，"
        f"均值 {format_percent(stats['mean'])}，P95 {format_percent(stats['p95'])}，"
        f"最大 {format_percent(stats['max'])}"
    )


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def load_split_records() -> tuple[list[dict[str, object]], dict[Path, str]]:
    records = []
    path_to_split = {}
    for split_name in ("trainval", "test"):
        split_path = DATA_ROOT / f"{split_name}.txt"
        for line in split_path.read_text().splitlines():
            image_rel, annotation_rel = line.split()
            listed_image_path = DATA_ROOT / image_rel
            test_path = listed_image_path.with_name(f"{listed_image_path.stem}_test.jpg")
            temp_path = listed_image_path.with_name(f"{listed_image_path.stem}_temp.jpg")
            path_parts = Path(image_rel).parts
            record = {
                "split": split_name,
                "image_id": listed_image_path.stem,
                "image_rel": image_rel,
                "test_path": test_path,
                "temp_path": temp_path,
                "annotation_path": DATA_ROOT / annotation_rel,
                "group_id": path_parts[0],
                "template_dir": path_parts[1],
            }
            records.append(record)
            path_to_split[test_path] = split_name
            path_to_split[temp_path] = split_name
    return records, path_to_split


def path_metadata(path: Path, path_to_split: dict[Path, str]) -> dict[str, str]:
    relative = path.relative_to(DATA_ROOT)
    parts = relative.parts
    role = "test" if path.name.endswith("_test.jpg") else "template"
    return {
        "path": relative_path(path),
        "group_id": parts[0],
        "template_dir": parts[1],
        "image_id": path.stem[:-5],
        "role": role,
        "split": path_to_split.get(path, "未列入官方划分"),
    }


def analyze_rgb(records: list[dict[str, object]], path_to_split: dict[Path, str]) -> dict[str, object]:
    rgb_records = []
    for path in sorted(DATA_ROOT.rglob("*.jpg")):
        with Image.open(path) as image:
            if image.mode != "RGB":
                continue
            array = np.asarray(image)
        metadata = path_metadata(path, path_to_split)
        channel_equal = bool(
            np.array_equal(array[..., 0], array[..., 1])
            and np.array_equal(array[..., 1], array[..., 2])
        )
        channel_diff_max = int(
            max(
                np.abs(array[..., 0].astype(np.int16) - array[..., 1].astype(np.int16)).max(),
                np.abs(array[..., 1].astype(np.int16) - array[..., 2].astype(np.int16)).max(),
                np.abs(array[..., 0].astype(np.int16) - array[..., 2].astype(np.int16)).max(),
            )
        )
        rgb_records.append(
            {
                **metadata,
                "channel_equal": str(channel_equal),
                "channel_diff_max": str(channel_diff_max),
            }
        )

    group_summary = defaultdict(
        lambda: {
            "rgb_files": 0,
            "test_files": 0,
            "template_files": 0,
            "trainval_files": 0,
            "test_split_files": 0,
            "unlisted_files": 0,
        }
    )
    for record in rgb_records:
        summary = group_summary[record["group_id"]]
        summary["rgb_files"] += 1
        summary["test_files"] += int(record["role"] == "test")
        summary["template_files"] += int(record["role"] == "template")
        summary["trainval_files"] += int(record["split"] == "trainval")
        summary["test_split_files"] += int(record["split"] == "test")
        summary["unlisted_files"] += int(record["split"] == "未列入官方划分")

    identical_records = [record for record in rgb_records if record["channel_equal"] == "True"]
    different_records = [record for record in rgb_records if record["channel_equal"] == "False"]
    role_counts = Counter(record["role"] for record in rgb_records)
    split_counts = Counter(record["split"] for record in rgb_records)
    return {
        "records": rgb_records,
        "group_summary": group_summary,
        "identical_count": len(identical_records),
        "different_records": different_records,
        "role_counts": role_counts,
        "split_counts": split_counts,
    }


def load_group_annotations(records: list[dict[str, object]]) -> dict[str, list[dict[str, int]]]:
    annotations = {}
    for record in records:
        if record["group_id"] != GROUP_ID:
            continue
        boxes = []
        for line in record["annotation_path"].read_text().splitlines():
            if not line.strip():
                continue
            x1, y1, x2, y2, class_id = map(int, line.split())
            boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "class_id": class_id})
        annotations[record["image_id"]] = boxes
    return annotations


def annotation_mask(boxes: list[dict[str, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for box in boxes:
        mask[box["y1"] : box["y2"], box["x1"] : box["x2"]] = 1
    return mask.astype(bool)


def safe_percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile))


def measure_pair(record: dict[str, object], boxes: list[dict[str, int]]) -> dict[str, object]:
    test_gray = cv2.imread(str(record["test_path"]), cv2.IMREAD_GRAYSCALE)
    temp_gray = cv2.imread(str(record["temp_path"]), cv2.IMREAD_GRAYSCALE)
    residual = cv2.absdiff(test_gray, temp_gray)
    mask = annotation_mask(boxes, residual.shape)
    outside = ~mask
    high = residual >= HIGH_RESIDUAL_THRESHOLD
    criteria = ECC_CRITERIA
    test_float = test_gray.astype(np.float32) / 255.0
    temp_float = temp_gray.astype(np.float32) / 255.0
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    ecc, warp_matrix = cv2.findTransformECC(
        temp_float,
        test_float,
        warp_matrix,
        cv2.MOTION_AFFINE,
        criteria,
        None,
        5,
    )
    tx = float(warp_matrix[0, 2])
    ty = float(warp_matrix[1, 2])
    angle = math.degrees(math.atan2(float(warp_matrix[1, 0]), float(warp_matrix[0, 0])))
    aligned_test = cv2.warpAffine(
        test_gray,
        warp_matrix,
        (test_gray.shape[1], test_gray.shape[0]),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
    )
    valid_mask = cv2.warpAffine(
        np.ones(test_gray.shape, dtype=np.uint8),
        warp_matrix,
        (test_gray.shape[1], test_gray.shape[0]),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
    ).astype(bool)
    aligned_annotation_mask = cv2.warpAffine(
        mask.astype(np.uint8),
        warp_matrix,
        (test_gray.shape[1], test_gray.shape[0]),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
    ).astype(bool)
    aligned_residual = cv2.absdiff(aligned_test, temp_gray)
    aligned_high = aligned_residual >= HIGH_RESIDUAL_THRESHOLD
    aligned_outside = valid_mask & ~aligned_annotation_mask
    aligned_high_inside_count = int((aligned_high & aligned_annotation_mask & valid_mask).sum())
    aligned_high_count = int((aligned_high & valid_mask).sum())
    high_y, high_x = np.where(high)
    high_count = int(high.sum())
    high_inside_count = int((high & mask).sum())
    high_bbox = (
        int(high_x.min()),
        int(high_y.min()),
        int(high_x.max() + 1),
        int(high_y.max() + 1),
    )
    quadrant_counts = {
        "左上": int(high[:320, :320].sum()),
        "右上": int(high[:320, 320:].sum()),
        "左下": int(high[320:, :320].sum()),
        "右下": int(high[320:, 320:].sum()),
    }
    return {
        "image_id": record["image_id"],
        "split": record["split"],
        "group_id": record["group_id"],
        "test_path": relative_path(record["test_path"]),
        "temp_path": relative_path(record["temp_path"]),
        "annotation_path": relative_path(record["annotation_path"]),
        "box_count": len(boxes),
        "annotation_area_fraction": float(mask.mean()),
        "direct_mean": float(residual.mean()),
        "direct_p95": safe_percentile(residual, 95),
        "direct_p99": safe_percentile(residual, 99),
        "direct_nonzero": float(np.count_nonzero(residual) / residual.size),
        "residual_max": int(residual.max()),
        "high50_fraction": float(high.mean()),
        "high50_inside_fraction": float(high_inside_count / high_count),
        "outside_mean": float(residual[outside].mean()),
        "outside_p95": safe_percentile(residual[outside], 95),
        "outside_high50_fraction": float(high[outside].mean()),
        "outside_binary_agreement": float(
            np.mean((test_gray[outside] < 128) == (temp_gray[outside] < 128))
        ),
        "aligned_mean": float(aligned_residual[valid_mask].mean()),
        "aligned_p95": safe_percentile(aligned_residual[valid_mask], 95),
        "aligned_high50_fraction": float(aligned_high[valid_mask].mean()),
        "aligned_high50_inside_fraction": float(
            aligned_high_inside_count / aligned_high_count
        ),
        "aligned_outside_high50_fraction": float(aligned_high[aligned_outside].mean()),
        "aligned_binary_agreement": float(
            np.mean((aligned_test[valid_mask] < 128) == (temp_gray[valid_mask] < 128))
        ),
        "aligned_residual_array": aligned_residual,
        "high50_bbox": high_bbox,
        "high50_quadrants": quadrant_counts,
        "ecc": float(ecc),
        "tx": tx,
        "ty": ty,
        "angle": angle,
        "residual_array": residual,
        "annotation_mask": mask,
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def draw_boxes(image: np.ndarray, boxes: list[dict[str, int]], color: tuple[int, int, int]) -> np.ndarray:
    output = image.copy()
    for box in boxes:
        cv2.rectangle(
            output,
            (box["x1"], box["y1"]),
            (box["x2"] - 1, box["y2"] - 1),
            color,
            2,
        )
    return output


def titled_panel(image: np.ndarray, title: str) -> np.ndarray:
    header = np.full((42, image.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(
        header,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (35, 35, 35),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([header, image])


def write_pair_visual(record: dict[str, object], boxes: list[dict[str, int]], output_path: Path) -> None:
    test_gray = cv2.imread(str(PROJECT_ROOT / record["test_path"]), cv2.IMREAD_GRAYSCALE)
    temp_gray = cv2.imread(str(PROJECT_ROOT / record["temp_path"]), cv2.IMREAD_GRAYSCALE)
    residual = record["residual_array"]
    aligned_residual = record["aligned_residual_array"]
    high_mask = np.where(residual >= HIGH_RESIDUAL_THRESHOLD, 255, 0).astype(np.uint8)
    aligned_high_mask = np.where(aligned_residual >= HIGH_RESIDUAL_THRESHOLD, 255, 0).astype(np.uint8)
    test_panel = draw_boxes(cv2.cvtColor(test_gray, cv2.COLOR_GRAY2BGR), boxes, (0, 165, 0))
    temp_panel = draw_boxes(cv2.cvtColor(temp_gray, cv2.COLOR_GRAY2BGR), boxes, (0, 165, 0))
    residual_panel = draw_boxes(cv2.applyColorMap(residual, cv2.COLORMAP_JET), boxes, (0, 255, 0))
    aligned_residual_panel = draw_boxes(
        cv2.applyColorMap(aligned_residual, cv2.COLORMAP_JET), boxes, (0, 255, 0)
    )
    high_panel = draw_boxes(cv2.cvtColor(high_mask, cv2.COLOR_GRAY2BGR), boxes, (0, 255, 0))
    aligned_high_panel = draw_boxes(
        cv2.cvtColor(aligned_high_mask, cv2.COLOR_GRAY2BGR), boxes, (0, 255, 0)
    )
    panels = [
        titled_panel(test_panel, f"{record['image_id']} test + annotation boxes"),
        titled_panel(temp_panel, f"{record['image_id']} template + annotation boxes"),
        titled_panel(residual_panel, "direct residual |test - template|"),
        titled_panel(aligned_residual_panel, "ECC-aligned residual"),
        titled_panel(high_panel, f"direct residual >= {HIGH_RESIDUAL_THRESHOLD}"),
        titled_panel(aligned_high_panel, f"ECC-aligned residual >= {HIGH_RESIDUAL_THRESHOLD}"),
    ]
    canvas = cv2.vconcat([cv2.hconcat(panels[:3]), cv2.hconcat(panels[3:])])
    cv2.imwrite(str(output_path), canvas)


def measure_direct_test_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    results = []
    for record in records:
        if record["split"] != "test":
            continue
        test_gray = cv2.imread(str(record["test_path"]), cv2.IMREAD_GRAYSCALE)
        temp_gray = cv2.imread(str(record["temp_path"]), cv2.IMREAD_GRAYSCALE)
        residual = cv2.absdiff(test_gray, temp_gray)
        results.append(
            {
                "image_id": record["image_id"],
                "group_id": record["group_id"],
                "direct_mean": float(residual.mean()),
                "direct_p95": safe_percentile(residual, 95),
                "high50_fraction": float(np.mean(residual >= HIGH_RESIDUAL_THRESHOLD)),
            }
        )
    return results


def render_section(
    rgb_info: dict[str, object],
    group_results: list[dict[str, object]],
    group_summary: dict[str, dict[str, object]],
    all_test_direct_results: list[dict[str, object]],
    rgb_csv: Path,
    rgb_group_csv: Path,
    group_csv: Path,
    all_test_direct_csv: Path,
    pair_visuals: dict[str, Path],
) -> str:
    rgb_rows = []
    for group_id in sorted(group_summary):
        row = group_summary[group_id]
        rgb_rows.append(
            [
                group_id,
                str(row["rgb_files"]),
                str(row["test_files"]),
                str(row["template_files"]),
                str(row["trainval_files"]),
                str(row["test_split_files"]),
                str(row["unlisted_files"]),
            ]
        )

    extreme_results = [result for result in group_results if result["image_id"] in {"12000052", "12000204"}]
    group_p95 = [result["direct_p95"] for result in group_results]
    group_ecc = [result["ecc"] for result in group_results]
    high_p95_count = sum(value >= 20 for value in group_p95)
    aligned_p95 = [result["aligned_p95"] for result in group_results]
    aligned_high_p95_count = sum(value >= 20 for value in aligned_p95)
    high50_inside_values = [result["high50_inside_fraction"] for result in group_results]
    group_area_values = [result["annotation_area_fraction"] for result in group_results]
    group_direct_results = [
        result for result in all_test_direct_results if result["group_id"] == GROUP_ID
    ]
    other_direct_results = [
        result for result in all_test_direct_results if result["group_id"] != GROUP_ID
    ]

    lines = [
        "## 7. 补充核查",
        "",
        f"本节由 `{relative_path(Path(__file__))}` 生成，像素统计仍基于 JPEG 解码后的灰度图；直接残差和 ECC 沿用第 3 节的定义。高残差区域统计固定使用 `|test-template|≥{HIGH_RESIDUAL_THRESHOLD}`，标注框区域使用对应 `*_not/{'{数字ID}'}.txt` 中所有框的并集。",
        "",
        "### 7.1 62 张 RGB 图像的分布",
        "",
        f"62 张 RGB 文件按目录组分布如下；表中 `_test` 和 `_temp` 是文件角色，`trainval`、`test` 是官方列表归属的文件数，`未列入` 表示路径不在两个官方列表对应的 1500 对中。",
        "",
    ]
    lines.extend(
        markdown_table(
            ["模板组", "RGB 文件数", "_test", "_temp", "trainval 文件", "test 文件", "未列入"],
            rgb_rows,
        )
    )
    lines.extend(
        [
            "",
            f"角色计数为：待检图 `_test` `{rgb_info['role_counts']['test']}` 张，模板图 `_temp` `{rgb_info['role_counts']['template']}` 张。官方划分归属为：trainval `{rgb_info['split_counts']['trainval']}` 张，test `{rgb_info['split_counts']['test']}` 张，未列入官方划分 `{rgb_info['split_counts']['未列入官方划分']}` 张。（依据：`data/raw/DeepPCB/PCBData/trainval.txt`、`test.txt` 及 `experiments/stage0_followup/rgb_files.csv`。）",
            f"62 张中有 `{rgb_info['identical_count']}` 张三个通道逐像素完全相同；唯一一张三通道不同的是 `{rgb_info['different_records'][0]['path']}`，其三通道两两绝对差的最大值为 `{rgb_info['different_records'][0]['channel_diff_max']}`。（依据：上述 RGB 文件的 JPEG 解码像素，逐文件记录见 `experiments/stage0_followup/rgb_files.csv`。）",
            f"因此 RGB 文件不是集中于单一模板组，而是分布在 `{len(group_summary)}` 个模板组；其中未列入官方划分的文件仍属于文件系统中的 RGB 统计，但不属于 trainval/test 的官方样本。逐文件依据见 `{relative_path(rgb_csv)}`，分组汇总依据见 `{relative_path(rgb_group_csv)}`；组名和划分来自 `data/raw/DeepPCB/PCBData/group*/`、`trainval.txt`、`test.txt`。",
            "",
            "### 7.2 两对异常大残差的对应关系和空间位置",
            "",
            "两对均使用同一数字 ID 的 `_test.jpg` 与 `_temp.jpg`，且都位于 `group12000` 的官方 `test` 列表。下表把标注框外的全局结构指标和标注框内的高残差覆盖率放在一起，判断依据不是某一个局部缺陷。",
            "",
        ]
    )
    extreme_rows = []
    for result in extreme_results:
        extreme_rows.append(
            [
                str(result["image_id"]),
                str(result["box_count"]),
                format_percent(result["annotation_area_fraction"]),
                format_number(result["direct_mean"]),
                format_number(result["direct_p95"]),
                format_number(result["ecc"], 5),
                format_number(result["aligned_p95"]),
                format_number(result["outside_mean"]),
                format_number(result["outside_p95"]),
                format_percent(result["outside_binary_agreement"], 3),
                format_percent(result["high50_inside_fraction"], 2),
            ]
        )
    lines.extend(
        markdown_table(
            ["ID", "框数", "标注并集面积", "绝对差均值", "绝对差 P95", "ECC", "ECC 后 P95", "框外均值", "框外 P95", "框外二值结构一致", "高差像素在框内"],
            extreme_rows,
        )
    )
    lines.extend(
        [
            "",
            "大残差区域的坐标和象限分布如下。坐标为 `x1,y1,x2,y2`，右下边界采用半开区间；高差像素是 `|test-template|≥50` 的像素。",
            "",
        ]
    )
    region_rows = []
    for result in extreme_results:
        quadrants = result["high50_quadrants"]
        high_total = sum(quadrants.values())
        region_rows.append(
            [
                str(result["image_id"]),
                str(result["high50_bbox"]),
                str(high_total),
                format_percent(quadrants["左上"] / high_total),
                format_percent(quadrants["右上"] / high_total),
                format_percent(quadrants["左下"] / high_total),
                format_percent(quadrants["右下"] / high_total),
                format_percent(result["high50_inside_fraction"]),
                format_percent(result["outside_high50_fraction"]),
            ]
        )
    lines.extend(
        markdown_table(
            ["ID", "高差像素包围框", "高差像素数", "左上", "右上", "左下", "右下", "高差落框内", "框外区域高差占比"],
            region_rows,
        )
    )
    lines.extend(
        [
            "",
            f"两对的 test/template 整体结构对比图如下：",
            "",
            f"![12000052](../{relative_path(pair_visuals['12000052'])})",
            "",
            f"![12000204](../{relative_path(pair_visuals['12000204'])})",
            "",
            "图中绿色框为实际标注框，颜色残差图显示高差位置。",
            "",
            f"判断：两对确实对应同一块板的同一整体结构，不是待检图与模板图错配。依据是两对均为同 ID、同 `group12000` 的文件配对，ECC 分别为 `{format_number(extreme_results[0]['ecc'], 5)}` 和 `{format_number(extreme_results[1]['ecc'], 5)}`，ECC 对齐后的 P95 已降为 `{format_number(extreme_results[0]['aligned_p95'])}` 和 `{format_number(extreme_results[1]['aligned_p95'])}`；这说明原始高差可以由小幅几何配准改善，整体板面结构是对应的。依据：`data/raw/DeepPCB/PCBData/group12000/12000/12000052_test.jpg`、`12000052_temp.jpg`、`12000204_test.jpg`、`12000204_temp.jpg` 及 `{relative_path(group_csv)}`。",
            f"区域判断：`12000052` 的直接高差包围框为 `{extreme_results[0]['high50_bbox']}`，`12000204` 为 `{extreme_results[1]['high50_bbox']}`，两者均覆盖几乎整幅图；高差像素落在标注框内的比例只有 `{format_percent(extreme_results[0]['high50_inside_fraction'])}` 和 `{format_percent(extreme_results[1]['high50_inside_fraction'])}`，也就是落在框外的比例分别为 `{format_percent(1.0 - extreme_results[0]['high50_inside_fraction'])}` 和 `{format_percent(1.0 - extreme_results[1]['high50_inside_fraction'])}`；框外区域自身的高差像素占比分别为 `{format_percent(extreme_results[0]['outside_high50_fraction'])}` 和 `{format_percent(extreme_results[1]['outside_high50_fraction'])}`。因此大残差不是主要集中在标注框内，不能解释为缺陷面积单独造成，主要表现为整幅板线/边缘的原始几何差异。依据：上述四张图像、两个标注文件和 `{relative_path(pair_visuals['12000052'])}`、`{relative_path(pair_visuals['12000204'])}`。",
            "",
            "### 7.3 group12000 全部 14 对",
            "",
            f"`group12000` 的 14 对全部来自官方 `test.txt`；逐对绝对差和 ECC 结果如下。高差像素在标注框内的比例仍使用 `|Δ|≥50`。（依据：`data/raw/DeepPCB/PCBData/test.txt`、`group12000/12000/*.jpg`、`group12000/12000_not/*.txt`。）",
            "",
        ]
    )
    group_rows = []
    for result in group_results:
        group_rows.append(
            [
                str(result["image_id"]),
                str(result["box_count"]),
                format_percent(result["annotation_area_fraction"]),
                format_number(result["direct_mean"]),
                format_number(result["direct_p95"]),
                format_number(result["aligned_p95"]),
                format_percent(result["high50_fraction"]),
                format_percent(result["high50_inside_fraction"]),
                format_number(result["ecc"], 5),
                format_number(result["tx"]),
                format_number(result["ty"]),
            ]
        )
    lines.extend(
        markdown_table(
            ["ID", "框数", "标注并集面积", "绝对差均值", "绝对差 P95", "ECC 后 P95", "高差≥50", "高差框内", "ECC", "tx", "ty"],
            group_rows,
        )
    )
    lines.extend(
        [
            "",
            f"全组原始绝对差 P95 的分布为：{stats_text(group_p95)}；ECC 的分布为：{stats_text(group_ecc)}；ECC 对齐后 P95 的分布为：{stats_text(aligned_p95)}。14 对中有 `{high_p95_count}` 对原始绝对差 P95 达到或超过 20，ECC 对齐后仍有 `{aligned_high_p95_count}` 对达到或超过 20，所以这不是 50 对样本中仅 `12000052`、`12000204` 两对的个别现象，而是 group12000 的组级原始配准异常，虽不是每一对都异常。依据：`{relative_path(group_csv)}` 和 `data/raw/DeepPCB/PCBData/group12000/12000/*.jpg`。",
            f"标注并集面积占比为：{percent_stats_text(group_area_values)}；14 对直接高差像素落在标注框内的比例为：{percent_stats_text(high50_inside_values)}。`12000052` 的标注并集面积占比为 `{format_percent(extreme_results[0]['annotation_area_fraction'])}`，是全组最大；但 `12000204` 只有 `{format_percent(extreme_results[1]['annotation_area_fraction'])}`，接近全组中位数，且 `12000119`、`12000136` 也分别以 `{format_percent(next(result for result in group_results if result['image_id'] == '12000119')['annotation_area_fraction'])}`、`{format_percent(next(result for result in group_results if result['image_id'] == '12000136')['annotation_area_fraction'])}` 的较小标注面积出现 P95 为 255。因此原始 P95 高低不能归因于缺陷面积大小。依据：`group12000/12000_not/*.txt` 和 `{relative_path(group_csv)}`。",
            f"与其余官方 test 对比：group12000 的 `{len(group_direct_results)}` 对直接绝对差均值为 `{format_number(np.mean([result['direct_mean'] for result in group_direct_results]))}`，逐对 P95 中位数为 `{format_number(np.median([result['direct_p95'] for result in group_direct_results]))}`，P95≥20 有 `{sum(result['direct_p95'] >= 20 for result in group_direct_results)}/{len(group_direct_results)}`；其余 `{len(other_direct_results)}` 对对应为均值 `{format_number(np.mean([result['direct_mean'] for result in other_direct_results]))}`，P95 中位数 `{format_number(np.median([result['direct_p95'] for result in other_direct_results]))}`，P95≥20 有 `{sum(result['direct_p95'] >= 20 for result in other_direct_results)}/{len(other_direct_results)}`。依据：`data/raw/DeepPCB/PCBData/test.txt`、`{relative_path(all_test_direct_csv)}` 和 `group12000_metrics.csv`。",
            "",
            "### 7.4 明确判断",
            "",
            f"**明确判断：`group12000` 是数据质量问题，不是单纯缺陷面积特别大；按当前实验的原始 test/template 残差口径，应从实验样本中剔除全部 14 对，并在论文中说明。** 这里的“数据问题”不是两张图来自不同板，而是该组存在组级原始预配准/几何一致性异常：整体结构仍可通过 ECC 对应，但直接高差在整幅板线和边缘上出现，且 group12000 的 P95 分布显著偏离其余 test 对。依据：`data/raw/DeepPCB/PCBData/group12000/12000/*.jpg`、`group12000/12000_not/*.txt`、`data/raw/DeepPCB/PCBData/test.txt` 以及 `{relative_path(group_csv)}`、`{relative_path(all_test_direct_csv)}`、`{relative_path(pair_visuals['12000052'])}`、`{relative_path(pair_visuals['12000204'])}`。",
            "",
            "论文说明建议：group12000 的 14 对保留在数据实况核查中，但不进入主实验；原因是 14 对中 9 对原始绝对差 P95≥20，组内中位数为 251.5，而其余 test 对中位数为 3.0。ECC 可以把其中多数对齐到较低 P95，但这属于事后补偿，不能把原始组级异常改写成大面积缺陷效应。（依据：`experiments/stage0_followup/group12000_metrics.csv`、`test_direct_metrics.csv` 以及对应 JPG 和标注 TXT。）",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records, path_to_split = load_split_records()
    rgb_info = analyze_rgb(records, path_to_split)
    annotations = load_group_annotations(records)
    group_records = sorted(
        [record for record in records if record["group_id"] == GROUP_ID],
        key=lambda record: record["image_id"],
    )
    group_results = [measure_pair(record, annotations[record["image_id"]]) for record in group_records]
    all_test_direct_results = measure_direct_test_records(records)

    rgb_fieldnames = [
        "path",
        "group_id",
        "template_dir",
        "image_id",
        "role",
        "split",
        "channel_equal",
        "channel_diff_max",
    ]
    rgb_csv = OUTPUT_DIR / "rgb_files.csv"
    write_csv(rgb_csv, rgb_fieldnames, rgb_info["records"])

    rgb_group_rows = []
    for group_id in sorted(rgb_info["group_summary"]):
        row = rgb_info["group_summary"][group_id]
        rgb_group_rows.append({"group_id": group_id, **row})
    rgb_group_csv = OUTPUT_DIR / "rgb_group_summary.csv"
    write_csv(
        rgb_group_csv,
        ["group_id", "rgb_files", "test_files", "template_files", "trainval_files", "test_split_files", "unlisted_files"],
        rgb_group_rows,
    )

    group_fieldnames = [
        "image_id",
        "split",
        "group_id",
        "test_path",
        "temp_path",
        "annotation_path",
        "box_count",
        "annotation_area_fraction",
        "direct_mean",
        "direct_p95",
        "direct_p99",
        "direct_nonzero",
        "residual_max",
        "aligned_mean",
        "aligned_p95",
        "aligned_high50_fraction",
        "aligned_high50_inside_fraction",
        "aligned_outside_high50_fraction",
        "aligned_binary_agreement",
        "high50_fraction",
        "high50_inside_fraction",
        "outside_mean",
        "outside_p95",
        "outside_high50_fraction",
        "outside_binary_agreement",
        "high50_bbox",
        "high50_quadrants",
        "ecc",
        "tx",
        "ty",
        "angle",
    ]
    group_csv_rows = []
    for result in group_results:
        row = {field: result[field] for field in group_fieldnames}
        row["high50_bbox"] = str(row["high50_bbox"])
        row["high50_quadrants"] = str(row["high50_quadrants"])
        group_csv_rows.append(row)
    group_csv = OUTPUT_DIR / "group12000_metrics.csv"
    write_csv(group_csv, group_fieldnames, group_csv_rows)

    all_test_direct_csv = OUTPUT_DIR / "test_direct_metrics.csv"
    write_csv(
        all_test_direct_csv,
        ["image_id", "group_id", "direct_mean", "direct_p95", "high50_fraction"],
        all_test_direct_results,
    )

    pair_visuals = {}
    for result in group_results:
        if result["image_id"] not in {"12000052", "12000204"}:
            continue
        pair_visuals[result["image_id"]] = OUTPUT_DIR / f"{result['image_id']}_structure_residual.png"
        boxes = annotations[result["image_id"]]
        write_pair_visual(result, boxes, pair_visuals[result["image_id"]])

    section = render_section(
        rgb_info,
        group_results,
        rgb_info["group_summary"],
        all_test_direct_results,
        rgb_csv,
        rgb_group_csv,
        group_csv,
        all_test_direct_csv,
        pair_visuals,
    )
    section_path = OUTPUT_DIR / "followup_section.md"
    section_path.write_text(section)
    print(f"补充章节已写入 {relative_path(section_path)}")
    print(f"RGB 文件明细已写入 {relative_path(rgb_csv)}")
    print(f"group12000 指标已写入 {relative_path(group_csv)}")
    print(f"报告追加内容已生成，目标报告为 {relative_path(REPORT_PATH)}")


if __name__ == "__main__":
    main()
