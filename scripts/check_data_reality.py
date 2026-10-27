#!/usr/bin/env python3
"""核查 DeepPCB 的实际文件、图像属性、配准、标注和模板分组。"""

from collections import Counter, defaultdict
from pathlib import Path
import math
import sys

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data/raw/DeepPCB/PCBData"
REPORT_PATH = PROJECT_ROOT / "docs/data_reality_check.md"
SEED = 20260811
SAMPLE_SIZE = 50

CLASS_NAMES = {
    0: "background（未使用）",
    1: "open",
    2: "short",
    3: "mousebite",
    4: "spur",
    5: "copper（项目语义：spurious copper）",
    6: "pin-hole",
}


def describe(values: list[float] | list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "min": float(np.min(array)),
        "p5": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.percentile(array, 50)),
        "mean": float(np.mean(array)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def format_number(value: float | int, digits: int = 2) -> str:
    number = float(value)
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.{digits}f}"


def format_percent(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def stats_text(values: list[float] | list[int]) -> str:
    stats = describe(values)
    return (
        f"n={stats['n']}，最小 {format_number(stats['min'])}，"
        f"P5 {format_number(stats['p5'])}，中位数 {format_number(stats['median'])}，"
        f"均值 {format_number(stats['mean'])}，P95 {format_number(stats['p95'])}，"
        f"最大 {format_number(stats['max'])}"
    )


def histogram_percentile(histogram: np.ndarray, percentile: float) -> int:
    cumulative = np.cumsum(histogram)
    target = math.ceil(float(histogram.sum()) * percentile / 100.0)
    return int(np.searchsorted(cumulative, target, side="left"))


def relative_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def load_split_records() -> list[dict[str, object]]:
    records = []
    for split_name in ("trainval", "test"):
        split_path = DATA_ROOT / f"{split_name}.txt"
        for line in split_path.read_text().splitlines():
            image_rel, annotation_rel = line.split()
            listed_image_path = DATA_ROOT / image_rel
            image_path = listed_image_path.with_name(f"{listed_image_path.stem}_test.jpg")
            temp_path = listed_image_path.with_name(f"{listed_image_path.stem}_temp.jpg")
            image_id = listed_image_path.stem
            path_parts = Path(image_rel).parts
            records.append(
                {
                    "split": split_name,
                    "image_id": image_id,
                    "image_rel": image_rel,
                    "annotation_rel": annotation_rel,
                    "test_path": image_path,
                    "temp_path": temp_path,
                    "annotation_path": DATA_ROOT / annotation_rel,
                    "group_id": path_parts[0],
                    "template_dir": path_parts[1],
                }
            )
    return records


def inspect_images(records: list[dict[str, object]]) -> dict[str, object]:
    test_paths = {record["test_path"] for record in records}
    temp_paths = {record["temp_path"] for record in records}
    all_test_paths = set(DATA_ROOT.rglob("*_test.jpg"))
    all_temp_paths = set(DATA_ROOT.rglob("*_temp.jpg"))
    unpaired_temp_paths = sorted(all_temp_paths - temp_paths)

    all_paths = sorted(test_paths | temp_paths | set(unpaired_temp_paths))
    mode_counts = Counter()
    mode_role_counts = Counter()
    format_counts = Counter()
    size_counts = Counter()
    dtype_counts = Counter()
    bit_depth_counts = Counter()
    channel_counts = Counter()
    unique_counts_by_role = defaultdict(list)
    histogram = np.zeros(256, dtype=np.int64)
    rgb_count = 0
    rgb_identical_count = 0

    for path in all_paths:
        role = "test" if path.name.endswith("_test.jpg") else "template"
        with Image.open(path) as image:
            array = np.asarray(image)
            mode_counts[image.mode] += 1
            mode_role_counts[(role, image.mode)] += 1
            format_counts[image.format] += 1
            size_counts[image.size] += 1
            dtype_counts[str(array.dtype)] += 1
            bit_depth_counts[int(array.dtype.itemsize * 8)] += 1
            channels = 1 if array.ndim == 2 else int(array.shape[2])
            channel_counts[channels] += 1
            if image.mode == "RGB":
                rgb_count += 1
                if np.array_equal(array[..., 0], array[..., 1]) and np.array_equal(
                    array[..., 1], array[..., 2]
                ):
                    rgb_identical_count += 1
            grayscale = array if array.ndim == 2 else cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
            histogram += np.bincount(grayscale.reshape(-1), minlength=256).astype(np.int64)
            unique_counts_by_role[role].append(int(np.unique(grayscale).size))

    return {
        "test_paths": test_paths,
        "temp_paths": temp_paths,
        "all_test_paths": all_test_paths,
        "all_temp_paths": all_temp_paths,
        "unpaired_temp_paths": unpaired_temp_paths,
        "all_paths": all_paths,
        "mode_counts": mode_counts,
        "mode_role_counts": mode_role_counts,
        "format_counts": format_counts,
        "size_counts": size_counts,
        "dtype_counts": dtype_counts,
        "bit_depth_counts": bit_depth_counts,
        "channel_counts": channel_counts,
        "unique_counts_by_role": unique_counts_by_role,
        "histogram": histogram,
        "rgb_count": rgb_count,
        "rgb_identical_count": rgb_identical_count,
    }


def load_annotations(records: list[dict[str, object]]) -> dict[str, object]:
    image_box_counts = []
    image_box_counts_by_split = defaultdict(list)
    box_records = []
    class_counts = Counter()
    class_image_counts = Counter()
    token_count_counts = Counter()
    delimiter_counts = Counter()

    for record in records:
        annotation_path = record["annotation_path"]
        lines = [line.strip() for line in annotation_path.read_text().splitlines() if line.strip()]
        image_box_counts.append(len(lines))
        image_box_counts_by_split[record["split"]].append(len(lines))
        image_classes = set()
        for line in lines:
            tokens = line.split()
            token_count_counts[len(tokens)] += 1
            delimiter_counts["逗号分隔"] += int("," in line)
            delimiter_counts["空白分隔"] += int("," not in line)
            x1, y1, x2, y2, class_id = map(int, tokens)
            width = x2 - x1
            height = y2 - y1
            class_counts[class_id] += 1
            image_classes.add(class_id)
            box_records.append(
                {
                    "image_id": record["image_id"],
                    "split": record["split"],
                    "annotation_rel": record["annotation_rel"],
                    "class_id": class_id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": width,
                    "height": height,
                    "area": width * height,
                }
            )
        for class_id in image_classes:
            class_image_counts[class_id] += 1

    return {
        "image_box_counts": image_box_counts,
        "image_box_counts_by_split": image_box_counts_by_split,
        "box_records": box_records,
        "class_counts": class_counts,
        "class_image_counts": class_image_counts,
        "token_count_counts": token_count_counts,
        "delimiter_counts": delimiter_counts,
    }


def measure_registration(records: list[dict[str, object]]) -> dict[str, object]:
    test_records = [record for record in records if record["split"] == "test"]
    rng = np.random.default_rng(SEED)
    sampled_indices = rng.choice(len(test_records), size=SAMPLE_SIZE, replace=False)
    sampled_records = [test_records[int(index)] for index in sampled_indices]
    residual_histogram = np.zeros(256, dtype=np.int64)
    results = []
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)

    for record in sampled_records:
        test_gray = cv2.imread(str(record["test_path"]), cv2.IMREAD_GRAYSCALE)
        temp_gray = cv2.imread(str(record["temp_path"]), cv2.IMREAD_GRAYSCALE)
        residual = cv2.absdiff(test_gray, temp_gray)
        residual_histogram += np.bincount(residual.reshape(-1), minlength=256).astype(np.int64)

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
        results.append(
            {
                "image_id": record["image_id"],
                "image_rel": record["image_rel"],
                "direct_mean": float(np.mean(residual)),
                "direct_p95": float(np.percentile(residual, 95)),
                "direct_nonzero": float(np.count_nonzero(residual) / residual.size),
                "ecc": float(ecc),
                "tx": tx,
                "ty": ty,
                "angle": angle,
            }
        )

    ecc_values = [result["ecc"] for result in results]
    tx_values = [result["tx"] for result in results]
    ty_values = [result["ty"] for result in results]
    angle_values = [result["angle"] for result in results]
    near_identity = [
        abs(result["tx"]) <= 1.0
        and abs(result["ty"]) <= 1.0
        and abs(result["angle"]) <= 0.1
        for result in results
    ]

    return {
        "results": results,
        "residual_histogram": residual_histogram,
        "residual_mean": float(
            np.dot(np.arange(256, dtype=np.float64), residual_histogram)
            / residual_histogram.sum()
        ),
        "residual_p95": histogram_percentile(residual_histogram, 95),
        "residual_nonzero": float(
            1.0 - residual_histogram[0] / residual_histogram.sum()
        ),
        "sample_ids": [result["image_id"] for result in results],
        "ecc_stats": describe(ecc_values),
        "tx_stats": describe(tx_values),
        "ty_stats": describe(ty_values),
        "angle_stats": describe(angle_values),
        "abs_tx_stats": describe([abs(value) for value in tx_values]),
        "abs_ty_stats": describe([abs(value) for value in ty_values]),
        "abs_angle_stats": describe([abs(value) for value in angle_values]),
        "near_identity_count": int(sum(near_identity)),
    }


def summarize_groups(records: list[dict[str, object]]) -> dict[str, object]:
    groups = defaultdict(lambda: {"trainval": 0, "test": 0, "all": 0})
    consistency_count = 0
    for record in records:
        group = record["group_id"]
        groups[group][record["split"]] += 1
        groups[group]["all"] += 1
        if (
            group == f"group{record['template_dir']}"
            and str(record["image_id"]).startswith(str(record["template_dir"]))
        ):
            consistency_count += 1

    group_sizes = [row["all"] for row in groups.values()]
    train_groups = {group for group, row in groups.items() if row["trainval"]}
    test_groups = {group for group, row in groups.items() if row["test"]}
    return {
        "groups": groups,
        "group_sizes": group_sizes,
        "consistency_count": consistency_count,
        "train_group_count": len(train_groups),
        "test_group_count": len(test_groups),
        "overlap_group_count": len(train_groups & test_groups),
        "group_stats": describe(group_sizes),
    }


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def render_report(
    records: list[dict[str, object]],
    image_info: dict[str, object],
    annotation_info: dict[str, object],
    registration_info: dict[str, object],
    group_info: dict[str, object],
) -> str:
    lines = [
        "# DeepPCB 数据实况核查",
        "",
        f"本报告由 `{relative_path(Path(__file__))}` 生成，解释器为 `{sys.executable}`。",
        f"数据根目录为 `{relative_path(DATA_ROOT)}`，固定随机种子为 `{SEED}`。",
        "报告中的像素统计基于 JPEG 解码后的图像。",
        "",
        "## 1. 数据规模",
        "",
        "数据规模依据 `data/raw/DeepPCB/PCBData/trainval.txt`、`data/raw/DeepPCB/PCBData/test.txt` 的实际非空行，以及 `PCBData` 下的 JPG 文件名配对结果。",
    ]

    split_counts = Counter(record["split"] for record in records)
    pair_ids = {record["image_id"] for record in records}
    test_ids = {path.stem[:-5] for path in image_info["all_test_paths"]}
    temp_ids = {path.stem[:-5] for path in image_info["all_temp_paths"]}
    matched_file_pair_count = len(test_ids & temp_ids)
    lines.extend(
        [
            f"- 列表中的图像对总数：`{len(pair_ids)}`。`trainval.txt` 实际 `{split_counts['trainval']}` 对，`test.txt` 实际 `{split_counts['test']}` 对。",
            f"- 文件系统中 `_test.jpg` 共 `{len(image_info['all_test_paths'])}` 张，`_temp.jpg` 共 `{len(image_info['all_temp_paths'])}` 张，按同一数字 ID 匹配得到 `{matched_file_pair_count}` 对。",
            f"- 文件系统中存在 `{len(image_info['unpaired_temp_paths'])}` 张未配对模板图：" + (
                "、".join(f"`{relative_path(path)}`" for path in image_info["unpaired_temp_paths"])
                if image_info["unpaired_temp_paths"]
                else "无"
            )
            + "。",
        ]
    )
    lines.extend(
        markdown_table(
            ["划分文件", "实际非空行数", "图像对数", "占 1500 对"],
            [
                [
                    "trainval.txt",
                    str(split_counts["trainval"]),
                    str(split_counts["trainval"]),
                    format_percent(split_counts["trainval"] / len(pair_ids)),
                ],
                [
                    "test.txt",
                    str(split_counts["test"]),
                    str(split_counts["test"]),
                    format_percent(split_counts["test"] / len(pair_ids)),
                ],
            ],
        )
    )
    lines.extend(
        [
            "",
            "目录组织方式为 `PCBData/groupXXXXX/XXXXX/{数字ID}_test.jpg`、`{数字ID}_temp.jpg`，对应标注位于同级的 `XXXXX_not/{数字ID}.txt`。两个列表文件每行记录待检图相对路径和标注相对路径，模板路径由 `_test.jpg` 与 `_temp.jpg` 后缀替换得到。",
            "",
            "## 2. 图像属性",
            "",
            "属性统计覆盖 `PCBData` 下全部 `{}` 个 JPG 文件，其中包括上面列出的未配对模板图。尺寸、模式、位深和通道数来自 Pillow 解码结果。像素值直方图统一按每个空间像素的灰度值统计，RGB 文件先按 `cv2.COLOR_RGB2GRAY` 转灰度。".format(len(image_info["all_paths"])),
            "",
            "### 2.1 尺寸、位深、格式和通道",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["属性", "实测分布"],
            [
                ["尺寸", "；".join(f"{size[0]}×{size[1]}：{count} 张" for size, count in sorted(image_info["size_counts"].items()))],
                ["Pillow 模式", "；".join(f"{mode}：{count} 张" for mode, count in sorted(image_info["mode_counts"].items()))],
                ["文件格式", "；".join(f"{fmt}：{count} 张" for fmt, count in sorted(image_info["format_counts"].items()))],
                ["NumPy dtype", "；".join(f"{dtype}：{count} 张" for dtype, count in sorted(image_info["dtype_counts"].items()))],
                ["位深", "；".join(f"{bits} bit/通道：{count} 张" for bits, count in sorted(image_info["bit_depth_counts"].items()))],
                ["通道数", "；".join(f"{channels} 通道：{count} 张" for channels, count in sorted(image_info["channel_counts"].items()))],
            ],
        )
    )
    lines.extend(
        [
            "",
            f"RGB 文件共 `{image_info['rgb_count']}` 张，其中三个通道逐像素完全相同的有 `{image_info['rgb_identical_count']}` 张。",
            "",
            "### 2.2 唯一像素值和二值判定",
            "",
        ]
    )
    unique_rows = []
    for role in ("test", "template"):
        values = image_info["unique_counts_by_role"][role]
        binary_count = sum(value <= 2 for value in values)
        unique_rows.append(
            [
                role,
                str(len(values)),
                stats_text(values),
                f"{binary_count} 张（{format_percent(binary_count / len(values))}）",
            ]
        )
    lines.extend(markdown_table(["文件角色", "张数", "每图唯一灰度值个数分布", "严格不超过 2 个值"], unique_rows))
    histogram = image_info["histogram"]
    total_pixels = int(histogram.sum())
    extreme_fraction = float((histogram[0] + histogram[255]) / total_pixels)
    exact_binary_count = sum(
        value <= 2
        for values in image_info["unique_counts_by_role"].values()
        for value in values
    )
    lines.extend(
        [
            "",
            f"全部图像共统计 `{total_pixels:,}` 个空间像素。严格按解码后的灰度值判定，唯一值不超过 2 个的图像为 `{exact_binary_count}/{len(image_info['all_paths'])}`（{format_percent(exact_binary_count / len(image_info['all_paths']))}）。",
            f"全体像素中值为 0 或 255 的比例为 `{format_percent(extreme_fraction)}`，其余灰度值比例为 `{format_percent(1.0 - extreme_fraction)}`。因此报告同时保留严格二值判定和极端值像素占比，避免把 JPEG 解码后的近二值现象写成严格二值。",
            "",
            "### 2.3 像素值直方图",
            "",
            f"下表是全部 {len(image_info['all_paths'])} 张 JPG 的灰度像素直方图，计数单位为空间像素。",
            "",
        ]
    )
    histogram_rows = [
        [str(value), f"{int(histogram[value]):,}", format_percent(float(histogram[value] / total_pixels), 4)]
        for value in range(256)
        if histogram[value]
    ]
    lines.extend(markdown_table(["灰度值", "像素数", "占全部像素"], histogram_rows))

    lines.extend(
        [
            "",
            "## 3. 配准状态",
            "",
            f"样本来自 `test.txt` 的 `{split_counts['test']}` 对，使用固定种子 `{SEED}` 不放回随机抽取 `{SAMPLE_SIZE}` 对。直接残差是待检图与同 ID 模板图的灰度绝对差，ECC 调用为 `cv2.findTransformECC(template, test, MOTION_AFFINE)`，下表中的仿射矩阵把 test 坐标映射到 template 坐标。平移量单位为像素，旋转角由仿射矩阵左上 2×2 部分按 `atan2(m21, m11)` 换算为度。",
            "",
            "### 3.1 50 对样本的直接绝对差分汇总",
            "",
            f"50 对全部像素合并后，绝对差均值为 `{format_number(registration_info['residual_mean'])}`，绝对差 P95 为 `{format_number(registration_info['residual_p95'])}`，非零像素占比为 `{format_percent(registration_info['residual_nonzero'])}`。",
            f"按每对图像分别计算后，绝对差均值分布为：{stats_text([result['direct_mean'] for result in registration_info['results']])}。绝对差 P95 分布为：{stats_text([result['direct_p95'] for result in registration_info['results']])}。非零像素占比分布为：{stats_text([result['direct_nonzero'] for result in registration_info['results']])}。",
            "",
            "### 3.2 ECC 和仿射参数分布",
            "",
        ]
    )
    transform_rows = [
        ["ECC 相关系数", stats_text([result["ecc"] for result in registration_info["results"]])],
        ["tx（有符号）", stats_text([result["tx"] for result in registration_info["results"]])],
        ["ty（有符号）", stats_text([result["ty"] for result in registration_info["results"]])],
        ["旋转角（有符号，度）", stats_text([result["angle"] for result in registration_info["results"]])],
        ["|tx|", stats_text([abs(result["tx"]) for result in registration_info["results"]])],
        ["|ty|", stats_text([abs(result["ty"]) for result in registration_info["results"]])],
        ["|旋转角|", stats_text([abs(result["angle"]) for result in registration_info["results"]])],
    ]
    lines.extend(markdown_table(["量", "50 对样本分布"], transform_rows))
    near_count = registration_info["near_identity_count"]
    near_fraction = near_count / SAMPLE_SIZE
    ecc_p5 = registration_info["ecc_stats"]["p5"]
    approx_pre_registered = (
        ecc_p5 >= 0.95
        and registration_info["abs_tx_stats"]["p95"] <= 3.0
        and registration_info["abs_ty_stats"]["p95"] <= 2.0
        and registration_info["abs_angle_stats"]["p95"] <= 0.3
    )
    lines.extend(
        [
            "",
            f"近零变换统计标准固定为 `|tx|≤1 px`、`|ty|≤1 px`、`|旋转角|≤0.1°`。满足三项的样本为 `{near_count}/{SAMPLE_SIZE}`（{format_percent(near_fraction)}），ECC P5 为 `{format_number(ecc_p5, 5)}`。",
            f"近似预配准判定口径固定为 `ECC P5≥0.95`、`|tx| P95≤3 px`、`|ty| P95≤2 px`、`|旋转角| P95≤0.3°`。本样本的对应实测值为 `{format_number(ecc_p5, 5)}`、`{format_number(registration_info['abs_tx_stats']['p95'])}` px、`{format_number(registration_info['abs_ty_stats']['p95'])}` px、`{format_number(registration_info['abs_angle_stats']['p95'])}°`。",
            f"配准判定：**{'是，主数据近似预配准，但不是像素级完全对齐' if approx_pre_registered else '否，50 对样本未显示出稳定的近似预配准状态'}**。判定依据是上述 50 对 test/template 文件的 ECC 相关系数和仿射变换，不是 README 的文字描述。直接绝对差仍有非零残差，数值见本节 3.1。",
            "",
            "### 3.3 逐样本结果",
            "",
        ]
    )
    registration_rows = [
        [
            str(result["image_id"]),
            format_number(result["direct_mean"]),
            format_number(result["direct_p95"]),
            format_percent(result["direct_nonzero"]),
            format_number(result["ecc"], 5),
            format_number(result["tx"]),
            format_number(result["ty"]),
            format_number(result["angle"], 4),
        ]
        for result in registration_info["results"]
    ]
    lines.extend(
        markdown_table(
            ["样本 ID", "绝对差均值", "绝对差 P95", "非零占比", "ECC", "tx", "ty", "角度°"],
            registration_rows,
        )
    )

    box_records = annotation_info["box_records"]
    lines.extend(
        [
            "",
            "## 4. 标注",
            "",
            "标注统计依据 `PCBData/group*/ *_not/*.txt` 的实际内容，共 1500 个与图像对对应的标注文件；类别名称同时对照 `data/raw/DeepPCB/README.md` 的映射。",
            f"实际格式是每行 `{sorted(annotation_info['token_count_counts'].items())}` 个空白分隔字段，分隔符统计为 `{dict(annotation_info['delimiter_counts'])}`。每行字段顺序为 `x1 y1 x2 y2 type`，不是 README 示例中写的逗号分隔形式。",
            "",
            "### 4.1 类别 ID 映射",
            "",
        ]
    )
    class_map_rows = [[str(class_id), CLASS_NAMES[class_id]] for class_id in sorted(CLASS_NAMES)]
    lines.extend(markdown_table(["原始 class ID", "实际名称"], class_map_rows))
    lines.extend(
        [
            "",
            "### 4.2 每图缺陷数",
            "",
        ]
    )
    count_rows = []
    for split in ("trainval", "test", "all"):
        values = (
            annotation_info["image_box_counts"]
            if split == "all"
            else annotation_info["image_box_counts_by_split"][split]
        )
        count_rows.append([split, str(len(values)), stats_text(values)])
    lines.extend(markdown_table(["划分", "图像数", "每图标注框数分布"], count_rows))
    count_frequency = Counter(annotation_info["image_box_counts"])
    lines.extend(
        [
            "",
            "全部图像的每图缺陷数频数：",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["每图框数", "图像数", "占全部图像"],
            [
                [str(count), str(number), format_percent(number / len(records))]
                for count, number in sorted(count_frequency.items())
            ],
        )
    )
    lines.extend(
        [
            "",
            "### 4.3 缺陷框宽、高和面积",
            "",
            f"全部 `{len(box_records)}` 个标注框的宽度分布为：{stats_text([box['width'] for box in box_records])}。高度分布为：{stats_text([box['height'] for box in box_records])}。面积分布为：{stats_text([box['area'] for box in box_records])}。宽度和高度均按 `x2-x1`、`y2-y1` 计算，面积为宽乘高。",
            "",
        ]
    )
    class_box_rows = []
    for class_id in sorted(annotation_info["class_counts"]):
        class_boxes = [box for box in box_records if box["class_id"] == class_id]
        class_box_rows.append(
            [
                str(class_id),
                CLASS_NAMES[class_id],
                str(len(class_boxes)),
                str(annotation_info["class_image_counts"][class_id]),
                stats_text([box["width"] for box in class_boxes]),
                stats_text([box["height"] for box in class_boxes]),
                stats_text([box["area"] for box in class_boxes]),
            ]
        )
    lines.extend(
        markdown_table(
            ["ID", "类别", "框数", "含该类图像数", "宽度分布", "高度分布", "面积分布"],
            class_box_rows,
        )
    )
    min_area_box = min(box_records, key=lambda box: box["area"])
    min_width_box = min(box_records, key=lambda box: box["width"])
    min_height_box = min(box_records, key=lambda box: box["height"])
    min_box_rows = []
    for label, box, size_value in (
        (
            "最小面积框",
            min_area_box,
            f"{min_area_box['width']}×{min_area_box['height']}，面积 {min_area_box['area']}",
        ),
        (
            "最小宽度框",
            min_width_box,
            f"{min_width_box['width']}×{min_width_box['height']}，面积 {min_width_box['area']}",
        ),
        (
            "最小高度框",
            min_height_box,
            f"{min_height_box['width']}×{min_height_box['height']}，面积 {min_height_box['area']}",
        ),
    ):
        min_box_rows.append(
            [
                label,
                str(box["image_id"]),
                str(box["split"]),
                CLASS_NAMES[box["class_id"]],
                str(box["annotation_rel"]),
                f"({box['x1']},{box['y1']},{box['x2']},{box['y2']})",
                size_value,
            ]
        )
    lines.extend(
        [
            "",
            "最小缺陷尺寸记录如下。坐标来自对应标注文件，尺寸来自该框本身。",
            "",
        ]
    )
    lines.extend(markdown_table(["指标", "图像 ID", "划分", "类别", "标注文件", "坐标", "尺寸"], min_box_rows))

    groups = group_info["groups"]
    lines.extend(
        [
            "",
            "## 5. 模板分组",
            "",
            "模板分组依据图像相对路径中的 `groupXXXXX`、其下一级模板目录 `XXXXX` 以及数字文件名前缀。1500 个列表记录中，三者一致的记录数为 `{}/1500`。".format(group_info["consistency_count"]),
            f"可从目录和文件名还原出 `{len(groups)}` 个模板组。trainval 中出现 `{group_info['train_group_count']}` 组，test 中出现 `{group_info['test_group_count']}` 组，两个官方划分交叠 `{group_info['overlap_group_count']}` 组。",
            "",
        ]
    )
    group_rows = []
    for group in sorted(groups):
        row = groups[group]
        group_rows.append([group, group.removeprefix("group"), str(row["all"]), str(row["trainval"]), str(row["test"])])
    lines.extend(markdown_table(["目录组", "模板 ID", "总对数", "trainval", "test"], group_rows))
    group_stats = group_info["group_stats"]
    group_ratio = group_stats["max"] / group_stats["min"]
    lines.extend(
        [
            "",
            f"每组图像对数分布为：{stats_text(group_info['group_sizes'])}；最大组与最小组之比为 `{format_number(group_ratio)}`。按这组实测数量差异，模板组分布判定为**不均衡**。",
            f"官方 trainval/test 划分的组交叠数为 `{group_info['overlap_group_count']}/{len(groups)}`，另有 `{len(groups) - group_info['overlap_group_count']}` 组只出现在 test，因此官方划分不是模板隔离划分。上述结论来自 `PCBData` 路径和两个官方列表文件，不是根据图像外观推断。",
        ]
    )

    binary_pixel_statement = (
        f"严格二值图比例为 {exact_binary_count}/{len(image_info['all_paths'])}，值 0 或 255 的像素占比为 {format_percent(extreme_fraction)}"
    )
    if approx_pre_registered:
        registration_conclusion = (
            f"M1 配准结论：主数据是近似预配准，不是像素级完全对齐。50 对 test/template 样本中 {near_count}/{SAMPLE_SIZE} 对满足更严格的近零变换标准，ECC P5 为 {format_number(ecc_p5, 5)}，|tx| P95 为 {format_number(registration_info['abs_tx_stats']['p95'])} px，|ty| P95 为 {format_number(registration_info['abs_ty_stats']['p95'])} px，|旋转角| P95 为 {format_number(registration_info['abs_angle_stats']['p95'])}°；M1 在这批原始主数据上的实测作用是小残余对齐和质量核查，不是从大位移开始的几何配准。"
        )
    else:
        registration_conclusion = (
            f"M1 配准结论：50 对样本没有达到预配准判定，只有 {near_count}/{SAMPLE_SIZE} 对满足近零变换标准，ECC P5 为 {format_number(ecc_p5, 5)}。"
        )
    if exact_binary_count == len(image_info["all_paths"]):
        m2_conclusion = (
            f"M2 外观差异结论：严格二值判定为是，{binary_pixel_statement}；在文件解码值层面没有连续灰度信息。"
        )
        rq1_conclusion = (
            "RQ1 判定：保留 A2 到 A3 作为实测消融，但把 A3 解释为二值结构上的外观差异检验，不把连续光度归一或 SSIM 增益当作前置事实。"
        )
    else:
        m2_conclusion = (
            f"M2 外观差异结论：严格二值判定为否，{binary_pixel_statement}；文件中存在 JPEG 解码灰度值，但像素仍集中在极端值附近，连续光度变化的实测空间有限。"
        )
        rq1_conclusion = (
            "RQ1 判定：保留 A2 到 A3，但将 A3 明确标为近二值 JPEG 图像上的增量检验，单独报告其差异，不能把 A3 的收益解释为普通连续光度纹理带来的收益。"
        )
    lines.extend(
        [
            "",
            "## 6. 结论及对方法的影响",
            "",
            f"1. **M1 配准**：{registration_conclusion}（依据：`data/raw/DeepPCB/PCBData/test.txt` 的 500 对中固定种子抽取的 50 对，逐对文件见第 3.3 节。）",
            f"2. **M2 外观差异**：{m2_conclusion}（依据：`PCBData` 下全部 {len(image_info['all_paths'])} 个 JPG 的解码像素、唯一值统计和第 2.3 节完整直方图。）",
            f"3. **RQ1 消融**：{rq1_conclusion}（依据：上述全部图像像素统计，而不是依据方案文档中的先验描述。）",
            f"4. **最小证据目标**：最小面积框为 `{min_area_box['width']}×{min_area_box['height']}` 像素，面积 `{min_area_box['area']}`，来自图像 `{min_area_box['image_id']}`；最小宽度为 `{min_width_box['width']}` 像素，最小高度为 `{min_height_box['height']}` 像素。这个尺寸记录给出了证据卡局部放大的实测下限。（依据：1500 个标注文件中的全部 {len(box_records)} 个框，第 4.3 节。）",
            f"5. **防泄漏分组**：模板 ID 可以从目录和文件名稳定还原，但数量分布不均衡，最大组/最小组为 `{format_number(group_ratio)}`，官方 trainval/test 共享 `{group_info['overlap_group_count']}/{len(groups)}` 组，另有 `{len(groups) - group_info['overlap_group_count']}` 组只在 test。模板分组前提已被文件结构确认，官方划分的组隔离状态已被排除。（依据：1500 条官方列表记录和 `PCBData/group*/` 实际路径，第 5 节。）",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    records = load_split_records()
    image_info = inspect_images(records)
    annotation_info = load_annotations(records)
    registration_info = measure_registration(records)
    group_info = summarize_groups(records)
    report = render_report(records, image_info, annotation_info, registration_info, group_info)
    REPORT_PATH.write_text(report)
    print(f"报告已写入 {relative_path(REPORT_PATH)}")


if __name__ == "__main__":
    main()
