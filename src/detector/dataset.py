"""DeepPCB manifest and GroupKFold split conversion for Ultralytics."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable

import cv2
import yaml

CLASS_NAMES = ["open", "short", "mousebite", "spur", "copper", "pinhole"]
SEED = 20260811


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_splits(path: Path) -> dict[str, dict[str, list[str]]]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_gray_three_channel(path: Path) -> tuple[object, int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    height, width = image.shape[:2]
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), width, height


def annotation_to_yolo(annotation_path: Path, width: int, height: int) -> list[str]:
    lines = []
    for line in annotation_path.read_text(encoding="utf-8").splitlines():
        x1, y1, x2, y2, class_id = line.split()
        x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
        class_index = int(class_id) - 1
        cx = ((x1 + x2) / 2.0) / width
        cy = ((y1 + y2) / 2.0) / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        lines.append(f"{class_index} {cx:.8f} {cy:.8f} {box_width:.8f} {box_height:.8f}")
    return lines


def _replace_symlink(path: Path, target: Path) -> None:
    if os.path.lexists(path):
        path.unlink()
    path.symlink_to(target.resolve())


def _split_rows(rows: Iterable[dict[str, str]], split_groups: list[str]) -> list[dict[str, str]]:
    group_set = set(split_groups)
    return [row for row in rows if row["group"] in group_set]


def build_fold_dataset(
    fold: int,
    manifest_path: Path = Path("data/manifests/deeppcb.csv"),
    split_path: Path = Path("data/splits/groupkfold5.json"),
    output_root: Path = Path("cache/detector_dataset"),
) -> Path:
    rows = read_manifest(manifest_path)
    splits = read_splits(split_path)
    fold_spec = splits[f"fold_{fold}"]
    fold_root = output_root / f"fold{fold}"
    for role in ("train", "val", "test"):
        image_dir = fold_root / "images" / role
        label_dir = fold_root / "labels" / role
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for row in sorted(_split_rows(rows, fold_spec[role]), key=lambda item: item["pair_id"]):
            source_image = Path(row["test_path"])
            source_label = Path(row["anno_path"])
            image_target = image_dir / source_image.name
            label_target = label_dir / f"{source_image.stem}.txt"
            _replace_symlink(image_target, source_image)
            label_lines = annotation_to_yolo(source_label, 640, 640)
            label_target.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
    data_yaml = fold_root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(fold_root.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "nc": len(CLASS_NAMES),
                "names": CLASS_NAMES,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return data_yaml


def rows_for_role(
    fold: int,
    role: str,
    manifest_path: Path = Path("data/manifests/deeppcb.csv"),
    split_path: Path = Path("data/splits/groupkfold5.json"),
) -> list[dict[str, str]]:
    rows = read_manifest(manifest_path)
    splits = read_splits(split_path)
    return sorted(_split_rows(rows, splits[f"fold_{fold}"][role]), key=lambda item: item["pair_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/deeppcb.csv"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits/groupkfold5.json"))
    parser.add_argument("--output-root", type=Path, default=Path("cache/detector_dataset"))
    args = parser.parse_args()
    yaml_path = build_fold_dataset(args.fold, args.manifest, args.splits, args.output_root)
    print(f"fold_{args.fold}: {yaml_path}")


if __name__ == "__main__":
    main()
