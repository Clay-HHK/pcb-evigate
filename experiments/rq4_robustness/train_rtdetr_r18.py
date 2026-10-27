"""Fine-tune official PekingU RT-DETR-R18VD on DeepPCB fold0."""

from __future__ import annotations

import csv
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

os.environ["HF_HOME"] = str(Path("cache/huggingface").resolve())
os.environ["MPLCONFIGDIR"] = "/tmp/zyp-rtdetr-matplotlib"

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForObjectDetection

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.detector.dataset import rows_for_role
from src.detector.predict import filter_records, read_ground_truth, select_working_confidence, write_jsonl


SEED = 20260811
MODEL_ID = "PekingU/rtdetr_r18vd"
OUTPUT = Path("cache/rtdetr_r18_fold0")
PROGRESS = Path("experiments/rq4_robustness/progress.csv")
CLASS_NAMES = ["open", "short", "mousebite", "spur", "pinhole", "spurious_copper"]


class DeepPCBDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], processor: AutoImageProcessor) -> None:
        self.rows = rows
        self.processor = processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image = Image.open(row["test_path"]).convert("RGB")
        annotations = []
        for annotation_id, (box, class_id) in enumerate(read_ground_truth(Path(row["anno_path"]))):
            x1, y1, x2, y2 = box
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": index,
                    "category_id": class_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                }
            )
        encoded = self.processor(
            images=image,
            annotations={"image_id": index, "annotations": annotations},
            return_tensors="pt",
        )
        return {"pixel_values": encoded["pixel_values"].squeeze(0), "labels": encoded["labels"][0]}


def collate(batch: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": [item["labels"] for item in batch],
    }


def append_progress(stage: str, completed: int, total: int, detail: str) -> None:
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{datetime.now().astimezone().isoformat(timespec='seconds')},{stage},{completed},{total},{detail}\n"
        )


def train() -> tuple[AutoImageProcessor, AutoModelForObjectDetection]:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID,
        size={"height": 640, "width": 640},
        local_files_only=True,
    )
    model = AutoModelForObjectDetection.from_pretrained(
        MODEL_ID,
        num_labels=6,
        id2label={index: name for index, name in enumerate(CLASS_NAMES)},
        label2id={name: index for index, name in enumerate(CLASS_NAMES)},
        ignore_mismatched_sizes=True,
        local_files_only=True,
    ).cuda()
    train_loader = DataLoader(
        DeepPCBDataset(rows_for_role(0, "train"), processor),
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(SEED),
    )
    validation_loader = DataLoader(
        DeepPCBDataset(rows_for_role(0, "val"), processor),
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=collate,
    )
    backbone = [parameter for name, parameter in model.named_parameters() if "backbone" in name]
    detector = [parameter for name, parameter in model.named_parameters() if "backbone" not in name]
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": 1.0e-5}, {"params": detector, "lr": 1.0e-4}],
        weight_decay=1.0e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    scaler = torch.amp.GradScaler("cuda")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    history = []
    best_loss = float("inf")
    for epoch in range(1, 31):
        model.train()
        training_losses = []
        for batch in train_loader:
            pixel_values = batch["pixel_values"].cuda(non_blocking=True)
            labels = [{key: value.cuda(non_blocking=True) for key, value in item.items()} for item in batch["labels"]]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(pixel_values=pixel_values, labels=labels).loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            training_losses.append(float(loss.detach().cpu()))
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for batch in validation_loader:
                pixel_values = batch["pixel_values"].cuda(non_blocking=True)
                labels = [{key: value.cuda(non_blocking=True) for key, value in item.items()} for item in batch["labels"]]
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = model(pixel_values=pixel_values, labels=labels).loss
                validation_losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(training_losses))
        validation_loss = float(np.mean(validation_losses))
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        append_progress("rtdetr_train", epoch, 30, f"train_loss={train_loss:.6f};val_loss={validation_loss:.6f}")
        if validation_loss < best_loss:
            best_loss = validation_loss
            model.save_pretrained(OUTPUT / "best")
            processor.save_pretrained(OUTPUT / "best")
    with (OUTPUT / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    best_processor = AutoImageProcessor.from_pretrained(OUTPUT / "best", local_files_only=True)
    best_model = AutoModelForObjectDetection.from_pretrained(OUTPUT / "best", local_files_only=True).cuda().eval()
    return best_processor, best_model


def predict(
    processor: AutoImageProcessor,
    model: AutoModelForObjectDetection,
    rows: list[dict[str, str]],
) -> list[dict]:
    records = []
    with torch.no_grad():
        for completed, row in enumerate(rows, start=1):
            image = Image.open(row["test_path"]).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(pixel_values=inputs["pixel_values"].cuda())
            target_sizes = torch.tensor([[image.height, image.width]], device="cuda")
            result = processor.post_process_object_detection(outputs, threshold=0.001, target_sizes=target_sizes)[0]
            records.append(
                {
                    "pair_id": row["pair_id"],
                    "boxes": result["boxes"].cpu().numpy().tolist(),
                    "classes": result["labels"].cpu().numpy().astype(int).tolist(),
                    "scores": result["scores"].cpu().numpy().tolist(),
                }
            )
            if completed % 25 == 0 or completed == len(rows):
                append_progress("rtdetr_predict", completed, len(rows), row["pair_id"])
    return records


def main() -> None:
    processor, model = train()
    validation_rows = rows_for_role(0, "val")
    test_rows = rows_for_role(0, "test")
    validation = predict(processor, model, validation_rows)
    threshold, validation_f1 = select_working_confidence(validation, validation_rows)
    test = predict(processor, model, test_rows)
    write_jsonl(OUTPUT / "fold0_lowconf.jsonl", test)
    write_jsonl(OUTPUT / "fold0.jsonl", filter_records(test, threshold))
    (OUTPUT / "working_point.json").write_text(
        json.dumps({"threshold": threshold, "validation_f1": validation_f1}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    append_progress("rtdetr_complete", 1, 1, f"threshold={threshold:.6f};validation_f1={validation_f1:.6f}")
    print(f"RT-DETR-R18VD 完成，阈值={threshold:.6f}，验证 F1={validation_f1:.6f}")


if __name__ == "__main__":
    main()
