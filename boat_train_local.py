#!/usr/bin/env python3
"""
=============================================================
 BOAT CLASSIFIER — LOCAL TRAINING (no internet needed)
 Uses the dataset collected with collect_dataset.py
 Output: best.pt → plug directly into boat_detection_jetson.py
================================================================
"""

import sys
import shutil
from pathlib import Path
import random

DATASET_DIR = Path.home() / "Desktop" / "vision_output" / "my_dataset"
TRAIN_OUT   = Path.home() / "Desktop" / "vision_output" / "boat_classifier"

CLASSES = [
    "Sailboat", "Speedboat", "Fishing_Boat", "Ferry", "Cargo_Ship",
    "Kayak_-_Canoe", "Military_Vessel", "Tugboat", "Cruise_Ship",
    "Inflatable_RIB", "Houseboat", "Small_Boat",
]


def check_dataset():
    print("="*55)
    print("  DATASET CHECK")
    print("="*55)
    total = 0
    ready = []
    warn  = []
    empty = []

    for cls in CLASSES:
        safe  = cls.replace(" ", "_").replace("/", "-")
        train = DATASET_DIR / "train" / safe
        valid = DATASET_DIR / "valid" / safe
        n_tr  = len(list(train.glob("*.jpg"))) if train.exists() else 0
        n_val = len(list(valid.glob("*.jpg"))) if valid.exists() else 0
        total += n_tr + n_val

        if n_tr >= 100:
            ready.append((cls, n_tr, n_val))
        elif n_tr > 0:
            warn.append((cls, n_tr, n_val))
        else:
            empty.append(cls)

    for cls, n_tr, n_val in ready:
        print(f"  ✓  {cls:22s}  train:{n_tr:4d}  val:{n_val:3d}")
    for cls, n_tr, n_val in warn:
        print(f"  △  {cls:22s}  train:{n_tr:4d}  val:{n_val:3d}  (LOW — need 100+)")
    for cls in empty:
        print(f"  ✗  {cls:22s}  (no images — will skip)")

    print(f"\n  Total images: {total}")
    active = [c for c,_,_ in ready+warn]
    print(f"  Active classes: {len(active)}")

    if total == 0:
        print("\n[ERROR] No images found!")
        print(f"  Run collect_dataset.py first to capture images")
        print(f"  Expected path: {DATASET_DIR}/train/ClassName/image.jpg")
        sys.exit(1)

    if len(active) < 2:
        print("\n[ERROR] Need at least 2 classes with images to train.")
        sys.exit(1)

    return active


def auto_split(active_classes, val_ratio=0.15, test_ratio=0.05):
    """Auto-create val/test splits from train if they don't exist."""
    print("\n[SPLIT] Checking train/val/test splits...")
    for cls in active_classes:
        safe      = cls.replace(" ", "_").replace("/", "-")
        train_dir = DATASET_DIR / "train" / safe
        val_dir   = DATASET_DIR / "valid" / safe
        test_dir  = DATASET_DIR / "test"  / safe

        if not train_dir.exists():
            continue

        all_imgs = list(train_dir.glob("*.jpg"))
        if len(all_imgs) < 10:
            continue

        # Only split if val is empty
        val_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        existing_val  = len(list(val_dir.glob("*.jpg")))
        existing_test = len(list(test_dir.glob("*.jpg")))

        if existing_val == 0:
            random.shuffle(all_imgs)
            n_val  = max(1, int(len(all_imgs) * val_ratio))
            n_test = max(1, int(len(all_imgs) * test_ratio))
            for img in all_imgs[:n_val]:
                shutil.copy2(img, val_dir / img.name)
            for img in all_imgs[n_val:n_val+n_test]:
                shutil.copy2(img, test_dir / img.name)
            print(f"  Split {cls}: val={n_val}, test={n_test}")


def write_yaml(active_classes):
    import yaml
    # Map class folder names back to display names
    safe_classes = [c.replace(" ", "_").replace("/", "-") for c in active_classes]
    data = {
        "path":  str(DATASET_DIR),
        "train": "train",
        "val":   "valid",
        "test":  "test",
        "nc":    len(active_classes),
        "names": active_classes,
    }
    yaml_path = DATASET_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"\n[OK]  data.yaml written: {yaml_path}")
    return str(yaml_path)


def train(data_yaml, active_classes):
    import torch
    from ultralytics import YOLO

    device = "0" if torch.cuda.is_available() else "cpu"
    if device == "0":
        gpu  = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n[GPU]  {gpu} | {vram:.1f}GB VRAM")
        batch = 32
    else:
        print("\n[WARN] No GPU — CPU training (slow)")
        batch = 8

    n_classes = len(active_classes)
    epochs    = 100 if n_classes >= 8 else 80

    print(f"[INFO] Classes  : {n_classes}")
    print(f"[INFO] Epochs   : {epochs}")
    print(f"[INFO] Batch    : {batch}")
    print(f"[INFO] Output   : {TRAIN_OUT}\n")

    model = YOLO("yolov8m-cls.pt")

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=224,
        batch=batch,
        device=device,
        project=str(TRAIN_OUT),
        name="boat_types",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.005,
        warmup_epochs=5,
        patience=25,
        cos_lr=True,
        augment=True,
        degrees=15.0,
        fliplr=0.5,
        flipud=0.05,
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        translate=0.1,
        scale=0.5,
        erasing=0.3,      # random erasing — helps with partial views
        auto_augment="randaugment",
        save=True,
        plots=True,
        verbose=True,
    )

    best = TRAIN_OUT / "boat_types" / "weights" / "best.pt"
    print(f"\n{'='*55}")
    print(f"  TRAINING COMPLETE")
    print(f"  Model: {best}")
    print(f"{'='*55}")
    print(f"\n  Plug into boat_detection_jetson.py:")
    print(f"  Change:")
    print(f"    CLASSIFY_MODEL = 'yolov8s-cls.pt'")
    print(f"  To:")
    print(f"    CLASSIFY_MODEL = '{best}'")
    return str(best)


def validate(model_path, data_yaml):
    from ultralytics import YOLO
    print("\n[VALIDATE] Running on test split...")
    model   = YOLO(model_path)
    metrics = model.val(data=data_yaml, split="test", verbose=False)
    print(f"  Top-1 accuracy: {metrics.top1*100:.1f}%")
    print(f"  Top-5 accuracy: {metrics.top5*100:.1f}%")


if __name__ == "__main__":
    try:
        import yaml
    except ImportError:
        print("Run: pip3 install pyyaml ultralytics")
        sys.exit(1)

    print("="*55)
    print("  BOAT CLASSIFIER — LOCAL TRAINING")
    print(f"  Dataset: {DATASET_DIR}")
    print("="*55)

    active    = check_dataset()
    auto_split(active)
    data_yaml = write_yaml(active)

    print("\nReady to train. Start? [y/N] ", end="")
    if input().strip().lower() != "y":
        print("Aborted.")
        sys.exit(0)

    best = train(data_yaml, active)
    validate(best, data_yaml)
