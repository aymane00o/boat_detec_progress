#!/usr/bin/env python3
"""
======================================================================
 BOAT DATASET COLLECTOR
 Captures images from your IMX477 CSI camera and saves them
 organized by boat class — ready for YOLOv8 training.

 HOW TO USE:
   1. Run this script
   2. Select the boat class you are filming (1-12)
   3. Point camera at a boat
   4. Press SPACE to capture (or A for auto-capture every 1s)
   5. Press N to switch to next class
   6. Repeat for all boat types you have access to
   7. When done, run boat_train_classifier_local.py to train

 TIPS FOR GOOD DATA:
   - Capture 100-300 images per class minimum
   - Vary: distance, angle, lighting, weather
   - Include partial views (not just perfect centered shots)
   - Capture morning, noon, sunset light conditions
================================================================
"""

import cv2
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import threading
import json

# ── Config ─────────────────────────────────────────────────────────────
OUTPUT_DIR   = Path.home() / "Desktop" / "vision_output" / "my_dataset"
AUTO_INTERVAL = 1.0   # seconds between auto-captures

# ── Classes ────────────────────────────────────────────────────────────
CLASSES = [
    "Sailboat",
    "Speedboat",
    "Fishing Boat",
    "Ferry",
    "Cargo Ship",
    "Kayak / Canoe",
    "Military Vessel",
    "Tugboat",
    "Cruise Ship",
    "Inflatable RIB",
    "Houseboat",
    "Small Boat",
]

CLASS_COLORS = [
    (  0, 220,  80), (  0, 200, 100), (255, 200,   0), (  0, 180, 255),
    (  0, 140, 255), (255, 180,   0), (  0,  60, 220), (  0, 160, 255),
    (180,   0, 255), (255, 120,   0), (160,   0, 220), (100, 100, 100),
]

# ── Camera thread ───────────────────────────────────────────────────────
class CameraThread(threading.Thread):
    def __init__(self, src, backend):
        super().__init__(daemon=True)
        self.src     = src
        self.backend = backend
        self.frame   = None
        self.lock    = threading.Lock()
        self.running = True
        self.opened  = False

    def run(self):
        cap = cv2.VideoCapture(self.src, self.backend)
        if not cap.isOpened():
            return
        self.opened = True
        for _ in range(10):
            cap.read()
        while self.running:
            ret, f = cap.read()
            if ret:
                with self.lock:
                    self.frame = f
        cap.release()

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False


def gstreamer_pipeline(cw=1920, ch=1080, dw=1280, dh=720, fps=60, flip=0):
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM),width={cw},height={ch},"
        f"format=NV12,framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw,width={dw},height={dh},format=BGRx ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink drop=1 max-buffers=1"
    )


def get_class_counts():
    counts = {}
    for i, cls in enumerate(CLASSES):
        safe = cls.replace(" ", "_").replace("/", "-")
        total = 0
        for split in ("train", "valid", "test"):
            d = OUTPUT_DIR / split / safe
            if d.exists():
                total += len(list(d.glob("*.jpg")))
        counts[cls] = total
    return counts


def save_image(frame, class_idx, split="train"):
    cls      = CLASSES[class_idx]
    safe     = cls.replace(" ", "_").replace("/", "-")
    save_dir = OUTPUT_DIR / split / safe
    save_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = save_dir / f"{safe}_{ts}.jpg"
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


def draw_ui(frame, class_idx, counts, auto_mode, last_saved, frame_n):
    fh, fw = frame.shape[:2]
    overlay = frame.copy()
    color   = CLASS_COLORS[class_idx]
    cls     = CLASSES[class_idx]
    count   = counts.get(cls, 0)

    # ── Top bar ─────────────────────────────────────────────────────
    cv2.rectangle(overlay, (0, 0), (fw, 70), (6, 6, 6), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    # Current class
    cv2.putText(frame, f"CLASS: {cls}", (14, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)
    cv2.putText(frame, f"Saved: {count} images", (14, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)

    # Auto mode badge
    if auto_mode:
        cv2.rectangle(frame, (fw-130, 8), (fw-8, 52), (0, 140, 50), -1)
        cv2.putText(frame, "AUTO ON", (fw-122, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    else:
        cv2.rectangle(frame, (fw-130, 8), (fw-8, 52), (50, 50, 50), -1)
        cv2.putText(frame, "MANUAL", (fw-118, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (160, 160, 160), 2)

    # ── Class list sidebar ───────────────────────────────────────────
    panel_w = 200
    cv2.rectangle(overlay, (0, 70), (panel_w, fh), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    for i, (c, col) in enumerate(zip(CLASSES, CLASS_COLORS)):
        y       = 90 + i * 38
        cnt     = counts.get(c, 0)
        is_cur  = (i == class_idx)

        if is_cur:
            cv2.rectangle(frame, (4, y-20), (panel_w-4, y+14), col, -1)
            txt_col = (0, 0, 0)
        else:
            txt_col = col if cnt > 0 else (60, 60, 60)

        short = c[:16]
        cv2.putText(frame, f"{i+1:2d}. {short}", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, txt_col, 1 if not is_cur else 2)
        # Mini count bar
        bar = min(panel_w - 20, cnt // 2)
        if bar > 0 and not is_cur:
            cv2.rectangle(frame, (8, y+4), (8+bar, y+8), col, -1)
        if cnt > 0:
            cv2.putText(frame, str(cnt), (panel_w-38, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (200,200,200) if not is_cur else (0,0,0), 1)

    # ── Flash when saved ─────────────────────────────────────────────
    if last_saved and (time.time() - last_saved) < 0.15:
        flash = frame.copy()
        cv2.rectangle(flash, (panel_w, 70), (fw, fh), color, -1)
        cv2.addWeighted(flash, 0.25, frame, 0.75, 0, frame)
        cv2.putText(frame, "CAPTURED", (fw//2-60, fh//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)

    # ── Bottom bar ───────────────────────────────────────────────────
    cv2.rectangle(overlay, (0, fh-36), (fw, fh), (6, 6, 6), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.putText(frame,
                "SPACE=capture  A=auto  N=next class  P=prev  1-9=jump  Q=quit",
                (panel_w+8, fh-10), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (140,140,140), 1)

    # ── Total progress bar ───────────────────────────────────────────
    total     = sum(counts.values())
    target    = len(CLASSES) * 200   # 200 per class = good dataset
    bar_w     = fw - panel_w - 16
    filled    = int(bar_w * min(1.0, total / target))
    cv2.rectangle(frame, (panel_w+8, fh-52), (fw-8,        fh-42), (30,30,30), -1)
    cv2.rectangle(frame, (panel_w+8, fh-52), (panel_w+8+filled, fh-42),
                  (0, 200, 100), -1)
    cv2.putText(frame, f"Total: {total}/{target} images",
                (panel_w+8, fh-56), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120,120,120), 1)


def print_summary(counts):
    print("\n" + "="*50)
    print("  DATASET SUMMARY")
    print("="*50)
    total = 0
    for cls in CLASSES:
        cnt   = counts.get(cls, 0)
        total += cnt
        bar   = "█" * min(30, cnt // 5)
        status = "✓ GOOD" if cnt >= 100 else ("△ LOW" if cnt > 0 else "✗ EMPTY")
        print(f"  {status}  {cls:20s} {cnt:4d}  {bar}")
    print(f"\n  Total images: {total}")
    print(f"  Dataset path: {OUTPUT_DIR}")
    print("\n  Recommendation:")
    print("  - Aim for 100-300 images per class")
    print("  - Run boat_train_classifier_local.py to train")
    print("="*50)


def run(use_gstreamer=True):
    # Setup output dirs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Start camera
    src     = gstreamer_pipeline() if use_gstreamer else 0
    backend = cv2.CAP_GSTREAMER if use_gstreamer else cv2.CAP_V4L2
    cam     = CameraThread(src, backend)
    cam.start()

    print("[INFO] Starting camera...")
    t_end = time.time() + 12
    while not cam.opened and time.time() < t_end:
        time.sleep(0.1)
    if not cam.opened:
        print("[ERROR] Camera failed. Run: sudo systemctl restart nvargus-daemon")
        return

    print("[OK]   Camera ready")
    print("\n  Controls:")
    print("  SPACE = capture image")
    print("  A     = toggle auto-capture (every 1s)")
    print("  N/P   = next/previous class")
    print("  1-9   = jump to class number")
    print("  Q     = quit and show summary\n")

    class_idx  = 0
    auto_mode  = False
    last_auto  = 0.0
    last_saved = None
    frame_n    = 0

    while True:
        frame = cam.read()
        if frame is None:
            time.sleep(0.005)
            continue

        frame_n += 1
        counts   = get_class_counts()

        # Auto-capture
        if auto_mode and (time.time() - last_auto) >= AUTO_INTERVAL:
            path = save_image(frame, class_idx)
            last_auto  = time.time()
            last_saved = time.time()
            print(f"[AUTO] {CLASSES[class_idx]} → {path.name}")

        draw_ui(frame, class_idx, counts, auto_mode, last_saved, frame_n)
        cv2.imshow("Boat Dataset Collector | Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            path = save_image(frame, class_idx)
            last_saved = time.time()
            cnt = counts.get(CLASSES[class_idx], 0) + 1
            print(f"[SNAP] {CLASSES[class_idx]} ({cnt}) → {path.name}")
        elif key == ord('a'):
            auto_mode = not auto_mode
            print(f"[AUTO] {'ON — capturing every 1s' if auto_mode else 'OFF'}")
        elif key == ord('n'):
            class_idx = (class_idx + 1) % len(CLASSES)
            auto_mode = False
            print(f"[CLASS] → {CLASSES[class_idx]}")
        elif key == ord('p'):
            class_idx = (class_idx - 1) % len(CLASSES)
            auto_mode = False
            print(f"[CLASS] → {CLASSES[class_idx]}")
        elif ord('1') <= key <= ord('9'):
            idx = key - ord('1')
            if idx < len(CLASSES):
                class_idx = idx
                auto_mode = False
                print(f"[CLASS] → {CLASSES[class_idx]}")

    cam.stop()
    cv2.destroyAllWindows()
    print_summary(get_class_counts())


if __name__ == "__main__":
    USE_GSTREAMER = True   # False = USB webcam
    run(use_gstreamer=USE_GSTREAMER)
