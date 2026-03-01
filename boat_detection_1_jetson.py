import cv2
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# ======================================================================
#  JETSON BOAT DETECTION — High Performance
#  Same structure as face_detection_jetson.py
#  Uses MobileNet-SSD (COCO) — detects boats via OpenCV DNN
#  CUDA backend auto-detected (no PyTorch needed)
#  Shows: FPS, inference time, boat count, GPU/CPU badge
#  Camera: IMX477 CSI via GStreamer
# ======================================================================

OUTPUT_DIR = Path.home() / "Desktop" / "vision_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── MobileNet-SSD COCO model ──────────────────────────────────────────
PROTOTXT_URL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
MODEL_URL    = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/mobilenet_iter_73000.caffemodel"
PROTOTXT     = Path.home() / ".cache" / "boat_detect" / "MobileNetSSD_deploy.prototxt"
MODEL_FILE   = Path.home() / ".cache" / "boat_detect" / "MobileNetSSD.caffemodel"

# MobileNet-SSD COCO class list (21 classes including background)
SSD_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
BOAT_CLASS_ID = 4   # "boat" is index 4 in SSD_CLASSES

# Boat type colors (used when classifier is added later)
BOAT_COLOR    = (0, 200, 255)   # cyan


# ── Download models ───────────────────────────────────────────────────

def download_models():
    import urllib.request
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not PROTOTXT.exists():
        print("[INFO] Downloading MobileNet-SSD prototxt...")
        try:
            urllib.request.urlretrieve(PROTOTXT_URL, PROTOTXT)
            print(f"[OK]   {PROTOTXT.name}")
        except Exception as e:
            print(f"[FAIL] prototxt: {e}")
            return False

    if not MODEL_FILE.exists():
        print("[INFO] Downloading MobileNet-SSD weights (~23MB)...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
            print(f"[OK]   {MODEL_FILE.name} ({MODEL_FILE.stat().st_size//1024}KB)")
        except Exception as e:
            print(f"[FAIL] weights: {e}")
            return False

    print("[OK]   Model files ready.")
    return True


# ── Load detector ─────────────────────────────────────────────────────

def load_detector():
    if not download_models():
        return None, "CPU"

    net = cv2.dnn.readNetFromCaffe(str(PROTOTXT), str(MODEL_FILE))

    # Try CUDA backend first, fall back to CPU
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        # Quick probe to confirm CUDA works
        test = np.zeros((300, 300, 3), dtype=np.uint8)
        blob = cv2.dnn.blobFromImage(test, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        net.forward()
        backend = "CUDA"
        print("[GPU]  CUDA backend active — running on Jetson GPU")
    except Exception:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        backend = "CPU"
        print("[WARN] CUDA not available — running on CPU")

    return net, backend


# ── Detect boats ──────────────────────────────────────────────────────

def detect_boats(net, frame, conf_threshold=0.5):
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        0.007843,           # scale factor for MobileNet-SSD
        (300, 300),
        127.5               # mean subtraction
    )

    t0 = time.perf_counter()
    net.setInput(blob)
    detections = net.forward()
    infer_ms = (time.perf_counter() - t0) * 1000

    boats = []
    for i in range(detections.shape[2]):
        class_id   = int(detections[0, 0, i, 1])
        confidence = float(detections[0, 0, i, 2])

        if class_id == BOAT_CLASS_ID and confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            # Clamp to frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)
            boats.append((x1, y1, x2, y2, confidence))

    return boats, infer_ms


# ── Draw detections ───────────────────────────────────────────────────

def draw_boats(frame, boats):
    fh, fw = frame.shape[:2]
    for (x1, y1, x2, y2, conf) in boats:
        color = BOAT_COLOR
        bw    = x2 - x1

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Corner accents
        c = 16
        for (cx, cy, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame, (cx, cy), (cx + c*dx, cy), color, 3)
            cv2.line(frame, (cx, cy), (cx, cy + c*dy), color, 3)

        # Confidence bar
        bar_y1 = min(y2 + 2, fh - 9)
        bar_y2 = min(y2 + 8, fh - 1)
        cv2.rectangle(frame, (x1, bar_y1), (x2,          bar_y2), (40,40,40), -1)
        cv2.rectangle(frame, (x1, bar_y1), (x1+int(bw*conf), bar_y2), color,  -1)

        # Label
        label = f"Boat  {conf*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ly = max(y1 - 6, th + 8)
        cv2.rectangle(frame, (x1, ly - th - 8), (x1 + tw + 10, ly + 2), color, -1)
        cv2.putText(frame, label, (x1 + 5, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


# ── HUD ───────────────────────────────────────────────────────────────

def draw_hud(frame, boats, fps, infer_ms, backend, frame_count):
    fh, fw  = frame.shape[:2]
    overlay = frame.copy()

    # Top bar
    cv2.rectangle(overlay, (0, 0), (fw, 56), (8, 8, 8), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Boat count
    cv2.putText(frame, f"Boats: {len(boats)}", (12, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 220, 80) if boats else (120, 120, 120), 2)

    # FPS
    fps_color = (0, 220, 80) if fps >= 25 else (0, 180, 255) if fps >= 15 else (0, 80, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (fw - 175, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, fps_color, 2)

    # Inference time
    cv2.putText(frame, f"Inf: {infer_ms:.1f}ms", (fw - 175, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)

    # Backend badge
    badge_color = (0, 160, 80) if backend == "CUDA" else (160, 80, 0)
    cv2.rectangle(frame, (fw//2 - 45, 8), (fw//2 + 45, 48), badge_color, -1)
    cv2.putText(frame, backend, (fw//2 - 32, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    # Bottom stats bar
    cv2.rectangle(overlay, (0, fh - 30), (fw, fh), (8, 8, 8), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, f"Frame: {frame_count}   Q=Quit  S=Save  +/-=Threshold",
                (10, fh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 150, 150), 1)


# ── GStreamer pipeline ────────────────────────────────────────────────

def gstreamer_pipeline(
    capture_width=1920, capture_height=1080,
    display_width=1280, display_height=720,
    framerate=60, flip_method=0
):
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, "
        f"format=NV12, framerate={framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={display_width}, height={display_height}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
    )


# ── Main ──────────────────────────────────────────────────────────────

def run(use_gstreamer=True):
    net, backend = load_detector()
    if net is None:
        return

    # Open camera
    if use_gstreamer:
        pipeline = gstreamer_pipeline()
        print("[INFO] Opening CSI camera via GStreamer...")
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        print("[INFO] Opening USB webcam at index 0...")
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        if use_gstreamer:
            print("  Test: gst-launch-1.0 nvarguscamerasrc ! nvvidconv ! videoconvert ! autovideosink")
        return

    if use_gstreamer:
        print("[INFO] Warming up camera (2s)...")
        time.sleep(2)
        for _ in range(10):
            cap.read()

    conf_thresh = 0.5
    frame_count = 0
    shot_n      = 0
    fps         = 0.0
    infer_ms    = 0.0
    fps_timer   = time.time()
    fps_frames  = 0

    print(f"[OK]   Running — Backend: {backend}")
    print("       Q=quit | S=screenshot | +/-=confidence threshold\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue

        frame_count += 1
        fps_frames  += 1

        boats, infer_ms = detect_boats(net, frame, conf_threshold=conf_thresh)
        draw_boats(frame, boats)

        # FPS — update every 0.5s for stable reading
        now = time.time()
        if now - fps_timer >= 0.5:
            fps        = fps_frames / (now - fps_timer)
            fps_timer  = now
            fps_frames = 0

        draw_hud(frame, boats, fps, infer_ms, backend, frame_count)
        cv2.imshow("Boat Detection — Jetson | Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            shot_n += 1
            path = OUTPUT_DIR / f"boat_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{shot_n}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"[SAVED] {path}")
        elif key in (ord('+'), ord('=')):
            conf_thresh = min(0.95, conf_thresh + 0.05)
            print(f"[CONF] {conf_thresh:.2f}")
        elif key == ord('-'):
            conf_thresh = max(0.10, conf_thresh - 0.05)
            print(f"[CONF] {conf_thresh:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] {frame_count} frames | FPS: {fps:.1f} | Backend: {backend}")


if __name__ == "__main__":
    USE_GSTREAMER = True   # False = USB webcam
    run(use_gstreamer=USE_GSTREAMER)
