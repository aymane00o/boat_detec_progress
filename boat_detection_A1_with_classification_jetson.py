import cv2
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import threading
import collections

# ================================================================
#  PROFESSIONAL BOAT DETECTION + CLASSIFICATION SYSTEM
#  Platform  : NVIDIA Jetson (Orin/Xavier) + IMX477 CSI camera
#  Detection : YOLOv8m — high precision, multi-scale
#  Classifier: YOLOv8s-cls — boat type per detection
#  Output    : Structured Detection objects — ready for tracker plug-in
#
#  Architecture (tracker-ready):
#    CameraThread → FrameBuffer → Detector → Classifier → Renderer
#
#  To add tracking later, replace detect_boats() output with:
#    tracker.update(boats) → returns boats with persistent IDs
#
#  Controls:
#    Q = quit        S = screenshot
#    C = classifier  A = augment (more accurate, slower)
#    + / - = confidence threshold
#    ] / [ = inference size (1280 = best for far boats)
# =================================================================

OUTPUT_DIR = Path.home() / "Desktop" / "vision_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────
DETECT_MODEL   = "yolov8m.pt"      # detector  — yolov8l.pt for more accuracy
CLASSIFY_MODEL = "yolov8s-cls.pt"  # classifier — swap for custom trained model
INFER_SIZE     = 1280              # 1280 catches far/small boats
CONF_THRESH    = 0.25              # low = catch far boats; raise if too many false positives
IOU_THRESH     = 0.40
MIN_BOX_AREA   = 100               # px² — filters sub-10px noise
CLASSIFY_EVERY = 4                 # re-classify every N frames (saves GPU)
BOAT_COCO_IDS  = {8, 9}           # 8=boat 9=ship in COCO

# ── Boat type taxonomy (from your reference image) ─────────────────────
BOAT_TYPES = {
    # label            display name        BGR color
    "speedboat":      ("Speedboat",        (  0, 220,  80)),
    "sailboat":       ("Sailboat",         (  0, 220,  80)),
    "catamaran":      ("Catamaran",        (  0, 200, 100)),
    "pontoon":        ("Pontoon Boat",     (  0, 200, 100)),
    "rowboat":        ("Rowboat",          (255, 200,   0)),
    "canoe":          ("Kayak / Canoe",    (255, 200,   0)),
    "kayak":          ("Kayak / Canoe",    (255, 200,   0)),
    "dinghy":         ("Dinghy",           (255, 180,   0)),
    "lifeboat":       ("Inflatable RIB",   (255, 120,   0)),
    "ferry":          ("Ferry",            (  0, 180, 255)),
    "tugboat":        ("Tugboat",          (  0, 160, 255)),
    "container_ship": ("Container Ship",   (  0, 140, 255)),
    "cargo":          ("Cargo Barge",      (  0, 140, 255)),
    "ocean_liner":    ("Cruise Ship",      (180,   0, 255)),
    "houseboat":      ("Houseboat",        (160,   0, 220)),
    "fireboat":       ("Patrol Boat",      (  0,  80, 255)),
    "warship":        ("Military Vessel",  (  0,  60, 220)),
    "submarine":      ("Military Vessel",  (  0,  60, 220)),
    "aircraft_carrier":("Military Vessel", (  0,  60, 220)),
}
BOAT_DEFAULT   = ("Boat",          (  0, 200, 255))

# ── Detection dataclass (tracker-ready) ───────────────────────────────
class Detection:
    """
    Structured detection result.
    Tracker plug-in: assign .track_id after tracker.update()
    """
    __slots__ = ["x1","y1","x2","y2","det_conf","boat_type","cls_conf","track_id","age"]
    def __init__(self, x1, y1, x2, y2, det_conf,
                 boat_type=None, cls_conf=0.0):
        self.x1        = x1
        self.y1        = y1
        self.x2        = x2
        self.y2        = y2
        self.det_conf  = det_conf
        self.boat_type = boat_type or BOAT_DEFAULT[0]
        self.cls_conf  = cls_conf
        self.track_id  = -1       # set by tracker
        self.age       = 0        # frames tracked
    @property
    def bbox(self):
        return (self.x1, self.y1, self.x2, self.y2)
    @property
    def center(self):
        return ((self.x1+self.x2)//2, (self.y1+self.y2)//2)
    @property
    def color(self):
        for key, (label, col) in BOAT_TYPES.items():
            if label == self.boat_type:
                return col
        return BOAT_DEFAULT[1]


# ══════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════

def load_models():
    import torch
    from ultralytics import YOLO

    if torch.cuda.is_available():
        device   = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram     = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU]  {gpu_name} | VRAM: {vram:.1f}GB | CUDA {torch.version.cuda}")
    else:
        device = "cpu"
        print("[WARN] No GPU — running on CPU (slow)")

    def load_or_export(model_name, imgsz, tag):
        engine = OUTPUT_DIR / model_name.replace(".pt", f"_{tag}.engine")
        if engine.exists():
            print(f"[TRT]  Loading {engine.name}")
            return YOLO(str(engine))
        print(f"[INFO] Loading {model_name}...")
        m = YOLO(model_name)
        m.to(device)
        if device == "cuda":
            try:
                print(f"[TRT]  Exporting {model_name} → TensorRT (one-time ~2min)...")
                m.export(format="engine", half=True, imgsz=imgsz,
                         device=0, simplify=True, workspace=4)
                src = Path(model_name.replace(".pt", ".engine"))
                if src.exists():
                    src.rename(engine)
                    m = YOLO(str(engine))
                    print(f"[TRT]  Saved: {engine.name}")
            except Exception as e:
                print(f"[WARN] TensorRT export failed: {e}")
                print("[INFO] Using PyTorch (still GPU-accelerated)")
        return m

    detector   = load_or_export(DETECT_MODEL,   INFER_SIZE, "det")
    classifier = load_or_export(CLASSIFY_MODEL, 224,        "cls")

    print(f"[OK]   Models ready | Device: {device.upper()}\n")
    return detector, classifier, device


# ══════════════════════════════════════════════════════════════════════
#  THREADED CAMERA — always holds latest frame, zero stall
# ══════════════════════════════════════════════════════════════════════

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
            print("[ERROR] Camera failed to open.")
            return
        self.opened = True
        for _ in range(15):    # flush stale buffer
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


# ══════════════════════════════════════════════════════════════════════
#  DETECTION
# ══════════════════════════════════════════════════════════════════════

def detect_boats(detector, frame, conf, iou, imgsz, augment):
    t0      = time.perf_counter()
    results = detector(
        frame,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
        agnostic_nms=True,   # reduces duplicate boxes across classes
        augment=augment,     # multi-scale TTA — better recall for far boats
    )
    ms = (time.perf_counter() - t0) * 1000

    detections = []
    fh, fw = frame.shape[:2]
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) not in BOAT_COCO_IDS:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0,x1), max(0,y1)
            x2, y2 = min(fw-1,x2), min(fh-1,y2)
            if (x2-x1)*(y2-y1) < MIN_BOX_AREA:
                continue
            detections.append(Detection(x1, y1, x2, y2, float(box.conf[0])))

    return detections, ms


# ══════════════════════════════════════════════════════════════════════
#  CLASSIFICATION — maps ImageNet class → boat type
# ══════════════════════════════════════════════════════════════════════

def classify_detections(classifier, frame, detections):
    fh, fw = frame.shape[:2]
    for d in detections:
        pad  = 12
        cx1  = max(0, d.x1-pad);  cy1 = max(0, d.y1-pad)
        cx2  = min(fw, d.x2+pad); cy2 = min(fh, d.y2+pad)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue
        try:
            r          = classifier(crop, imgsz=224, verbose=False)[0]
            raw_name   = r.names[r.probs.top1].lower().replace(" ","_")
            cls_conf   = float(r.probs.top1conf)

            # Map ImageNet name → boat type
            matched = False
            for key, (label, _) in BOAT_TYPES.items():
                if key in raw_name or raw_name in key:
                    d.boat_type = label
                    d.cls_conf  = cls_conf
                    matched     = True
                    break

            if not matched:
                # Keep generic "Boat" label but store conf
                d.cls_conf = cls_conf

        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════
#  RENDERING — professional overlay
# ══════════════════════════════════════════════════════════════════════

def render_detection(frame, d, show_cls):
    fh, fw = frame.shape[:2]
    color  = d.color
    x1, y1, x2, y2 = d.x1, d.y1, d.x2, d.y2
    bw = x2 - x1

    # ── Box ────────────────────────────────────────────────────────────
    thickness = 2 if d.det_conf < 0.6 else 3
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)

    # ── Corner accents ─────────────────────────────────────────────────
    c = max(10, min(20, bw//5))
    for (cx,cy,dx,dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (cx,cy), (cx+c*dx, cy),     color, 3)
        cv2.line(frame, (cx,cy), (cx,      cy+c*dy), color, 3)

    # ── Confidence bar ─────────────────────────────────────────────────
    by1 = min(y2+2, fh-9)
    by2 = min(y2+8, fh-1)
    cv2.rectangle(frame, (x1,by1), (x2,              by2), (25,25,25), -1)
    cv2.rectangle(frame, (x1,by1), (x1+int(bw*d.det_conf), by2), color, -1)

    # ── Track ID (if tracker is active) ────────────────────────────────
    if d.track_id >= 0:
        tid_label = f"#{d.track_id}"
        cv2.putText(frame, tid_label, (x2+4, y1+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # ── Label pill ─────────────────────────────────────────────────────
    if show_cls:
        label = f"{d.boat_type}  {d.det_conf*100:.0f}%"
    else:
        label = f"Boat  {d.det_conf*100:.0f}%"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    ly1 = max(y1-th-10, 0)
    ly2 = max(y1-1,     th+10)
    cv2.rectangle(frame, (x1,ly1), (x1+tw+10, ly2), color, -1)
    cv2.putText(frame, label, (x1+5, ly2-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0,0,0), 2)

    # ── Center dot (useful for tracker) ────────────────────────────────
    cv2.circle(frame, d.center, 3, color, -1)


def render_hud(frame, detections, fps, det_ms, cls_ms,
               device, conf, imgsz, augment, show_cls, frame_n):
    fh, fw  = frame.shape[:2]
    overlay = frame.copy()

    # ── Top bar ────────────────────────────────────────────────────────
    cv2.rectangle(overlay, (0,0), (fw,62), (6,6,6), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    boat_n = len(detections)
    cv2.putText(frame, f"Boats: {boat_n}", (14,42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0,220,80) if boat_n else (80,80,80), 2)

    # FPS
    fc = (0,220,80) if fps>=20 else (0,180,255) if fps>=10 else (0,60,255)
    cv2.putText(frame, f"FPS  {fps:5.1f}", (fw-195,22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, fc, 2)
    cv2.putText(frame, f"Det  {det_ms:5.1f}ms", (fw-195,42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (160,160,160), 1)
    cv2.putText(frame, f"Cls  {cls_ms:5.1f}ms", (fw-195,58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (140,140,140), 1)

    # Centre badge: device + mode flags
    flags  = ""
    flags += " AUG" if augment  else ""
    flags += " CLS" if show_cls else ""
    badge  = f"{'GPU' if device=='cuda' else 'CPU'}{flags}"
    bc     = (0,140,50) if device=="cuda" else (130,50,0)
    bw     = len(badge)*11+16
    bx     = fw//2
    cv2.rectangle(frame, (bx-bw//2,8), (bx+bw//2,54), bc, -1)
    cv2.putText(frame, badge, (bx-bw//2+7,38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # ── Bottom bar ─────────────────────────────────────────────────────
    cv2.rectangle(overlay, (0,fh-34),(fw,fh),(6,6,6),-1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    status = f"Conf:{conf:.2f}  Size:{imgsz}  F:{frame_n}  " \
             f"Q=Quit S=Save C=Cls A=Aug +/-=Conf ]/[=Size"
    cv2.putText(frame, status, (10,fh-9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (140,140,140), 1)

    # ── Confidence color legend ────────────────────────────────────────
    for i,(lbl,col) in enumerate([
        (">70% High", (  0,220, 80)),
        ("50% Med",   (  0,180,255)),
        ("<50% Low",  (  0,120,255)),
    ]):
        y = fh-90+i*24
        cv2.rectangle(frame,(fw-115,y-10),(fw-101,y+4),col,-1)
        cv2.putText(frame,lbl,(fw-96,y+2),
                    cv2.FONT_HERSHEY_SIMPLEX,0.40,col,1)


# ══════════════════════════════════════════════════════════════════════
#  GSTREAMER PIPELINE
# ══════════════════════════════════════════════════════════════════════

def gstreamer_pipeline(cw=1920,ch=1080,dw=1280,dh=720,fps=60,flip=0):
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM),width={cw},height={ch},"
        f"format=NV12,framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw,width={dw},height={dh},format=BGRx ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink drop=1 max-buffers=1"
    )


# ══════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════

def run(use_gstreamer=True):
    detector, classifier, device = load_models()

    # ── Camera ────────────────────────────────────────────────────────
    src     = gstreamer_pipeline() if use_gstreamer else 0
    backend = cv2.CAP_GSTREAMER if use_gstreamer else cv2.CAP_V4L2
    cam     = CameraThread(src, backend)
    cam.start()

    print("[INFO] Waiting for camera...")
    t_end = time.time() + 12
    while not cam.opened and time.time() < t_end:
        time.sleep(0.1)
    if not cam.opened:
        print("[ERROR] Camera failed to open. Reset with:")
        print("        sudo systemctl restart nvargus-daemon")
        cam.stop()
        return

    print("[INFO] Warming up (2s)...")
    time.sleep(2)

    # ── State ─────────────────────────────────────────────────────────
    conf         = CONF_THRESH
    imgsz        = INFER_SIZE
    augment      = False
    show_cls     = True
    shot_n       = 0
    fps          = 0.0
    det_ms       = 0.0
    cls_ms       = 0.0
    fps_t        = time.time()
    fps_n        = 0
    frame_n      = 0
    last_dets    = []     # cached detections (for classify_every)

    print(f"[OK]   Running | {device.upper()} | {DETECT_MODEL} + {CLASSIFY_MODEL}")
    print("       Q=quit S=save C=classifier A=augment +/-=conf ]/[=size\n")

    while True:
        frame = cam.read()
        if frame is None:
            time.sleep(0.005)
            continue

        frame_n += 1
        fps_n   += 1

        # ── Detection ────────────────────────────────────────────────
        last_dets, det_ms = detect_boats(
            detector, frame, conf, IOU_THRESH, imgsz, augment)

        # ── Classification (every N frames) ──────────────────────────
        t_cls = time.perf_counter()
        if show_cls and frame_n % CLASSIFY_EVERY == 0:
            classify_detections(classifier, frame, last_dets)
        cls_ms = (time.perf_counter() - t_cls) * 1000

        # ── Render ───────────────────────────────────────────────────
        for d in last_dets:
            render_detection(frame, d, show_cls)

        # ── FPS ───────────────────────────────────────────────────────
        now = time.time()
        if now - fps_t >= 0.5:
            fps   = fps_n / (now - fps_t)
            fps_t = now
            fps_n = 0

        render_hud(frame, last_dets, fps, det_ms, cls_ms,
                   device, conf, imgsz, augment, show_cls, frame_n)

        cv2.imshow("Professional Boat Detection | Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            shot_n += 1
            p = OUTPUT_DIR / f"boat_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{shot_n}.jpg"
            cv2.imwrite(str(p), frame)
            print(f"[SAVED] {p}")
        elif key == ord('c'):
            show_cls = not show_cls
            print(f"[CLS] {'ON' if show_cls else 'OFF'}")
        elif key == ord('a'):
            augment = not augment
            print(f"[AUG] {'ON — better recall, slower' if augment else 'OFF — faster'}")
        elif key in (ord('+'), ord('=')):
            conf = round(min(0.90, conf+0.05), 2)
            print(f"[CONF] {conf:.2f} — raising reduces false positives")
        elif key == ord('-'):
            conf = round(max(0.05, conf-0.05), 2)
            print(f"[CONF] {conf:.2f} — lowering catches more distant boats")
        elif key == ord(']'):
            imgsz = min(1280, imgsz+64)
            print(f"[SIZE] {imgsz}px — larger catches smaller/far boats (slower)")
        elif key == ord('['):
            imgsz = max(320, imgsz-64)
            print(f"[SIZE] {imgsz}px — smaller = faster")

    cam.stop()
    cv2.destroyAllWindows()
    print(f"\n[DONE] {frame_n} frames | FPS: {fps:.1f} | {device.upper()}")


if __name__ == "__main__":
    # ── Reset camera if needed ─────────────────────────────────────────
    # If you see "Failed to create CaptureSession", run:
    #   sudo systemctl restart nvargus-daemon
    # then re-run this script.

    USE_GSTREAMER = True   # False = USB webcam
    run(use_gstreamer=USE_GSTREAMER)
