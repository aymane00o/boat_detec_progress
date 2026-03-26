the boat_detec_A1_with_classification_jetson is the latest version of the code it s good it precision and clasify by need more precision in the classification part.

the boat_detec_beta_jetson is a version of the code it s good but it need more precision and clasification.

the boat_detec_beta_jetson is a good version of the code but not enough.

the boat_detec_1_jetson is the first version not good enought.

# 🚤 Boat Detection System


**Real-time boat detection and classification using YOLOv8 on Jetson Orin Nano Super**  
*Arducam IMX219 CSI Camera · TensorRT Accelerated · Tracker-Ready Architecture*

[Features](#-features) · [Hardware](#-hardware) · [Installation](#-installation) · [Usage](#-usage) · [Controls](#-controls) · [Architecture](#-architecture)

</div>

---

## 📸 Demo

```
┌─────────────────────────────────────────────────────────┐
│  Boats: 3                    GPU  CLS        FPS  28.4  │
│                                               Det  34ms  │
│  ┌─────────────────┐                         Cls   8ms  │
│  │ Speedboat  87%  │                                     │
│  └─────────────────┘                                     │
│                    ┌──────────────┐                      │
│                    │  Ferry  72%  │                      │
│                    └──────────────┘                      │
│                                  ┌───────────────────┐  │
│                                  │  Cargo Barge  65% │  │
│                                  └───────────────────┘  │
│  Conf:0.25  Size:1280  Q=Quit S=Save C=Cls A=Aug        │
└─────────────────────────────────────────────────────────┘
```

---

## ✨ Features

- 🎯 **Real-time detection** — YOLOv8m at 25–30 FPS (GPU accelerated)
- 🚢 **14+ boat types** — Speedboat, Ferry, Cargo, Sailboat, Military Vessel, and more
- ⚡ **TensorRT export** — automatic one-time optimization on first run
- 🧵 **Threaded camera** — zero-stall frame buffer, always fresh frames
- 📊 **Live HUD** — FPS, inference time, confidence bar, detection count
- 🔌 **Tracker-ready** — `Detection` objects have `.track_id` and `.age` slots
- 📸 **Screenshot** — save annotated frames with one key press
- 🎛️ **Runtime controls** — adjust confidence and inference size on the fly

---

## 🔧 Hardware

| Component | Details |
|-----------|---------|
| **Board** | NVIDIA Jetson Orin Nano Super |
| **Camera** | Arducam IMX219 8MP (CSI) |
| **JetPack** | 6.x (L4T R36.4.7) |
| **Kernel** | 5.15.148-tegra OOT |
| **RAM** | 8GB LPDDR5 |
| **Storage** | NVMe SSD recommended |

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/boat-detection.git
cd boat-detection
```

### 2. Install system dependencies

```bash
sudo apt update && sudo apt install -y \
  python3-pip python3-dev python3-opencv \
  gstreamer1.0-tools gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly gstreamer1.0-libav \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  v4l-utils
```

### 3. Install PyTorch for Jetson ⚠️

> Do **not** use `pip install torch` — it won't have CUDA support on Jetson.

```bash
wget https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.3.0a0+ebedce2.nv24.2-cp310-cp310-linux_aarch64.whl
pip3 install torch-2.3.0a0+ebedce2.nv24.2-cp310-cp310-linux_aarch64.whl
pip3 install torchvision==0.18.0
```

### 4. Install Python dependencies

```bash
pip3 install -r requirements.txt
```

### 5. Verify installation

```bash
python3 verify.py
```

Expected output:
```
PyTorch     : 2.3.0
CUDA        : True | 12.2
OpenCV      : 4.8.1
Ultralytics : 8.3.0
GStreamer   : YES
GPU         : Orin (nvgpu)
✅ All checks passed — ready to run!
```

---

## 🚀 Usage

### Reset camera daemon (run before each session)

```bash
sudo systemctl restart nvargus-daemon
```

### Run the detection system

```bash
python3 boat_detection.py
```

### First run note
On first launch, YOLOv8 models are downloaded automatically (~50MB each).  
TensorRT engine export runs once (~2 min) and is cached for future runs.

### Use USB webcam instead of CSI

```python
# In boat_detection.py, change the last line:
run(use_gstreamer=False)
```

---

## 🎛️ Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Save screenshot to `~/Desktop/vision_output/` |
| `C` | Toggle boat type classifier ON/OFF |
| `A` | Toggle augmentation (better recall, slower) |
| `+` / `=` | Increase confidence threshold (fewer detections) |
| `-` | Decrease confidence threshold (more detections) |
| `]` | Increase inference size (catches far/small boats) |
| `[` | Decrease inference size (faster) |

---

## 🏗️ Architecture

```
CSI Camera (IMX219)
      │
      ▼
 CameraThread  ──── daemon thread, always holds latest frame
      │
      ▼
  FrameBuffer  ──── thread-safe lock, zero stall
      │
      ▼
   Detector    ──── YOLOv8m → filters COCO IDs {8=boat, 9=ship}
      │
      ▼
  Classifier   ──── YOLOv8s-cls → maps ImageNet → boat type
      │              (runs every 4 frames to save GPU)
      ▼
  [Tracker]    ──── plug-in slot: tracker.update(detections)
      │              assigns Detection.track_id + Detection.age
      ▼
   Renderer    ──── bounding boxes, labels, HUD overlay
      │
      ▼
  cv2.imshow   ──── display / screenshot
```

### Detection Object (tracker-ready)

```python
class Detection:
    x1, y1, x2, y2  # bounding box
    det_conf         # detector confidence
    boat_type        # classified boat type string
    cls_conf         # classifier confidence
    track_id         # set by tracker (-1 if no tracker)
    age              # frames tracked
    
    # Properties
    .bbox    → (x1, y1, x2, y2)
    .center  → (cx, cy)
    .color   → BGR color for this boat type
```

---

## 🚢 Supported Boat Types

| Type | Label |
|------|-------|
| Speedboat | `Speedboat` |
| Sailboat | `Sailboat` |
| Catamaran | `Catamaran` |
| Pontoon | `Pontoon Boat` |
| Kayak / Canoe | `Kayak / Canoe` |
| Dinghy | `Dinghy` |
| Inflatable RIB | `Inflatable RIB` |
| Ferry | `Ferry` |
| Tugboat | `Tugboat` |
| Container Ship | `Container Ship` |
| Cargo Barge | `Cargo Barge` |
| Cruise Ship | `Cruise Ship` |
| Houseboat | `Houseboat` |
| Military Vessel | `Military Vessel` |

---

## ⚙️ Configuration

Edit `configs/config.yaml` to tune the system:

```yaml
camera:
  width: 1920
  height: 1080
  fps: 60
  flip: 0              # 0=none 2=180° flip

detection:
  model: yolov8m.pt    # yolov8l.pt for more accuracy
  conf_thresh: 0.25    # lower = detect far boats
  iou_thresh: 0.40
  infer_size: 1280     # 1280 = best for small/far boats
  min_box_area: 100    # px² noise filter

classifier:
  model: yolov8s-cls.pt
  classify_every: 4    # re-classify every N frames
  enabled: true
```

---

## 🐛 Troubleshooting

### Camera not found
```bash
sudo systemctl restart nvargus-daemon
ls /dev/video*
```

### `Failed to create CaptureSession`
```bash
sudo systemctl restart nvargus-daemon && sleep 2
python3 boat_detection.py
```

### No GPU detected
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
# If False → reinstall PyTorch from NVIDIA wheel (Step 3)
```

### Low FPS
- Lower inference size: press `[` during runtime
- Disable augmentation: press `A`
- Disable classifier: press `C`
- Switch to `yolov8s.pt` for faster detection

### TensorRT export failed
```bash
# Delete cached engines and retry
rm ~/Desktop/vision_output/*.engine
python3 boat_detection.py
```

---

## 🗂️ Project Structure

```
boat-detection/
├── boat_detection.py     # main detection script
├── verify.py             # installation checker
├── requirements.txt      # Python dependencies
├── git_setup.sh          # repo setup script
├── README.md
├── configs/
│   └── config.yaml       # tunable parameters
├── models/               # cached .pt / .engine files (gitignored)
├── outputs/              # screenshots (gitignored)
└── logs/                 # run logs (gitignored)
```

---

## 📋 Dependencies

| Package | Version |
|---------|---------|
| PyTorch | 2.3.0 (NVIDIA aarch64) |
| Ultralytics | 8.3.0 |
| OpenCV | 4.8.1 |
| NumPy | 1.24.4 |
| Python | 3.10.x |
| CUDA | 12.2 |
| JetPack | 6.x |

---

## 🔮 Roadmap

- [ ] Add DeepSORT / ByteTrack tracker
- [ ] Custom trained model on maritime dataset
- [ ] RTSP stream output
- [ ] REST API for detection results
- [ ] Multi-camera support (CAM0 + CAM1)
- [ ] Alarm trigger on vessel class detection

---


---

<div align="center">
Built for real-time maritime surveillance on NVIDIA Jetson Orin Nano Super
</div>
