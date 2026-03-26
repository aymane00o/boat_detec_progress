the boat_detec_A1_with_classification_jetson is the latest version of the code it s good it precision and clasify by need more precision in the classification part.

the boat_detec_beta_jetson is a version of the code it s good but it need more precision and clasification.

the boat_detec_beta_jetson is a good version of the code but not enough.

the boat_detec_1_jetson is the first version not good enought.

#!/bin/bash
# ================================================================
#  GIT SETUP SCRIPT — Boat Detection Project
#  Platform : NVIDIA Jetson Orin Nano Super
#  Run once : bash git_setup.sh
# ================================================================

set -e  # stop on any error

# ── Colors ────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "=================================================="
echo "   Boat Detection — Git Repository Setup"
echo "=================================================="
echo -e "${NC}"

# ── 1. Git config ─────────────────────────────────────────────────
echo -e "${YELLOW}[1/6] Configuring Git identity...${NC}"
read -p "  Enter your Git username : " GIT_USER
read -p "  Enter your Git email    : " GIT_EMAIL

git config --global user.name  "$GIT_USER"
git config --global user.email "$GIT_EMAIL"
git config --global core.editor nano
git config --global init.defaultBranch main
echo -e "${GREEN}  ✓ Git identity set${NC}"

# ── 2. Init repo ──────────────────────────────────────────────────
echo -e "${YELLOW}[2/6] Initializing repository...${NC}"

PROJECT_DIR="$HOME/boat_detection"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Create project folder structure
mkdir -p models outputs logs configs tests

echo -e "${GREEN}  ✓ Project created at: $PROJECT_DIR${NC}"

# ── 3. Create .gitignore ──────────────────────────────────────────
echo -e "${YELLOW}[3/6] Creating .gitignore...${NC}"

cat > .gitignore << 'EOF'
# ── Models & Weights ──────────────────────────────────────────────
*.pt
*.pth
*.engine
*.onnx
*.trt
models/

# ── Output files ──────────────────────────────────────────────────
outputs/
*.jpg
*.jpeg
*.png
*.mp4
*.avi
*.raw
*.h264

# ── Logs ──────────────────────────────────────────────────────────
logs/
*.log
runs/

# ── Python ────────────────────────────────────────────────────────
__pycache__/
*.py[cod]
*.so
*.egg
*.egg-info/
dist/
build/
.eggs/
.venv/
venv/
env/

# ── Jupyter ───────────────────────────────────────────────────────
.ipynb_checkpoints/
*.ipynb

# ── System ────────────────────────────────────────────────────────
.DS_Store
Thumbs.db
*.swp
*.swo
*~

# ── NVIDIA / CUDA ─────────────────────────────────────────────────
*.cubin
*.fatbin
*.ptx

# ── IDE ───────────────────────────────────────────────────────────
.vscode/
.idea/
*.code-workspace

# ── Secrets ───────────────────────────────────────────────────────
.env
*.key
secrets.yaml
EOF

echo -e "${GREEN}  ✓ .gitignore created${NC}"

# ── 4. Copy main script ───────────────────────────────────────────
echo -e "${YELLOW}[4/6] Setting up project files...${NC}"

# Create main detection script placeholder if not present
if [ ! -f "boat_detection.py" ]; then
cat > boat_detection.py << 'PYEOF'
# Place your boat_detection.py content here
# or copy it with: cp /path/to/boat_detection.py .
PYEOF
fi

# Create requirements.txt
cat > requirements.txt << 'EOF'
# ── Core ──────────────────────────────────────────────────────────
# PyTorch: install separately via NVIDIA wheel (see README)
ultralytics==8.3.0
numpy==1.24.4
opencv-python==4.8.1.78
Pillow==10.3.0

# ── Utilities ─────────────────────────────────────────────────────
PyYAML==6.0.1
requests==2.31.0
tqdm==4.66.1
psutil==5.9.8
py-cpuinfo==9.0.0
scipy==1.11.4
matplotlib==3.7.5
seaborn==0.13.2
pandas==2.0.3
EOF

# Create README
cat > README.md << 'EOF'
# 🚤 Boat Detection System
**Platform:** NVIDIA Jetson Orin Nano Super  
**Camera:** Arducam IMX219 (CSI)  
**Model:** YOLOv8m + YOLOv8s-cls  

## Setup

### 1. Install PyTorch (Jetson)
```bash
wget https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.3.0a0+ebedce2.nv24.2-cp310-cp310-linux_aarch64.whl
pip3 install torch-2.3.0a0+ebedce2.nv24.2-cp310-cp310-linux_aarch64.whl
```

### 2. Install dependencies
```bash
pip3 install -r requirements.txt
```

### 3. Reset camera daemon
```bash
sudo systemctl restart nvargus-daemon
```

### 4. Run
```bash
python3 boat_detection.py
```

## Controls
| Key | Action |
|-----|--------|
| Q | Quit |
| S | Screenshot |
| C | Toggle classifier |
| A | Toggle augmentation |
| + / - | Confidence threshold |
| ] / [ | Inference size |

## Output
Screenshots saved to `~/Desktop/vision_output/`
EOF

# Create config file
cat > configs/config.yaml << 'EOF'
camera:
  width: 1920
  height: 1080
  fps: 60
  flip: 0
  display_width: 1280
  display_height: 720

detection:
  model: yolov8m.pt
  conf_thresh: 0.25
  iou_thresh: 0.40
  infer_size: 1280
  min_box_area: 100

classifier:
  model: yolov8s-cls.pt
  classify_every: 4
  enabled: true
EOF

echo -e "${GREEN}  ✓ Project files created${NC}"

# ── 5. Init git and first commit ──────────────────────────────────
echo -e "${YELLOW}[5/6] Creating first commit...${NC}"

git init
git add .
git commit -m "🚀 Initial commit — Boat detection system (Jetson Orin Nano)"

echo -e "${GREEN}  ✓ Repository initialized with first commit${NC}"

# ── 6. Connect to remote (optional) ──────────────────────────────
echo -e "${YELLOW}[6/6] Remote repository (GitHub/GitLab)...${NC}"
read -p "  Do you have a remote repo URL? (leave blank to skip): " REMOTE_URL

if [ -n "$REMOTE_URL" ]; then
    git remote add origin "$REMOTE_URL"
    git branch -M main
    git push -u origin main
    echo -e "${GREEN}  ✓ Pushed to remote: $REMOTE_URL${NC}"
else
    echo -e "  Skipped — add remote later with:"
    echo -e "  ${BLUE}git remote add origin <your-repo-url>${NC}"
    echo -e "  ${BLUE}git push -u origin main${NC}"
fi

# ── Done ──────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}=================================================="
echo "   ✓ Setup Complete!"
echo "=================================================="
echo -e "${NC}"
echo -e "  Project location : ${BLUE}$PROJECT_DIR${NC}"
echo ""
echo -e "  Useful commands:"
echo -e "  ${BLUE}cd $PROJECT_DIR${NC}"
echo -e "  ${BLUE}git status${NC}              — check changes"
echo -e "  ${BLUE}git add .${NC}               — stage all changes"
echo -e "  ${BLUE}git commit -m 'msg'${NC}     — commit"
echo -e "  ${BLUE}git push${NC}                — push to remote"
echo -e "  ${BLUE}git log --oneline${NC}       — view history"
echo ""
