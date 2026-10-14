#!/bin/bash
# ===========================================================================
#  Shuk HaTikvah 360 - environment setup (macOS, Apple Silicon). Mac version of setup.bat; see README.md.
#  Creates:  data/      folder structure (the raw data is downloaded separately - README section 2)
#            .venv      Python 3.13 main environment (PyTorch with MPS, YOLO, CLIP, OCR, telemetry-parser)
#            models/    YOLO11x + YOLO-World weights
#  Unlike Windows there is no .venv-imu: telemetry-parser is built from source (Rust) into .venv.
#  Safe to run again: existing environments are reused, existing files are kept.
#  Double-click in Finder, or run:  ./setup_mac.command
# ===========================================================================
cd "$(dirname "$0")" || exit 1
FAIL=0
error() { echo; echo "[ERROR] Setup stopped at the step above. Fix the message shown and run setup_mac.command again."; read -r -p "Press Enter to close..."; exit 1; }

echo
echo "=== 1/8  Checking required programs ==="
PY=""
for c in python3.13 "$HOME/miniconda3/bin/python3.13" /opt/homebrew/bin/python3.13; do
  if command -v "$c" >/dev/null 2>&1; then PY="$(command -v "$c")"; break; fi
done
[ -n "$PY" ] || { echo "[MISSING] Python 3.13 - brew install python@3.13"; FAIL=1; }
command -v curl >/dev/null || { echo "[MISSING] curl"; FAIL=1; }
command -v ffmpeg >/dev/null || { echo "[MISSING] ffmpeg - brew install ffmpeg"; FAIL=1; }
if ! command -v tesseract >/dev/null; then
  echo "[MISSING] Tesseract OCR - brew install tesseract tesseract-lang"; FAIL=1
else
  tesseract --list-langs 2>/dev/null | grep -qx heb || echo "[WARN] Tesseract has no Hebrew data - brew install tesseract-lang"
fi
command -v cargo >/dev/null || echo "[WARN] Rust (cargo) not found - gyro extraction for stabilisation needs it: brew install rust"
command -v git >/dev/null || echo "[WARN] git not found - needed to fetch telemetry-parser"
if [ "$FAIL" = 1 ]; then
  echo; echo "Install the missing programs above, then run setup_mac.command again."
  read -r -p "Press Enter to close..."; exit 1
fi

echo
echo "=== 2/8  Folder structure ==="
for d in data/raw/LABAI/Photos data/raw/LABAI/Sketches data/interim data/processed models; do
  if [ -d "$d" ]; then echo "  exists  $d"; else mkdir -p "$d" && echo "  created $d"; fi
done
# If the LABAI files sit in the folder above the repo, link them in instead of copying 5 GB.
for f in "1_Hatikvah St.insv" "2_Hamevasser St.insv" "Shops with Icons.pdf" "shuk_360_frame_coding_v01.docx"; do
  [ ! -e "data/raw/LABAI/$f" ] && [ -f "../$f" ] && ln -s "$(cd .. && pwd)/$f" "data/raw/LABAI/$f" && echo "  linked  data/raw/LABAI/$f"
done
for sub in Photos Sketches; do
  [ -d "../$sub" ] || continue
  for f in "../$sub"/*; do
    [ -f "$f" ] && [ ! -e "data/raw/LABAI/$sub/$(basename "$f")" ] && ln -s "$(cd "../$sub" && pwd)/$(basename "$f")" "data/raw/LABAI/$sub/"
  done
done

echo
echo "=== 3/8  Main environment .venv (Python 3.13) ==="
[ -x .venv/bin/python ] || "$PY" -m venv .venv || error
.venv/bin/python -m pip install --upgrade pip || error

echo
echo "=== 4/8  PyTorch (Apple MPS build) ==="
.venv/bin/python -m pip install torch==2.14.1 torchvision==0.29.1 || error

echo
echo "=== 5/8  Pipeline packages ==="
.venv/bin/python -m pip install -r requirements.txt || error
[ -f models/clip.zip ] || curl -L -f -o models/clip.zip https://github.com/ultralytics/CLIP/archive/refs/heads/main.zip || error
.venv/bin/python -m pip install models/clip.zip || error

echo
echo "=== 6/8  Model weights ==="
[ -f models/yolo11x.pt ] || curl -L -f -o models/yolo11x.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11x.pt || error
[ -f models/yolov8x-worldv2.pt ] || curl -L -f -o models/yolov8x-worldv2.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8x-worldv2.pt || error
# CLIP ViT-B/32 + ViT-L/14 and the docTR text detector download themselves on first use.

echo
echo "=== 7/8  Gyro reader telemetry-parser (built from source, needs Rust) ==="
if .venv/bin/python -c "import telemetry_parser" 2>/dev/null; then
  echo "  already installed"
elif command -v cargo >/dev/null && command -v git >/dev/null; then
  # PyPI has no macOS wheel and its old sdist does not build; the GitHub python-module does.
  [ -d models/telemetry-parser ] || git clone --depth 1 https://github.com/AdrianEddy/telemetry-parser.git models/telemetry-parser || error
  .venv/bin/python -m pip install models/telemetry-parser/bin/python-module || error
else
  echo "[WARN] Skipped: install Rust (brew install rust) and rerun - stabilisation needs the gyro data"
fi

echo
echo "=== 8/8  Checking the installation and the data ==="
.venv/bin/python -c "import torch, ultralytics, clip, cv2, py360convert, pytesseract, symspellpy, doctr; print('torch', torch.__version__, '| MPS', torch.backends.mps.is_available()); print('ultralytics', ultralytics.__version__, '| opencv', cv2.__version__)" || error
.venv/bin/python -c "import telemetry_parser; print('telemetry-parser OK')"
DATAOK=1
for f in "1_Hatikvah St.insv" "2_Hamevasser St.insv" "Shops with Icons.pdf" "shuk_360_frame_coding_v01.docx"; do
  if [ -e "data/raw/LABAI/$f" ]; then echo "  [data OK]      $f"; else echo "  [data MISSING] data/raw/LABAI/$f"; DATAOK=0; fi
done

echo
echo "Setup complete."
if [ "$DATAOK" = 0 ]; then
  echo "  The raw data is not complete yet: download it as described in README.md, section 2,"
  echo "  into data/raw/LABAI/  - the folders were created above - then run the pipeline."
fi
echo "  Run the pipeline:  .venv/bin/python pipeline/run_all.py"
echo "  Open the viewer:   start_viewer_mac.command"
echo
read -r -p "Press Enter to close..."
