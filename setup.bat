@echo off
rem ===========================================================================
rem  Shuk HaTikvah 360 - environment setup (Windows)
rem  Creates:  .venv      Python 3.13 main environment (PyTorch CUDA 12.4, YOLO, CLIP, OCR)
rem            .venv-imu  Python 3.9 helper for gyro extraction (telemetry-parser has cp39 wheels only)
rem            models\    YOLO11x + YOLO-World weights
rem  Safe to run again: existing environments are reused, existing files are kept.
rem  Downloads use curl.exe (Python/git downloads can fail on certificate checks here).
rem ===========================================================================
setlocal EnableExtensions
cd /d "%~dp0"
set "FAIL=0"

echo.
echo === 1/7  Checking required programs ===
where py >nul 2>nul || (echo [MISSING] Python launcher "py" - install Python 3.13 and 3.9 from python.org & set FAIL=1)
py -3.13 --version >nul 2>nul || (echo [MISSING] Python 3.13 - install from python.org & set FAIL=1)
where curl.exe >nul 2>nul || (echo [MISSING] curl.exe - included with Windows 10/11 & set FAIL=1)
where ffmpeg >nul 2>nul || (echo [MISSING] ffmpeg on PATH - install a full build from https://www.gyan.dev/ffmpeg/builds/ and add its bin folder to PATH & set FAIL=1)
if not exist "C:\Program Files\Tesseract-OCR\tesseract.exe" (
  echo [MISSING] Tesseract OCR - install from https://github.com/UB-Mannheim/tesseract/wiki and tick Hebrew + Arabic language data
  set FAIL=1
) else (
  "C:\Program Files\Tesseract-OCR\tesseract.exe" --list-langs 2>nul | findstr /x "heb" >nul || echo [WARN] Tesseract has no Hebrew data - rerun its installer and tick Hebrew
)
where nvidia-smi >nul 2>nul || echo [WARN] nvidia-smi not found - GPU stages will be very slow on CPU
py -3.9 --version >nul 2>nul || echo [WARN] Python 3.9 not found - gyro extraction for stabilisation will be skipped
if "%FAIL%"=="1" (
  echo.
  echo Install the missing programs above, then run setup.bat again.
  goto :end
)

echo.
echo === 2/7  Main environment .venv (Python 3.13) ===
if not exist ".venv\Scripts\python.exe" py -3.13 -m venv .venv || goto :error
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :error

echo.
echo === 3/7  PyTorch with CUDA 12.4 (large download, ~2.5 GB) ===
".venv\Scripts\python.exe" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 || goto :error

echo.
echo === 4/7  Pipeline packages ===
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error
if not exist models mkdir models
if not exist "models\clip.zip" curl.exe -L -f -o "models\clip.zip" https://github.com/ultralytics/CLIP/archive/refs/heads/main.zip || goto :error
".venv\Scripts\python.exe" -m pip install "models\clip.zip" || goto :error

echo.
echo === 5/7  Model weights ===
if not exist "models\yolo11x.pt" curl.exe -L -f -o "models\yolo11x.pt" https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11x.pt || goto :error
if not exist "models\yolov8x-worldv2.pt" curl.exe -L -f -o "models\yolov8x-worldv2.pt" https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8x-worldv2.pt || goto :error
rem CLIP ViT-B/32 + ViT-L/14 and the docTR text detector download themselves on first use.

echo.
echo === 6/7  Gyro helper .venv-imu (Python 3.9) ===
py -3.9 --version >nul 2>nul || goto :skipimu
if not exist ".venv-imu\Scripts\python.exe" py -3.9 -m venv .venv-imu || goto :error
if not exist "models\telemetry_parser-0.3.0-cp39-none-win_amd64.whl" (
  curl.exe -L -f -o "models\telemetry_parser-0.3.0-cp39-none-win_amd64.whl" https://files.pythonhosted.org/packages/7e/89/7d86d1bd0823c35bd82482629ad1330ec9845f114c9e6696fd6c779f200c/telemetry_parser-0.3.0-cp39-none-win_amd64.whl || (
    echo [WARN] Could not download the telemetry-parser wheel. Get telemetry_parser-0.3.0-cp39-none-win_amd64.whl from
    echo        https://pypi.org/project/telemetry-parser/0.3.0/#files and put it in the models folder, then rerun.
    goto :skipimu
  )
)
".venv-imu\Scripts\python.exe" -m pip install --no-index "models\telemetry_parser-0.3.0-cp39-none-win_amd64.whl" || goto :error
:skipimu

echo.
echo === 7/7  Checking the installation ===
".venv\Scripts\python.exe" -c "import torch, ultralytics, clip, cv2, py360convert, pytesseract, symspellpy, doctr; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '(no GPU)'); print('ultralytics', ultralytics.__version__, '| opencv', cv2.__version__)" || goto :error
if exist ".venv-imu\Scripts\python.exe" ".venv-imu\Scripts\python.exe" -c "import telemetry_parser; print('telemetry-parser OK')"

echo.
echo Setup complete.
echo   Put the raw data in data\raw\LABAI\ (1_Hatikvah St.insv, 2_Hamevasser St.insv, ...)
echo   Run the pipeline:  .venv\Scripts\python pipeline\run_all.py
echo   Open the viewer:   start_viewer.bat
goto :end

:error
echo.
echo [ERROR] Setup stopped at the step above. Fix the message shown and run setup.bat again.

:end
echo.
pause
endlocal
