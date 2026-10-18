@echo off
rem ===========================================================================
rem  Shuk HaTikvah 360 - environment setup (Windows). See README.md.
rem  Creates:  data\      folder structure (the raw data is downloaded separately - README section 2)
rem            .venv      Python 3.13 main environment (PyTorch CUDA 12.4, YOLO, CLIP, OCR)
rem            .venv-imu  Python 3.9 helper for gyro extraction (telemetry-parser has cp39 wheels only)
rem            models\    YOLO11x + YOLO-World weights
rem  Safe to run again: existing environments are reused, existing files are kept.
rem  Downloads use curl.exe (Python/git downloads can fail on certificate checks on some networks).
rem ===========================================================================
setlocal EnableExtensions
cd /d "%~dp0"
set "FAIL=0"

echo.
echo === 1/8  Checking required programs (missing ones are installed with winget) ===
where curl.exe >nul 2>nul || (echo [MISSING] curl.exe - included with Windows 10/11 & set FAIL=1)
where winget >nul 2>nul
if errorlevel 1 (
  set "HAVE_WINGET=0"
  echo [WARN] winget not found - missing programs cannot be installed automatically. Update "App Installer" from the Microsoft Store.
) else (
  set "HAVE_WINGET=1"
)

call :find_tess

rem --- Python 3.13 (main) and 3.9 (gyro helper) ---
py -3.13 --version >nul 2>nul || call :winget_install Python.Python.3.13 "Python 3.13"
py -3.9 --version >nul 2>nul  || call :winget_install Python.Python.3.9 "Python 3.9"
rem --- ffmpeg (full build, with NVENC) ---
where ffmpeg >nul 2>nul || call :winget_install Gyan.FFmpeg "ffmpeg"
rem --- Tesseract ---
if not defined TESS call :winget_install UB-Mannheim.TesseractOCR "Tesseract OCR"

rem Newly installed programs are not on this window's PATH yet: reload PATH from the registry.
call :refresh_path
call :find_tess

py -3.13 --version >nul 2>nul || (echo [MISSING] Python 3.13 - install from python.org, tick "Add to PATH" & set FAIL=1)
where ffmpeg >nul 2>nul || (echo [MISSING] ffmpeg on PATH - install a full build from https://www.gyan.dev/ffmpeg/builds/ and add its bin folder to PATH & set FAIL=1)
if not defined TESS (
  echo [MISSING] Tesseract OCR - install from https://github.com/UB-Mannheim/tesseract/wiki
  set FAIL=1
) else (
  rem Hebrew and Arabic language data are not in the default installer: fetch them if absent.
  for %%L in (heb ara) do if not exist "%TESSDATA%\%%L.traineddata" (
    echo   downloading Tesseract %%L language data
    curl.exe -L -f -s -o "%TESSDATA%\%%L.traineddata" https://github.com/tesseract-ocr/tessdata_best/raw/main/%%L.traineddata || echo [WARN] Could not write %%L.traineddata - rerun setup.bat as Administrator
  )
  "%TESS%" --list-langs 2>nul | findstr /x "heb" >nul || echo [WARN] Tesseract has no Hebrew data - rerun setup.bat as Administrator, or copy heb.traineddata into its tessdata folder
)
where nvidia-smi >nul 2>nul || echo [WARN] nvidia-smi not found - install the NVIDIA driver from nvidia.com; GPU stages will be very slow on CPU
py -3.9 --version >nul 2>nul || echo [WARN] Python 3.9 not found - gyro extraction for stabilisation will be skipped
if "%FAIL%"=="1" (
  echo.
  echo Install the missing programs above, then run setup.bat again.
  goto :end
)

echo.
echo === 2/8  Folder structure ===
for %%D in ("data\raw\LABAI\Photos" "data\raw\LABAI\Sketches" "data\interim" "data\processed" "models") do (
  if not exist "%%~D" (mkdir "%%~D" && echo   created %%~D) else (echo   exists  %%~D)
)

echo.
echo === 3/8  Main environment .venv (Python 3.13) ===
if not exist ".venv\Scripts\python.exe" py -3.13 -m venv .venv || goto :error
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :error

echo.
echo === 4/8  PyTorch with CUDA 12.4 (large download, ~2.5 GB) ===
".venv\Scripts\python.exe" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 || goto :error

echo.
echo === 5/8  Pipeline packages ===
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error
if not exist "models\clip.zip" curl.exe -L -f -o "models\clip.zip" https://github.com/ultralytics/CLIP/archive/refs/heads/main.zip || goto :error
".venv\Scripts\python.exe" -m pip install "models\clip.zip" || goto :error

echo.
echo === 6/8  Model weights ===
if not exist "models\yolo11x.pt" curl.exe -L -f -o "models\yolo11x.pt" https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11x.pt || goto :error
if not exist "models\yolov8x-worldv2.pt" curl.exe -L -f -o "models\yolov8x-worldv2.pt" https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8x-worldv2.pt || goto :error
if not exist "models\yolov8x-oiv7.pt" curl.exe -L -f -o "models\yolov8x-oiv7.pt" https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8x-oiv7.pt || goto :error
rem OSNet-AIN x1.0 trained on MSMT17 (person ReID for reid_embed.py; torchreid model zoo, Google Drive)
if not exist "models\osnet_ain_x1_0_msmt17.pth" curl.exe -L -f -o "models\osnet_ain_x1_0_msmt17.pth" "https://drive.usercontent.google.com/download?id=1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal&export=download&confirm=t" || goto :error
if not exist "models\segformer-b2-ade" mkdir "models\segformer-b2-ade"
for %%F in (config.json preprocessor_config.json pytorch_model.bin) do if not exist "models\segformer-b2-ade\%%F" curl.exe -L -f -o "models\segformer-b2-ade\%%F" https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512/resolve/main/%%F || goto :error
rem CLIP ViT-B/32 + ViT-L/14 and the docTR text detector download themselves on first use.

echo.
echo === 7/8  Gyro helper .venv-imu (Python 3.9) ===
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
echo === 8/8  Checking the installation and the data ===
".venv\Scripts\python.exe" -c "import torch, ultralytics, clip, cv2, py360convert, pytesseract, symspellpy, doctr; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '(no GPU)'); print('ultralytics', ultralytics.__version__, '| opencv', cv2.__version__)" || goto :error
if exist ".venv-imu\Scripts\python.exe" ".venv-imu\Scripts\python.exe" -c "import telemetry_parser; print('telemetry-parser OK')"
set "DATAOK=1"
for %%F in ("1_Hatikvah St.insv" "2_Hamevasser St.insv" "Shops_In_hatikva_30_7_26.xlsx") do (
  if exist "data\raw\LABAI\%%~F" (echo   [data OK]      %%~F) else (echo   [data MISSING] data\raw\LABAI\%%~F & set "DATAOK=0")
)

echo.
echo Setup complete.
if "%DATAOK%"=="0" (
  echo   The raw data is not complete yet: download it as described in README.md, section 2,
  echo   into data\raw\LABAI\  - the folders were created above.
)
echo   Next: put the data in data\raw\LABAI\, then double-click process_data.bat
echo   Open the viewer:   start_viewer.bat
goto :end

:error
echo.
echo [ERROR] Setup stopped at the step above. Fix the message shown and run setup.bat again.

:end
echo.
pause
endlocal
exit /b

rem --- helpers (only reached through "call") ---
:winget_install
if "%HAVE_WINGET%"=="0" exit /b 1
echo   installing %~2 with winget ...
winget install --id %~1 -e --silent --accept-package-agreements --accept-source-agreements
exit /b 0

:find_tess
rem Locate tesseract.exe: PATH, then the usual install folders (per-machine, 32-bit, per-user).
set "TESS="
for /f "delims=" %%T in ('where tesseract 2^>nul') do if not defined TESS set "TESS=%%T"
for %%T in ("%ProgramFiles%\Tesseract-OCR\tesseract.exe" "%ProgramFiles(x86)%\Tesseract-OCR\tesseract.exe" "%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe") do if not defined TESS if exist %%T set "TESS=%%~T"
if defined TESS for %%T in ("%TESS%") do set "TESSDATA=%%~dpTtessdata"
exit /b 0

:refresh_path
for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('Path','Machine')+';'+[Environment]::GetEnvironmentVariable('Path','User')"`) do set "PATH=%%P"
exit /b 0
