@echo off
rem ===========================================================================
rem  Cliff - process the Shuk HaTikvah data (Windows). Double-click it after setup.bat. See README.md.
rem  1. checks the raw data in data\raw\LABAI\ and gives both projects their files
rem  2. runs the whole analysis (every finished step is kept: stop any time, run again to continue)
rem ===========================================================================
setlocal EnableExtensions
cd /d "%~dp0"
title Cliff - processing the data (keep this window open)

if not exist ".venv\Scripts\python.exe" (
  echo The program is not installed yet. Double-click setup.bat first, then run this again.
  goto :end
)

echo.
echo === 1/2  Preparing the data ===
".venv\Scripts\python.exe" pipeline\prepare_data.py || goto :end

where nvidia-smi >nul 2>nul || (
  echo.
  echo [NOTE] No NVIDIA graphics card found: the analysis runs on the processor and takes days instead of hours.
  echo        That is fine - you can stop at any time and run process_data.bat again; it continues where it stopped.
)
echo.
echo Which project should be processed?
echo   1 = Shuk HaTikvah only          (recommended - the one the viewer opens; about 2-3 hours the first time)
echo   2 = Shuk HaTikvah and Legacy    (about 5-6 hours the first time)
choice /c 12 /n /m "Press 1 or 2: "
set "BOTH=0"
if errorlevel 2 set "BOTH=1"

echo.
echo === 2/2  Analysis ===
echo Keep the computer plugged in and do not let it sleep. You can close this window to stop;
echo run process_data.bat again later and it continues where it stopped (finished steps are kept,
echo and the long detection steps save their progress every 5 minutes).
echo.
echo --- Shuk HaTikvah ---
".venv\Scripts\python.exe" pipeline\run_all.py --project shuk_hatikvah_v2 || goto :failed
if "%BOTH%"=="1" (
  echo.
  echo --- Shuk HaTikvah Legacy ---
  ".venv\Scripts\python.exe" pipeline\run_all.py --project shuk_hatikvah || goto :failed
)

echo.
echo ==========================================================
echo  Done. Open the viewer: double-click start_viewer.bat
echo ==========================================================
goto :end

:failed
echo.
echo [ERROR] The analysis stopped at the step shown above. The full log is in job.log, in the project's
echo         data\processed folder. Fix the message, or send the log to the team, then run this again.

:end
echo.
pause
endlocal
