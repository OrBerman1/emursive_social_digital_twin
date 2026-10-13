@echo off
rem Viewer design 2: Lens - focus circle + 360 radar strip
rem Starts its own local server on port 8002 and opens the page. Close this window to stop it.
cd /d "%~dp0"
title Shuk HaTikvah viewer 2 (lens) - close this window to stop
echo Viewer 2: Lens - focus circle + 360 radar strip
echo http://localhost:8002/viewer/v2_lens.html
start "" "http://localhost:8002/viewer/v2_lens.html"
".venv\Scripts\python.exe" pipeline\serve.py --port 8002
pause
