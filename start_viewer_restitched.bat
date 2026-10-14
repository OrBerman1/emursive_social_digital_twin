@echo off
rem Viewer ui_alex on the restitched project (shuk_hatikvah_v2: seam fixed). Port 8006; close this window to stop.
cd /d "%~dp0"
title Shuk HaTikvah viewer - restitched - close this window to stop
echo Viewer ui_alex - restitched project
echo http://localhost:8006/viewer/ui_alex.html?project=shuk_hatikvah_v2
start "" "http://localhost:8006/viewer/ui_alex.html?project=shuk_hatikvah_v2"
rem if a viewer server is already running on 8006, the page above uses it and this one just exits
".venv\Scripts\python.exe" pipeline\serve.py --port 8006
pause
