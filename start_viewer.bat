@echo off
rem Main viewer (ui_alex): 360 exploration with pointer dots, info windows, round map + full map, Ask Jeeves
rem Starts its own local server on port 8006 and opens the page. Close this window to stop it.
cd /d "%~dp0"
title Shuk HaTikvah viewer ui_alex - close this window to stop
echo Viewer ui_alex - 360 exploration
echo http://localhost:8006/viewer/ui_alex.html
start "" "http://localhost:8006/viewer/ui_alex.html"
".venv\Scripts\python.exe" pipeline\serve.py --port 8006
pause
