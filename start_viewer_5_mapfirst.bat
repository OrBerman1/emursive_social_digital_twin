@echo off
rem Viewer design 5: Map-first - map canvas + 360 inset
rem Starts its own local server on port 8005 and opens the page. Close this window to stop it.
cd /d "%~dp0"
title Shuk HaTikvah viewer 5 (mapfirst) - close this window to stop
echo Viewer 5: Map-first - map canvas + 360 inset
echo http://localhost:8005/viewer/v5_mapfirst.html
start "" "http://localhost:8005/viewer/v5_mapfirst.html"
".venv\Scripts\python.exe" pipeline\serve.py --port 8005
pause
