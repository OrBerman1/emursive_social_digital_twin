@echo off
rem Viewer design 1: Atelier - calm, light, editorial
rem Starts its own local server on port 8001 and opens the page. Close this window to stop it.
cd /d "%~dp0..\.."
title Shuk HaTikvah viewer 1 (atelier) - close this window to stop
echo Viewer 1: Atelier - calm, light, editorial
echo http://localhost:8001/viewer/old_viewer/v1_atelier.html
start "" "http://localhost:8001/viewer/old_viewer/v1_atelier.html"
".venv\Scripts\python.exe" pipeline\serve.py --port 8001
pause
