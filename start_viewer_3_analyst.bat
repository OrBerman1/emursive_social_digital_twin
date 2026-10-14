@echo off
rem Viewer design 3: Analyst - video + dashboard + heat timeline
rem Starts its own local server on port 8003 and opens the page. Close this window to stop it.
cd /d "%~dp0"
title Shuk HaTikvah viewer 3 (analyst) - close this window to stop
echo Viewer 3: Analyst - video + dashboard + heat timeline
echo http://localhost:8003/viewer/v3_analyst.html
start "" "http://localhost:8003/viewer/v3_analyst.html"
".venv\Scripts\python.exe" pipeline\serve.py --port 8003
pause
