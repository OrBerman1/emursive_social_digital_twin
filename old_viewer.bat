@echo off
rem Starts the local viewer server and opens the viewer in the default browser.
rem Close this window (or press Ctrl+C in it) to stop the server.
cd /d "%~dp0"
title Shuk HaTikvah viewer - close this window to stop
echo Starting the viewer at http://localhost:8000/viewer/old_viewer/index.html
echo Keep this window open while viewing. Close it to stop the viewer.
start "" "http://localhost:8000/viewer/old_viewer/index.html"
".venv\Scripts\python.exe" pipeline\serve.py --port 8000
pause
