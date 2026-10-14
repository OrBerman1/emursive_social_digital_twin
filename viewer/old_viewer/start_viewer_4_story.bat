@echo off
rem Viewer design 4: Story - fullscreen, one mode at a time
rem Starts its own local server on port 8004 and opens the page. Close this window to stop it.
cd /d "%~dp0..\.."
title Shuk HaTikvah viewer 4 (story) - close this window to stop
echo Viewer 4: Story - fullscreen, one mode at a time
echo http://localhost:8004/viewer/old_viewer/v4_story.html
start "" "http://localhost:8004/viewer/old_viewer/v4_story.html"
".venv\Scripts\python.exe" pipeline\serve.py --port 8004
pause
