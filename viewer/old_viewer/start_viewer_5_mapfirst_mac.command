#!/bin/bash
# Viewer design 5: Map-first - map canvas + 360 inset (Mac version of start_viewer_5_mapfirst.bat)
# Starts its own local server on port 8005 and opens the page. Close this Terminal window to stop it.
cd "$(dirname "$0")/../.." || exit 1
echo "5: Map-first - map canvas + 360 inset"
echo "http://localhost:8005/viewer/old_viewer/v5_mapfirst.html"
(sleep 1 && open "http://localhost:8005/viewer/old_viewer/v5_mapfirst.html") &
.venv/bin/python pipeline/serve.py --port 8005
read -r -p "Press Enter to close..."
