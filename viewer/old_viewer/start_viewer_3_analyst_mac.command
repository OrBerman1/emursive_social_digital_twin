#!/bin/bash
# Viewer design 3: Analyst - video + dashboard + heat timeline (Mac version of start_viewer_3_analyst.bat)
# Starts its own local server on port 8003 and opens the page. Close this Terminal window to stop it.
cd "$(dirname "$0")/../.." || exit 1
echo "3: Analyst - video + dashboard + heat timeline"
echo "http://localhost:8003/viewer/old_viewer/v3_analyst.html"
(sleep 1 && open "http://localhost:8003/viewer/old_viewer/v3_analyst.html") &
.venv/bin/python pipeline/serve.py --port 8003
read -r -p "Press Enter to close..."
