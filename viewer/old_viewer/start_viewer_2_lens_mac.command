#!/bin/bash
# Viewer design 2: Lens - focus circle + 360 radar strip (Mac version of start_viewer_2_lens.bat)
# Starts its own local server on port 8002 and opens the page. Close this Terminal window to stop it.
cd "$(dirname "$0")/../.." || exit 1
echo "2: Lens - focus circle + 360 radar strip"
echo "http://localhost:8002/viewer/old_viewer/v2_lens.html"
(sleep 1 && open "http://localhost:8002/viewer/old_viewer/v2_lens.html") &
.venv/bin/python pipeline/serve.py --port 8002
read -r -p "Press Enter to close..."
