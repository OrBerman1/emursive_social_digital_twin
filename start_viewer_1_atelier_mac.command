#!/bin/bash
# Viewer design 1: Atelier - calm, light, editorial (Mac version of start_viewer_1_atelier.bat)
# Starts its own local server on port 8001 and opens the page. Close this Terminal window to stop it.
cd "$(dirname "$0")" || exit 1
echo "1: Atelier - calm, light, editorial"
echo "http://localhost:8001/viewer/v1_atelier.html"
(sleep 1 && open "http://localhost:8001/viewer/v1_atelier.html") &
.venv/bin/python pipeline/serve.py --port 8001
read -r -p "Press Enter to close..."
