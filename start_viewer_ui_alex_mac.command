#!/bin/bash
# Viewer ui_alex: 360° exploration with pointer dots, leader-line cards, radar and event timeline.
# Starts its own local server on port 8006 and opens the page. Close this Terminal window to stop it.
cd "$(dirname "$0")" || exit 1
echo "ui_alex - 360° exploration"
echo "http://localhost:8006/viewer/ui_alex.html"
(sleep 1 && open "http://localhost:8006/viewer/ui_alex.html") &
.venv/bin/python pipeline/serve.py --port 8006
read -r -p "Press Enter to close..."
