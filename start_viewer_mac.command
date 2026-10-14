#!/bin/bash
# Main viewer (ui_alex): 360° exploration with pointer dots, leader-line cards, map/radar and event timeline.
# Starts a local server on port 8006 (or reuses one that is already running) and opens the page.
# Close this Terminal window to stop the server.
cd "$(dirname "$0")" || exit 1
PORT=8006
URL="http://localhost:$PORT/viewer/ui_alex.html?v=$(date +%s)"   # ?v= makes the browser load the latest file, not a cached copy
echo "ui_alex - 360° exploration"
if lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
  echo "A viewer server is already running on port $PORT - opening the page with it."
  open "$URL"
  exit 0
fi
echo "$URL"
(sleep 1 && open "$URL") &
.venv/bin/python pipeline/serve.py --port $PORT
read -r -p "Press Enter to close..."
