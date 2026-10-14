#!/bin/bash
# Starts the local viewer server and opens the viewer in the default browser (Mac version of start_viewer.bat).
# Close this Terminal window (or press Ctrl+C in it) to stop the server.
cd "$(dirname "$0")/../.." || exit 1
echo "Starting the viewer at http://localhost:8000/viewer/old_viewer/index.html"
echo "Keep this window open while viewing. Close it to stop the viewer."
(sleep 1 && open "http://localhost:8000/viewer/old_viewer/index.html") &
.venv/bin/python pipeline/serve.py --port 8000
read -r -p "Press Enter to close..."
