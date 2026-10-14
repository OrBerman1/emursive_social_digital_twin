#!/bin/bash
# Viewer design 4: Story - fullscreen, one mode at a time (Mac version of start_viewer_4_story.bat)
# Starts its own local server on port 8004 and opens the page. Close this Terminal window to stop it.
cd "$(dirname "$0")/../.." || exit 1
echo "4: Story - fullscreen, one mode at a time"
echo "http://localhost:8004/viewer/old_viewer/v4_story.html"
(sleep 1 && open "http://localhost:8004/viewer/old_viewer/v4_story.html") &
.venv/bin/python pipeline/serve.py --port 8004
read -r -p "Press Enter to close..."
