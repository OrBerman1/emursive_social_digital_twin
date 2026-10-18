#!/bin/bash
# ===========================================================================
#  Cliff - process the Shuk HaTikvah data (macOS). Mac version of process_data.bat; double-click it after
#  setup_mac.command. Every finished step is kept: stop any time, run again to continue. See README.md.
# ===========================================================================
cd "$(dirname "$0")" || exit 1
done_() { echo; read -r -p "Press Enter to close..."; exit "${1:-0}"; }
failed() {
  echo; echo "[ERROR] The analysis stopped at the step shown above. The full log is in job.log, in the project's"
  echo "        data/processed folder. Fix the message, or send the log to the team, then run this again."; done_ 1
}
[ -x .venv/bin/python ] || { echo "The program is not installed yet. Double-click setup_mac.command first, then run this again."; done_ 1; }

echo; echo "=== 1/2  Preparing the data ==="
.venv/bin/python pipeline/prepare_data.py || done_ 1

echo; echo "Which project should be processed?"
echo "  1 = Shuk HaTikvah only          (recommended - the one the viewer opens; a few hours the first time)"
echo "  2 = Shuk HaTikvah and Legacy    (about twice as long)"
read -r -p "Press 1 or 2, then Enter: " pick

echo; echo "=== 2/2  Analysis ==="
echo "Keep the computer plugged in and awake. You can close this window to stop;"
echo "run process_data_mac.command again later and it continues from the last finished step."
echo; echo "--- Shuk HaTikvah ---"
.venv/bin/python pipeline/run_all.py --project shuk_hatikvah_v2 || failed
if [ "$pick" = 2 ]; then
  echo; echo "--- Shuk HaTikvah Legacy ---"
  .venv/bin/python pipeline/run_all.py --project shuk_hatikvah || failed
fi
echo; echo "Done. Open the viewer: double-click start_viewer_mac.command"
done_ 0
