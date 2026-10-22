# Shuk HaTikvah 360° — Social Immersive Digital Twin (hackathon prototype)

360° walks through Shuk HaTikvah (Tel Aviv) → stitching and offline stabilisation → AI detection
(people, objects, perceived gender and age group, text on signs) → tracking with persistent IDs → an
immersive 360° viewer with the observations anchored in the scene, plus GIS-ready GeoJSON.

| Document | What it is |
|---|---|
| `README.md` | This file: how to prepare the data, install, run the pipeline and open the viewer |
| `METHODS.txt` | Which algorithm/model solves each problem (stitching, stabilisation, detection, tracking, OCR…) |
| `PLAN.md` | Project plan, brief requirements, progress log |
| `CLAUDE.md` | Working notes for the AI assistant (conventions, gotchas) |

---

## Quick start: no programming needed (Windows)

You only double-click files; no Python knowledge is needed. Three files do everything:

| File | What it does | How long |
|---|---|---|
| **`setup.bat`** | Installs the programs and AI models. It does **not** touch the data. | 30–60 min (large downloads) |
| **`process_data.bat`** | Analyses the videos (people, objects, signs, map…). | 2–3 h the first time, minutes afterwards |
| **`start_viewer.bat`** | Opens the viewer (Cliff) in your browser. | seconds |

1. **Get the program.** On the GitHub page press **Code → Download ZIP** and unzip it to a folder with
   enough space (about 25 GB free on that drive, see section 1). Or, with git: `git clone <repository address>`.
2. **Double-click `setup.bat`.** It works on an empty computer: if Python, ffmpeg or Tesseract are missing it
   installs them itself (with winget, part of Windows 10/11). Answer "Yes" if Windows asks for permission.
   At the end it lists the data files that are still missing; that is expected the first time.
   - It does **not** install the **NVIDIA graphics driver**. Without it everything still works, but runs on the
     CPU and takes many hours more. If setup warns "nvidia-smi not found", install the driver from nvidia.com.
   - If it warns about Tesseract's Hebrew data: right-click `setup.bat` → **Run as administrator** once.
   - If something fails, fix the message it shows and simply run `setup.bat` again; finished parts are kept.
3. **Put the data in `data\raw\LABAI\`** (the folder is created by setup): the two videos and the store table,
   with their exact names (section 2). It does not matter whether you do this before or after setup.
4. **Double-click `process_data.bat`.** It checks the data (and tells you if a video is missing or only half
   downloaded), then asks:
   - **1 = Shuk HaTikvah only** (recommended: this is the project the viewer opens), or
   - **2 = Shuk HaTikvah and Shuk HaTikvah Legacy** (the original stitch, for comparison; twice as long).

   Keep the computer plugged in and awake. You can close the window at any time (or it may stop on an error); run
   `process_data.bat` again and it continues where it stopped: finished steps are kept, and the long steps (detection,
   floor masks) save their progress every few minutes, so at most a few minutes of work are repeated.
   - **Without an NVIDIA graphics card** everything runs on the processor: it works, but takes **days** instead of
     hours (`process_data.bat` says so at the start). Stopping and continuing later is safe.
5. **Double-click `start_viewer.bat`.** The viewer opens in your browser. Keep the black window open while you use it.

**Mac (Apple Silicon):** the same three steps with `setup_mac.command`, `process_data_mac.command` and
`start_viewer_mac.command` (double-click in Finder). Install first, in Terminal: `brew install python@3.13 ffmpeg
tesseract tesseract-lang rust` (Homebrew: brew.sh). The GPU stages use Apple MPS, video is encoded with VideoToolbox,
and the gyro reader is built from source into `.venv` (no `.venv-imu`). If the LABAI files sit in the folder above
the program folder, `setup_mac.command` links them into `data/raw/LABAI/`.

### Updating a computer that is already set up
For example, a computer installed before the people-ReID update (15 Oct 2026):
1. Get the new program: `git pull` in the program folder, or download the ZIP again and unzip it **over** the old
   folder (replace the files). Your data, results and installed programs are in folders the ZIP does not contain,
   so they stay.
2. Run `setup.bat` again. It keeps everything already installed and only downloads what is new (for ReID: the
   OSNet model file, `models\osnet_ain_x1_0_msmt17.pth`; no new Python packages).
3. Run `process_data.bat` again. Finished steps are skipped, so only new or changed steps run (for ReID: a few
   minutes per project).
4. Close the viewer's black window, start `start_viewer.bat` once, and press **Ctrl+F5** in the browser.

### Good to know
- **Two projects.** "Shuk HaTikvah" (`projects\shuk_hatikvah_v2\`) uses the improved, seamless stitch and is the one
  the viewer opens. "Shuk HaTikvah Legacy" is the same walks with the first stitch (data in `data\`), kept for
  comparison; the viewer opens it with `?project=shuk_hatikvah`. Both read the same raw files: `process_data.bat`
  gives the second project a link to them (no extra disk space; a copy if links are not possible on that drive).
- **Sign text.** The text on signs was read once by Claude. Those readings are kept in the program
  (`config\signs_read\`, `projects\shuk_hatikvah_v2\config\signs_read\`) and are put back automatically, so a new
  computer shows the same sign text without a Claude key and without new Claude calls.
- **Claude key (optional).** Ask Jeeves, and Claude checks on new projects, need a key, entered on the projects page.
  It is stored only on that computer (`%LOCALAPPDATA%\ShukTwin`), never in the program folder.

The sections below are the detailed reference.

## 1. Requirements

- **Windows 10/11** with an **NVIDIA GPU** (tested on an RTX 4060 Laptop, 8 GB). A CPU works but detection is many times slower.
- About **20 GB free disk space per project** (measured for one project): raw data 5.0 GB, pipeline cache
  `data\interim` 5.8 GB, viewer outputs `data\processed` 2.0 GB, environments + weights 6.0 GB (shared). Both
  projects together need about 35 GB (the raw videos are shared).
- Internet during setup and the first run (model downloads, OpenStreetMap streets), and for the map tiles in the viewer.
- Programs (`setup.bat` installs the missing ones with winget; install them by hand only if that fails):

| Program | Where | Notes |
|---|---|---|
| Python **3.13** and **3.9** (64-bit) | python.org | Install both; the `py` launcher must be available. 3.9 is only for reading the camera's gyro data. |
| **ffmpeg** (full build) | gyan.dev/ffmpeg/builds | Add its `bin` folder to PATH. |
| **Tesseract OCR 5** | github.com/UB-Mannheim/tesseract/wiki | Default path `C:\Program Files\Tesseract-OCR`; tick **Hebrew** and **Arabic** language data. |
| curl, winget | built into Windows 10/11 | winget missing: update "App Installer" from the Microsoft Store. |
| NVIDIA driver | nvidia.com | **Not** installed by setup. Without it everything runs on the CPU (many times slower). |

## 2. Prepare the data (not in git)

The raw data is shared by the research team (Boaz) as the **LABAI** folder on OneDrive.

1. Run `setup.bat` once (section 3), or create the folders yourself. It creates:
   ```
   data\raw\LABAI\            <- raw data goes here
   data\raw\LABAI\Photos\
   data\raw\LABAI\Sketches\
   data\interim\              <- pipeline cache (created automatically, large)
   data\processed\            <- viewer and GIS outputs
   models\                    <- model weights (downloaded by setup.bat)
   ```
2. From the shared LABAI folder, download these into `data\raw\LABAI\`, keeping the exact file names:

| File | Size | Content |
|---|---|---|
| `1_Hatikvah St.insv` | 2.63 GB | Walk 1, raw Insta360 ONE RS 360° video (3:28) |
| `2_Hamevasser St.insv` | 2.56 GB | Walk 2, raw Insta360 ONE RS 360° video (3:23) |
| `Shops with Icons.pdf` | 1.1 MB | GIS shop map (PDF export of layer `Hatikvah_Shops_30_7_26`) |
| `Shops_In_hatikva_30_7_26.xlsx` | small | Store table with X/Y (WGS84) and attributes, used for the map |
| `shuk_360_frame_coding_v01.docx` | 5.8 MB | The team's frame-coding scheme (9 coded frames) |
| `Photos\*.jpg` | 12 files | Field photos (2024–2026) |
| `Sketches\*.png` | 37 files | Boaz's drawings + style images |

> **Download the files one by one, not as a zipped folder.** OneDrive's "download folder as zip" of
> several GB stopped half-way twice; the broken zip looks empty in Windows and the last video is cut off.
> Check that the two `.insv` sizes match the table (`process_data.bat` checks this too). Needed for the
> analysis: the two videos and the store table (for the map); the other files are reference material.
> The data can be added before or after `setup.bat`; nothing is processed until you run `process_data.bat`.

## 3. Install the environment

Double-click **`setup.bat`** (or run it in a terminal). It:

1. checks the programs from section 1 and installs the missing ones with winget (Python 3.13 and 3.9, ffmpeg,
   Tesseract + its Hebrew and Arabic data),
2. creates the folder structure from section 2,
3. creates `.venv` (Python 3.13) and installs **PyTorch with CUDA 12.4** (~2.5 GB download),
4. installs the pinned packages from `requirements.txt` plus CLIP,
5. downloads the model weights into `models\` (YOLO11x, YOLO-World, YOLOv8 Open Images, SegFormer, OSNet-AIN for people ReID),
6. creates `.venv-imu` (Python 3.9) with telemetry-parser for the gyro data,
7. verifies the installation (prints whether CUDA works) and lists any missing data files.

It does **not** process the data (that is `process_data.bat`, section 4). It is safe to run again; existing
environments and downloads are reused, and only new items are downloaded.

## 4. Run the pipeline

**Easiest: double-click `process_data.bat`** (Mac: `process_data_mac.command`). It runs
`pipeline\prepare_data.py` (checks the raw files and their sizes, links them into the Shuk HaTikvah project folder,
puts back the saved sign readings) and then the pipeline for the project(s) you choose.

The same by hand, in a terminal in the program folder:
```
.venv\Scripts\python pipeline\prepare_data.py                              (check + link the data)
.venv\Scripts\python pipeline\run_all.py --project shuk_hatikvah_v2        (Shuk HaTikvah, the viewer's default)
.venv\Scripts\python pipeline\run_all.py                                   (Shuk HaTikvah Legacy)
.venv\Scripts\python pipeline\run_all.py walk1                             (one walk only)
```

- **First run: about 2–3 hours per project** on the RTX 4060 (mostly GPU detection). Every expensive step is
  **cached** (`data\interim\`, or `projects\<id>\data\interim\`) with a `manifest.json` and is skipped next time,
  so later runs, updates and parameter changes take seconds to minutes. Stopping half-way is safe: the next run
  continues from the last finished step.
- Resuming: each step saves its result per walk with a `manifest.json`; a rerun skips everything finished. Inside a
  step, `detect.py` saves the boxes so far every 5 minutes (`partial.parquet` + `partial.json`, same cache key) and
  `waste.py` keeps the floor masks already written, so an interrupted step continues instead of starting again.
- Without an NVIDIA GPU every step runs on the CPU (or Apple MPS); the viewer video is then encoded with libx264
  (the NVIDIA encoder is tested with a short clip, not just looked up), and a video already finished with either
  encoder is kept.
- Progress and the full log: `job.json` / `job.log` in the project's processed folder (the projects page shows them too).
- After people tracking, `reid_embed.py` + `link_persons.py --reid` (people ReID, OSNet-AIN) decide which track
  pieces are one person; it is skipped, falling back to the reviewed links, when `models\osnet_ain_x1_0_msmt17.pth`
  is missing (rerun `setup.bat`).
- If you have a newer Claude reading of the signs on one computer, `prepare_data.py --save` copies it into
  `config\signs_read\` (and the Shuk HaTikvah project's) so it can be committed and shared.
- Main stages, in order (each can also be run alone; `METHODS.txt` explains every one, including the later
  ones not listed here: Claude sign reading, map, motion, food, crowds, person linking and ReID):

| Stage | Script | Output |
|---|---|---|
| Stitch + frames (5 fps) | `extract_frames.py` | `data\interim\<walk>\frames\` |
| Browser video | `encode_video.py` | `data\processed\<walk>\video\equirect.mp4` |
| Detection L1 / L2 (unstabilised) | `detect.py` | `data\interim\<walk>\det_*\` |
| Gyro data | `imu_extract.py` (in `.venv-imu`) | `data\interim\<walk>\imu.csv` |
| Stabilisation rotation | `stabilize.py` | `data\processed\<walk>\stab.json` |
| Tracking + box smoothing | `tracks.py` | `data\interim\<walk>\tracks*\` |
| Bake stabilisation | `bake_stab.py` | `video_stab\`, `frames_stab\` |
| Detection L1 / L2 on stabilised frames | `detect.py --frames frames_stab` | `det_*_stab\` |
| Perceived gender + age group | `attributes.py` | `attributes_stab\` |
| Text on signs (OCR) | `ocr_signs.py` | `text_det_stab\`, `ocr_stab\` |
| Viewer data + GeoJSON | `export_overlays.py --variant stab` | `data\processed\<walk>\overlays_stab.json`, `observations_stab.geojson` |
| Hand-count sample | `make_validation.py` | `data\processed\validation\` |

To force a stage to recompute, run it alone with `--force`.

## 5. Open the viewer

**Main viewer (design `ui_alex`):** double-click **`start_viewer.bat`**
(Mac: `start_viewer_mac.command`) → http://localhost:8006/viewer/ui_alex.html (http://localhost:8006/viewer/ opens it too).
It opens the restitched project (`config/viewer.json`); the original results stay at `?project=shuk_hatikvah`.
Click a dot for its card, the round map for the full map, 🎩 Ask Jeeves at the top.

1. A black window (the local server) opens and the viewer opens in your browser. Keep the black window open while
   viewing; if it is closed, the page stops loading.
2. To stop: close the browser tab and the black window.

The viewer must be opened through this server; opening the html file directly does not work.
**Earlier viewers** (the original viewer and the five design variants) are kept in `viewer/old_viewer/` with their
launchers: the first version by **`old_viewer.bat`** in the main folder (Mac: `old_viewer_mac.command`, port 8000); the five
designs by `viewer\old_viewer\start_viewer_1..5_*.bat` (ports 8001–8005).
The controls below describe the original viewer.

**Controls**
- Play / pause (or space bar); drag the timeline; **Speed** ×0.25 – ×2.
- Look around: drag the image or arrow keys; mouse wheel zooms.
- Walk 1 / Walk 2 buttons (top right).
- Click a person's box for an information card (track ID, time seen, distance).
- **📎 Attach file** in a person's card: attach an interview, note or photo to that person. People with files get an
  orange 📎 on their box and on the timeline; click the timeline 📎 to jump to them. Files stay on this computer
  (`data/processed/attachments/`, not in git).
- **🎩 Ask Jeeves** (top right): ask about the project ("How many women are in the video?") or give it a task
  ("Count all the people wearing blue shirts"). It answers from the analysed data and can look at the video; blue
  buttons in the answer jump to that person or moment. Needs a Claude key (projects page: Anthropic `sk-ant-…` or
  Amazon Bedrock `ABSK…`). Chats are kept per project.
- **Layers:** people, IDs & distance, level-1 objects, level-2 objects, text on signs (OCR), hide tracks < 1 s,
  **Stabilised video (offline)** on/off (off = original video and detections).
- **Min. confidence** slider (default 0.60).
- Side panel: people in view, perceived gender and age groups in view (aggregate only), text in view,
  objects in view.
- **People totals** (whole walk, so far) count linked persons: track pieces confirmed as one person
  (`pipeline\link_persons.py`, vision-reviewed in `config\person_links_review.json`) count once and share one
  perceived gender / age label (smoothed over the pieces). Undo: `.venv\Scripts\python pipeline\restore_person_linking_backup.py`.

After updating the code, restart `start_viewer.bat` and press **Ctrl+F5** once in the browser.

**Hand-count validation:** http://localhost:8006/viewer/validate.html — 30 frames; type how many people
you see in each, and the page compares your counts with the model's.

### Projects: add data, or start a new area

Open **http://localhost:8006/viewer/projects.html** (link "projects" in the viewer's title).
- **Add data to a project** (also to Shuk HaTikvah): another 360 video (Insta360 ONE RS `.insv`, one walk), a
  store table (`.xlsx` with at least `Shop_Name`, `X`, `Y` in WGS84, like `Shops_In_hatikva_30_7_26.xlsx`), or a
  `.txt` of anchor points for one walk, e.g. `Hasaluf f104 yaw -8; Rosita Bar f163 yaw 150` (store name or
  GlobalID, frame `f…` and view yaw from the viewer's top-left box; optional `pitch` and `fov`).
- **New project** (another area): a name and a 360 video are required; the store table and anchors are optional.
- **Run analysis**: runs the pipeline once for that project in the background, with a bar per step. Finished steps
  are cached, so adding anchors later only re-runs the map steps. A new walk takes about 1–1.5 h on the GPU.
- Without a store table **and** anchors for a walk there is no map for it (no paths, crowds, speed map,
  segment summary); the 360 view, people, objects, signs and food still work.
- **Claude API key** (optional, left panel): if saved, sign text is read by Claude and the pictured-people and
  stall-material checks are reviewed by Claude. Without it, Tesseract and the automatic fallbacks run, and stall
  materials are skipped. The key is stored only on this computer (`%LOCALAPPDATA%\ShukTwin`), never in the repo.
- Each project lives in `projects/<id>/` (its data folder is not in git). The viewer opens a project with
  `viewer/?project=<id>`.

## 6. Outputs for GIS

`projects\shuk_hatikvah_v2\data\processed\<walk>\observations_stab.geojson` (Legacy: `data\processed\<walk>\…`):
one feature per persistent entity (ID, entity type, frames and times seen, confidence, model, position on the map
from the store anchors and the gyro; the videos contain no GPS).

**Five alternative viewer designs** (branch `design`; now in `viewer/old_viewer/`, launchers there too). Each has
its own launcher and port, so several can run side by side:

| Launcher | Design | Idea |
|---|---|---|
| `start_viewer_1_atelier.bat` (port 8001) | Atelier | calm light editorial look; quiet dots instead of boxes, fixed insight cards, settings behind one "Layers" button |
| `start_viewer_2_lens.bat` (port 8002) | Lens | boxes only inside a central focus circle; a 360° radar strip shows what is around you; category dock |
| `start_viewer_3_analyst.bat` (port 8003) | Analyst | video + tabbed dashboard (People · Stalls · Movement · Map); heat timeline of people density; nearest-N boxes |
| `start_viewer_4_story.bat` (port 8004) | Story | fullscreen; one mode at a time (People · Food · Signs · Materials · Map, keys 1–5); one-sentence caption; auto-hiding controls |
| `start_viewer_5_mapfirst.bat` (port 8005) | Map-first | the map is the main canvas (people, stores, frontage, route); 360 video as an inset; click the route to jump |

All five share `viewer/engine.js` (360 rendering, data, tracking blend, counts, map position, controls) and use
one colour-blind-safe palette (Okabe–Ito: people blue, food green, objects orange, market items purple, signs sky
blue, pictured people yellow, frontage vermilion).

**People's paths and motion.** `pipeline\motion.py` puts every tracked person on the map (camera position +
ground distance from their feet) and measures their walking speed and direction with the camera's own motion
removed (`data\processed\<walk>\motion_stab.json`), combining map geometry with optical flow of the feet
against the ground beside them (`pipeline\person_flow.py`, cached). Arrows are drawn only where both agree. In the viewer, the panel's *Motion & paths* section
chooses who gets motion arrows; follow people with the card's **Follow** button, **Shift + drag** a box on the
video, or **Follow everyone in view**. Followed people get one colour on their box, arrow and path on both maps.

**Pictured people and food.** `pipeline\depicted.py` separates pictured people (rabbi portraits, posters,
adverts) from real people: CLIP proposes candidates and a vision review in `config\depicted_review.json`
decides; they become the category `possible_rabi_sign`, are not counted as people and have their own viewer
layer. `pipeline\food.py` finds food displays (YOLOv8x Open Images V7 + COCO produce) and names them with CLIP
from a market vocabulary; menu/sign photos of food are kept apart. The Open Images weights
(`models\yolov8x-oiv7.pt`) are downloaded by `setup.bat`.

**Waste on the floor.** `pipeline\waste.py` keeps open-vocabulary waste detections only where the ground around
and under them is walkway floor (SegFormer, downloaded by `setup.bat` into `models\segformer-b2-ade\`), checks
each item with CLIP, and writes items with map position and waste per 10 m of route
(`data\processed\<walk>\waste_stab.json`). In the viewer: magenta boxes, a panel section, dots on both maps.

**Crowds.** `pipeline\crowds.py` finds crowds from overlapping person boxes of similar size and puts them on
the maps as orange markers (number = most people at once). Click one on the full map for its statistics
(approximate size, speed mean ± std, walking/standing, age group, perceived gender) and **▶ Go to video**. The
thresholds are adjustable live under **👥 Crowd settings** on the full map.

**Map layers and filters.** In the full map, **🗂 Layers & filters** chooses what is shown (default: stores and
crowds; optional: walk routes, frontage materials, everyone's paths), filters stores by any field of the store
table (trade, shop type, ownership, patronage, facade, structure, informal vendor, generation, seen on a walk),
and colours stores by one of them. The **🌡 Speed map** layer paints the mean walking speed per area as a smooth
colour surface with its own legend (smoothing radius and minimum people adjustable in the same panel).

**Comments.** Press **🚩 Comment** next to Play to pin a note to the current moment; choose the flag colour and save.
Flags appear above the timeline: hover to read, click to jump there and edit or delete. Comments are saved to
`data\processed\comments.json` by the viewer's local server (copy that file to share them).

**Segment summary.** In the full map, press **🖱 Select segment** (top right), click a start store and an end
store: the stretch of the walk between them is highlighted and summarised (people, ages, movement, stores by
trade, food, signs, objects, frontage materials), with **⬇ JSON** and **⬇ Excel** export. Esc cancels.

**Map.** `pipeline\georef.py` places both walks on the map from the store anchors in `config\anchors.json`
(frame + view yaw where the camera looks at a known store) and the gyro heading; it writes
`data\processed\<walk>\georef.json` and `data\processed\shops.json`. It keeps the walker on real streets
(never inside a building, never through a stall) using OpenStreetMap streets and buildings, downloaded once
into `data\raw\osm\` on the first run (needs internet; © OpenStreetMap contributors, ODbL). Add anchors there and rerun it to
improve the fit. The viewer shows a heading-up minimap (bottom right); click it for the full store map (a red
"you are here" marker with a view cone follows the video live) and click a store to jump the video there (it switches walk if needed). Map tiles come from OpenStreetMap,
so the map needs internet.

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| Windows shows the data zip as empty | The download was cut off. Download the files one by one (section 2). |
| `process_data.bat` says `[MISSING]` or `[INCOMPLETE]` | Put that file into `data\raw\LABAI\` with its exact name, or download it again (the size must match section 2). |
| `process_data.bat` says "The program is not installed yet" | Run `setup.bat` first. |
| Viewer opens but is empty / "Could not load" | The data was not processed yet: run `process_data.bat` (choice 1 for the default project). |
| Processing stopped with `[ERROR]` | Read the last lines above the error; the full log is `job.log` in `projects\shuk_hatikvah_v2\data\processed\` (Legacy: `data\processed\`). Fix it and run `process_data.bat` again; it continues where it stopped. |
| Setup: "winget not found" | Update "App Installer" from the Microsoft Store, or install the programs in section 1 by hand, then run `setup.bat` again. |
| Setup: "nvidia-smi not found" | Install the NVIDIA driver from nvidia.com (setup does not install it). |
| Certificate / SSL errors during downloads | Some networks intercept HTTPS. `setup.bat` uses curl, which works; if a pip install fails, rerun on another network. |
| Viewer shows "Could not load overlays.json" | The pipeline has not been run for that walk, or the page was opened without `start_viewer.bat`. |
| Video stays black, still frames show instead | Normal while the large video loads, or if the browser tab is in the background. |
| "Address already in use" in the black window | The viewer is already running in another window; close it or just open the browser address. |
| Changes don't appear in the viewer | Restart `start_viewer.bat`, then press Ctrl+F5. |
| Detection is extremely slow | No GPU found: check that `setup.bat` step 8 prints `CUDA True`. |
