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

## 1. Requirements

- **Windows 10/11** with an **NVIDIA GPU** (tested on an RTX 4060 Laptop, 8 GB). A CPU works but detection is many times slower.
- About **20 GB free disk space** (measured): raw data 5.0 GB, pipeline cache `data\interim` 5.8 GB, viewer outputs `data\processed` 2.0 GB, environments + weights 6.0 GB.
- Programs to install first (`setup.bat` checks them and says what is missing):

| Program | Where | Notes |
|---|---|---|
| Python **3.13** and **3.9** (64-bit) | python.org | Install both; the `py` launcher must be available. 3.9 is only for reading the camera's gyro data. |
| **ffmpeg** (full build) | gyan.dev/ffmpeg/builds | Add its `bin` folder to PATH. |
| **Tesseract OCR 5** | github.com/UB-Mannheim/tesseract/wiki | Default path `C:\Program Files\Tesseract-OCR`; tick **Hebrew** and **Arabic** language data. |
| curl | built into Windows 10/11 | |

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
> Check that the two `.insv` sizes match the table. Only the two videos are needed to run the pipeline;
> the other files are reference material.

## 3. Install the environment

Double-click **`setup.bat`** (or run it in a terminal). It:

1. checks the programs from section 1,
2. creates the folder structure from section 2,
3. creates `.venv` (Python 3.13) and installs **PyTorch with CUDA 12.4** (~2.5 GB download),
4. installs the pinned packages from `requirements.txt` plus CLIP,
5. downloads the YOLO11x and YOLO-World weights into `models\`,
6. creates `.venv-imu` (Python 3.9) with telemetry-parser for the gyro data,
7. verifies the installation (prints whether CUDA works) and lists any missing data files.

It is safe to run again; existing environments and downloads are reused.

## 4. Run the pipeline

```
.venv\Scripts\python pipeline\run_all.py            (both walks)
.venv\Scripts\python pipeline\run_all.py walk1      (one walk)
```

- **First run: about 2–2.5 hours** on the RTX 4060 (mostly GPU detection). Every expensive step is
  **cached** in `data\interim\` with a `manifest.json` and is skipped next time, so later runs and
  parameter changes take seconds to minutes.
- Stages, in order (each can also be run alone; see `METHODS.txt` for what each one does):

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

1. Double-click **`start_viewer.bat`**. A black window (the local server) opens and the viewer opens in
   your browser at **http://localhost:8000/viewer/**. Keep the black window open while viewing.
2. To stop: close the browser tab and the black window.

The viewer must be opened through this server; opening `viewer\index.html` directly does not work.

**Controls**
- Play / pause (or space bar); drag the timeline; **Speed** ×0.25 – ×2.
- Look around: drag the image or arrow keys; mouse wheel zooms.
- Walk 1 / Walk 2 buttons (top right).
- Click a person's box for an information card (track ID, time seen, distance).
- **Layers:** people, IDs & distance, level-1 objects, level-2 objects, text on signs (OCR), hide tracks < 1 s,
  **Stabilised video (offline)** on/off (off = original video and detections).
- **Min. confidence** slider (default 0.60).
- Side panel: people in view, perceived gender and age groups in view (aggregate only), text in view,
  objects in view.

After updating the code, restart `start_viewer.bat` and press **Ctrl+F5** once in the browser.

**Hand-count validation:** http://localhost:8000/viewer/validate.html — 30 frames; type how many people
you see in each, and the page compares your counts with the model's.

## 6. Outputs for GIS

`data\processed\<walk>\observations_stab.geojson`: one feature per persistent entity (ID, entity type,
frames and times seen, confidence, model). Positions along the route currently use an assumed walking
pace; real coordinates follow once anchor points on the GIS map are available (the videos contain no GPS).

**Map.** `pipeline\georef.py` places both walks on the map from the store anchors in `config\anchors.json`
(frame + view yaw where the camera looks at a known store) and the gyro heading; it writes
`data\processed\<walk>\georef.json` and `data\processed\shops.json`. Add anchors there and rerun it to
improve the fit. The viewer shows a heading-up minimap (bottom right); click it for the full store map
and click a store to jump the video there (it switches walk if needed). Map tiles come from OpenStreetMap,
so the map needs internet.

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| Windows shows the data zip as empty | The download was cut off. Download the files one by one (section 2). |
| Certificate / SSL errors during downloads | Some networks intercept HTTPS. `setup.bat` uses curl, which works; if a pip install fails, rerun on another network. |
| Viewer shows "Could not load overlays.json" | The pipeline has not been run for that walk, or the page was opened without `start_viewer.bat`. |
| Video stays black, still frames show instead | Normal while the large video loads, or if the browser tab is in the background. |
| "Address already in use" in the black window | The viewer is already running in another window; close it or just open the browser address. |
| Changes don't appear in the viewer | Restart `start_viewer.bat`, then press Ctrl+F5. |
| Detection is extremely slow | No GPU found: check that `setup.bat` step 8 prints `CUDA True`. |
