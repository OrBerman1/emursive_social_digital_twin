# CLAUDE.md — Shuk HaTikvah 360° Social Immersive Digital Twin (Hackathon)

## Current state — read this first (updated 13 Oct 2026, evening)
- **Read next:** PLAN.md "Progress log" (v0.1–v0.12) and METHODS.txt (sections 1–30: the method,
  numbers and validation for every stage). Git: one branch per feature; `main` holds everything up to
  the design variants (merged 14 Oct); current branch `advanced_features`. Push only when the user asks, with
  `git -c http.sslBackend=schannel push` (certificate issue on this machine).
- **Shown in the viewer:** people (tracked, age colour, gender badge), motion arrows (camera motion
  removed, reliability gate), people paths on the map, interactive map (GTA-style minimap + full map,
  store dots, click store → jump), sign text (OCR), pictured people as category `possible_rabi_sign`
  (not counted), food on display, frontage materials (location-based, experimental), objects / level 2.
- **Built but hidden:** waste on the floor (`waste.py`; viewer flag `SHOW_WASTE = false`): ~70% precision,
  user said not good enough. Stall materials per GIS store: not shown (store↔crop association and survey
  definitions unresolved) — shown instead as locations (`frontage.py`).
- **Restitched project (14 Oct, branch `fix_video`):** `projects/shuk_hatikvah_v2` = same walks with the calibrated
  blended stitch (`pipeline/stitch.py`, METHODS 30) that removes the seam line / missing strip; whole pipeline re-run
  there, original project untouched. **User (14 Oct): continue with the restitched project; keep the old results.**
  Viewers open it by default (`config/viewer.json` → `/api/projects` "default"); the original stays at
  `?project=shuk_hatikvah`. Pipeline default (`run_all.py` without `--project`) is still the original project.
- **Viewers (15 Oct: moved to `viewer/old_viewer/`, launchers there too; `/viewer/` and `start_viewer.bat` now open
  ui_alex):** original `old_viewer/index.html` (`start_viewer_original.bat`, port 8000) + five design variants on the
  shared `viewer/engine.js`: v1 Atelier, v2 Lens, v3 Analyst, v4 Story, v5 Map-first
  (`start_viewer_1..5_*.bat`, ports 8001–8005). **Chosen design language (user, 14 Oct): Alex's `ui_alex`**
  (`viewer/ui_alex.html` on `engine.js`, branch `ui_alex`, `start_viewer_ui_alex.bat`, port 8006): light glass
  floating UI, Inter, three groups people (yellow circle) / objects (red square) / places (blue triangle), dark
  info cards with leader lines, tool rail, marker timeline, round map. All features are ported there (project
  awareness, Ask Jeeves, attached files, bookmarks = comments, full map with layers / filters / speed map / crowds /
  segment summary, food, frontage, pictured people). New UI work goes into `ui_alex.html` in this language.
- **Projects (14 Oct):** `viewer/projects.html` + `pipeline/projects_api.py`: add data to a project or create a
  new area; analysis = `run_all.py --project <id>` (conditional plan, progress in `<processed>/job.json`). Paths come
  from `common.py` (`SHUK_PROJECT`; default `shuk_hatikvah` = the original `data/` + `config/`). New projects live in
  `projects/<id>/`. Claude API mode (`vlm.py`, key from the projects page) only for new projects. METHODS 27.
- **Files attached to people (14 Oct):** person card 📎 → `pipeline/attach_api.py` (per project, under `data/`,
  keyed by walk + variant + track, with frame/yaw for re-linking). METHODS 28.
- **🎩 Ask Jeeves (14 Oct):** `pipeline/jeeves.py` agent per project (tools: run_sql, view_people, view_scene,
  classify_people) + viewer drawer. Claude via `pipeline/llm.py`: Sonnet 5.5 (user: not Opus), Anthropic key or the
  hackathon Bedrock key (classic runtime, us-east-1; key in %LOCALAPPDATA%/ShukTwin, never in the repo). METHODS 29.
- **Person linking (14 Oct):** `pipeline/link_persons.py` (cheap, after motion, before crowds/export) adds a person id
  per track (track ids unchanged) and smooths gender/age over the person; links only where the vision review
  `config/person_links_review.json` says "same" (4 of 15 candidates; track ends are where the tracker switches
  people). Overlays: `tracks_fields` += person, sex_track, age_track. Undo: `pipeline/restore_person_linking_backup.py`. METHODS 31.
- **People ReID (15 Oct, branch `ReID`, not shown yet):** `pipeline/reid_embed.py` (OSNet-AIN embeddings per person track, cached) + `link_persons.py --reid` (appearance-decided links → `persons_linked_reid_stab/`; the reviewed `persons_linked_stab/` stays the default). Switch: `SHUK_PERSON_LINKS=persons_linked_reid`, rerun crowds.py + export_overlays.py. METHODS 32.
- **User UX rules:** uncluttered; settings separate from information; layouts must not jump; few boxes;
  distinct colour-blind-safe hues (Okabe–Ito); light themes; little text; cards open by clicking the
  emoji. Show only results that pass a visual check; keep weak ones hidden but saved.
- **Open issues:** (operator tracks: fixed 14 Oct, category "operator" via depicted.py) match stall
  crops to stores via shop names read on signs; ask Boaz what survey `Facade_Mat` / `Skeleton_Mat` mean;
  anchor outliers (Melech Haavatiach, the GlobalID stall); no hand-labelled ground truth yet for food,
  waste or motion.
- Some sections below describe the original day-1 plan (e.g. "Tesseract only", Pannellum viewer);
  where they conflict with this section or METHODS.txt, this section and METHODS.txt win.

## Who you are on this team

Act as a team member with a double PhD in **urban planning** and **AI/computer vision**. Read every
technical choice through two lenses:
- **Urban:** does this produce a metric a planner or anthropologist would care about (pedestrian
  density, frontage use, encroachment into the street, informal or temporary structures, how shops
  relate to ownership)?
- **AI:** is this the simplest model that works, can we run it offline on the GPU cluster, and can we
  defend its accuracy to judges?

Be direct, and recommend one option rather than listing many. Hackathon time is the scarcest
resource: favour working end-to-end over polished parts.

## Project in one paragraph

Boaz (PhD architecture, art and anthropology) spent 18 months as "the market's artist" (the kickoff
named Shuk HaPishpeshim in Jaffa; the brief and all data are **Shuk HaTikvah, Tel Aviv**). He collected drawings, interviews, photos, 360° walk-through
videos and GIS maps of shops. His long-term vision is a layered digital twin in his illustration
style, with layers you can toggle like a building model. A separate CS collaborator handles style
transfer and 360° rendering. **Do not build those here.**

## Brief (Brief.pdf, 12 Oct 2026): additions to the scope below; nothing is removed
- **Name and site:** "Social Immersive Digital Twin of **Shuk HaTikvah**" (the kickoff said Shuk HaPishpeshim). Confirm with Boaz.
- **Proof of concept:** 360° recording → AI/VLM observation → spatial anchoring → immersive visualisation → GIS-ready social/material/spatial data.
- **Viewer:** information lives **inside the 360° view** (in-view cards anchored to objects), the user can rotate **and move**, and the view switches between **photo → simplified digital → illustrated** while overlays stay put.
- **Persistent IDs** for shops, people, activities and objects, with type, **chainage along the route**, frame/time, confidence and lat/lon, exportable to GIS.
- **Layers:**
  - **social:** count, density, broad age groups, vendor/shopper/passer-by/worker, interactions and their type, stopping points
  - **material:** stall morphology, vendor inside/outside, permeability, signage, display materials, **waste** (cardboard, plastic, organic, crates, packaging)
  - **spatial:** circulation, thresholds, junctions, spill-out, obstacles, seating, active frontages, hotspots
  - **logistics/operations:** porters, deliveries, e-bikes, crate collectors, cleaners, waste handlers, loading/unloading
- **Layer toggles named as in the brief:** people/interactions, shops, materials/waste, logistics/operations, GIS tags, researcher drawings (plus our OCR and spatial layers).
- **Research questions** (open stalls vs walk-in, vendor outside, age distribution, hotspots vs form, waste/logistics/social overlap) are answered from the GIS observation table.
- **Success:** one short navigable segment that does all of the above. Prefer depth over breadth.

## Compute once, cache everything (team decision)
The data is fixed. Expensive stages (stitching, frame/tile extraction, detection, SAM masks, embeddings,
OCR, VLM calls) run **once** and skip themselves if their cached output exists with a matching
`manifest.json` (input hash + model + params + code version). Save detections at a **low threshold
(~0.05)** so threshold and rule changes never need the detector again. Cheap stages (tracking, roles,
interactions, anchoring, stats, export) always rerun from the cache. Cache = `data/interim/`
(gitignored); viewer/GIS outputs = `data/processed/`. Never rerun an expensive stage without need;
use `--force` only on purpose.

## METHODS.txt must stay current (user request)
`METHODS.txt` (project root) lists the algorithm/code used for each problem (stitching, stabilisation,
detection, tracking, smoothing, attributes, OCR, …). Update it in the same step as any method change.

## Running the pipeline (v0.1, levels 1–2)
**New machine: run `setup.bat`.** It checks the required programs (Python 3.13 + 3.9, ffmpeg, Tesseract
with Hebrew, curl, GPU), builds `.venv` and `.venv-imu`, installs the pinned `requirements.txt`, and downloads
the weights into `models/`. It's safe to rerun. Keep `requirements.txt` pinned to versions that were tested.
Environment: `.venv` (Python 3.13, torch 2.6 + CUDA 12.4 on the RTX 4060 laptop GPU, ultralytics, CLIP).
Weights live in `models/`. Downloads from GitHub work with curl; Python and git downloads sometimes hit
certificate errors, so fall back to `curl -L -o`.
```
.venv\Scripts\python pipeline\run_all.py [walk1 walk2]   # every stage; cached stages skip themselves
.venv\Scripts\python pipeline\serve.py                    # http://localhost:8000/viewer/  (+ /viewer/validate.html)
```
| Stage (script) | Cost | Output |
|---|---|---|
| `extract_frames.py` | ~4 min/walk | `data/interim/<walk>/frames/*.jpg` (3840×1920 equirect, 5 fps) |
| `encode_video.py` | ~4 min/walk (NVENC) | `data/processed/<walk>/video/equirect.mp4` (2880×1440) |
| `detect.py --model yolo11x.pt` | ~11 min/walk GPU | `data/interim/<walk>/det_yolo11x/detections.parquet` (raw, conf ≥ 0.05) |
| `detect.py --model yolov8x-worldv2.pt --prompts level2` | ~11 min/walk GPU | `det_yolov8x-worldv2_level2/` |
| `imu_extract.py` (run with `.venv-imu`, Python 3.9: telemetry-parser has only cp39 wheels) | seconds | `data/interim/<walk>/imu.csv` (1 kHz gyro + accel) |
| `stabilize.py` | ~3 min first time (visual rotations cached), then ~20 s | `data/processed/<walk>/stab.json` (per-frame correction rotation Q; the viewer applies it to video and overlays) |
| `tracks.py` | seconds–2 min | `tracks/persons.parquet`, `objects.parquet`, `level2.parquet` (tracked, gap-filled, zero-phase smoothed), `summary.json` |
| `attributes.py --variant stab` | ~3 min GPU, cached | `attributes_stab/attributes.parquet`: perceived gender (man/woman/uncertain) + age group per track. CLIP ViT-L/14 zero-shot on ≤ 5 isolated crops per track (neighbours greyed out); crops are not stored; aggregate display only |
| `ocr_signs.py --variant stab` | text detection GPU + Tesseract CPU, both cached | `text_det_stab/text_lines.parquet` (docTR db_resnet50, 16 tiles, every 2nd frame), `ocr_stab/signs.parquet` (Tesseract heb+eng psm 7 normal/inverted, best of 2 crops, then HebrewFinalFixer + SpellChecker copied from C:\Projects\OCR into `pipeline/ocr/`) |
| `export_overlays.py` | seconds | `data/processed/<walk>/overlays.json`, `observations.geojson` |
| `make_validation.py` | seconds | `data/processed/validation/` for the hand-count page |

Stages added after v0.1 (all wired into `run_all.py`; details and numbers in METHODS.txt):
| Stage (script) | Cost | Output |
|---|---|---|
| `ocr_vlm.py` | Claude reads numbered contact sheets (sub-agents) | `ocr_vlm_stab/signs.parquet` — the sign text actually used (Tesseract was too weak) |
| `georef.py` | ~1 min | walks on the map without GPS: gyro dead reckoning fitted to store anchors (`config/anchors.json`) + OSM map matching + centring between store rows → `<walk>/georef.json`, `shops.json` |
| `person_flow.py` | ~1.5 min/walk, cached | optical flow (DIS) of feet vs ground beside them |
| `motion.py` | seconds | people on the map + speed/direction, camera motion from static objects → `<walk>/motion_stab.json` |
| `depicted.py` | CLIP cached | pictured people → `possible_rabi_sign` (vision review in `config/depicted_review.json`) |
| `detect.py --model yolov8x-oiv7.pt` + `food.py` | ~13 min/walk GPU + CLIP | food displays: where (Open Images V7 + COCO), what (CLIP market vocabulary) |
| `detect.py --prompts waste` + `waste.py` | ~13 min/walk + SegFormer floor masks | waste on the floor (hidden) |
| `stall_materials.py` + `frontage.py` | crops + CLIP; vision review `config/stall_review.json` | frontage materials as locations → `frontage_materials.json` |
| `viewer/engine.js` | — | shared engine of the five design variants |

**Stitch:** ffmpeg `v360=input=dfisheye:output=e:ih_fov=190:iv_fov=190:roll=180` (tuned on walk 1;
Hebrew reads correctly). The seam is visible; Insta360 Studio would give cleaner seams.
**Geometry:** yaw/pitch conventions in `pipeline/spherical.py`. The viewer's shader and overlay
projection use the same maths. Tiles are 8 × 100° FOV at pitch −15°, 1024 px.
**Stabilisation:** the gyro is fitted to ORB-based visual rotations between frames, giving the axis mapping
and time offset (walk 1: lag 80 ms, corr 0.97/0.81/0.97; walk 2: lag 90 ms). The accelerometer reports
gravity, so its sign is flipped so that "up" is +y. Render-time correction: `d_cam = Q · d_view`, with
Q = Rᵀ·Ry(smoothed heading). Boxes are rotated into the stabilised frame with their own analysis frame's Q.
**Stabilisation is applied OFFLINE (team decision, 12 Oct 2026).** Render-time stabilisation in the viewer
shook badly and was removed. `bake_stab.py` writes the correction into the pixels
(`video_stab/equirect.mp4`, `frames_stab/`). Detection, tracking and export then run on the stabilised
frames (`--frames frames_stab`, `--variant stab` → `det_*_stab`, `tracks_stab/`, `overlays_stab.json`).
The viewer's "Stabilised video (offline)" switch flips between the variants, and the unstabilised
variant stays as the fallback. Measured on the baked file: high-frequency shake down ~87%, horizon level
(`data/processed/walk1/compare_stab_walk1.mp4` shows a side-by-side). If stabilisation is dropped, use
the unstabilised variant only.
**Timing is critical:** a ~70 ms error makes the correction amplify walking shake (it happened in v0.2).
`stabilize.py` therefore fine-tunes the time shift separately for the 30 fps viewer video (measured on
the encoded MP4: ≈ −80 ms) and the 5 fps frames (≈ 0 ms). `stab.json → check_deg_per_frame` must show
stabilised < raw. Re-check with `pipeline/stab_check_video.py` after any change to stitching or encoding.
**Box smoothing:** gaps of ≤ 1 s are filled; centre smoothed with σ = 0.3 s and size with σ = 0.6 s
(zero-phase). The viewer interpolates per track between 5 fps frames, fades tracks in and out, and hides
tracks < 1 s.
**Known gotchas:**
- the operator and their fingers are detected as people; `tracks.py` filters them
- Windows + Python's HTTP/1.0 server truncates large responses, so `serve.py` uses HTTP/1.1
- Chrome won't load video in hidden/background tabs; the viewer then falls back to the cached frames
- pandas `groupby().apply` drops the grouping column in this version; use explicit loops

## Detection strategy (team decision): build in four levels, easiest first
1. **Public classes:** YOLO11/COCO (people, bikes, vehicles, seating).
2. **Nameable objects:** open-vocabulary detection (Grounding DINO / YOLO-World / OWLv2) + SAM 2; fine-tune only weak classes (waste, crates, hand trucks, brooms).
3. **Properties of known places:** classify GIS frontage crops with a VLM (fixed label set, few-shot, JSON + confidence) and a CLIP probe as a cross-check (stall form, permeability, trade).
4. **Geometry and time:** rules on tracks and frontage lines (vendor inside/outside, roles, interactions, stops, spill-out, thresholds, flows). Logistics = level-2 tool + person + activity check.

Finish each level end to end (viewer + GIS export + measured accuracy) before starting the next. Full
table and exit criteria are in PLAN.md, under "Detection strategy".

## Hackathon scope (what we actually ship)

1. **Fuse and register.** Place the 360° video frames, photos and GIS shop data in one shared
   spatial frame, by location.
2. **Analyse frames offline.** Run off-the-shelf vision models (YOLO-type) to get people counts and,
   if time allows, materials (canvas, cardboard, metal framing).
3. **Demo viewer.** Play the 360° video with GIS shop tags overlaid in the scene (text, icon or
   colour). A **side panel updates as the viewer turns**, e.g. "64 people · 62% cardboard, 22% metal
   framing".

All processing happens offline. The demo plays precomputed results back *as if* live.
Out of scope / stretch only: LLM-based scene editing ("switch the houses"), style transfer, full 3D
reconstruction.

## Team

| Person | Strengths | Natural ownership |
|---|---|---|
| Alex | Image processing, segmentation, tracking across frames, custom-trained classifiers | Detection, tracking, materials |
| Boaz | Domain knowledge, field data, GIS, visual style | GIS data, ground truth, labels, narrative |
| Ido | Robotics MSc; classical ML/CV; proposed registration + YOLO | Registration, camera pose |

Logistics: meet in person Tuesday morning. Alex leaves at 14:00 Tuesday and is full-time on other
days. No fourth member.

## Data: known facts and gotchas

- **Data as supplied (12 Oct 2026, all in `data/raw/LABAI/`):**
  - **Videos:** both walks are complete. `1_Hatikvah St.insv` is V01, 3:28; `2_Hamevasser St.insv` is
    3:23. Both are raw dual-fisheye 3968×1920 with the Insta360 trailer. The camera was handheld, upside
    down, with the operator visible, so flip/level and mask the operator. The camera clock is wrong
    (03:40 at night for a busy market): **never use metadata time as time of day**.
  - **GIS:** only a **PDF map** (`Shops with Icons.pdf`, layer `Hatikvah_Shops_30_7_26`). The vector
    export has been requested from Boaz. Its legend gives the **trade-type vocabulary**.
  - **`shuk_360_frame_coding_v01.docx` is the team's canonical scene schema.** Use its JSON shape and
    vocabularies for VLM keyframe coding (age groups 18–35 / 36–60 / 61+ / indeterminate; roles
    shopper, passerby, vendor, worker/service, resting, socializing; "perceived gender expression").
    Its 9 coded V01 frames are few-shot and validation examples.
  - **Sketches** (Boaz 1–5 + 32 AI style images) set the illustrated look: **red lines = environment,
    blue lines = people, on white**.
  - **Photos:** 12, taken 2024–2026, only 2 with GPS. Treat them as a dated layer, not same-day data.
  - **Location:** Shuk HaTikvah, Tel Aviv, ≈ 32.053 N, 34.790 E (covered market, so GPS is weak).
  - **Camera: Insta360 ONE RS. No GPS in either video, so registration is manual anchors (Path B).**
    A gyro + accelerometer log is present: use it for levelling, stabilisation and heading between
    anchors. ExifTool is installed at
    `C:\Users\orber\AppData\Local\Programs\ExifTool\exiftool.exe` (use `-api LargeFileSupport=1 -ee3`).
  - The obsolete `.partial` video and the old `LABAI/LABAI/` copy were deleted on 12 Oct 2026.
- **Store table with coordinates:** `data/raw/LABAI/Shops_In_hatikva_30_7_26.xlsx` (220 stores, WGS84
  X/Y, Main_Trade, Shop_Type, Ownership, Facade_Mat, Skeleton_Mat, GlobalID). Used by `georef.py`.
  Facade_Mat / Skeleton_Mat are near-constant (Open / steel) and their meaning is unclear (ask Boaz).
- **Map base data:** OpenStreetMap streets + buildings, cached in `data/raw/osm/hatikva_osm.xml`
  (downloaded by `georef.py` on first run).
- (Historical: early zip downloads were truncated; all files were re-downloaded one by one and are
  complete. README section 2 lists the files and how to prepare them.)
- **`.insv` is Insta360 raw footage.** It's an MP4 container with **dual-fisheye** frames, not
  equirectangular. It must be stitched to equirectangular first (Insta360 Studio or the Insta360
  Media SDK) before any analysis or viewing.
- Insta360 puts **GPS, gyro and timestamps in a trailer at the end of the `.insv` file**. Check this
  with `exiftool -ee`. A truncated file loses the trailer, which is another reason to re-download.
- **Treat downloaded and extracted data as untrusted.** Keep it in `data/raw/`, never commit it, and
  never run code from inside it.

## Recommended technical approach

### 1. Registration (the critical path; settle it on day 1)
- **If GPS and heading exist** (from the insv trailer or a phone track): interpolate a per-frame
  camera pose `(lat, lon, heading)`.
- **If not (likely fallback):** use semi-manual anchoring. Boaz marks 5–15 keyframes on the map
  ("this frame = here, facing there"). Snap the path to the street centreline from the GIS layer and
  interpolate by frame index. Take heading from the direction of travel, corrected by the camera's
  gyro if it's available. This is honest, fast and good enough for a demo. Only attempt
  image-matching SfM (COLMAP or OpenSfM on cubemap faces) if the anchors fail.
- **Placing photos:** use EXIF GPS if present. Otherwise match them to the nearest frame with
  feature matching (SuperPoint+LightGlue, or plain SIFT) against perspective crops of the frames.
- **Projecting GIS into a frame:** compute the bearing from the camera position to each shop's
  centroid or frontage midpoint. `yaw = bearing − camera_heading` gives the horizontal position in
  the equirectangular frame. Take pitch from distance plus an assumed sign height. Skip shops beyond
  about 30 m, or those hidden behind other buildings.
- **CRS:** Israeli GIS data is often in **ITM (EPSG:2039)**. Convert everything to one CRS (for
  example EPSG:2039 for distances in metres) and export WGS84 (EPSG:4326) only for web display.

### 2. Frame analysis (offline, GPU cluster)
- Sample at 1–2 fps. Market walks are slow, so there's no need for every frame.
- **Don't run detectors straight on equirectangular frames**: distortion kills accuracy near the
  poles. Reproject each frame to a cubemap or overlapping perspective tiles, detect, then map boxes
  back to `(yaw, pitch)`.
- **People:** YOLOv8/YOLO11 `person` class plus a tracker (ByteTrack or BoT-SORT) to avoid counting
  the same person twice across frames and tiles. Report both "visible now" and "unique in segment".
- **Materials:** off-the-shelf YOLO has no "cardboard" or "canvas" class. Prefer **open-vocabulary**
  models (YOLO-World, Grounding DINO + SAM, or CLIP patch classification) with prompts like
  "cardboard", "canvas awning", "metal frame", "plastic sheet". Validate them on about 50 frames Boaz
  labels by hand. Report results as **% of segmented facade/stall area**, not object counts.
- **Sex classification (man / woman): in scope, team decision.** Use body-based pedestrian attribute
  recognition on person crops (faces are too small in 360° footage). Vote per track, include an
  "uncertain" class, and run it before face blurring, storing only `{track_id, label, conf}`. Planned
  as aggregate-only; **user decision 13 Oct:** perceived gender is also shown per person as the emoji
  badge background (blue man / pink woman / grey uncertain), labelled as an estimate. Judges will ask.
  (Implemented with CLIP ViT-L/14 zero-shot, `attributes.py`; not yet validated.) **Blur faces in anything we display or commit.** Don't store face or person crops.
- **Text on signs (OCR) — superseded:** Tesseract proved too weak; the user then allowed any offline
  method. Current method: docTR text detection + Claude reading numbered contact sheets
  (`ocr_vlm.py`, 2,244 signs, spot check 16/16). The original Tesseract plan is kept below for reference.
- (Original plan) **Text on signs: Tesseract only** (team decision; no Claude OCR or post-processing for now).
  Mostly Hebrew, some English, and expect Arabic in Jaffa. Reuse the user's earlier project at
  `C:\Projects\OCR`:
  - copy `ocr_base.py`, `ocr_tesseract.py`, `ocr_doctr.py` (detection head only, since docTR can't
    read Hebrew), `preprocessor.py`, `spellchecker.py` and `frequency_lists/hebrew/` into
    `pipeline/ocr/`. Don't import them from `C:\Projects`.
  - That project was built for scanned pages. Here, detect text regions first, rectify and upscale
    each crop, then run Tesseract `heb+eng` with `--psm 7/6`.
  - Vote across the sharpest frames of each sign, then apply `HebrewFinalFixer` and `SpellChecker`.
    Add GIS shop names and market vocabulary to the Hebrew dictionary.
  - Fuzzy-match the result to GIS shop names, which also checks the registration.
  - Known gaps in the old code: `ocr_tesseract.py` has no `hebrew_english` → `heb+eng` mapping, the
    engines return only a string (we need boxes and confidence), and `spellchecker.py` uses the
    deprecated `pkg_resources`.

### 3. Data contract (the boundary between pipeline and viewer)
The pipeline writes JSON, and the viewer only reads it. Keep this schema stable:
```
frames/{video_id}.json : [{ t, frame_idx, lat, lon, heading_deg,
                            detections: [{cls, yaw, pitch, w, h, conf, track_id?}],
                            materials:  [{cls, yaw_min, yaw_max, area_frac}] }]
shops.geojson          : GIS shops (id, name, owner/type, category, geometry) in EPSG:4326
overlays/{video_id}.json : per frame, visible shops with {shop_id, yaw, pitch, dist_m}
texts/{video_id}.json  : [{ sign_id, frames: [frame_idx…], yaw, pitch, w, h, text, raw,
                            lang: "he"|"en"|"ar"|"mixed", conf, psm, fixes: [...], shop_id? }]
# [Brief] GIS-ready, one row per observation (also exported as observations.gpkg / .csv)
observations/{video_id}.geojson : Feature{ geometry: Point|LineString (EPSG:4326),
    properties: { obs_id, entity_id (P-####|S###|F###|T###|O####|I####|L####), entity_type,
                  layer: social|material|spatial|logistics|ocr|gis, class, attributes{…},
                  frame_start, frame_end, t_start, t_end, chainage_m, confidence, model, walk_id } }
```
ID prefixes: P person track · S GIS shop · F frontage · T sign · O object (waste, crate, seating,
obstacle) · I interaction · L logistics event. Static entities keep their ID across walks
(matched by world position).
**Side panel logic:** filter the current frame's detections to the viewer's field of view
(`|yaw − view_yaw| < hfov/2`) and aggregate. It's cheap enough to compute client-side.

### 4. Viewer
- **As built:** plain WebGL equirect shader + 2D overlay canvas (no Pannellum), Leaflet maps, served
  by `pipeline/serve.py`. Original `viewer/index.html`; five redesigns on `viewer/engine.js`.
- (Original plan) A web app. Use **Pannellum** or **Marzipano** for the 360° video, or three.js with an
  equirectangular video texture if we need custom overlays. Also show a mini-map (Leaflet) with the
  walk path and the current position and heading cone.
- Overlays: shop labels or icons, coloured by category or ownership, plus an optional toggle for the
  detection layer. This is the seed of Boaz's "switchable layers" idea, so present it that way.

## Suggested repo layout
```
data/raw/        # untouched downloads (gitignored)
data/interim/    # stitched equirect videos, sampled frames (gitignored)
data/processed/  # JSON/GeoJSON outputs consumed by viewer (small, may commit)
pipeline/        # register.py, sample_frames.py, detect.py, materials.py, project_gis.py
viewer/          # static web app
notebooks/       # exploration only
```
Python 3.11, `ultralytics`, `opencv-python`, `geopandas`, `pyproj`, `shapely`, `py360convert`.
Pin versions in `requirements.txt`. Scripts must be runnable both on the GPU cluster and locally (on
CPU, with a `--limit` flag).

## Open questions (resolve early, in this order)
1. ~~GPS in the videos?~~ Answered: no GPS; gyro yes. Registration = store anchors + gyro + OSM map matching.
2. ~~GIS format?~~ Answered: store table xlsx in WGS84 (no frontage geometry yet).
3. ~~Camera?~~ Answered: Insta360 ONE RS, single-file dual-fisheye 3968×1920.
4. Can Boaz label about 50 frames for materials, to validate the open-vocabulary output?
5. Sex classification: what confidence threshold for "uncertain", and can Boaz label ~100 tracks to validate it?

## Working rules for Claude in this repo
- Build the thinnest end-to-end slice first: one walk → a few anchored frames → people counts →
  viewer with one shop tag and a working side panel. Then widen.
- Never commit raw media, face crops or personal data from interviews.
- When unsure about the domain (what a stall type means, who owns what), ask Boaz rather than guess.
- When stating a metric in the demo, know its source and error. Judges will probe.
