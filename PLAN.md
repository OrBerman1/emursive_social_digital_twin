# PLAN — Shuk HaPishpeshim 360° Digital Twin (Hackathon)

_Source: hackathon_kickoff_summary.pdf (11 Oct 2026). Conventions & data contract: see CLAUDE.md. Visual mockup: mockup/viewer_mockup.html_

_Updated from **Brief.pdf** (12 Oct 2026): additions are marked **[Brief]**. Nothing from the earlier plan was removed._

> **[Brief] Which market?** The brief names **Shuk HaTikvah** (Tel Aviv, a covered produce market) and calls the project a **"Social Immersive Digital Twin"**. The kickoff said Shuk HaPishpeshim (Jaffa). The video file name `1_Hatikvah St.insv` fits the brief. **Confirm with Boaz.** If it's HaTikvah, the material and waste classes lean towards produce (crates, organic waste, cardboard boxes). The Arabic-signage rationale (Jaffa) gets weaker, but Arabic stays in the OCR, which costs nothing.
> New data has arrived: `OneDrive_2_10-12-2026.zip` (1.3 GB, not yet extracted or checked).

## [Data] What the supplied data contains, and what it changes (reviewed 12 Oct 2026)
All data is in `data/raw/LABAI/` (7.4 GB). Additions from this review are marked **[Data]**.

| Item | What it is | Key facts |
|---|---|---|
| `1_Hatikvah St.insv` | Walk 1, raw Insta360 | 3 min 28 s (6,244 frames at 29.97 fps); **dual fisheye side by side, 3968×1920**, H.264; Insta360 trailer present. Coded as `V01` (original name `VID_20250825_034000_00_001`) |
| `2_Hamevasser St.insv` | Walk 2, raw Insta360 | 3 min 23 s (6,078 frames); same format; trailer present. *(The `.partial` copy next to it is obsolete.)* |
| `Shops with Icons.pdf` | GIS shop layer **as a PDF map only** | Layer `Hatikvah_Shops_30_7_26`, 1:1,944, an Esri basemap with Hebrew street names. Shop points carry a **trade category** (≈40 classes, below). The market is a **T-shaped spine along Hatikvah St and Hamevasser St**, which matches the two walks |
| `shuk_360_frame_coding_v01.docx` | **The research team's frame-coding scheme**: 9 frames from V01 coded, with a full JSON appendix | Sections: frame metadata · spatial map · cultural map (+ a per-person table) · material map · composite indices (legibility / imageability) · confidence notes. 79 people coded |
| `Sketches/` | 5 drawings by Boaz + 32 AI images in his style | **Two-colour pencil: red = built environment, blue = people**, on white; Hebrew signs, price tags, time/date stamp (e.g. "12:40 21/9/25") |
| `Photos/` | 12 field photos | Taken **2024-03 → 2026-02** (not the walk date). **Only 2 of 12 have GPS** (32.0524–32.0539 N, 34.7903–34.7908 E, i.e. the Hatikvah quarter) |

**[Data] Metadata check with ExifTool 13.59 (12 Oct 2026). This settles Phase 0's key question:**
- **Camera: Insta360 ONE RS** (firmware v2.0.11). Supported by Insta360 Studio, the Insta360 MediaSDK and Gyroflow.
- **No GPS records in either video.** → **Registration is Path B (manual anchors)**, as the covered market already suggested. Path A is off the table.
- **Gyroscope + accelerometer present** (high-rate IMU log in a 5.8 MB trailer per file, plus per-frame exposure and timecode). Use it for:
  1. horizon levelling and stabilisation (Insta360 Studio FlowState or Gyroflow)
  2. **camera heading between anchors** by integrating gyro yaw, which is much better than "direction of travel" when the operator turns the camera
  3. detecting when the operator stopped (accelerometer), to time the walk and check pace for chainage interpolation
- ExifTool is installed at `C:\Users\orber\AppData\Local\Programs\ExifTool\exiftool.exe`. Use `-api LargeFileSupport=1 -ee3` to dump the IMU. telemetry-parser/Gyroflow read the full-rate log directly.

**[Data] What it changes in the plan**
- **Location:** the data confirms **Shuk HaTikvah, Tel Aviv** (≈ 32.053 N, 34.790 E), not Jaffa. The open question in the brief note above is answered by the data; still worth a quick confirmation with Boaz.
- **Camera timestamps are unreliable.** The files say 25 Aug 2025 **03:39 and 03:44 local**, but the footage shows a busy daytime market, so the camera clock is wrong. Don't derive time of day from metadata; the coding scheme also leaves `time_of_day` as "indeterminate". Ask Boaz for the real time.
- **Phase 1, stitching:**
  - The camera was **handheld, mounted upside down, with the operator's hand and body in view**. Stitch and **flip/level the horizon** (Insta360 Studio does this; Gyroflow can use the trailer's gyro data).
  - **Mask the operator.** Remove the fixed rear/nadir region, and drop the track that stays at a constant small distance in every frame. Otherwise the operator is counted as a person in all 6,000 frames.
  - The effective equirectangular resolution is only about **3840×1920**, so OCR and small-object detection work only on nearby stalls. Choose the best frames per sign or object, as planned.
- **Phase 2, lining up video and map:**
  - It's a **covered market** (permanent roof, as the coding scheme says), so any GPS in the trailer will be weak or missing under the roof. **Expect Path B (manual anchors).**
  - Good anchors: the **T-junction** where Hatikvah and Hamevasser meet, both street ends, and distinctive GIS shops (fish monger, herbs, cafés).
  - Camera height is unknown (handheld, probably 1.2–1.5 m). Estimate it once from a known dimension (counter ≈ 0.9 m, aisle width from GIS) before using flat-ground distances.
- **GIS: the vector layer is missing.** Ask Boaz to export `Hatikvah_Shops_30_7_26` (ArcGIS) as **GeoPackage, shapefile or GeoJSON, or CSV with lat/lon** and all fields (name? category, ownership?).
  - *Fallback:* georeference the PDF in QGIS using OSM street intersections, then digitise the ~150 points along the two streets (~2 h). Or extract the vector symbols with PyMuPDF `get_drawings()` if the export kept them as vectors.
- **Trade-type vocabulary = the GIS legend.** Use these as the label set for frontage trade type (level 3) instead of the invented list: Acting school, Bakery, Barber, BBQ equipment, Bikes & repair, Butcher, Café, Change / money transfer / small electrics, Cheese, Clothing, Coffee tea & dried goods, Deli, Dentist, Dried fruit, Drinks, Dry goods, Eggs, Electronics, Fish monger, Flowers, Former-USSR mini-market, Fruit/vegetables, **Hamara (informal)**, Hardware, Herbs, Household, Jewellery, Kitchen, Ladies', Laundromat, Mail, Members, Minimarket, Mixed, Nuts… *(the legend is cut off in the PDF; get the full list with the export).*
- **The coding scheme becomes the canonical scene-level schema** (it complements our per-entity observations):
  - The VLM fills **exactly this JSON** for sampled keyframes (e.g. every 5 m of route). The input is **perspective reframes** (forward / left / right / back), like the coder's 16:9 screenshots, not the warped equirectangular image. Each keyframe is anchored at its chainage, so legibility, imageability, density, materials and so on become **profiles along the route**.
  - **The 9 coded frames are the few-shot examples and the first validation set.** Locate them in V01 by image matching (DINOv2/CLIP retrieval over reframed frames, then a LightGlue check) to get their timestamps.
  - **Align our vocabularies with the team's:**
    - age groups: *young adult 18–35 · middle adult 36–60 · older adult 61+ · indeterminate*
    - roles: *shopper · passerby · vendor · worker/service · resting · socializing · indeterminate*
    - gender: their term is **"perceived gender expression"** (masculine / feminine / indeterminate). Use that name for our men/women task; the method stays the same.
  - **Attributes in the scheme that weren't in the plan:** `informal_vendor_present` (also a GIS class, "Hamara – Informal"), `seating_infrastructure` (moveable / informal perching), `accessibility` (step-free / steps / obstructions), `safety_cues`, `adaptability_cues`, `depth_layers`, `walkability_notes`, `walking_tour_potential`, `dominant_colour_palette`, `surface_condition`, `structural_skeleton_indicators`, and the **legibility and imageability indices** (Lynch).
    - `dominant_colour_palette` can be computed from pixels (k-means), no VLM needed.
- **New level-2 detection classes, prominent in the footage and coding:** **walkers/rollators, mobility scooters, wheeled shopping trolleys** (central to accessibility and the older-shopper population), plus **flags** and overhead lighting (legibility cues).
- **Illustrated view, made concrete:** Boaz's style is **red lines for the built environment and blue lines for people, on white paper**. That gives a cheap, deterministic, flicker-free renderer:
  - **edges of the environment** (Canny/HED, or informative-drawings line art) in red
  - **edges of person masks** (YOLO-seg) in blue
  - paper-texture background
  - Diffusion with IP-Adapter, using his 5 drawings plus the 32 style images, stays as the stretch goal for a richer look.
- **Photos and sketches are a temporal layer.** They span 2024–2026, so treat them as **"researcher photos & drawings" pins with their date**, placed by Boaz (only 2 photos have GPS). They're not used for counts. This supports the brief's "changing system" framing.
- **Analysis volume:** ~12,300 video frames in total. At 2 fps that's ~830 analysed frames, and at 5 fps ~2,000, which is minutes to under an hour on the GPU cluster.

**[Data] Requests to Boaz (send Tuesday morning)**
1. Vector export of `Hatikvah_Shops_30_7_26` with all attribute fields (and the full legend).
2. The real time of day of both walks, and where they start and end on the map.
3. Approximate camera height (handheld?). *(GPS is settled: the files contain none.)*
4. Locations (and dates) for the 10 photos without GPS and the 5 drawings.
5. Is the frame-coding scheme v01 final? Who or what produced it (a VLM?), and can the team code ~20 more frames for validation?

## [Brief] What the brief adds, at a glance
Proof-of-concept chain to demo: **360° field recording → AI/VLM observation → spatial anchoring → immersive visualisation → GIS-ready social/material/spatial data.**

| Brief requirement | Already in plan | Added now |
|---|---|---|
| 1. One integrated immersive viewer, info anchored in the scene | 360° viewer, in-scene shop tags, people boxes, OCR, layers | **Info cards open inside the view, anchored to the object**; an immersive mode that hides the side panel; **move along the route**, not only rotate; **switching between 360° photo, simplified digital and illustrated views** while keeping all overlays |
| 2. Persistent spatial identification | track IDs, shop IDs, sign IDs, JSON contract | A **unified observation table** with stable IDs for every entity type: shop ID/type, **position along the route (chainage, m)**, timestamp/frame, confidence, lat/lon. **Export to GeoJSON, GeoPackage and CSV** |
| 3a. Social layer | count, density, tracking, men/women | **Broad age groups**, **role** (vendor / shopper / passer-by / worker), **interactions and their type**, **gathering and stopping points** |
| 3b. Material layer | awning materials, signage (OCR) | **Stall morphology / trade type** (walk-in shop, open stall, counter, temporary table), **vendor inside/outside**, **shop-front permeability**, display materials, **waste: cardboard, plastic, organic, crates, packaging** |
| 3c. Spatial layer | GIS streets, frontages | **Circulation** (flow lines), **thresholds**, **junctions**, **shop spill-out**, **obstacles**, **seating**, **active frontages**, **interaction hotspots** |
| 4. Operational/logistics detection | none | **Delivery workers, porters, crate collectors, sweepers/cleaners, waste handlers, loading/unloading, stock movement, e-bike deliveries**, tagged in space and time ("operational metabolism") |
| 5. Interaction with the information | cards in the side panel; layer checkboxes | Cards **in the immersive view** for shops, people, objects and signs. **Layer names follow the brief:** people/interactions, shops, materials/waste, logistics/operations, GIS tags, researcher drawings, plus our OCR and spatial layers |
| Key research test | none | A **research questions** module computed from the GIS table (see Phase 5b) |

## Detection strategy: four levels, built in this order (team decision, 12 Oct 2026)
Development of the analysis layers follows these levels **from easiest to hardest**. Get each level working end to end in the viewer and the GIS export before starting the next, so there is always a working demo.

| Level | What it covers | Method | Brief items |
|---|---|---|---|
| **1. Public classes** | Objects pre-trained detectors already know | YOLO11 (COCO) + ByteTrack on undistorted cubemap tiles | people, bicycles / e-bikes, motorcycles, vans / trucks, chairs / benches (seating) |
| **2. Nameable objects without a public class** | Objects you can describe in a text prompt | Open-vocabulary detectors (Grounding DINO / YOLO-World / OWLv2) + SAM 2 masks. Validate on ~50 labelled frames; fine-tune YOLO-seg on 100–300 SAM-assisted labels only for weak classes | crates, cardboard, plastic, organic waste, packaging, hand trucks, brooms, wheelie bins, delivery boxes, display tables, parked carts; awning / display materials |
| **3. Properties of known places** | Things that aren't objects but attributes of a frontage | Crop each GIS frontage from its 2–3 best frames → **classify** (VLM with a fixed label set + Boaz's few-shot examples, JSON + confidence; cross-check with a CLIP linear probe trained on ~20–30 labels per class) | stall form (walk-in shop / open stall / counter / temporary table), trade type, shop-front permeability, display materials, signage (OCR) |
| **4. Geometry and time** | Facts computed from tracks, ground positions and the GIS frontage lines; no new model. The VLM only labels what rules can't decide | Rules on track position, speed, distance and duration | vendor inside/outside, vendor / shopper / passer-by, interactions (+ type via VLM), stopping points, spill-out depth, obstacles / clear width, threshold crossings, circulation flows, active frontages, hotspots; **logistics roles** = level-2 tool + person moving together + activity check (X-CLIP / VideoMAE) |

**Build order and exit criteria**
1. **Level 1:** people tracked, counted and placed on the route; seating and vehicles detected; IDs persist; GeoJSON export works. *Exit: people count within ±15% of a manual count on 30 frames.*
2. **Level 2:** waste, crates and logistics tools detected and anchored on the ground. *Exit: per-class precision measured on ~50 labelled frames; weak classes either fine-tuned or marked "experimental".*
3. **Level 3:** every frontage in the segment has a stall form, permeability and trade label with a confidence. *Exit: agreement with Boaz's labels on ~50 frontages reported.*
4. **Level 4:** roles, interactions, stops, spill-out, thresholds, flows and logistics events computed, which unlocks the research questions. *Exit: rules checked on ~100 tracks and ~20 logistics events; at least 2 research questions answered from the GIS table.*

**Rules for every level:** every label carries a confidence and an "uncertain" option, VLM answers must come from the fixed label list, rules cross-check model labels, and the accuracy measured on Boaz's labels is shown in the demo. **Labelling starts Tuesday**, since it's the main constraint for levels 2–4.

## Progress log
- **12 Oct 2026, v0.1 (levels 1–2) built on real data.** Pipeline in `pipeline/` (see CLAUDE.md → "Running the pipeline"). Both walks stitched (ffmpeg dual-fisheye, 5 fps cache) and encoded for the browser. YOLO11x level-1 detection on 8 undistorted tiles per frame, cached. Persistent person tracks with operator removal and camera-height self-calibration (≈1.0–1.1 m, handheld). Real-data viewer at `viewer/index.html` (WebGL 360° + anchored boxes, IDs, distances, in-view card, layers, confidence slider). Hand-count page at `viewer/validate.html`. Level 2 (YOLO-World, 28 prompts) running.
  - First numbers: walk 1 has 15.6 people per frame on average (max 30), 916 tracks; walk 2 has 4.9 per frame, 354 tracks. Track fragmentation is still noticeable (53% of tracks < 2 s), so **unique-people counts are overestimates until validated**.
  - Open: hand-count validation (30 frames), level-2 quality check, Boaz's anchors → real chainage and lat/lon, cleaner stitch (Insta360 Studio) if the seam hurts.
- **12 Oct 2026, v0.2:** level 2 done on both walks (trolleys, cardboard, crates, bins, e-bikes, mobility scooters, hand trucks, lamps, flags). Boxes are now tracked, gap-filled and smoothed (people and objects), interpolated to 30 fps in the viewer, with fades, a 1 s minimum track length and decluttered labels. **Gyro stabilisation + horizon levelling** comes from the 1 kHz IMU (axis mapping and 80–90 ms time offset fitted to visual rotations), applied at render time to the video and overlays together (toggle in the viewer).
  - Next: use the stabilised frame inside the tracker (camera-motion compensation) to cut fragmentation; level-2 class cleanup (lamps/flags as separate "legibility" layer); validation counts.
- **12 Oct 2026, v0.3, offline stabilisation:** render-time stabilisation shook badly (timing mismatch in the browser) and was removed. Stabilisation is now **baked into the video and frames** (`bake_stab.py`, high-frequency shake −87%, horizon level; side-by-side `compare_stab_walk1.mp4` approved by the user). Detection L1+L2, tracking and smoothing were **rerun on the stabilised frames**. Tracking improved: walk 1 has 675 tracks instead of 916 (median length 18 frames instead of 14), walk 2 has 284 instead of 354. The camera-height estimate on levelled frames is 1.37 m (walk 1) / 1.20 m (walk 2), more trustworthy than the tilted-frame values. The viewer defaults to the stabilised variant; the original remains as a switchable fallback.
- **13 Oct 2026, v0.4:** perceived gender + age group per track (CLIP ViT-L/14 zero-shot on isolated crops; walk 1: 387 man / 83 woman / 194 uncertain, aggregate only, not yet validated); operator filter in camera directions also on stabilised frames. OCR built (docTR text detection + Tesseract heb+eng + user's final-letter fixer/spellcheck), but recognition is weak (spot check ~3/24 fully correct), so a Claude-vision reader is next (needs an API key). Viewer: default confidence 0.6, speed ×0.25–×2, gender/age panel, OCR layer, `METHODS.txt` added.
- **13 Oct 2026, v0.5, sign reading by vision model:** Tesseract was too weak, so signs are now read by Claude vision (background agents in the session, no API key) from one crop per sign on numbered contact sheets. 2,244 signs read (walk 1 1,151 / walk 2 1,093) instead of 177; spot check 16/16 correct. Operator's T-shirt (683 crops) and no-text crops removed. Viewer counts now show in-view / whole-walk totals by unique track ID (e.g. 13 / 565).
- **13 Oct 2026, v0.6, interactive map (branch `mapping`):** the walks are placed on the map without GPS: gyro dead reckoning fitted to 14 team-provided store anchors (`georef.py`, median bearing error ~2 deg). The viewer has a GTA-style heading-up minimap (route, all store dots, view cone) and a full map of all 220 stores; clicking a store jumps to the walk/time/yaw where it is seen best (137 stores reachable). To improve: more anchors in walk 1's middle stretch; check the two outlier anchors.

## [Next] Food, floor waste, stall materials, emoji labels, age on boxes (planned 13 Oct 2026; not implemented)
All three detection items are offline, cached stages on the stabilised frames, following the existing pattern
(`detect.py`-style tiles → raw detections at low conf → tracking + smoothing → export → viewer layer).

### A. Food on display (level 2 → a "Produce & food" layer)
| Option | What | Fit |
|---|---|---|
| **1. Open Images V7 YOLOv8x** (`yolov8x-oiv7.pt`, Ultralytics, 600 classes) | Checked in the installed config: **Apple, Artichoke, Banana, Bell pepper, Broccoli, Cabbage, Cantaloupe, Carrot, Common fig, Cucumber, Grape, Grapefruit, Lemon, Mango, Orange, Peach, Pear, Pineapple, Pomegranate, Potato, Pumpkin, Radish, Strawberry, Tomato, Watermelon, Zucchini, Vegetable, Fruit**, plus **Bread, Bagel, Baked goods, Cake, Cookie, Muffin, Pastry, Candy, Cheese, Egg (Food), Fish, Seafood, Honeycomb, Juice, Coffee, Drink, Fast food, Food** | **Recommended first.** Same runner as level 1; ~11 min/walk GPU. Also gives Box, Plastic bag, Tin can, Waste container, Shelf, Table, Tent (useful for B and C). Box mAP is modest (36.3 for the whole model), so validate per class |
| 2. Already in our cache | YOLO11x (COCO) found banana, apple, orange, carrot, broccoli (25k banana boxes on walk 1) | Free: just expose them as a layer now |
| 3. YOLO-World prompts | Market items OIV7 lacks: "eggplant", "herbs bunch", "dates", "nuts and seeds", "spices in sacks", "olives", "pita bread", "raw meat", "whole fish on ice" | Add to the level-2 prompt set; check on samples |
| 4. Vision model on stall crops (level 3) | "Which goods does this stall sell?" per GIS frontage | Gives the **trade type** per stall, matching the GIS legend (Fruit/Vegetables, Butcher, Fish Monger, Bakery, Spices, Nuts…). Reuse the contact-sheet reading method from OCR |

- **Unit of analysis:** count produce as **display areas per frontage** (dominant goods + share), not individual fruits. Hundreds of oranges in one stack mean nothing as a count.
- **Validation:** Boaz/team label ~30 stall crops with their goods; check top-3 agreement.

### B. Waste on the floor (level 2, "Materials & waste" layer)
| Option | What | Fit |
|---|---|---|
| **1. Ground restriction + existing detections** | Keep only waste boxes whose bottom is on the walkway. Pitch below the horizon, and inside the floor mask from semantic segmentation (SegFormer/OneFormer on ADE20K: classes floor, road, sidewalk) | **Required for any option:** separates waste on the floor from goods on stalls |
| 2. Our YOLO-World prompts (already cached) | cardboard box, flattened cardboard, plastic bag, organic waste, fruit and vegetable waste, packaging, garbage bag, trash bin | Already computed; re-filter with option 1 |
| 3. TACO-trained litter models | [CatSat/yolov11-litter-materials](https://huggingface.co/CatSat/yolov11-litter-materials) (YOLOv11-seg; 60 TACO classes → plastic / metal / glass / paper / other; mask mAP50-95 0.19); [turhancan97/yolov8-segment-trash-detection](https://huggingface.co/turhancan97/yolov8-segment-trash-detection) (YOLOv8-seg, TACO + TrashNet, MIT; pickle file flagged as unsafe → load only after inspection); [jeremy-rico/litter-detection](https://github.com/jeremy-rico/litter-detection) (YOLOv8 on TACO) | TACO is small (1,500 images) and photographed close up, so expect weak recall on 360 frames. Use as a second opinion; ensemble with option 2 |
| 4. OIV7 classes | Plastic bag, Box, Tin can, Waste container | Comes free with A1 |
| 5. Organic waste | Squashed produce and leaves on the floor: hard for every model | Open-vocabulary detection ("crushed vegetables on the ground", "lettuce leaves on the floor") inside the floor mask, plus a vision-model check on the crops |

- **Output:** waste objects with a ground position and approximate area (m²), and **waste density per 10 m of route**. This feeds the brief's question "where do waste, logistics and social activity overlap".
- **Validation:** ~50 floor crops labelled waste/clean by type. Reading time of day from the footage matters too: waste builds up through the day.

### C. What the stalls are made of (level 3, per GIS frontage)
| Option | What | Fit |
|---|---|---|
| **1. Vision model on frontage crops** | Fixed label set: corrugated metal, painted render, steel frame, wood/timber, canvas/fabric awning, plastic sheeting, crates as display, glass, ceramic tile, composite panel. The vocabulary is taken from the team's coding scheme (`facade_materials`, `structural_skeleton_indicators`, `surface_condition`) | **Recommended.** The same contact-sheet workflow as OCR, no API key. Output: multi-label + dominant material per stall + confidence |
| 2. Apple **DMS** dense material segmentation | Pretrained model, 46 materials (metal, wood, fabric, plastic, cardboard, glass, ceramic, concrete…); [paper](https://arxiv.org/pdf/2207.10614): 0.42 mean IoU | Pixel-level shares per frontage ("62% metal, 22% fabric"). Non-commercial licence; weights from github.com/apple/ml-dms-dataset |
| 3. MINC (Materials in Context, 23 categories) | Classic patch classifier ([paper](https://ar5iv.labs.arxiv.org/html/1412.0623)) | Older; fallback/cross-check |
| 4. CLIP linear probe | ~20–30 labelled frontage crops per material | Cheap cross-check of option 1 |

- **Depends on:** GIS frontage geometry (vector export requested from Boaz) to crop each stall consistently. Until then, use frontage crops around the detected stall areas.
- **Validation:** compare with the 9 coded frames' `facade_materials`, plus ~30 labelled frontages.

### D. Viewer: emoji instead of text tags on boxes
- **Status 13 Oct 2026: DONE** (emoji badges on all box types; numbers and sign text moved to the click card; card flips to stay inside the view).
- **Change:** each box gets a **small emoji badge** at its top-left instead of the text label. The full label (class, ID, confidence, distance) moves to the **hover tooltip and click card**. A legend lists emoji → class. This cuts clutter, which is the main readability problem now.
- **Rendering:** canvas `fillText` with the system emoji font ("Segoe UI Emoji" on Windows). Badge size scales with box size and is hidden below a minimum box height. Fall back to a 2-letter code if a glyph is missing.
- **Proposed mapping** (to review with the team):

| Classes | Emoji |
|---|---|
| person | 🧍 (base), age variants: see E |
| bicycle / e-bike / motorcycle / car / truck / bus | 🚲 / 🛵 / 🏍️ / 🚗 / 🚚 / 🚌 |
| chair, plastic chair, stool, bench | 🪑 |
| handbag / backpack / suitcase / umbrella | 👜 / 🎒 / 🧳 / ☂️ |
| wheeled shopping trolley, trolley cart, hand truck | 🛒 |
| walker / wheelchair / mobility scooter | 🦯 / 🦽 / 🦼 |
| cardboard box, flattened cardboard, packaging, delivery box, plastic crate, stack of crates | 📦 |
| plastic bag / garbage bag | 🛍️ / 🗑️ |
| trash bin, waste container | 🗑️ |
| organic waste, fruit and vegetable waste | 🍂 |
| broom | 🧹 |
| pallet, display table | 🪵 |
| flag / hanging lamp / awning | 🚩 / 💡 / ⛱️ |
| price sign / shop sign / OCR text | 🏷️ / 🪧 / 🪧 |
| food (OIV7/COCO) | 🍎 apple · 🍌 banana · 🍊 orange · 🍋 lemon · 🍇 grape · 🍓 strawberry · 🍅 tomato · 🥔 potato · 🥕 carrot · 🥒 cucumber · 🎃 pumpkin · 🍉 watermelon · 🍐 pear · 🍑 peach · 🍍 pineapple · 🥭 mango · 🫑 bell pepper · 🥬 cabbage · 🥦 broccoli · 🍈 cantaloupe · 🍞 bread · 🥯 bagel · 🧁 muffin/pastry · 🍰 cake · 🍪 cookie · 🍬 candy · 🧀 cheese · 🥚 egg · 🐟 fish · 🦐 seafood · 🍯 honeycomb · 🧃 juice · ☕ coffee · 🥤 drink · 🍔 fast food · 🧺 fruit/vegetable/food (generic) |

### E. Viewer: age group shown on each person's box
- **Status 13 Oct 2026: DONE.** Box colour = age group on a single green ramp (light → dark with age; grey = uncertain), same colours in the panel's age graph; age emoji badge; age group in the person card; switch "Colour people by age". Update 13 Oct (user decision): perceived gender is now also shown per person as the emoji badge background (blue man / pink woman / grey uncertain), with switch "Gender on emoji badge"; mention on the ethics slide together with validation status.
- **Change:** **box colour = age group**, with a legend in the panel. Optionally, the person emoji varies by age: 🧒 child · 🧑 young adult 18–35 · 🧔 middle adult 36–60 · 🧓 older adult 61+ · 👤 uncertain. Colour palette (colour-blind safe, distinct from the other layers): child violet, young adult green, middle adult blue, older adult orange, uncertain grey.
- **Policy note (decision changed by the user, 13 Oct):** age was previously shown only in aggregate. It now appears per person as a **broad, estimated group**. Keep it out of exports at the individual level beyond the anonymous track ID. **Sex stays aggregate-only** (not requested on boxes). Mention it in the ethics/privacy slide, with the "uncertain" share and validation status.

### Materials / links for A–E
- Weights: `yolov8x-oiv7.pt` ([Ultralytics Open Images V7 docs](https://docs.ultralytics.com/datasets/detect/open-images-v7)); TACO litter models (links in B); Apple DMS (github.com/apple/ml-dms-dataset); SegFormer/OneFormer ADE20K via Hugging Face `transformers` for the floor mask.
- Datasets for validation/fine-tuning: TACO ([paper](https://arxiv.org/abs/2003.06975)), ZeroWaste, DMS, MINC.
- Effort (estimate): A1 + B1/B2 ≈ 2–3 h including GPU time; C via the vision-model workflow ≈ 2 h once frontages are defined; D + E (viewer only) ≈ 1–1.5 h.

## Compute once, cache everything (team decision, 12 Oct 2026)
The data is fixed, so every **expensive** stage runs once, writes its output to disk, and is **skipped on later runs** unless its input, model or parameters change. **Cheap** downstream stages rerun from the cache in seconds.

| Stage | Cost | Cache (run once) | Recompute freely from cache |
|---|---|---|---|
| Stitch + level + frame/tile extraction | minutes per walk | equirect frames, cubemap tiles (`data/interim/{walk}/frames/`, `tiles/`) | — |
| Detection: YOLO11, YOLO-World, SAM 2 | GPU | **raw detections at a low threshold (~0.05)** + masks (`detections.parquet`, `masks/`) | confidence thresholds, class filters, NMS |
| Embeddings: CLIP / DINOv2 per crop | medium | `embeddings/*.npy` + index | level-3 classifiers, matching the 9 coded frames |
| OCR: Tesseract per sign crop | medium | raw text, confidence, alternatives per PSM/variant | spellcheck, lexicon, voting |
| VLM calls (level 3+) | slow / costly | answers keyed by **crop hash + prompt version + model** | parsing, aggregation |
| Tracking, roles, interactions, anchoring, stats, GIS export | seconds | — | everything: rule tuning, Boaz's anchors, research questions |

**Mechanics**
- Each stage is a script that checks for its output first (make-style; a `Makefile` or a simple `pipeline.py` runner).
- Every output folder carries a `manifest.json` recording the input file hash, model weights and version, parameters, code version and timestamp. The cache key is `stage@version + hash(params)`. Changing any of these reruns only that stage and the ones after it, and `--force` overrides.
- **Formats:** Parquet for detections and tracks; `.npy` for embeddings; JSON/GeoJSON only for what the viewer and GIS consume.
- `data/interim/` = cache (gitignored, can be shared from the cluster); `data/processed/` = small viewer/GIS outputs (may be committed).
- Heavy GPU stages can run on the cluster; all downstream analysis runs on a laptop from the copied cache.

**Effect:** the heavy compute for levels 1–2 happens once (≈1–2 h in total for both walks). Every later iteration takes seconds.

## Guiding principle
Thinnest end-to-end slice first (Day 1): **one walk → anchored poses → people counts → viewer with shop tags + live panel.** Then widen (more walks, materials, photos, polish). Everything offline; viewer reads precomputed JSON.

## Phase 0 — Data & environment (Tue morning, all; blocks everything)
- Re-download data **file-by-file** (not as one zip); verify sizes. Inventory: walks, photos, drawings, GIS layers.
- Run `exiftool -ee -G3 -api LargeFileSupport=1 file.insv` and `telemetry-parser` on a complete file → **is there GPS / gyro / timestamps?** Decides Phase 2 path.
- Identify camera model (single-file vs `_00_/_10_` dual files), GIS format + CRS (likely ITM EPSG:2039), shop attributes.
- GitHub repo with layout from CLAUDE.md; GPU-cluster sign-up; `requirements.txt`.

## Phase 1 — Stitch & stabilise video (Ido; ~0.5 day)
- Dual-fisheye `.insv` → equirectangular MP4: **Insta360 Studio export** (fastest, with FlowState stabilisation + horizon lock) or **Insta360 MediaSDK** for batch on cluster. Fallback: FFmpeg `v360=input=dfisheye:output=e` (no seam blending).
- Stabilisation / horizon levelling: FlowState in Studio, or **Gyroflow** with the camera gyro. Level horizon matters: yaw→bearing math assumes upright frames.
- Sample frames at 1–2 fps (`ffmpeg -vf fps=2`) → `data/interim/{walk}/frames/`.

## Phase 2 — Registration: pose per frame + GIS projection (Ido + Boaz; Day 1–2)
- **Path A (GPS present):** parse GPS/gyro → per-frame `(lat, lon, heading)`; smooth (GPS in a narrow market is noisy) and snap to street centreline.
- **Path B (no GPS, likely):** small web/notebook tool where Boaz clicks 5–15 keyframes on the GIS map ("frame 1234 is here, facing there"); interpolate along the street polyline by frame index; heading = travel direction + gyro yaw delta.
- **Path C (only if B fails):** SfM on perspective crops (COLMAP / OpenSfM spherical camera, or hloc with SuperPoint+LightGlue), then align to map with a few anchors.
- **Photos:** EXIF GPS if present; else match to nearest frame via LightGlue against cubemap faces → attach as clickable thumbnails at that frame/yaw.
- **GIS → frame:** `yaw = bearing(cam→shop frontage midpoint) − heading`; pitch from distance + assumed sign height; drop shops > ~30 m or occluded (simple ray test against building footprints with shapely).
- Output: `poses/{walk}.json`, `overlays/{walk}.json`, `shops.geojson` (EPSG:4326).

## Phase 3 — Frame analysis (Alex; Day 1–3, GPU cluster)
- **Reproject** equirect → cubemap / 8 overlapping perspective tiles (`py360convert`), detect, map boxes back to `(yaw, pitch)`; de-duplicate across tile overlaps (NMS in spherical coords). SAHI as an alternative tiler.
- **People:** Ultralytics YOLO11 `person` + **ByteTrack/BoT-SORT** tracking → per frame "visible now", per segment "unique". Dense crowds: optional P2PNet density counting as cross-check.
- **Materials (stretch → aim for it):** zero-shot first — **YOLO-World / Grounding DINO + SAM 2** with prompts ("cardboard", "canvas awning", "corrugated metal", "plastic sheet", "wood pallet"); alternative **Apple DMS pretrained material segmenter** (46 classes incl. cardboard/fabric/metal). Output area fractions per yaw bin. Validate on ~50 frames Boaz labels in CVAT/Label Studio; fine-tune YOLO-seg only if zero-shot is poor and time allows.
- **Sex classification (man / woman), owner Alex, Day 2–3:** gives the panel's "38% men" stat from the kickoff.
  - **Model:** body-based **pedestrian attribute recognition** (PAR) on each person crop. In 360° street footage faces are usually too small or turned away, so a face classifier sees little. Use the `Female` attribute of a PAR model trained on PA-100K/PETA/RAP: OpenPAR (github.com/Event-AHU/OpenPAR) or Rethinking_of_PAR (github.com/valencebond/Rethinking_of_PAR). Zero-shot fallback: CLIP / OpenCLIP prompts ("a photo of a man / a woman, full body"). Optionally add FairFace (github.com/dchen236/FairFace) only when a face is large enough (e.g. > 40 px).
  - **Per track, not per frame:** average the probabilities over all of a ByteTrack track's frames. Use three classes: man / woman / **uncertain** (confidence below a threshold, e.g. 0.7, or too few frames). Children and occluded people mostly land in "uncertain".
  - **Run it on raw frames on the cluster, before face blurring.** Store only `{track_id, label, conf}`, never crops.
  - **Output:** add `sex: {label, conf}` per detection in `frames/{walk}.json`. The panel shows "38% men · 47% women · 15% uncertain", aggregated over the field of view or segment, never on individual boxes.
  - **Validation:** Boaz labels ~100 tracks → report accuracy and the uncertain rate. Watch for bias from clothing such as head coverings and long coats; show the error next to the stat in the demo.
- **Text on signs (OCR), Hebrew-first + English, owner: whoever brings `C:\Projects\OCR`, Day 2–3.** Reads shop signs, banners and price boards, links them to GIS shops and shows them in the scene.
  - **Key difference from the old project:** `C:\Projects\OCR` was built for *scanned document pages*: it runs `extract_text(page) → str` on a whole page. Market footage is *scene text*: small, angled, painted, on cluttered backgrounds. So we add a **detection + rectification** stage in front and feed the old pipeline **crops of single signs**.
  - **Pipeline:**
    1. **Detect text regions** on the same perspective tiles used for YOLO. Use docTR's detector, which doesn't depend on language (`db_resnet50`, already wrapped in `src/ocr_doctr.py`; use only its detection head, since docTR can't *read* Hebrew). Alternatives: CRAFT (github.com/clovaai/CRAFT-pytorch), or Grounding DINO "shop sign" to find signboards first.
    2. **Rectify the crop**: perspective-warp the quad to a rectangle and upscale ×2–3. Reuse `src/preprocessor.py`'s CLAHE + unsharp-mask steps, and probably skip the NL-means step, which is for scanner noise.
    3. **Read the text with Tesseract only (team decision; Claude OCR is out for now).** Use `heb+eng` with `--psm 7` (one line) or `--psm 6` (block), which gives per-word confidence. For a hard crop, try both PSMs, plus a binarised and an inverted version of the crop (light text on dark boards), and keep the result with the highest mean confidence. Crops below the confidence threshold are marked "detected, unreadable" rather than guessed.
    4. **One text per sign, not per frame:** a sign appears in dozens of frames. Group detections by position (yaw and pose) or track them, then OCR only the 3–5 sharpest and largest crops and vote word by word. This is both faster and more accurate than OCR on every frame.
    5. **Correction (reused as is):** `HebrewFinalFixer` (final-letter forms ך ם ן ף ץ, ס↔ם swaps), then `SpellChecker` (symspellpy) with `frequency_lists/hebrew/hebrew_100000.txt` plus the English dictionary.
    6. **Domain lexicon:** the frequency list comes from conversational text (its top words are לא, את, אני), so shop vocabulary such as עתיקות, רהיטים and נחושת and proper names are underweighted. Add **all GIS shop names plus a market vocabulary list from Boaz** to the symspell dictionary with high counts.
    7. **Link to GIS:** fuzzy-match the read text to GIS shop names (rapidfuzz). A match **auto-links a sign to its shop and independently checks the registration.** If a sign's matched shop sits more than 10° from where we projected the GIS tag, the pose is off.
  - **Changes needed in the old code:**
    - Make `OCREngine` return boxes and confidence (`extract_regions(image) → [{text, conf, box}]`), not just a string.
    - Add `"hebrew_english": "heb+eng"` to `_LANG_MAP` in `ocr_tesseract.py`. The README claims mixed mode, but the code only supports `english` and `hebrew`.
    - Make the PSM configurable per crop.
    - Replace `pkg_resources` in `spellchecker.py` (deprecated) with `importlib.resources`.
  - **Arabic:** Jaffa is a mixed city, so expect Arabic signs. Run Tesseract with `ara` too (`heb+eng+ara`, or a separate `ara` pass when a crop contains Arabic script), or at minimum label them "Arabic". That makes signage language share a real urban-planning metric.
  - **Output:** `texts/{walk}.json`: `[{sign_id, yaw, pitch, w, h, text, raw, lang: he|en|ar|mixed, conf, engine, fixes, shop_id?}]`.
  - **Validation:** Boaz transcribes about 50 signs. Report character error rate (CER) and word accuracy per language, and the GIS-match rate.
- **Privacy:** `deface` face blurring on every frame shown in the viewer. Results are aggregate-only, with the ethics and privacy slide.
- Output: `frames/{walk}.json` (schema in CLAUDE.md).

## Phase 2b — [Brief] Spatial anchoring of every observation (Ido; Day 2)
- **World position of any detection**, beyond the shop tags. Foot point of the box → pitch below the horizon → ground distance `d = cam_h / tan(pitch)` (flat-ground assumption; camera height measured on site, about 1.6–1.8 m). Bearing comes from yaw + camera heading, giving lat/lon in EPSG:2039, then WGS84. Cross-check with a monocular depth model on the cubemap faces (Depth Anything V2) for objects that don't touch the ground, such as hanging goods and awnings.
- **Chainage (position along the route, m):** project each world point onto the walk polyline. This is the brief's "position along route" and the natural x-axis for market morphology plots.
- **Persistent IDs across frames and walks:** people `P-####` (ByteTrack track), shops `S###` (from GIS), signs `T###`, frontages `F###`, objects `O####` (waste, crates, seating, obstacles: same object if it is within 0.75 m on the ground and the same class), interactions `I####`, logistics events `L####`. When a later walk revisits a place, static objects (frontages, seating, waste piles) are matched by world position, so they keep their ID across walks.

## Phase 3b — [Brief] AI/VLM observation layers (Alex + Boaz; Day 2–3)
The brief explicitly asks for **AI/VLM** analysis. Detectors give the boxes; a **vision-language model (VLM)** gives the labels a detector has no class for (role, activity, stall type, permeability).
- **VLM choice:** an open model on the GPU cluster, because the volume is many crops and it keeps footage local: **Qwen2.5-VL-7B** (or Qwen3-VL), InternVL, or **Florence-2** for fast captioning and grounding. Ask for **structured JSON** with a fixed label set and a confidence, and validate the result against the schema. A hosted VLM (e.g. Claude) is an optional switch for a quality comparison on a sample. The "Tesseract only" decision applies to OCR, not here.
- **Run the VLM per entity, not per frame:** one call per track or per frontage on its 2–3 best crops, plus a short clip strip for activities. This keeps cost and time manageable.

**Social / human layer**
- **Age group (broad, aggregate only):** child / young adult / adult / older adult. Take it from the same pedestrian-attribute model as sex (PA-100K has `AgeLess18`, `Age18-60`, `AgeOver60`), or from MiVOLO (face + body age estimation) when the person is close. Vote per track, with an "uncertain" class. Show it only in aggregate and on the map, never on individual cards. Same validation as sex (~100 tracks).
- **Role:** vendor / shopper / passer-by / worker. Start with rules that use space and time, then let the VLM confirm. A person standing behind the frontage line or counter for more than 60 s is a vendor. One stationary 0.5–2 m in front of a stall and facing it is a shopper. One moving steadily along the route is a passer-by. A person with a hand truck, crates, broom or delivery box is a worker (see Phase 3c).
- **Interactions:** two tracks within 1.5 m, roughly facing each other, for at least 5 s, form an interaction. Its **type** (transaction / conversation / assistance / group) comes from rules (vendor + shopper across a counter = transaction) and the VLM on a crop of the pair. Store `I####` with its participants, duration, position and frontage ID.
- **Gathering and stopping points:** dwell = track speed < 0.3 m/s for at least 10 s. Cluster stop points with DBSCAN and compute hotspots with kernel density estimation (KDE) or Getis-Ord Gi* (PySAL).

**Material layer**
- **Stall morphology / trade type** per frontage `F###`: walk-in shop / open stall / counter / temporary table / kiosk. Use VLM classification of the best frontage crop plus Boaz's GIS attributes. Trade type (produce, spices, fish, meat, bakery, household, clothing, café) comes from the VLM, OCR'd signage and GIS.
- **Vendor inside/outside:** the vendor's ground position relative to the frontage line (negative = inside).
- **Shop-front permeability:** open share of the frontage (opening / total width, from segmentation of opening vs wall/shutter), plus a VLM score from 0 to 3 (closed, window, partly open, fully open).
- **Display and construction materials:** the existing awning-material pipeline, extended with display classes (crates, tables, shelving, sacks).
- **Waste materials (emphasised in the brief):** cardboard, plastic, organic waste, crates, packaging. Use open-vocabulary detection (Grounding DINO / YOLO-World prompts) and validate it on waste datasets: **TACO** (litter in context) and **ZeroWaste** (cardboard, soft and rigid plastic). Fine-tune YOLO-seg on about 200 labelled market crops if zero-shot is weak. Output `O####` objects with class, ground position, approximate area in m², and first/last frame.

**Spatial layer**
- **Circulation:** aggregate trajectories with MovingPandas into **flow lines** per street segment (direction and volume) and desire lines.
- **Thresholds:** shop entrance lines from GIS frontage plus the opening. Count crossings (inbound/outbound) per frontage.
- **Junctions:** taken from the GIS street network, with flows counted per junction arm.
- **Shop spill-out:** goods, tables and crates whose ground position lies beyond the frontage line. Store the encroachment depth (m) and the width taken from the walkway.
- **Obstacles:** static objects in the walkway (spill-out, parked carts, piles). Report the remaining clear width per segment.
- **Seating:** COCO `chair` / `bench` plus the VLM (plastic stools, crates used as seats).
- **Active frontages:** a score per frontage combining permeability, vendor present, interactions per minute and spill-out.
- **Areas of concentrated interaction:** KDE of interactions and stops, as above.

## Phase 3c — [Brief] Operational / logistics activity detection (Alex; Day 2–3)
- **Classes:** delivery worker, porter, box/crate collector, road sweeper/cleaner, waste handler, loading/unloading, stock movement, e-bike delivery, other service.
- **How:**
  1. Detect **tools and vehicles**: COCO `bicycle`, `motorcycle`, `truck`, plus open-vocabulary prompts for "hand truck", "trolley with crates", "broom", "garbage bin", "delivery box", "pallet jack", "e-bike".
  2. **Associate person and object**: their boxes overlap and they move together for at least 3 s.
  3. **Recognise the activity** on the track's clip: zero-shot **X-CLIP**, or **VideoMAE** fine-tuned on Kinetics, whose classes include *sweeping floor*, *pushing cart* and *carrying*. Confirm with the VLM.
  4. Rules on top: a van plus people moving crates between van and shop = loading/unloading.
- **Output:** events `L####` with class, start/end frame and time, a polyline of their path (porters, e-bikes) or a point (loading), the frontage they serve, and confidence. This is the market's **operational metabolism**: when and where logistics happens, and how it overlaps waste and social activity.

## Phase 4 — Demo viewer (Alex/Ido UI, Boaz styling; Day 2–3)
- Static web app: **Photo Sphere Viewer** (Equirectangular Video adapter + Video, Markers, Plan/Map, Compass plugins) — covers 360° video, in-scene tags and mini-map out of the box. Fallback: Pannellum or three.js video texture.
- Each `timeupdate`: look up nearest analysed frame → update shop markers (positioned by yaw/pitch), people boxes layer, side panel = aggregate detections with `|yaw − view_yaw| < hfov/2`.
- **Data selection** — checkboxes for what to show *in the scene* (people detection boxes, track ID + confidence, GIS tags, materials segmentation, Boaz's photos & drawings) and *in the panel* (people count, men/women split, materials share, shops in view, mini-map); choices remembered per viewer. Explicit nod to Boaz's "switchable layers" twin.
- **Text (OCR) presentation**: a dashed box on each detected sign with the read text (rendered RTL for Hebrew), its language and confidence. The panel's "Text in view" lists each sign with a language chip, a ✓ when it matched a GIS shop, and the share of Hebrew / English / Arabic signs in view. Clicking a sign shows the raw OCR → corrected text and which engine and corrections were used.
- **People detection presentation** — boxes around each detected person, `#track_id confidence` label, detected people as dots on the mini-map, counts in the panel.
- Leaflet mini-map with path, current position and view cone; shop click → info card (name, owner, category, Boaz's drawing if any).
- LLM side feature (only if everything else works): "ask the market" chat over the JSON stats (Claude API).

- **[Brief] Three representations of the same viewpoint:** **360° photo/video → simplified digital → fully illustrated**, with a crossfade between them. Every overlay (tags, people and activity markers, cards) stays anchored, because overlays are positioned by yaw/pitch and don't depend on the image.
  - *Photo:* the stitched equirectangular video.
  - *Digital:* a semantic-segmentation render (OneFormer / Mask2Former on ADE20K, which has classes for awning, stall and person), drawn as flat class colours plus edges, in the "blueprint" look of the brief's cover image. Alternatively, a GIS-extruded box model in three.js.
  - *Illustrated:* precomputed stylised keyframes. Use ControlNet (lineart/canny) + SDXL with **IP-Adapter using Boaz's drawings as the style reference** (no training), or a **LoRA trained on his drawings** (kohya_ss) if time allows. informative-drawings gives a fast line-art pass. Reduce flicker with keyframes at 1–2 fps, crossfades, or EbSynth. Coordinate with the CS collaborator who is already working on style transfer and 360° rendering, and use their output if it's ready.
- **[Brief] In-view information cards:** clicking a shop, person, object, sign or logistics event opens a small card **inside the 360° view**, anchored to it, which follows it as you rotate. The side-panel cards stay as a secondary place to read details.
- **[Brief] Immersive mode:** hide the side panel so the view fills the screen, with in-view counters for the layers that are on.
- **[Brief] Move through the scene:** step forward and back along the route (↑/↓ keys and buttons). Show chainage in the HUD.
- **[Brief] Layer names follow the brief:** people/interactions · shops (morphology, frontages) · materials/waste · logistics/operations · GIS tags · researcher drawings, plus OCR text and spatial (circulation, thresholds, seating, obstacles) as our extra layers. All switch independently.
- **[Brief] GIS export from the viewer:** an "Export observations" control that writes or copies the observation table as **GeoJSON** (the artifact sandbox blocks downloads, so the demo copies to the clipboard; the real app writes **GeoPackage + CSV** via the pipeline).

## Phase 5b — [Brief] GIS-ready data and the research questions (Boaz + Ido; Day 3)
- **One observation table** (`observations.gpkg`, with `observations.csv` alongside). Columns: `obs_id, entity_id, entity_type, layer, class, attributes(json), frame_start, frame_end, t_start, t_end, chainage_m, lat, lon, geom (point/line), confidence, model, walk_id`. Load it in QGIS for **market morphology** maps.
- **The brief's questions, as GIS operations** (show 2–3 of them computed live in the demo):
  1. *Do open stalls produce more interaction than walk-in shops?* Interactions per frontage-minute, grouped by morphology, with a Mann-Whitney U test.
  2. *Do vendors positioned outside interact more?* The same, by vendor inside/outside.
  3. *Are older and younger users concentrated in different parts?* Age-group share per 10 m route segment, with a chi-square test or a location quotient map.
  4. *Do interaction hotspots match spatial forms, signage, materials or shop types?* Overlay Gi* hotspots with frontage attributes (morphology, permeability, sign count and language, materials). Use a simple regression of interactions per frontage on its attributes.
  5. *Where do waste, logistics and social activity overlap?* A 5 m grid; count cells where waste objects, logistics events and interaction hotspots coincide, and map them.
- **Caveat for the pitch:** one walk is one moment in time. The method is the result, and statistical claims need repeated walks at different times of day.

## Phase 5 — Validation, story, pitch (all; last day)
- Accuracy table: manual people count on 30 frames vs model; sex classification accuracy + uncertain rate on ~100 labelled tracks; OCR CER / word accuracy per language on ~50 transcribed signs + GIS-match rate; materials IoU/accuracy on Boaz's labels; registration error (tag offset in degrees on 20 shops).
- Urban-planning story: pedestrian density per street segment, informal-material share per frontage, shops vs footfall map; ethics & privacy slide.
- Record backup screen-capture of the demo.

## Timeline (team can only meet Tue morning; Alex leaves 14:00 Tue)
| When | Goal |
|---|---|
| Tue AM (together) | Phase 0 complete; decide registration path; split work; agree JSON schema |
| Tue PM | Stitched walk + frames; YOLO people on frames; viewer skeleton plays video |
| Day 2 | Poses + GIS overlays; people JSON → viewer panel; first end-to-end demo |
| Day 3 | Materials, photos layer, tracking, styling, validation numbers |
| Final | Freeze features noon; pitch + backup video |
| **[Brief] Day 3 success test** | A **short navigable market segment**: rotate and move through it, switch photo / digital / illustrated, see persistent anchored observations, open in-view cards, export the observation table to GIS (GeoPackage/GeoJSON) and answer at least 2 research questions on it |

**[Brief] Scope rule:** pick **one short segment (~50–100 m)** and make every layer work there end to end. That beats covering the whole market thinly.

## Materials / resources needed
**Data:** complete `.insv` walks (or Studio-exported equirect MP4), photos with EXIF, Boaz's drawings, GIS layers (shops polygons/points, ownership, street centrelines, building footprints), ~50 hand-labelled frames.
**Compute:** GPU cluster (YOLO/SAM/GroundingDINO), laptop with Insta360 Studio (Win/Mac).

**Software & repos**
| Need | Tool / repo |
|---|---|
| Stitch .insv → equirect | Insta360 Studio (free); Insta360 MediaSDK (developer application at insta360.com); FFmpeg `v360` filter (ffmpeg.org/ffmpeg-filters.html#v360); 360toolkit (github, uses MediaSDK) |
| Stabilisation / horizon | Insta360 FlowState (in Studio); Gyroflow — github.com/gyroflow/gyroflow |
| Metadata GPS/gyro | ExifTool (`-ee`, GPX `gpx.fmt` from exiftool GitHub); telemetry-parser — github.com/AdrianEddy/telemetry-parser (PyPI) |
| Equirect ↔ cubemap/perspective | py360convert — github.com/sunset1995/py360convert |
| Detection + tracking + counting | Ultralytics YOLO11 — github.com/ultralytics/ultralytics (built-in ByteTrack/BoT-SORT, ObjectCounter); Roboflow supervision — github.com/roboflow/supervision |
| Tiled inference | SAHI — github.com/obss/sahi |
| Dense crowd counting (optional) | P2PNet — github.com/TencentYoutuResearch/CrowdCounting-P2PNet |
| Open-vocab materials | YOLO-World (in ultralytics); Grounding DINO — github.com/IDEA-Research/GroundingDINO; Grounded-SAM-2 — github.com/IDEA-Research/Grounded-SAM-2; SAM 2 — github.com/facebookresearch/sam2; OpenCLIP — github.com/mlfoundations/open_clip |
| Material segmentation | Apple DMS dataset + pretrained model — github.com/apple/ml-dms-dataset (CC-BY-NC) |
| Image matching / photo registration | LightGlue — github.com/cvg/LightGlue; hloc — github.com/cvg/Hierarchical-Localization; OpenCV SIFT / `cv2.Stitcher` for photo panoramas |
| SfM fallback | COLMAP — github.com/colmap/colmap; OpenSfM — github.com/mapillary/OpenSfM |
| GIS | QGIS; geopandas, shapely, pyproj (EPSG:2039 ↔ 4326) |
| OCR: our own pipeline | `C:\Projects\OCR`: copy `ocr_base.py`, `ocr_tesseract.py`, `ocr_doctr.py` (detection only), `preprocessor.py`, `spellchecker.py` (`HebrewFinalFixer`, `SpellChecker`) and `frequency_lists/hebrew/` into `pipeline/ocr/` |
| OCR: engines and detectors | Tesseract 5 + `heb`, `eng`, `ara` traineddata (github.com/UB-Mannheim/tesseract/wiki); docTR detection (github.com/mindee/doctr); CRAFT (github.com/clovaai/CRAFT-pytorch); Surya OCR (github.com/VikParuchuri/surya), a modern multilingual alternative that reads Hebrew line by line, as a backup only if Tesseract fails badly |
| OCR: correction and matching | symspellpy (github.com/mammothb/symspellpy); Hebrew frequency dictionary (github.com/Dotan-Peleh/hebrew-frequency-dictionary, already in the old project); rapidfuzz (github.com/rapidfuzz/RapidFuzz) for sign → GIS name matching |
| **[Brief]** Vision-language models | Qwen2.5-VL — github.com/QwenLM/Qwen2.5-VL (Qwen3-VL if available on the cluster); InternVL — github.com/OpenGVLab/InternVL; Florence-2 — huggingface.co/microsoft/Florence-2-large; vLLM — github.com/vllm-project/vllm for fast batch serving; (optional) Claude API for a quality comparison |
| **[Brief]** Age estimation | Pedestrian-attribute models above (PA-100K age attributes); MiVOLO — github.com/WildChlamydia/MiVOLO (face + body) |
| **[Brief]** Activity / logistics recognition | X-CLIP (zero-shot video) — huggingface.co/microsoft/xclip-base-patch32; VideoMAE (Kinetics) — github.com/MCG-NJU/VideoMAE; PySlowFast — github.com/facebookresearch/SlowFast |
| **[Brief]** Waste datasets | TACO — github.com/pedropro/TACO; ZeroWaste — github.com/dbash/zerowaste |
| **[Brief]** Depth and anchoring | Depth Anything V2 — github.com/DepthAnything/Depth-Anything-V2 (run on cubemap faces) |
| **[Brief]** Digital view (segmentation) | OneFormer — github.com/SHI-Labs/OneFormer; Mask2Former — github.com/facebookresearch/Mask2Former (ADE20K weights via HF `transformers`) |
| **[Brief]** Illustrated view (style) | diffusers — github.com/huggingface/diffusers; ControlNet v1.1 — github.com/lllyasviel/ControlNet-v1-1-nightly; IP-Adapter — github.com/tencent-ailab/IP-Adapter; kohya_ss (LoRA training) — github.com/bmaltais/kohya_ss; informative-drawings — github.com/carolineec/informative-drawings; EbSynth (temporal coherence) — ebsynth.com; controlnet_aux (HED/lineart preprocessors) — github.com/huggingface/controlnet_aux |
| **[Brief]** Trajectories, hotspots, GIS stats | MovingPandas — github.com/movingpandas/movingpandas; PySAL / esda (Getis-Ord Gi*) — github.com/pysal/esda; scikit-learn (DBSCAN, KDE); scipy.stats; GeoPackage via geopandas; QGIS for maps |
| **[Brief]** Researcher input | Boaz's drawings (style reference + "researcher drawings" layer), a labelled set of ~50 frontages (morphology/permeability), ~100 tracks (role/age), ~50 waste crops, ~20 logistics events for validation |
| OCR: extra links (checklist below) | tessdata_best — github.com/tesseract-ocr/tessdata_best (Apache-2.0); FrequencyWords — github.com/hermitdave/FrequencyWords (CC-BY-SA-4.0) |
| Face blurring | deface — github.com/ORB-HD/deface |
| Labelling | CVAT — github.com/cvat-ai/cvat or Label Studio |
| Viewer | Photo Sphere Viewer — github.com/mistic100/Photo-Sphere-Viewer; Pannellum — github.com/mpetroff/pannellum; Leaflet — leafletjs.com; three.js |
| Sex classification | OpenPAR — github.com/Event-AHU/OpenPAR; Rethinking_of_PAR — github.com/valencebond/Rethinking_of_PAR (PA-100K/PETA/RAP weights); OpenCLIP zero-shot; FairFace — github.com/dchen236/FairFace (close-range faces only) |

### OCR materials checklist (Tesseract only; checked on this machine 12 Oct 2026)
| Item | Status | Where / action |
|---|---|---|
| Tesseract binary | ✅ v5.5.0 installed | `C:\Program Files\Tesseract-OCR\tesseract.exe` |
| `eng`, `heb`, `ara` traineddata | ✅ installed (standard/fast models) | `…\Tesseract-OCR\tessdata\` |
| Higher-accuracy models | ⬜ recommended | Get `eng`, `heb`, `ara` from **tessdata_best** (github.com/tesseract-ocr/tessdata_best, Apache-2.0) into a separate folder; point to it with `--tessdata-dir`. Slower, but we only OCR small crops. |
| **English frequency dictionary** | ✅ already have it | Bundled with symspellpy 6.9.0: `frequency_dictionary_en_82_765.txt` (82,765 words) in `C:\Projects\OCR\venv\Lib\site-packages\symspellpy\`. The old `SpellChecker._load_english()` loads it automatically, but via `pkg_resources`, which needs to change to `importlib.resources`. |
| English bigram dictionary | ✅ already have it, unused | `frequency_bigramdictionary_en_243_342.txt` in the same folder. Use it with `SymSpell.lookup_compound` / `word_segmentation` to split words OCR runs together ("OLDJAFFARUGS" → "OLD JAFFA RUGS"). Signs often have tight letter spacing. |
| Hebrew frequency dictionary | ✅ have it | `C:\Projects\OCR\frequency_lists\hebrew\hebrew_{10k,20k,50k,100k}.txt` |
| Domain lexicon (Hebrew + English) | ⬜ to build | GIS shop names + Boaz's market vocabulary + place names (Jaffa, Yefet, Olei Zion, Beit Eshel …), each with a high count, merged into both dictionaries. Proper nouns are missing from both general lists. |
| Arabic frequency dictionary | ⬜ optional | **FrequencyWords** (github.com/hermitdave/FrequencyWords, OpenSubtitles-based, `word count` per line, CC-BY-SA-4.0). Check the exact path in `content/` (expected `…/ar/ar_50k.txt`). Without it, Arabic is OCR'd but not spell-corrected. |
| Tesseract on the Linux GPU cluster | ⬜ to install | `apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-heb tesseract-ocr-ara` (or conda-forge `tesseract`), or run OCR on a laptop, since it's CPU work. |

## Risks & mitigations
| Risk | Mitigation |
|---|---|
| No GPS in videos | Path B manual anchoring (half a day, robust) |
| Data still truncated / late | Work on any complete walk; Studio export on Boaz's laptop |
| Materials zero-shot weak | Show people + GIS as core; materials as "experimental" layer with honest metrics |
| Equirect distortion breaks detection | Cubemap tiling from the start |
| Scene text too small / blurry in 360° frames | OCR only close, sharp crops (choose the best frames per sign); report "detected but unreadable" honestly |
| Tesseract weak on painted / stylised Hebrew | Rectify + upscale + try binarised/inverted crops and two PSMs; vote across frames; domain lexicon + final-letter fixer from the old project. Revisit Claude OCR only if the team re-enables it |
| **[Brief]** Scope explosion (3 layers + logistics + 3 representations) | One short segment end to end; the VLM labels entities that are already detected (no new detectors per class); the illustrated view uses precomputed keyframes |
| **[Brief]** VLM labels unreliable (role, activity, morphology) | Fixed label sets + JSON schema + confidence; rules first, VLM second; validate on Boaz's labels; show confidence in every card |
| **[Brief]** Age groups ethically and visually shaky | Broad groups only, "where visually defensible" (brief); uncertain class; aggregate/map only, never on a person card |
| **[Brief]** Style-transfer flicker in 360° | Keyframes plus crossfade; or the CS collaborator's renderer; the digital view as fallback |
| **[Brief]** Research questions from one walk = anecdote | Present them as a method demo with explicit n; plan repeated walks |
| Sex classification inaccurate at distance / judges question it | Body-based PAR + per-track voting + explicit "uncertain" class; show measured accuracy next to the stat; aggregate-only; privacy slide |
| Team availability Tue | Agree JSON schema Tue AM so work parallelises |

---
