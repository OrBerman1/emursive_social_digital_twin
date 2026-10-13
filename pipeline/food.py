"""Stage (CLIP part cached): food on display - WHERE from detectors, WHAT from CLIP on the display crops.

Inputs : data/interim/<walk>/det_yolov8x-oiv7<_stab>/detections.parquet   (Open Images V7, 600 classes)
         data/interim/<walk>/det_yolo11x<_stab>/detections.parquet        (COCO: banana, apple, orange, carrot, broccoli)
Why two steps: on market displays Open Images labels ~98% of food boxes only generically ("Vegetable", "Food",
         "Fruit"), so it is good at WHERE food is but rarely says WHAT. Each display is therefore tracked as "food"
         and named afterwards by CLIP ViT-L/14 zero-shot over a market vocabulary, voted over its best crops.
Method : 1. food boxes = OIV7 food classes (conf >= 0.15) + COCO produce (conf >= 0.3); per-frame de-dup; tracked as one
            class (no label flicker) with the level-2 tracker -> persistent F-#### display IDs.
         2. CLIP on up to N_CROPS widest crops per display (with context) -> label = most probable item, plus the
            runner-up; a display is a stack or crate area, not a count of single fruits.
Output : data/interim/<walk>/tracks<_stab>/food.parquet (cls = CLIP item, group, p_item, second, det_cls)

usage:  python pipeline/food.py [walk1 walk2] [--variant stab] [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

import cv2
import numpy as np
import pandas as pd
import torch

from attributes import MODEL, crop, text_features
from common import INTERIM, WALKS, Stage
from tracks import dedup_frame, track_per_class

VERSION = 3
N_CROPS = 3
OIV7_FOOD = {"Apple", "Banana", "Orange", "Lemon", "Grapefruit", "Grape", "Strawberry", "Watermelon", "Winter melon", "Cantaloupe",
             "Mango", "Pineapple", "Peach", "Pear", "Pomegranate", "Common fig", "Coconut", "Fruit", "Tomato", "Potato", "Carrot",
             "Cucumber", "Zucchini", "Bell pepper", "Broccoli", "Cabbage", "Artichoke", "Radish", "Pumpkin", "Squash (Plant)",
             "Garden Asparagus", "Vegetable", "Bread", "Bagel", "Baked goods", "Cake", "Cookie", "Muffin", "Pastry", "Croissant",
             "Doughnut", "Pretzel", "Candy", "Cheese", "Egg (Food)", "Fish", "Seafood", "Honeycomb", "Juice", "Drink", "Fast food",
             "Food", "Dessert"}
COCO_FOOD = {"banana", "apple", "orange", "carrot", "broccoli"}
# market vocabulary: item -> (group, CLIP prompts)
VOCAB = {
    "Tomatoes": ("vegetables", ["a pile of red tomatoes at a market stall"]),
    "Cucumbers": ("vegetables", ["a pile of green cucumbers at a market stall"]),
    "Bell peppers": ("vegetables", ["colourful bell peppers at a market stall"]),
    "Chili peppers": ("vegetables", ["hot chili peppers at a market stall"]),
    "Eggplants": ("vegetables", ["purple eggplants at a market stall"]),
    "Zucchini": ("vegetables", ["green zucchini at a market stall"]),
    "Potatoes": ("vegetables", ["potatoes at a market stall"]),
    "Onions": ("vegetables", ["onions at a market stall"]),
    "Garlic": ("vegetables", ["garlic bulbs at a market stall"]),
    "Carrots": ("vegetables", ["orange carrots at a market stall"]),
    "Cabbage & lettuce": ("vegetables", ["heads of cabbage and lettuce at a market stall"]),
    "Herbs": ("vegetables", ["bunches of fresh green herbs, parsley and coriander at a market stall"]),
    "Corn": ("vegetables", ["corn cobs at a market stall"]),
    "Mushrooms": ("vegetables", ["mushrooms at a market stall"]),
    "Watermelons": ("fruit", ["watermelons at a market stall", "cut watermelon slices"]),
    "Melons": ("fruit", ["yellow melons at a market stall"]),
    "Bananas": ("fruit", ["bunches of bananas at a market stall"]),
    "Apples": ("fruit", ["red and green apples at a market stall"]),
    "Oranges & citrus": ("fruit", ["oranges, clementines and lemons at a market stall"]),
    "Grapes": ("fruit", ["grapes at a market stall"]),
    "Strawberries": ("fruit", ["strawberries in boxes at a market stall"]),
    "Mangoes": ("fruit", ["mangoes at a market stall"]),
    "Peaches & plums": ("fruit", ["peaches, nectarines and plums at a market stall"]),
    "Pomegranates": ("fruit", ["pomegranates at a market stall"]),
    "Pears": ("fruit", ["pears at a market stall"]),
    "Dates & dried fruit": ("dry goods", ["dates and dried fruit at a market stall"]),
    "Nuts & seeds": ("dry goods", ["nuts and seeds in bins at a market stall"]),
    "Spices": ("dry goods", ["spices in sacks and bowls at a market stall"]),
    "Olives & pickles": ("dry goods", ["olives and pickles in barrels at a market stall"]),
    "Sweets & halva": ("bakery & sweets", ["halva, sweets and candy at a market stall"]),
    "Bread & pita": ("bakery & sweets", ["bread and pita at a bakery stall"]),
    "Pastries & cakes": ("bakery & sweets", ["pastries, burekas and cakes in a display"]),
    "Cheese & dairy": ("meat, fish & dairy", ["cheese and dairy products in a refrigerator display"]),
    "Meat": ("meat, fish & dairy", ["raw meat at a butcher counter"]),
    "Fish": ("meat, fish & dairy", ["whole fish on ice at a fish stall"]),
    "Eggs": ("meat, fish & dairy", ["trays of eggs at a market stall"]),
    "Drinks & juice": ("drinks & prepared food", ["bottles of drinks and fresh juice"]),
    "Prepared food": ("drinks & prepared food", ["prepared food in trays at a deli counter", "falafel and street food"]),
    "Packaged groceries": ("groceries", ["packaged groceries on shelves"]),
    "Mixed fruit & vegetables": ("mixed produce", ["a market stall with many kinds of fruit and vegetables",
                                                   "a wide view of a fruit and vegetable stall"]),
    "Food picture (menu/sign)": ("pictured", ["a printed picture of fruit on a sign", "a drawing of fruit on a shop sign",
                                              "a photo of a dish on a restaurant menu board", "a printed menu photo of food with Hebrew text"]),
}
UNSURE_P = 0.5             # below this the item name is not shown (sample checks: p >= 0.5 -> ~80% correct, < 0.5 -> ~20%)
PICTURE_P = 0.8           # food pictures (menus, signs): own category above this (sample: >= 0.8 -> 7/8 correct, 0.6-0.8 mostly real stalls)
STRICT = {"Prepared food": 0.7}   # catch-all labels that need more confidence; otherwise the runner-up is used


def fhash(p) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def regions(walk: str, sfx: str) -> pd.DataFrame:
    parts = []
    po = INTERIM / walk / f"det_yolov8x-oiv7{sfx}" / "detections.parquet"
    if po.exists():
        d = pd.read_parquet(po); parts.append(d[d.cls.isin(OIV7_FOOD) & (d.conf >= 0.15)].assign(det_cls=lambda x: x.cls))
    pc = INTERIM / walk / f"det_yolo11x{sfx}" / "detections.parquet"
    if pc.exists():
        d = pd.read_parquet(pc); parts.append(d[d.cls.isin(COCO_FOOD) & (d.conf >= 0.3)].assign(det_cls=lambda x: x.cls))
    d = pd.concat(parts, ignore_index=True)
    d = d[d.pitch_top > -50].copy()
    d["ang_h"] = d.pitch_top - d.pitch_bot
    d = d[d.ang_h < 40]
    d["cls"] = "food"                                                   # one class while tracking: no label flicker
    return pd.concat([dedup_frame(g) for _, g in d.groupby("frame")]) if len(d) else d


def run(walk: str, variant: str, force: bool) -> dict:
    sfx = f"_{variant}" if variant else ""
    tdir = INTERIM / walk / f"tracks{sfx}"
    cam_h = json.loads((tdir / "summary.json").read_text(encoding="utf-8"))["camera_height_used_m"]
    d = regions(walk, sfx)
    det_cls = d.groupby(["frame", "yaw"]).det_cls.first() if len(d) else None
    f = track_per_class(d.drop(columns=["det_cls"]), id_offset=0)
    if det_cls is not None:
        f = f.merge(d[["frame", "yaw", "det_cls"]], on=["frame", "yaw"], how="left")
    f["dist_m"] = np.where(f.pitch_bot < -1, cam_h / np.tan(np.radians(-f.pitch_bot.clip(upper=-1))), np.nan)
    f = f[f.groupby("track").frame.transform("count") >= 3]

    # CLIP naming per display (cached)
    out_dir = INTERIM / walk / f"food{sfx}"
    st = Stage("food_clip", VERSION, out_dir, {"model": MODEL, "n_crops": N_CROPS, "vocab": {k: v[1] for k, v in VOCAB.items()}},
               {"tracks_hash": hashlib.sha256(pd.util.hash_pandas_object(f[["track", "frame", "yaw", "pitch_top"]]).values.tobytes()).hexdigest()[:16]})
    cache = out_dir / "clip.parquet"
    if st.done() and not force and cache.exists():
        lab = pd.read_parquet(cache)
    else:
        st.start()
        import clip
        from PIL import Image
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(MODEL, device=dev)
        names, tf = text_features(model, clip, {k: v[1] for k, v in VOCAB.items()}, dev)
        real = f[~f.interp] if "interp" in f.columns else f
        picks = real.assign(area=(real.yaw_r - real.yaw_l) % 360 * real.ang_h).sort_values("area", ascending=False).groupby("track").head(N_CROPS)
        probs = {}
        for fr, g in picks.groupby("frame"):
            equi = cv2.imread(str(INTERIM / walk / f"frames{sfx}" / f"{int(fr):06d}.jpg"))
            ims = [preprocess(Image.fromarray(cv2.cvtColor(crop(equi, r, isolate=False), cv2.COLOR_BGR2RGB))) for r in g.itertuples()]
            with torch.no_grad():
                im = model.encode_image(torch.stack(ims).to(dev)).float()
                im = im / im.norm(dim=-1, keepdim=True)
                p = (100 * im @ tf.T).softmax(-1).cpu().numpy()
            for r, pp in zip(g.itertuples(), p):
                probs.setdefault(int(r.track), []).append(pp)
        rows = []
        for tid, ps in probs.items():
            m = np.mean(ps, axis=0); o = np.argsort(m)[::-1]
            rows.append((tid, names[o[0]], float(m[o[0]]), names[o[1]], float(m[o[1]])))
        lab = pd.DataFrame(rows, columns=["track", "item", "p_item", "second", "p_second"])
        lab.to_parquet(cache, index=False)
        st.finish({"displays": len(lab)})
    f = f.merge(lab, on="track", how="inner")
    PIC = "Food picture (menu/sign)"                                     # printed pictures of food: own category, not food on display
    weak = ((f["item"] == PIC) & (f.p_item < PICTURE_P)) | f.apply(lambda r: r["item"] in STRICT and r.p_item < STRICT[r["item"]], axis=1)
    f.loc[weak, "item"], f.loc[weak, "p_item"] = f.loc[weak, "second"], f.loc[weak, "p_second"]   # unsure -> runner-up
    unsure = f.p_item < UNSURE_P
    f["cls"] = np.where(unsure, "Produce (unsure)", f["item"])
    f["group"] = np.where(unsure, "unsure", f["item"].map(lambda k: VOCAB[k][0]))
    f.to_parquet(tdir / "food.parquet", index=False)
    per = f.groupby("track").first()
    return {"walk": walk, "displays": int(len(per)), "by_group": per.group.value_counts().to_dict(),
            "top_items": per.cls.value_counts().head(12).to_dict(), "food_pictures": int((per.cls == "Food picture (menu/sign)").sum()), "median_p_item": round(float(per.p_item.median()), 2)}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("walks", nargs="*", default=list(WALKS))
    ap.add_argument("--variant", default="stab")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for w in a.walks:
        print(json.dumps(run(w, a.variant, a.force), ensure_ascii=False))


if __name__ == "__main__":
    main()
