#!/usr/bin/env python3
"""
generate_wlasl_dataset.py

Assumes the following files/dirs live in the current working directory:
  • nslt_1000.json
  • wlasl_class_list.txt
  • videos/            (contains files like 12345.mp4, 67890.mp4, …)

Output:
  • data/
      ├─ train/
      │    └─ <GLossName>/
      ├─ val/
      │    └─ <GlossName>/
      └─ test/
           └─ <GlossName>/
"""

import os
import sys
import json
import shutil

# --- CONFIG (all paths are relative to cwd) ---
NSLT_PATH        = "nslt_1000.json"
CLASS_LIST_PATH  = "wlasl_class_list.txt"
VIDEOS_DIR       = "videos"
OUTPUT_ROOT      = "data"
SPLITS           = ["train", "val", "test"]
# ---------------------------------------------

def load_nslt(path):
    with open(path, "r") as f:
        return json.load(f)

def load_classes(path):
    with open(path, "r") as f:
        return [l.strip() for l in f if l.strip()]

def make_dirs(root, splits, classes):
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(root, split, cls), exist_ok=True)

def main():
    # sanity checks
    if not os.path.isfile(NSLT_PATH):
        print(f"[ERROR] Missing NSLT file: {NSLT_PATH}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(CLASS_LIST_PATH):
        print(f"[ERROR] Missing class list: {CLASS_LIST_PATH}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(VIDEOS_DIR):
        print(f"[ERROR] Missing videos directory: {VIDEOS_DIR}", file=sys.stderr)
        sys.exit(1)

    # load metadata
    nslt       = load_nslt(NSLT_PATH)
    class_list = load_classes(CLASS_LIST_PATH)

    # create output folders
    print(f"Creating dataset folders under '{OUTPUT_ROOT}'…")
    make_dirs(OUTPUT_ROOT, SPLITS, class_list)

    # link or copy videos
    total = len(nslt)
    found = 0
    missing = 0

    for vid, info in nslt.items():
        vid_str = str(vid)
        src = os.path.join(VIDEOS_DIR, vid_str + ".mp4")
        if not os.path.isfile(src):
            missing += 1
            if missing <= 10:
                print(f"[WARN] missing video file: {src}")
            continue

        split    = info["subset"]           # "train" / "val" / "test"
        cls_idx  = info["action"][0]        # [class_idx, start, end]
        cls_name = class_list[cls_idx]

        dest_dir = os.path.join(OUTPUT_ROOT, split, cls_name)
        dest     = os.path.join(dest_dir, vid_str + ".mp4")

        try:
            os.symlink(os.path.abspath(src), dest)
        except OSError:
            shutil.copy2(src, dest)

        found += 1

    # summary
    print(f"Done. Total entries: {total}")
    print(f"  ✓ Videos linked/copied: {found}")
    print(f"  ⚠ Videos missing:     {missing}")
    if missing:
        print("Make sure your 'videos/' folder contains all the .mp4 files named by video_id.")

if __name__ == "__main__":
    main()
