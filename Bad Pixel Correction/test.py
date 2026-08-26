import os
import re
import sys
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Processors"))
from BadPixelCorrector import BadPixelCorrector

STACK_SRC = "PATH"
DEFECT_MAP_SRC = "PATH"
DARK_FIELD_STACK_SRC = "PATH"
FLAT_FIELD_STACK_SRC = "PATH"
DST_STACK_SRC = "PATH"

def natural_key(name):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def load_images(path: str):
    """Read all images in `path`, in ascending natural filename order, in parallel."""
    files = sorted(
        (f for f in os.listdir(path) if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"))),
        key=natural_key,
    )
    if not files:
        raise RuntimeError(f"No images found in: {path}")

    first = cv2.imread(os.path.join(path, files[0]), cv2.IMREAD_UNCHANGED)
    if first is None:
        raise RuntimeError(f"Could not read: {files[0]}")

    images = [None] * len(files)
    images[0] = first

    def load_one(idx, filename):
        full_path = os.path.join(path, filename)
        img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Could not read: {full_path}")
        return idx, img

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(load_one, i, f): i for i, f in enumerate(files) if i != 0}
        for future in as_completed(futures):
            idx, img = future.result()
            images[idx] = img

    return images, files


if __name__ == "__main__":
    # load images
    stack_imgs, stack_files = load_images(STACK_SRC)
    dark_imgs, _ = load_images(DARK_FIELD_STACK_SRC)
    flat_imgs, _ = load_images(FLAT_FIELD_STACK_SRC)

    defect_map = cv2.imread(DEFECT_MAP_SRC, cv2.IMREAD_UNCHANGED)
    if defect_map is None:
        raise RuntimeError(f"Could not read: {DEFECT_MAP_SRC}")
    mask = defect_map.astype(bool)  # False = bad pixel

    stack = np.stack(stack_imgs).astype(np.float64)
    dark_mean = np.mean(np.stack(dark_imgs).astype(np.float64), axis=0)
    flat_mean = np.mean(np.stack(flat_imgs).astype(np.float64), axis=0)

    # undergo correction
    flat_corrected = (stack - dark_mean) / (flat_mean - dark_mean)

    corrector = BadPixelCorrector(mask=mask)
    corrected = corrector.process(flat_corrected)

    # save images
    os.makedirs(DST_STACK_SRC, exist_ok=True)
    for filename, img in zip(stack_files, corrected):
        out_path = os.path.join(DST_STACK_SRC, filename)
        ok = cv2.imwrite(out_path, img.astype(np.float32))
        if not ok:
            raise RuntimeError(f"Failed to save: {out_path}")