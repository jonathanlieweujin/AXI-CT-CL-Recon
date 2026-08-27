import os
import re
import sys
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.ndimage import median_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Processors"))
from BadPixelCorrector import BadPixelCorrector

STACK_SRC = r"D:\data\FlatDarkCorrectionSample\OnXray Have Unit All Correction"
DARK_FIELD_STACK_SRC = r"D:\data\FlatDarkCorrectionSample\OffXray No Unit All Correction"
FLAT_FIELD_STACK_SRC = r"D:\data\FlatDarkCorrectionSample\OnXray No Unit All Correction"
DST_STACK_SRC = r"D:\data\FlatDarkCorrectionSample\Result"

# Hot/cold pixel detection (median-filter local outlier test)
MEDIAN_SIZE = 3
MAD_MULTIPLIER = 8

# Beer-Lambert log conversion (final step)
TO_LOG_BEER_LAMBERT = True
BETA = 1e-5


def hot_cold_map(img: np.ndarray) -> np.ndarray:
    """Flag pixels that deviate too far from their local neighborhood median."""
    median = median_filter(img, size=MEDIAN_SIZE)
    residual = img - median
    mad = np.median(np.abs(residual - np.median(residual)))
    threshold = MAD_MULTIPLIER * 1.4826 * mad
    return np.abs(residual) > threshold

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

    stack = np.stack(stack_imgs).astype(np.float64)
    dark_mean = np.mean(np.stack(dark_imgs).astype(np.float64), axis=0)
    flat_mean = np.mean(np.stack(flat_imgs).astype(np.float64), axis=0)

    # build defect map from dark-field / flat-field hot/cold outliers
    defects = hot_cold_map(dark_mean) | hot_cold_map(flat_mean)
    mask = ~defects  # False = bad pixel
    print(f"Defect pixels: {int(defects.sum())} / {defects.size}")

    # undergo correction
    # denom is ~0 at bad-pixel locations (already flagged in `mask`); those
    # values get overwritten by BadPixelCorrector below, so just suppress
    # the divide-by-zero warning and zero out the resulting inf/nan.
    denom = flat_mean - dark_mean
    with np.errstate(divide="ignore", invalid="ignore"):
        flat_corrected = np.where(denom != 0, (stack - dark_mean) / denom, 0.0)

    corrector = BadPixelCorrector(mask=mask)
    corrected = corrector.process(flat_corrected)

    if TO_LOG_BEER_LAMBERT:
        corrected = -np.log(corrected + BETA)

    corrected = corrected.astype(np.float32)

    # save images
    os.makedirs(DST_STACK_SRC, exist_ok=True)
    for filename, img in zip(stack_files, corrected):
        out_path = os.path.join(DST_STACK_SRC, filename)
        ok = cv2.imwrite(out_path, img)  # already float32, no need to convert back
        if not ok:
            raise RuntimeError(f"Failed to save: {out_path}")