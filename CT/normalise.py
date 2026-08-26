import os
import re
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

INPUT_FOLDER  = r"C:\Users\P3084\Desktop\Test\Projections"
OUTPUT_FOLDER = r"C:\Users\P3084\Desktop\Test\Projections_norm"
EPS = 1e-5
LOG = True
BAD_PIXEL_CORRECTION= True

def natural_key(name):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def _get_stack_info(folder):
    tifs = sorted((f for f in os.listdir(folder) if f.lower().endswith(('.tif', '.tiff'))), key=natural_key)
    if not tifs:
        raise RuntimeError(f"No .tif files found in: {folder}")
    img = cv2.imread(os.path.join(folder, tifs[0]), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Could not read: {tifs[0]}")
    h, w = img.shape[:2]
    return w, h, tifs


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    src_w, src_h, image_files = _get_stack_info(INPUT_FOLDER)
    n = len(image_files)
    print(f"Found {n} images  ({src_w} x {src_h})")

    # Load in the exact same natural-sorted order used for saving.
    print("Loading stack ...")
    stack = np.zeros((n, src_h, src_w), dtype=np.float32)

    def load_one(idx, filename):
        path = os.path.join(INPUT_FOLDER, filename)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Could not read: {path}")
        return idx, img.astype(np.float32)

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(load_one, i, f): i for i, f in enumerate(image_files)}
        done = 0
        for future in as_completed(futures):
            idx, img = future.result()
            stack[idx] = img
            done += 1
            print(f"  {done}/{n}", end='\r')

    print(f"\nStack shape: {stack.shape}  dtype: {stack.dtype}")

    g_min = float(stack.min())
    g_max = float(stack.max())
    print(f"Global min={g_min:.4f}  max={g_max:.4f}")

    denom = g_max - g_min if g_max != g_min else 1.0
    processed = [None] * n

    def normalise(idx):
        norm = (stack[idx] - g_min) / denom
        out = -np.log(norm + EPS) if LOG else norm
        if BAD_PIXEL_CORRECTION:
            out = cv2.medianBlur(out.astype(np.float32), 5)
        return idx, out.astype(np.float32)

    print("Normalising ...")
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(normalise, i): i for i in range(n)}
        done = 0
        for future in as_completed(futures):
            idx, log_img = future.result()
            processed[idx] = log_img
            done += 1
            print(f"  {done}/{n}", end='\r')

    print("\nSaving ...")
    for idx, log_img in enumerate(processed):
        out_name = f"slice_{idx:04d}.tif"
        out_path = os.path.join(OUTPUT_FOLDER, out_name)
        ok = cv2.imwrite(out_path, log_img.astype(np.float32))
        if not ok:
            raise RuntimeError(f"Failed to save: {out_path}")
        print(f"  {idx+1}/{n}", end='\r')

    print(f"\nDone - saved {n} images to:\n  {OUTPUT_FOLDER}")


if __name__ == '__main__':
    main()
