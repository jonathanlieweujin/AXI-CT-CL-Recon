import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Manager import VxManager

# --- Parameters ---
_cli = argparse.ArgumentParser()
_cli.add_argument("--dataset", default=r"output\Synthetic\MICRO_JIG",
                   help="Dataset root; SRC/SRC_CONFIG/DST default to its Corrected/, "
                        "Config/geometry.config and Recon subpaths")
_cli.add_argument("--src", default=None, help="Override: folder of Corrected/*.tif projections")
_cli.add_argument("--src-config", default=None, help="Override: path to geometry.config")
_cli.add_argument("--dst", default=None, help="Override: output folder for reconstructed slices")
_args, _ = _cli.parse_known_args()

SRC = _args.src or os.path.join(_args.dataset, "Corrected")
SRC_CONFIG = _args.src_config or os.path.join(_args.dataset, "Config", "geometry.config")
DST = _args.dst or os.path.join(_args.dataset, "Recon")

# --- Reconstruct ---
manager = VxManager()
manager.LoadConfig(SRC_CONFIG, inputFolder=SRC)
print(f"Loaded: {manager.Images.shape}  flow={manager.FlowMethod}  algo={manager.ReconMethod}")

manager.Run()
recon = manager.Result
print(f"Result: {recon.shape}  range [{recon.min():.4f}, {recon.max():.4f}]")

# --- Scale to the pixel format the config asked for (global contrast stretch) ---
depth = np.uint16 if manager.Param.dst_pixel_format == "u16" else np.uint8
peak = float(np.iinfo(depth).max)

r_min = float(recon.min())
r_max = float(recon.max())
denom = r_max - r_min if r_max != r_min else 1.0
slices = (((recon - r_min) / denom) * peak).clip(0, peak).astype(depth)

# --- Save one .tif per slice ---
manager.SaveImages(DST, slices)
print(f"Saved {len(slices)} {manager.Param.dst_pixel_format} slices to {DST}")
