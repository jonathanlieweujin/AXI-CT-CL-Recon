import numpy as np

from Manager import VxManager

# --- Parameters ---
SRC = r"output\Synthetic\PCB_PANEL\Corrected"
SRC_CONFIG = r"output\Synthetic\PCB_PANEL\Config\geometry.config"
DST = r"output\Synthetic\PCB_PANEL\Recon"

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
