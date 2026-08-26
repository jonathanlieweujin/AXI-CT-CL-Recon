import numpy as np
import os
import ctypes
# -----------------------------
# Load CUDA library
# -----------------------------
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
os.add_dll_directory(cuda_path)
ctypes.WinDLL("cudart64_12.dll")

import matplotlib.pyplot as plt

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, r"C:\Users\P3084\Documents\Projects\GitHub\TIGRE\Python")

from Util.param import VxParam
from Util.load_files import FileLoader
from TigreLib.get_structure import VxTool as TigreTool
from AstraLib.get_structure import VxTool as AstraTool

from matplotlib.widgets import Slider

# --- Parameters ---
SRC = r"C:\Users\P3084\Desktop\Test\Projections_norm"
DetU = 1536
DetV = 1536
DetZ = 256

LEFT_PAD = int(DetU / 2)
RIGHT_PAD = int(DetU / 2)

# LEFT_PAD = 0
# RIGHT_PAD = 0

DetPitch = 0.084
Binning = 2
Interval = 1
Mode = "tigre" # astra / tigre
DetectorParallel = True # True: planar (flat) detector geometry, False: tilted detector geometry
ProjectionAngles = np.linspace(0, 360, DetZ, endpoint=True)
ProjectionAngles = ProjectionAngles[::Interval]
NumImgs = len(ProjectionAngles)

VolX = 768
VolY = 768
VolZ = 150

SOD = 350
SDD = 476

DetTiltX = 30
DetTiltY = 0

DetOffsetU = 0
DetOffsetV = 0

VolMidX = 0
VolMidY = 0
VolMidZ = 0

# --- Load images ---
images = FileLoader.loadImages(SRC, DetU, DetV, NumImgs, interval=Interval, binning=Binning)
print(f"Loaded: {images.shape}")

# --- Build param ---
param = VxParam(
    sod=SOD,
    sdd=SDD,
    angles=ProjectionAngles,
    tilt_x=DetTiltX,
    tilt_y=DetTiltY,
    det_width=DetU,
    det_height=DetV,
    det_pitch=DetPitch,
    offset_u=DetOffsetU,
    offset_v=DetOffsetV,
    binning=Binning,
    volume_mid_x=VolMidX,
    volume_mid_y=VolMidY,
    volume_mid_z=VolMidZ,
    dst_x=VolX,
    dst_y=VolY,
    dst_z=VolZ,
)

# --- Reconstruct ---
if Mode == "astra":
    tool = AstraTool(param, 
                     detector_parallel=DetectorParallel,
                     left_pad=LEFT_PAD,
                     right_pad=RIGHT_PAD)
else:
    tool = TigreTool(param,
                     detector_parallel=DetectorParallel,
                     left_pad=LEFT_PAD,
                     right_pad=RIGHT_PAD)

# tool.plot_geometry()
recon = tool.run(images, algo="cgls")

# recon = np.transpose(recon, (2,0,1))

# --- Normalise to 8-bit (global contrast stretch) ---
r_min = float(recon.min())
r_max = float(recon.max())
denom = r_max - r_min if r_max != r_min else 1.0
recon = (((recon - r_min) / denom) * 255.0).clip(0, 255).astype(np.uint8)

# --- Display with slice slider ---
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.12)

mid = recon.shape[0] // 2
img = ax.imshow(recon[mid], cmap='gray', vmin=0, vmax=255)
ax.set_title(f"Slice {mid}")
plt.colorbar(img, ax=ax)

ax_slider = plt.axes([0.2, 0.04, 0.6, 0.03])
slider = Slider(ax_slider, 'Slice', 0, recon.shape[0] - 1, valinit=mid, valstep=1)

def update(val):
    z = int(val)
    img.set_data(recon[z])
    ax.set_title(f"Slice {z}")
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()