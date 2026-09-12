import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Manager import VxManager
from Manager.Param import VxParam
from Manager.Constants.recon_method import ReconMethodConstants
from Manager.Constants.laminography_method import LaminographyMethodConstants
from Manager.Constants.flow_method import VxFlowMethod

# --- Parameters ---
_cli = argparse.ArgumentParser()
_cli.add_argument("--src", default=r"path", help="Folder of raw projection images to reconstruct")
_cli.add_argument("--det-u", type=int, default=2803, help="Detector width in pixels")
_cli.add_argument("--det-v", type=int, default=2401, help="Detector height in pixels")
_cli.add_argument("--det-z", type=int, default=721, help="Number of projection angles")
_cli.add_argument("--left-pad", type=int, default=0)
_cli.add_argument("--right-pad", type=int, default=0)
_cli.add_argument("--det-pitch", type=float, default=0.1, help="Detector pixel pitch, mm")
_cli.add_argument("--binning", type=int, default=4)
_cli.add_argument("--interval", type=int, default=2, help="Use every Nth projection angle")
_cli.add_argument("--vol-x", type=int, default=700)
_cli.add_argument("--vol-y", type=int, default=700)
_cli.add_argument("--vol-z", type=int, default=600)
_cli.add_argument("--sod", type=float, default=34.0, help="Source-to-object distance, mm")
_cli.add_argument("--sdd", type=float, default=622.0, help="Source-to-detector distance, mm")
_cli.add_argument("--tilt-x", type=float, default=90.0, help="DetTiltX in degrees")
_cli.add_argument("--tilt-y", type=float, default=0.0, help="DetTiltY in degrees")
_cli.add_argument("--offset-u", type=float, default=-38.52, help="DetOffsetU in pixels")
_cli.add_argument("--offset-v", type=float, default=0.0, help="DetOffsetV in pixels")
_cli.add_argument("--iterations", type=int, default=1)
_args, _ = _cli.parse_known_args()

SRC = _args.src
DetU = _args.det_u
DetV = _args.det_v
DetZ = _args.det_z

LEFT_PAD = _args.left_pad
RIGHT_PAD = _args.right_pad

DetPitch = _args.det_pitch
Binning = _args.binning
Interval = _args.interval
Mode = VxFlowMethod.QUALITY
LaminographyMethod = LaminographyMethodConstants.INCLINED
ProjectionAngles = np.linspace(0, 360, DetZ, endpoint=True)
ProjectionAngles = ProjectionAngles[::Interval]
NumImgs = len(ProjectionAngles)

VolX = _args.vol_x
VolY = _args.vol_y
VolZ = _args.vol_z
DstPixelFormat = "u8"

SOD = _args.sod
SDD = _args.sdd

DetTiltX = _args.tilt_x
DetTiltY = _args.tilt_y

DetOffsetU = _args.offset_u
DetOffsetV = _args.offset_v

# Offset in mm
VolMidX = 0
VolMidY = 0
VolMidZ = 0

Iterations = _args.iterations

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
    dst_pixel_format=DstPixelFormat,
    recon_method=ReconMethodConstants.FDK,
    laminography_method=LaminographyMethod,
    iterations=Iterations
)

# --- Reconstruct ---
manager = VxManager()
manager.SetParam(param)
manager.SetFlowMethod(Mode)
manager.SetLeftPad(LEFT_PAD)
manager.SetRightPad(RIGHT_PAD)

images = manager.LoadImages(SRC, interval=Interval)
print(f"Loaded: {images.shape}")

manager.Run()
recon = manager.Result

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