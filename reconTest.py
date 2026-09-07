import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from Manager import VxManager
from Manager.Param import VxParam
from Manager.Constants.recon_method import ReconMethodConstants
from Manager.Constants.laminography_method import LaminographyMethodConstants
from Manager.Constants.flow_method import VxFlowMethod

# --- Parameters ---
SRC = r"C:\Users\User\Documents\Dataset\Button Cell\SOD 34mm SDD 622mm 130kV 500uA 3000 speed Cu Filter\Corrected"
DetU = 2803
DetV = 2401
DetZ = 721

# LEFT_PAD = int(DetU / 2)
LEFT_PAD = 0
# RIGHT_PAD = int(DetU / 2)
RIGHT_PAD = 0

DetPitch = 0.1
Binning = 4
Interval = 2
Mode = VxFlowMethod.QUALITY
LaminographyMethod = LaminographyMethodConstants.INCLINED
ProjectionAngles = np.linspace(0, 360, DetZ, endpoint=True)
ProjectionAngles = ProjectionAngles[::Interval]
NumImgs = len(ProjectionAngles)

VolX = 700
VolY = 700
VolZ = 600

SOD = 34
SDD = 622

DetTiltX = 90
DetTiltY = 0

DetOffsetU = -38.52
DetOffsetV = 0

# Offset in mm
VolMidX = 0
VolMidY = 0
VolMidZ = 0

Iterations = 1

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
    recon_method=ReconMethodConstants.FDK,
    laminography_method=LaminographyMethod,
    iterations=Iterations
)

# --- Reconstruct ---
manager = VxManager(param,
                    flow_method=Mode,
                    left_pad=LEFT_PAD,
                    right_pad=RIGHT_PAD)

images = manager.LoadImages(SRC, interval=Interval)
print(f"Loaded: {images.shape}")

recon = manager.Run()

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