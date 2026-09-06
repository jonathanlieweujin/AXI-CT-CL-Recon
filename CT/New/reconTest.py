import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from Util.param import VxParam
from Util.load_files import FileLoader
from Util.recon_method import ReconMethodConstants
from Util.laminography_method import LaminographyMethodConstants
from Util.flow_method import VxFlowMethod
from QualityFlow.get_structure import VxTool as QualityTool
from PerformanceFlow.get_structure import VxTool as PerformanceTool

# --- Parameters ---
SRC = r"C:\Users\User\Documents\Dataset\Excillum\LED 70kv\Projections_norm_90cw"
DetU = 1200
DetV = 1401
DetZ = 2000

# LEFT_PAD = int(DetU / 2)
LEFT_PAD = 0
# RIGHT_PAD = int(DetU / 2)
RIGHT_PAD = 0

DetPitch = 0.2
Binning = 2
Interval = 4
Mode = VxFlowMethod.QUALITY
LaminographyMethod = LaminographyMethodConstants.INCLINED
ProjectionAngles = np.linspace(0, -360, DetZ, endpoint=True)
ProjectionAngles = ProjectionAngles[::Interval]
NumImgs = len(ProjectionAngles)

VolX = 600
VolY = 600
VolZ = 300

SOD = 4.9
SDD = 490

DetTiltX = 51
DetTiltY = 0

DetOffsetU = -14.1
DetOffsetV = 0

# Offset in mm
VolMidX = 0
VolMidY = 0
VolMidZ = 0

Iterations = 1

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
    iterations=Iterations
)

# --- Reconstruct ---
if Mode == VxFlowMethod.PERFORMANCE:
    tool = PerformanceTool(param,
                     laminography_method=LaminographyMethod,
                     left_pad=LEFT_PAD,
                     right_pad=RIGHT_PAD)
else:
    tool = QualityTool(param,
                     laminography_method=LaminographyMethod,
                     left_pad=LEFT_PAD,
                     right_pad=RIGHT_PAD)

# tool.plot_geometry()
recon = tool.run(images, algo=ReconMethodConstants.FDK)

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