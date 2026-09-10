import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from SyntheticDataGenerator import VxSyntheticDataGenerator, VxPhantomConstants
from Manager.Constants.laminography_method import LaminographyMethodConstants

# --- Parameters ---
Phantom = VxPhantomConstants.PCB_PANEL
DST = rf"output\Synthetic\{Phantom}"
DetU = 500
DetV = 500
DetZ = 721

DetPitch = 0.3
Binning = 1
LaminographyMethod = LaminographyMethodConstants.COPLANAR

ProjectionAngles = np.linspace(0, 360, DetZ, endpoint=True)

VolX = 500
VolY = 500
VolZ = 100

SOD = 17
SDD = 476

DetTiltX = 51
DetTiltY = 0

DetOffsetU = 0
DetOffsetV = 0

# Offset in mm
VolMidX = 0
VolMidY = 0
VolMidZ = 0

# --- Generate ---
generator = VxSyntheticDataGenerator(phantom=Phantom)

param = generator.SetParams(
    angles=ProjectionAngles,
    sod=SOD,
    sdd=SDD,
    det_width=DetU,
    det_height=DetV,
    det_pitch=DetPitch,
    volume=(VolX, VolY, VolZ),
    tilt_x=DetTiltX,
    tilt_y=DetTiltY,
    binning=Binning,
    offset_u=DetOffsetU,
    offset_v=DetOffsetV,
    volume_mid=(VolMidX, VolMidY, VolMidZ),
    laminography_method=LaminographyMethod,
)

generator.Run()
print(generator.Describe())

generator.Save(DST)
print(f"Saved: {DST}")