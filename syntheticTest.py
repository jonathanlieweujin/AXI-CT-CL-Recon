import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from SyntheticDataGenerator import VxSyntheticDataGenerator, VxPhantomConstants
from Manager.Constants.laminography_method import LaminographyMethodConstants

# --- Parameters ---
DST = r"F:\\Dataset\\Synthetic\\HBM"
DetU = 1200
DetV = 1401
DetZ = 721

DetPitch = 0.084 * 2
Binning = 1
LaminographyMethod = LaminographyMethodConstants.INCLINED
Phantom = VxPhantomConstants.HBM
ProjectionAngles = np.linspace(0, 360, DetZ, endpoint=True)

VolX = 1200
VolY = 1200
VolZ = 700

SOD = 4.9
SDD = 490

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