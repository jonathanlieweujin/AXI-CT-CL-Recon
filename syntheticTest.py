import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from SyntheticDataGenerator import VxSyntheticDataGenerator, VxPhantomConstants
from Manager.Constants.laminography_method import LaminographyMethodConstants

# --- Parameters ---
DST = r"path"
DetU = 1401
DetV = 1200
DetZ = 721

DetPitch = 0.2
Binning = 2
LaminographyMethod = LaminographyMethodConstants.INCLINED
Phantom = VxPhantomConstants.SOLID
ProjectionAngles = np.linspace(0, 360, DetZ, endpoint=True)

VolX = 700
VolY = 700
VolZ = 600

SOD = 4.9
SDD = 490

DetTiltX = 90
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