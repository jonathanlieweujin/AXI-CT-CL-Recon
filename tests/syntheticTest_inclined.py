import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SyntheticDataGenerator import VxSyntheticDataGenerator, VxPhantomConstants
from Manager.Constants.laminography_method import LaminographyMethodConstants

# --- Parameters ---
_cli = argparse.ArgumentParser()
_cli.add_argument("--tilt", type=float, default=51.0,
                   help="DetTiltX in degrees (each tilt/offset combination gets its own output subfolder)")
_cli.add_argument("--offset-u", type=float, default=0.0, help="DetOffsetU in pixels")
_cli.add_argument("--offset-v", type=float, default=0.0, help="DetOffsetV in pixels")
# Note: noise is to replicate real application characteristics (a noisy
# image, air not landing on an exact 0); suggested to not be enabled for
# ground truth testing, e.g. validating a geometry-recovery optimiser
# against a known-exact tilt/offset.
_cli.add_argument("--no-noise", action="store_true",
                   help="Disable synthetic detector noise (on by default)")
_args, _ = _cli.parse_known_args()

Phantom = VxPhantomConstants.MICRO_JIG
ToAddNoise = not _args.no_noise

DetTiltX = _args.tilt
_suffix = f"_tilt{DetTiltX:g}"
if _args.offset_u or _args.offset_v:
    _suffix += f"_offU{_args.offset_u:g}_offV{_args.offset_v:g}"
if not ToAddNoise:
    _suffix += "_noNoise"
DST = rf"output\Synthetic\{Phantom}{_suffix}"
DetU = 600
DetV = 700
DetZ = 721

DetPitch = 0.4
Binning = 1

LaminographyMethod = LaminographyMethodConstants.INCLINED
ProjectionAngles = np.linspace(0, 360, DetZ, endpoint=True)

VolX = 600
VolY = 600
VolZ = 350

SOD = 10
SDD = 490

DetTiltY = 0

DetOffsetU = _args.offset_u
DetOffsetV = _args.offset_v

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
    toAddNoise=ToAddNoise,
)

generator.Run()
print(generator.Describe())

generator.Save(DST)
print(f"Saved: {DST}")