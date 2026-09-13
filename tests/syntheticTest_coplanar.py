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
_cli.add_argument("--tilt", type=float, default=30.0, help="DetTiltX in degrees")
_cli.add_argument("--offset-u", type=float, default=0.0, help="DetOffsetU in pixels")
_cli.add_argument("--offset-v", type=float, default=0.0, help="DetOffsetV in pixels")
_cli.add_argument("--det-u", type=int, default=768, help="Detector width in pixels")
_cli.add_argument("--det-v", type=int, default=768, help="Detector height in pixels")
_cli.add_argument("--det-z", type=int, default=721, help="Number of projection angles")
_cli.add_argument("--det-pitch", type=float, default=0.085 * 2, help="Detector pixel pitch, mm")
_cli.add_argument("--binning", type=int, default=1)
_cli.add_argument("--vol-x", type=int, default=768)
_cli.add_argument("--vol-y", type=int, default=768)
_cli.add_argument("--vol-z", type=int, default=250)
_cli.add_argument("--sod", type=float, default=17.0, help="Source-to-object distance, mm")
_cli.add_argument("--sdd", type=float, default=476.0, help="Source-to-detector distance, mm")
# Note: noise is to replicate real application characteristics (a noisy
# image, air not landing on an exact 0); suggested to not be enabled for
# ground truth testing, e.g. validating a geometry-recovery optimiser
# against a known-exact tilt/offset.
_cli.add_argument("--no-noise", action="store_true",
                   help="Disable synthetic detector noise (on by default)")
_args, _ = _cli.parse_known_args()

Phantom = VxPhantomConstants.PCB_PANEL
ToAddNoise = not _args.no_noise

DetTiltX = _args.tilt
_suffix = f"_tilt{DetTiltX:g}"
if _args.offset_u or _args.offset_v:
    _suffix += f"_offU{_args.offset_u:g}_offV{_args.offset_v:g}"
if not ToAddNoise:
    _suffix += "_noNoise"
DST = rf"output\Synthetic\{Phantom}{_suffix}"
DetU = _args.det_u
DetV = _args.det_v
DetZ = _args.det_z

DetPitch = _args.det_pitch
Binning = _args.binning
LaminographyMethod = LaminographyMethodConstants.COPLANAR

ProjectionAngles = np.linspace(0, 360, DetZ, endpoint=True)

VolX = _args.vol_x
VolY = _args.vol_y
VolZ = _args.vol_z

SOD = _args.sod
SDD = _args.sdd

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