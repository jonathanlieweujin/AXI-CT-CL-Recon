import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SyntheticDataGenerator import VxSyntheticDataGenerator, VxPhantomConstants
from SyntheticDataGenerator.Structures.materials import MU_MASK
from SyntheticDataGenerator.Structures.mesh import MeshStructure
from Manager.Constants.laminography_method import LaminographyMethodConstants

# --- Parameters ---
_cli = argparse.ArgumentParser()
_cli.add_argument("--model", default=r"tests\model\Greymon3D.obj", help=".stl or .obj model path")
_cli.add_argument("--model-scale", type=float, default=2.0,
                   help="model units -> mm (STL/OBJ carry no units)")
_cli.add_argument("--tilt", type=float, default=90.0,
                   help="DetTiltX in degrees (90 = upright CT for a free-standing object)")
_cli.add_argument("--offset-u", type=float, default=0.0, help="DetOffsetU in pixels")
_cli.add_argument("--offset-v", type=float, default=0.0, help="DetOffsetV in pixels")
_cli.add_argument("--det-u", type=int, default=700, help="Detector width in pixels")
_cli.add_argument("--det-v", type=int, default=700, help="Detector height in pixels")
_cli.add_argument("--det-z", type=int, default=360, help="Number of projection angles")
_cli.add_argument("--det-pitch", type=float, default=0.084, help="Detector pixel pitch, mm")
_cli.add_argument("--binning", type=int, default=1)
_cli.add_argument("--vol-x", type=int, default=384)
_cli.add_argument("--vol-y", type=int, default=384)
_cli.add_argument("--vol-z", type=int, default=512)
_cli.add_argument("--sod", type=float, default=100.0, help="Source-to-object distance, mm")
_cli.add_argument("--sdd", type=float, default=420.0, help="Source-to-detector distance, mm")
# Note: noise is to replicate real application characteristics (a noisy
# image, air not landing on an exact 0); suggested to not be enabled for
# ground truth testing, e.g. validating a geometry-recovery optimiser
# against a known-exact tilt/offset.
_cli.add_argument("--no-noise", action="store_true",
                   help="Disable synthetic detector noise (on by default)")
_args, _ = _cli.parse_known_args()

LaminographyMethod = LaminographyMethodConstants.INCLINED
ToAddNoise = not _args.no_noise
_suffix = f"_tilt{_args.tilt:g}"
if _args.offset_u or _args.offset_v:
    _suffix += f"_offU{_args.offset_u:g}_offV{_args.offset_v:g}"
if not ToAddNoise:
    _suffix += "_noNoise"
DST = rf"output\Model{_suffix}"

Model = _args.model
ModelScale = _args.model_scale
ModelMu = MU_MASK          # resin / plastic, 0.042 mm^-1 at 60 keV

DetU = _args.det_u
DetV = _args.det_v
DetZ = _args.det_z

DetPitch = _args.det_pitch
Binning = _args.binning

# Ignored while a model path is given, but SetParams still wants a phantom.
Phantom = VxPhantomConstants.SOLID
ProjectionAngles = np.linspace(0, 360, DetZ, endpoint=True)

# Reconstruction volume only: the phantom box comes from the model bounding
# box. 384 x 384 x 512 at the 0.02 mm voxel below covers 7.68 x 7.68 x 10.24 mm.
VolX = _args.vol_x
VolY = _args.vol_y
VolZ = _args.vol_z

# voxel = DetPitch * SOD / SDD = 0.084 * 100 / 420 = 0.02 mm
SOD = _args.sod
SDD = _args.sdd

# A free-standing object, so upright CT rather than a laminographic tilt.
DetTiltX = _args.tilt
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

# What the model looks like before it is voxelised. Load cleans and
# triangulates, so a watertight model reports 0 open edges here; anything else
# means the solid fill will leak and the phantom will be wrong.
poly = MeshStructure.Load(Model, scale=ModelScale)
print(f"Model: {poly.GetNumberOfCells()} triangles, "
      f"{MeshStructure.IsWatertight(poly)} open edges, "
      f"bounds %.2f x %.2f x %.2f mm" % MeshStructure.BoundsMm(poly))
print(f"GPU: {VxSyntheticDataGenerator.GpuMemoryMb()} MB, "
      f"phantom budget {generator.MaxPhantomVoxels:,} voxels")

generator.Run(model3DPath=Model, model_scale=ModelScale, model_mu=ModelMu)
print(generator.Describe())
print(f"Filled: {generator.Manifest['filled_mm3']:.2f} mm^3")

generator.Save(DST)
print(f"Saved: {DST}")
