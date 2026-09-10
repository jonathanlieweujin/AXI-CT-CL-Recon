import numpy as np

from SyntheticDataGenerator import VxSyntheticDataGenerator, VxPhantomConstants
from SyntheticDataGenerator.Structures.materials import MU_MASK
from SyntheticDataGenerator.Structures.mesh import MeshStructure
from Manager.Constants.laminography_method import LaminographyMethodConstants

# --- Parameters ---
LaminographyMethod = LaminographyMethodConstants.INCLINED
DST = rf"ouput\Model"

Model = r"tests\model\Greymon3D.obj"
# STL and OBJ carry no units. This model measures 3.494 x 3.500 x 5.033 in its
# own units, so 2.0 makes it a ~10 mm tall part.
ModelScale = 2.0
ModelMu = MU_MASK          # resin / plastic, 0.042 mm^-1 at 60 keV

DetU = 700
DetV = 700
DetZ = 360

DetPitch = 0.084
Binning = 1

# Ignored while a model path is given, but SetParams still wants a phantom.
Phantom = VxPhantomConstants.SOLID
ProjectionAngles = np.linspace(0, 360, DetZ, endpoint=True)

# Reconstruction volume only: the phantom box comes from the model bounding
# box. 384 x 384 x 512 at the 0.02 mm voxel below covers 7.68 x 7.68 x 10.24 mm.
VolX = 384
VolY = 384
VolZ = 512

# voxel = DetPitch * SOD / SDD = 0.084 * 100 / 420 = 0.02 mm
SOD = 100
SDD = 420

# A free-standing object, so upright CT rather than a laminographic tilt.
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
