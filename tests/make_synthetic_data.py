"""
Generate the three synthetic datasets used by test_flow_equivalence.py.

Geometry imitates the Excillum LED setup: 1401x1200 detector at 0.2 mm pitch,
721 projections, SOD 4.9 / SDD 490. Projections are stored at full detector
resolution and binned by 2 on load, as a real acquisition is, giving a 0.004 mm
voxel and a 700x700x600 volume that fills the 2.8 x 2.4 mm field of view.

    python tests/make_synthetic_data.py

Roughly 4.5 GB of projections per dataset.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Manager.Param import VxParam
from Manager.Constants.laminography_method import LaminographyMethodConstants
from GenSyntheticData.get_structure import VxSyntheticData, VxPhantomConstants

DATASET_ROOT = r"C:\Users\User\Documents\Dataset"

DET_U = 1401          # stored (unbinned) detector width
DET_V = 1200          # stored (unbinned) detector height
DET_PITCH = 0.2       # unbinned pitch
BINNING = 2
N_PROJ = 721
SOD = 4.9
SDD = 490.0
VOL = (700, 700, 600)

INCLINED = LaminographyMethodConstants.INCLINED
COPLANAR = LaminographyMethodConstants.COPLANAR

DATASETS = [
    ("Synthetic CT", INCLINED, 90.0, VxPhantomConstants.SOLID),
    ("Synthetic CL", INCLINED, 51.0, VxPhantomConstants.SLAB),
    ("Synthetic Coplanar", COPLANAR, 51.0, VxPhantomConstants.SLAB),
]


def build(tilt_x, angles, binning, method):
    vol_x, vol_y, vol_z = VOL
    return VxParam(
        sod=SOD, sdd=SDD, angles=angles,
        tilt_x=tilt_x, tilt_y=0.0,
        det_width=DET_U, det_height=DET_V, det_pitch=DET_PITCH,
        offset_u=0.0, offset_v=0.0, binning=binning,
        volume_mid_x=0.0, volume_mid_y=0.0, volume_mid_z=0.0,
        dst_x=vol_x, dst_y=vol_y, dst_z=vol_z,
        laminography_method=method, iterations=1,
    )


def main():
    angles = np.linspace(0, 360, N_PROJ, endpoint=True)

    for name, method, tilt_x, phantom in DATASETS:
        recon = build(tilt_x, angles, BINNING, method)   # binned: drives the volume grid
        acq = build(tilt_x, angles, 1, method)           # unbinned: the stored detector

        gen = VxSyntheticData(recon, phantom=phantom, acquisition_param=acq)
        gen.write(os.path.join(DATASET_ROOT, name))

        voxel = gen.voxel_size
        print(f"{name:20s} tilt={tilt_x:g} {str(method):9s} "
              f"det {int(acq.det_width)}x{int(acq.det_height)} -> "
              f"{int(recon.det_width)}x{int(recon.det_height)} binned  "
              f"voxel={voxel:.4f}mm  vol={gen.phantom.shape} "
              f"({VOL[0]*voxel:.2f}x{VOL[1]*voxel:.2f}x{VOL[2]*voxel:.2f}mm)  "
              f"proj range=[{gen._sinogram.min():.3f}, {gen._sinogram.max():.3f}]")


if __name__ == "__main__":
    main()
