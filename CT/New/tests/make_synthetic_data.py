"""
Generate the three synthetic datasets used by test_flow_equivalence.py.

    python tests/make_synthetic_data.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Util.param import VxParam
from Util.laminography_method import LaminographyMethodConstants
from GenSyntheticData.get_structure import VxSyntheticData, VxPhantomConstants

DATASET_ROOT = r"C:\Users\User\Documents\Dataset"

DET_U = 256
DET_V = 256
DET_PITCH = 0.2
N_PROJ = 120
SOD = 152.0
SDD = 511.0

INCLINED = LaminographyMethodConstants.INCLINED
COPLANAR = LaminographyMethodConstants.COPLANAR

DATASETS = [
    ("Synthetic CT", INCLINED, 90.0, (192, 192, 128), VxPhantomConstants.SOLID),
    ("Synthetic CL", INCLINED, 55.0, (192, 192, 48), VxPhantomConstants.SLAB),
    ("Synthetic Coplanar", COPLANAR, 55.0, (192, 192, 48), VxPhantomConstants.SLAB),
]


def main():
    angles = np.linspace(0, 360, N_PROJ, endpoint=False)

    for name, method, tilt_x, (vol_x, vol_y, vol_z), phantom in DATASETS:
        param = VxParam(
            sod=SOD, sdd=SDD, angles=angles,
            tilt_x=tilt_x, tilt_y=0.0,
            det_width=DET_U, det_height=DET_V, det_pitch=DET_PITCH,
            offset_u=0.0, offset_v=0.0, binning=1,
            volume_mid_x=0.0, volume_mid_y=0.0, volume_mid_z=0.0,
            dst_x=vol_x, dst_y=vol_y, dst_z=vol_z, iterations=1,
        )

        gen = VxSyntheticData(param, laminography_method=method, phantom=phantom)
        gen.write(os.path.join(DATASET_ROOT, name))

        print(f"{name:20s} {gen.projections.shape} projections  vol={gen.phantom.shape}  "
              f"range=[{gen.projections.min():.3f}, {gen.projections.max():.3f}]")


if __name__ == "__main__":
    main()
