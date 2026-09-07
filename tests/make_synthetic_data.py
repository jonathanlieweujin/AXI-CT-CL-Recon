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

from Manager.Constants.laminography_method import LaminographyMethodConstants
from SyntheticDataGenerator import VxSyntheticDataGenerator, VxPhantomConstants

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


def main():
    angles = np.linspace(0, 360, N_PROJ, endpoint=True)

    for name, method, tilt_x, phantom in DATASETS:
        manager = VxSyntheticDataGenerator(phantom=phantom)
        manager.SetParams(
            angles=angles,
            sod=SOD, sdd=SDD,
            det_width=DET_U, det_height=DET_V, det_pitch=DET_PITCH,
            volume=VOL,
            tilt_x=tilt_x,
            binning=BINNING,
            laminography_method=method,
        )
        manager.Generate(os.path.join(DATASET_ROOT, name))

        print(f"{name:20s} {manager.Describe()}")


if __name__ == "__main__":
    main()
