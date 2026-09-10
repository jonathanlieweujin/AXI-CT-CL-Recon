"""
Linear attenuation coefficients in mm^-1 at 60 keV, from the NIST XCOM mass
attenuation coefficient (mu/rho) times the material density.

Phantoms label voxels with these rather than with arbitrary numbers, so a
forward projection through a volume gridded in mm is a physical line integral.

  air                 -                  0.000
  solder mask / epoxy 1.30 g/cc          0.042
  FR4 laminate        1.85               0.055
  BT resin substrate  1.90               0.060
  silicon             2.33 x 0.3207      0.075
  copper              8.96 x 1.593       1.427
  SAC305 solder       7.37 x ~6.51       4.790   (96.5Sn/3.0Ag/0.5Cu)

Solder is ~64x silicon and ~3.4x copper, which is why a ball reads as opaque
and a void is the only thing visible through it.
"""

MU_AIR    = 0.000
MU_MASK   = 0.042
MU_FR4    = 0.055
MU_BT     = 0.060
MU_SI     = 0.075
MU_CU     = 1.427
MU_SOLDER = 4.790
