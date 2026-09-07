import numpy as np

from Manager.Param import VxParam

class VxComputeGeometry:
    """Laminography geometry builder"""
    
    # This planar flow does not use the old rotation-matrix extraction:
    #             U = row0 * pixel, V = -row1 * pixel,
    #             S = -row2 * SOD, D = row2 * ODD.
    #
    #         Instead, SOD and ODD are interpreted as axial distances along Z.
    #         The laminography tilt creates only an XY orbit shift. For a point at
    #         axial distance z from the object plane, the in-plane shift is:
    #             r = z * tan(tilt_x)
    #
    #         With phi = -deg2rad(angle), the source and detector centers are:
    #             Sx = -SOD * tan(tilt_x) * cos(phi)
    #             Sy = -SOD * tan(tilt_x) * sin(phi)
    #             Sz = -SOD
    #
    #             Dx =  ODD * tan(tilt_x) * cos(phi)
    #             Dy =  ODD * tan(tilt_x) * sin(phi)
    #             Dz =  ODD
    #
    #         The detector plane is kept flat in the world frame, so the pixel axes
    #         are constant for all projections:
    #             U = (det_pitch, 0, 0)
    #             V = (0, +det_pitch, 0)
    #
    #         Detector offsets move D along those U/V axes:
    #             D = D + offset_u * U + offset_v * V
    @staticmethod
    def computeCoplanarTranslationalLaminographyGeometry(param: VxParam) -> np.ndarray:
        """Compute coplanar laminography setup"""
        p = param

        tilt_x_rad = -np.deg2rad(p.tilt_x)  # tilt x negated
        angles_rad = -np.deg2rad(p.angles)

        eff_sod = p.sod
        eff_odd = p.sdd - p.sod
        eff_pixel = p.det_pitch

        tan_tx = np.tan(tilt_x_rad)
        cos_phi = np.cos(angles_rad)
        sin_phi = np.sin(angles_rad)

        N = p.num_of_imgs
        vecs = np.zeros((N, 12), dtype=np.float64)

        # Source center: axial SOD plus tilt-induced XY shift.
        vecs[:, 0] = eff_sod * tan_tx * cos_phi
        vecs[:, 1] = eff_sod * tan_tx * sin_phi
        vecs[:, 2] = eff_sod

        # Detector center: axial ODD plus matching tilt-induced XY shift.
        vecs[:, 3] = -eff_odd * tan_tx * cos_phi
        vecs[:, 4] = -eff_odd * tan_tx * sin_phi
        vecs[:, 5] = -eff_odd

        # Flat detector basis in world coordinates.
        vecs[:, 6] = eff_pixel
        vecs[:, 7] = 0.0
        vecs[:, 8] = 0.0

        vecs[:,  9] = 0.0
        vecs[:, 10] = +eff_pixel
        vecs[:, 11] = 0.0

        # Detector offsets
        vecs[:, 3] += p.offset_u * vecs[:, 6] + p.offset_v * vecs[:, 9]
        vecs[:, 4] += p.offset_u * vecs[:, 7] + p.offset_v * vecs[:, 10]
        vecs[:, 5] += p.offset_u * vecs[:, 8] + p.offset_v * vecs[:, 11]

        return vecs.reshape(N, 3, 4)

    # ref: https://doi.org/10.1088/1361-6501/aafcae
    # Laminography in the lab: imaging planar objects using a conventional x-ray CT scanner
    # θ = detTiltXRad,  β = detTiltYRad,  φ = angleRadContribution
    #
    # M1 = Rx(-θ) = [ 1      0       0   ]
    #               [ 0    cosθ    sinθ   ]
    #               [ 0   -sinθ    cosθ   ]
    #
    # M2 = Ry(β)  = [ cosβ    0   -sinβ  ]
    #               [  0      1     0    ]
    #               [ sinβ    0    cosβ  ]
    #
    # M3 = Rz(-φ) = [ cosφ   sinφ    0   ]
    #               [-sinφ   cosφ    0   ]
    #               [  0      0      1   ]
    #
    # Step 1 — A = M1·M2:
    #
    #   A[0][0] = 1·cosβ  + 0·0 + 0·sinβ   =  cosβ
    #   A[0][1] = 1·0     + 0·1 + 0·0      =  0
    #   A[0][2] = 1·(-sinβ) + 0·0 + 0·cosβ =  -sinβ
    #
    #   A[1][0] = 0·cosβ  + cosθ·0 + sinθ·sinβ  =  sinθ sinβ
    #   A[1][1] = 0·0     + cosθ·1 + sinθ·0     =  cosθ
    #   A[1][2] = 0·(-sinβ) + cosθ·0 + sinθ·cosβ =  sinθ cosβ
    #
    #   A[2][0] = 0·cosβ  + (-sinθ)·0 + cosθ·sinβ  =  cosθ sinβ
    #   A[2][1] = 0·0     + (-sinθ)·1 + cosθ·0     = -sinθ
    #   A[2][2] = 0·(-sinβ) + (-sinθ)·0 + cosθ·cosβ =  cosθ cosβ
    #
    #        [  cosβ      0       -sinβ  ]
    # A  =   [  sinθ sinβ  cosθ   sinθ cosβ ]
    #        [  cosθ sinβ -sinθ   cosθ cosβ ]
    #
    # Step 2 — R = A·M3:
    #
    #   R[0][0] = cosβ·cosφ   + 0·(-sinφ)      + (-sinβ)·0   =  cosβ cosφ
    #   R[0][1] = cosβ·sinφ   + 0·cosφ         + (-sinβ)·0   =  cosβ sinφ
    #   R[0][2] = cosβ·0      + 0·0            + (-sinβ)·1   = -sinβ
    #
    #   R[1][0] = sinθ sinβ·cosφ + cosθ·(-sinφ) + sinθ cosβ·0  =  sinθ sinβ cosφ - cosθ sinφ
    #   R[1][1] = sinθ sinβ·sinφ + cosθ·cosφ   + sinθ cosβ·0  =  sinθ sinβ sinφ + cosθ cosφ
    #   R[1][2] = sinθ sinβ·0   + cosθ·0       + sinθ cosβ·1  =  sinθ cosβ
    #
    #   R[2][0] = cosθ sinβ·cosφ + (-sinθ)·(-sinφ) + cosθ cosβ·0  =  cosθ sinβ cosφ + sinθ sinφ
    #   R[2][1] = cosθ sinβ·sinφ + (-sinθ)·cosφ   + cosθ cosβ·0  =  cosθ sinβ sinφ - sinθ cosφ
    #   R[2][2] = cosθ sinβ·0   + (-sinθ)·0       + cosθ cosβ·1  =  cosθ cosβ
    #
    #        [  cosβ cosφ                  cosβ sinφ                 -sinβ      ]  <- row0
    # R  =   [  sinθ sinβ cosφ - cosθ sinφ   sinθ sinβ sinφ + cosθ cosφ   sinθ cosβ  ]  <- row1
    #        [  cosθ sinβ cosφ + sinθ sinφ   cosθ sinβ sinφ - sinθ cosφ   cosθ cosβ  ]  <- row2
    #
    # Extraction:
    #   U  = +row0 · effPixelSize
    #   V  = -row1 · effPixelSize
    #   S  = -row2 · effSOD
    #   D  = +row2 · effODD
    @staticmethod
    def computeDefaultInclinedLaminographyGeometry(param: VxParam) -> np.ndarray:
        """Compute inclined laminography setup - Default"""
        p = param

        # The inclined flow takes alpha - 180 and a negated theta before the rig rotation is built.
        tilt_x_deg = p.tilt_x - 180.0
        angles_deg = -np.asarray(p.angles)

        tilt_x_rad = np.deg2rad(tilt_x_deg)  # θ, already alpha - 180
        tilt_y_rad = np.deg2rad(p.tilt_y)    # β

        angles_rad = -np.deg2rad(angles_deg)  # negated: matches -angleDeg[i] * PI / 180

        # Effective geometry (voxel domain, voxel_size = 1)
        eff_sod = p.sod
        eff_odd = p.sdd - p.sod
        eff_pixel = p.det_pitch

        sin_tx = np.sin(tilt_x_rad)   # sinθ
        cos_tx = np.cos(tilt_x_rad)   # cosθ
        sin_ty = np.sin(tilt_y_rad)   # sinβ
        cos_ty = np.cos(tilt_y_rad)   # cosβ

        cos_phi = np.cos(angles_rad)   # (N,)
        sin_phi = np.sin(angles_rad)   # (N,)

        # R = Rx(-θ) · Ry(β) · Rz(-φ)  (broadcast over N angles)
        # row0
        r00 =  cos_ty * cos_phi
        r01 =  cos_ty * sin_phi
        r02 = -sin_ty * np.ones_like(cos_phi)

        # row1
        r10 =  sin_tx * sin_ty * cos_phi - cos_tx * sin_phi
        r11 =  sin_tx * sin_ty * sin_phi + cos_tx * cos_phi
        r12 =  sin_tx * cos_ty * np.ones_like(cos_phi)

        # row2
        r20 =  cos_tx * sin_ty * cos_phi + sin_tx * sin_phi
        r21 =  cos_tx * sin_ty * sin_phi - sin_tx * cos_phi
        r22 =  cos_tx * cos_ty * np.ones_like(cos_phi)

        N = p.num_of_imgs
        vecs = np.zeros((N, 12), dtype=np.float64)

        # S = -row2 · eff_sod
        vecs[:, 0] = -eff_sod * r20
        vecs[:, 1] = -eff_sod * r21
        vecs[:, 2] = -eff_sod * r22

        # D = +row2 · eff_odd
        vecs[:, 3] = eff_odd * r20
        vecs[:, 4] = eff_odd * r21
        vecs[:, 5] = eff_odd * r22

        # U = +row0 · eff_pixel
        vecs[:, 6] = eff_pixel * r00
        vecs[:, 7] = eff_pixel * r01
        vecs[:, 8] = eff_pixel * r02

        # V = -row1 · eff_pixel
        vecs[:, 9]  = -eff_pixel * r10
        vecs[:, 10] = -eff_pixel * r11
        vecs[:, 11] = -eff_pixel * r12

        # Apply detector offsets along U and V axes
        vecs[:, 3] += p.offset_u * vecs[:, 6] + p.offset_v * vecs[:, 9]
        vecs[:, 4] += p.offset_u * vecs[:, 7] + p.offset_v * vecs[:, 10]
        vecs[:, 5] += p.offset_u * vecs[:, 8] + p.offset_v * vecs[:, 11]

        # Return as (N, 3, 4): each matrix is [D | U | V | S]
        return vecs.reshape(N, 3, 4)
