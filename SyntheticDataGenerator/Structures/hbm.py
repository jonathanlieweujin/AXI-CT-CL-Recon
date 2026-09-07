import numpy as np


class HBMStructure:
    """
    Representative HBM (High Bandwidth Memory) package cross-section phantom
    for computed laminography validation at high magnification.

    Three bump generations at physical scale (midpoint of spec ranges):

      μbumps   : dia 12.5 μm, pitch 27.5 μm  — DRAM layer-to-layer via TSVs
      C4 bumps : dia 27.5 μm, pitch 47.5 μm  — base die → silicon interposer
      BGA balls: dia  450 μm, pitch  900 μm  — package substrate → PCB

    Stack layout (bottom → top, ascending z):
      silicon interposer  100 μm
      C4 bump array        60 μm
      base logic die      100 μm
      8 × (μbump layer + DRAM die)  8 × (12.5 + 50) μm = 500 μm
      total stack height            ≈ 760 μm

    Material attenuation labels (arbitrary linear scale, forward-projection safe):
      0.0  background / air
      0.6  silicon (die / interposer)
      1.2  solder (SnAg μbumps and C4 bumps)
      1.8  copper (BGA balls — thicker shell, higher Z)
    """

    _UBUMP_DIA_UM  = 12.5
    _UBUMP_PITCH_UM = 27.5
    _C4_DIA_UM     = 27.5
    _C4_PITCH_UM   = 47.5
    _BGA_DIA_UM    = 450.0
    _BGA_PITCH_UM  = 900.0
    _DRAM_T_UM     = 50.0    # DRAM die thickness per layer
    _BASE_T_UM     = 100.0   # base logic die thickness
    _INTERP_T_UM   = 100.0   # silicon interposer thickness
    _C4_H_UM       = 60.0    # C4 bump standoff height
    _N_DRAM        = 8       # number of DRAM die layers

    @staticmethod
    def GetStructure(vol_shape: tuple[int, int, int], voxel_size_mm: float) -> np.ndarray:
        """
        Build a float32 (z, y, x) phantom representing an HBM package
        cross-section at the given voxel resolution.

        Args:
            vol_shape:     (vol_z, vol_y, vol_x) in voxels.
            voxel_size_mm: isotropic voxel edge length in mm.

        Returns:
            np.ndarray, float32, shape (vol_z, vol_y, vol_x).
        """
        vol_z, vol_y, vol_x = vol_shape
        vs_um = voxel_size_mm * 1_000.0  # mm → μm per voxel

        vol = np.zeros((vol_z, vol_y, vol_x), dtype=np.float32)

        H = HBMStructure

        # 1-D coordinate arrays in μm, centred at volume midpoint
        y1d = (np.arange(vol_y, dtype=np.float32) - vol_y / 2.0) * vs_um
        x1d = (np.arange(vol_x, dtype=np.float32) - vol_x / 2.0) * vs_um

        def bump_grid_2d(pitch_um: float, radius_um: float) -> np.ndarray:
            """
            Boolean (vol_y, vol_x) mask — True inside any bump of an infinite
            regular grid with the given pitch, centred at the volume midpoint.

            Uses modular arithmetic so no loop over bump centres is needed:
            map each coordinate into [-pitch/2, pitch/2) to get the distance to
            the nearest grid point, then apply a radius threshold.
            """
            xm = (x1d % pitch_um + pitch_um * 0.5) % pitch_um - pitch_um * 0.5
            ym = (y1d % pitch_um + pitch_um * 0.5) % pitch_um - pitch_um * 0.5
            return xm[None, :] ** 2 + ym[:, None] ** 2 < radius_um ** 2  # (vol_y, vol_x)

        def z_slice(z_um_start: float, thickness_um: float) -> tuple[int, int]:
            """μm offsets from stack bottom → clamped voxel slice bounds."""
            lo = z_bot_vox + int(z_um_start / vs_um)
            hi = z_bot_vox + int((z_um_start + thickness_um) / vs_um)
            return max(0, min(lo, vol_z)), max(0, min(hi, vol_z))

        # Total physical height of the stack
        stack_um = (H._INTERP_T_UM
                    + H._C4_H_UM
                    + H._BASE_T_UM
                    + H._N_DRAM * (H._UBUMP_DIA_UM + H._DRAM_T_UM))

        # Centre the stack in z
        z_bot_vox = int(vol_z / 2.0 - stack_um / vs_um / 2.0)

        # Build layers from bottom (interposer) upward
        z = 0.0  # cursor in μm from stack bottom

        # Silicon interposer
        lo, hi = z_slice(z, H._INTERP_T_UM)
        vol[lo:hi, :, :] = 0.6
        z += H._INTERP_T_UM

        # C4 flip-chip bumps (solder, SnAg)
        c4 = bump_grid_2d(H._C4_PITCH_UM, H._C4_DIA_UM * 0.5)
        lo, hi = z_slice(z, H._C4_H_UM)
        vol[lo:hi, c4] = 1.2
        z += H._C4_H_UM

        # Base logic die (silicon)
        lo, hi = z_slice(z, H._BASE_T_UM)
        vol[lo:hi, :, :] = 0.6
        z += H._BASE_T_UM

        # 8 DRAM layers — μbump array then DRAM die, repeated
        ubump = bump_grid_2d(H._UBUMP_PITCH_UM, H._UBUMP_DIA_UM * 0.5)
        for _ in range(H._N_DRAM):
            lo, hi = z_slice(z, H._UBUMP_DIA_UM)
            vol[lo:hi, ubump] = 1.2          # solder μbumps
            z += H._UBUMP_DIA_UM
            lo, hi = z_slice(z, H._DRAM_T_UM)
            vol[lo:hi, :, :] = 0.6          # DRAM silicon die
            z += H._DRAM_T_UM

        # BGA balls — only render if the pitch fits inside the FOV
        # (at high-mag they are typically outside the FOV; include when visible)
        fov_x_um = vol_x * vs_um
        fov_y_um = vol_y * vs_um
        if H._BGA_PITCH_UM < fov_x_um * 0.9 and H._BGA_PITCH_UM < fov_y_um * 0.9:
            bga = bump_grid_2d(H._BGA_PITCH_UM, H._BGA_DIA_UM * 0.5)
            # Place BGA balls one diameter below the interposer bottom
            bga_top_um    = -(H._BGA_DIA_UM * 0.5)
            bga_bottom_um = -(H._BGA_DIA_UM * 1.5)
            lo, hi = z_slice(bga_bottom_um + stack_um / 2,
                             H._BGA_DIA_UM)
            # recompute relative to stack bottom
            lo_abs = z_bot_vox + int(bga_top_um / vs_um) - int(H._BGA_DIA_UM / vs_um)
            hi_abs = z_bot_vox + int(bga_top_um / vs_um)
            lo_abs = max(0, min(lo_abs, vol_z))
            hi_abs = max(0, min(hi_abs, vol_z))
            if lo_abs < hi_abs:
                vol[lo_abs:hi_abs, bga] = 1.8  # copper shell

        # ASTRA treats array index 0 as the source-facing (top) slice.
        # Physical order built above is bottom-up (interposer→DRAM), so flip z
        # to place DRAM at index 0 (top/source-facing) and interposer at the end.
        return np.ascontiguousarray(vol[::-1])
