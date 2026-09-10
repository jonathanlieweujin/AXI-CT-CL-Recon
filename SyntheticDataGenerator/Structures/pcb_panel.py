import numpy as np

from SyntheticDataGenerator.Structures.materials import (
    MU_CU, MU_FR4, MU_MASK, MU_SOLDER,
)



class PcbPanelStructure:
    """
    PCB panel phantom sample size
    (4.608 x 4.608 x 0.900 mm).

    That is the box the phantom is designed for - the FID_2 reference volume,
    1536 x 1536 x 300 at 3.0 um. The board fills it laterally; in z the 0.573 mm
    stack sits centred with ~0.16 mm of air either side, so the surfaces project
    cleanly. DEFAULT_SIZE_MM and STACK_THICKNESS_MM carry both numbers.

    Every in-plane dimension below was measured off that reconstruction rather
    than taken from a datasheet:

      bump array pitch     200 um   (nearest-neighbour median, both sites)
      bump diameter         83 um   (equivalent-area diameter, 103 objects,
                                     p10 80 um / p90 86 um)
      main site            7 x 8 bumps, x 1710..2910, y 1610..3010 um
      second site          7 x 3 bumps, x 1710..2910, y 4104..4504 um
      plated through hole  110 um drill, 165 um outer -> 27 um barrel wall
      via positions        28 of them, measured centres, mostly perimeter

    Feature coordinates are held in um relative to the volume centre, so the
    panel renders correctly at any voxel size or volume shape and simply clips
    at the box - only the crop changes, never the part.

    The z stack is not measured: a 30 deg laminography reconstruction smears
    depth badly (a 60 um bump reads as a ~190 um streak in the reference), so
    copying its z profile would bake the point-spread into the phantom. These
    are ordinary board values that sum to the ~570 um of metal the reference
    does show, bottom -> top:

      solder mask    25 um
      bottom copper  30 um   plane, antipads around the vias
      FR4 core      400 um
      top copper     30 um   plane, keep-out over each component site, pads
      solder mask    25 um   opened over every pad
      solder bump    60 um   83 um dia, truncated sphere on the pad

    Defects are off by default so the panel is a faithful copy of the
    reference; raise defect_rate to seed the same population BgaStructure uses
    (void, gross_void, missing, misaligned) with a ground-truth manifest.
    """

    # box the phantom is designed for, and the material thickness inside it
    DEFAULT_SIZE_MM    = (4.608, 4.608, 0.900)
    STACK_THICKNESS_MM = 0.573

    _REF_VOXEL_UM = 3.0
    _REF_SHAPE    = (300, 1536, 1536)   # (z, y, x) the constants were measured at

    # bump array: pitch and diameter measured off the reference
    _BUMP_PITCH_UM = 200.0
    _BUMP_DIA_UM   = 83.0
    _PAD_DIA_UM    = 100.0
    _MASK_OPEN_UM  = 110.0

    # component sites: (centre_x, centre_y, cols, rows) in um from volume centre
    _SITES = (
        (6.0,    6.0, 7, 8),
        (6.0, 2000.0, 7, 3),
    )
    _SITE_MARGIN_UM = 150.0   # copper keep-out beyond the bump array bounding box

    # plated through holes
    _VIA_DRILL_UM = 110.0
    _VIA_OUTER_UM = 165.0
    _VIA_RING_UM  = 250.0     # annular ring pad on each copper layer

    # measured via centres, um from the volume centre (x, y)
    _VIAS = (
        (-2099, -1834), (-2093,  1421), (-2087,  -271), (-1998,  1113),
        (-1883,  1840), (-1639,   509), (-1424,  -275), (-1385,  2066),
        (-1348,  1290), (-1200, -1834), ( -305, -1851), (   84, -1474),
        (  611, -1850), ( 1494, -1858), ( 1519,  1270), ( 1529,   575),
        ( 1709,   195), ( 1874, -1439), ( 1905,  -334), ( 1932, -1059),
        ( 1962,  1298), ( 1979,   485), ( 2020,   108), ( 2071, -1686),
        ( 2107,   989), ( 2113,  1570), ( 2200,  -858), ( 2244,   385),
    )

    # z stack, um from the bottom of the board
    _MASK_BOT_UM = 25.0
    _CU_BOT_UM   = 30.0
    _CORE_UM     = 400.0
    _CU_TOP_UM   = 30.0
    _MASK_TOP_UM = 25.0
    _BUMP_H_UM   = 60.0

    _DEFECTS = ("void", "gross_void", "missing", "misaligned")

    @staticmethod
    def GetStructure(vol_shape: tuple[int, int, int], voxel_size_mm: float) -> np.ndarray:
        """Phantom only, in ASTRA (z, y, x) order - the Structures interface."""
        vol, _ = PcbPanelStructure.Build(vol_shape, voxel_size_mm)
        return vol

    @staticmethod
    def Build(
        vol_shape: tuple[int, int, int],
        voxel_size_mm: float,
        seed: int = 0,
        defect_rate: float = 0.0,
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Build the panel and the ground-truth manifest.

        Args:
            vol_shape:     (vol_z, vol_y, vol_x) in voxels.
            voxel_size_mm: isotropic voxel edge, mm. 0.003 reproduces the
                           reference; anything else rescales the crop, not the
                           part.
            seed:          RNG seed - the same seed gives the same defects.
            defect_rate:   fraction of bumps carrying a defect. 0 (default)
                           renders the panel as measured.

        Returns:
            (volume float32 (z, y, x), manifest list of per-bump dicts).
        """
        H = PcbPanelStructure
        vol_z, vol_y, vol_x = vol_shape
        vs = voxel_size_mm * 1_000.0                  # um per voxel
        vol = np.zeros((vol_z, vol_y, vol_x), dtype=np.float32)
        rng = np.random.default_rng(seed)

        # z plan, um from the bottom of the board, built bottom-up
        layers, z = [], 0.0

        def layer(name, thickness):
            nonlocal z
            layers.append((name, z, z + thickness))
            z += thickness

        layer("mask_bot", H._MASK_BOT_UM)
        layer("cu_bot",   H._CU_BOT_UM)
        layer("core",     H._CORE_UM)
        layer("cu_top",   H._CU_TOP_UM)
        layer("mask_top", H._MASK_TOP_UM)
        layer("bump",     H._BUMP_H_UM)
        span = {n: (lo, hi) for n, lo, hi in layers}
        stack_um = z

        z_bot_vox = vol_z / 2.0 - stack_um / vs / 2.0

        def zrange(name):
            lo, hi = span[name]
            k0 = max(0, min(int(z_bot_vox + lo / vs), vol_z))
            k1 = max(0, min(int(z_bot_vox + hi / vs), vol_z))
            return k0, k1

        # 1-D coordinates in um, centred on the volume midpoint
        y1d = (np.arange(vol_y, dtype=np.float32) - vol_y / 2.0) * vs
        x1d = (np.arange(vol_x, dtype=np.float32) - vol_x / 2.0) * vs

        def discs(centres, radius_um):
            """(vol_y, vol_x) mask, True inside any listed disc."""
            m = np.zeros((vol_y, vol_x), dtype=bool)
            r2 = radius_um ** 2
            for cx, cy in centres:
                # only touch the rows/cols the disc can reach
                i0, i1 = np.searchsorted(x1d, [cx - radius_um, cx + radius_um])
                j0, j1 = np.searchsorted(y1d, [cy - radius_um, cy + radius_um])
                if i0 >= i1 or j0 >= j1:
                    continue
                dx = x1d[i0:i1] - cx
                dy = y1d[j0:j1] - cy
                m[j0:j1, i0:i1] |= (dx[None, :] ** 2 + dy[:, None] ** 2) < r2
            return m

        # bump / pad centres, and the copper keep-out around each site
        pads, keepout = [], np.zeros((vol_y, vol_x), dtype=bool)
        for sx, sy, cols, rows in H._SITES:
            xs = (np.arange(cols) - (cols - 1) / 2.0) * H._BUMP_PITCH_UM + sx
            ys = (np.arange(rows) - (rows - 1) / 2.0) * H._BUMP_PITCH_UM + sy
            pads += [(px, py) for py in ys for px in xs]
            m = H._SITE_MARGIN_UM
            keepout |= (((x1d >= xs[0] - m) & (x1d <= xs[-1] + m))[None, :]
                        & ((y1d >= ys[0] - m) & (y1d <= ys[-1] + m))[:, None])

        vias = [(float(vx), float(vy)) for vx, vy in H._VIAS]
        via_ring  = discs(vias, H._VIA_RING_UM * 0.5)
        via_outer = discs(vias, H._VIA_OUTER_UM * 0.5)
        via_drill = discs(vias, H._VIA_DRILL_UM * 0.5)
        via_anti  = discs(vias, H._VIA_RING_UM * 0.5 + 60.0)
        barrel    = via_outer & ~via_drill
        pad_disc  = discs(pads, H._PAD_DIA_UM * 0.5)
        mask_open = discs(pads, H._MASK_OPEN_UM * 0.5)

        # solder mask, bottom
        k0, k1 = zrange("mask_bot")
        vol[k0:k1, :, :] = MU_MASK
        vol[k0:k1, via_drill] = 0.0

        # bottom copper: plane with an antipad around every via, ring pad, barrel
        k0, k1 = zrange("cu_bot")
        plane = np.full((vol_y, vol_x), MU_CU, dtype=np.float32)
        plane[via_anti] = MU_FR4
        plane[via_ring] = MU_CU
        plane[via_drill] = 0.0
        vol[k0:k1, :, :] = plane

        # FR4 core, drilled, with the plated barrel through it
        k0, k1 = zrange("core")
        vol[k0:k1, :, :] = MU_FR4
        vol[k0:k1, via_drill] = 0.0
        vol[k0:k1, barrel] = MU_CU

        # top copper: plane, keep-out over each component site, pads and vias
        k0, k1 = zrange("cu_top")
        plane = np.full((vol_y, vol_x), MU_CU, dtype=np.float32)
        plane[keepout] = MU_FR4
        plane[via_anti] = MU_FR4
        plane[via_ring] = MU_CU
        plane[pad_disc] = MU_CU
        plane[via_drill] = 0.0
        vol[k0:k1, :, :] = plane

        # solder mask, top, opened over every pad and every via
        k0, k1 = zrange("mask_top")
        vol[k0:k1, :, :] = MU_MASK
        vol[k0:k1, mask_open] = 0.0
        vol[k0:k1, via_drill] = 0.0

        # solder bumps
        manifest = []
        blo, bhi = span["bump"]
        r = H._BUMP_DIA_UM * 0.5
        for px, py in pads:
            kind = (str(rng.choice(H._DEFECTS))
                    if rng.random() < defect_rate else "none")
            entry = PcbPanelStructure.paintBump_Internal(
                vol, rng, kind, px, py, r, vs, z_bot_vox, x1d, y1d, blo, bhi)
            entry.update(centre_um=(px, py), defect=kind)
            manifest.append(entry)

        # ASTRA index 0 is the source-facing slice; the stack was built
        # bottom-up, so flip z to put the bumps at the top.
        return np.ascontiguousarray(vol[::-1]), manifest

    # private / internal
    @staticmethod
    def paintBump_Internal(vol, rng, kind, px, py, r, vs, z_bot_vox,
                           x1d, y1d, blo, bhi) -> dict:
        """
        Paint one bump as a truncated sphere seated on the pad, and return its
        ground truth. A reflowed bump is wider than it is tall, so the sphere
        is cut by the bump layer rather than sitting whole inside it.
        """
        vol_z, vol_y, vol_x = vol.shape
        if kind == "missing":
            return dict(void_pct=0.0)

        off = PcbPanelStructure._BUMP_PITCH_UM * 0.18 if kind == "misaligned" else 0.0
        reach = r * 1.2

        i0 = max(0, int((px + off - reach) / vs + vol_x / 2.0))
        i1 = min(vol_x, int((px + off + reach) / vs + vol_x / 2.0) + 1)
        j0 = max(0, int((py - reach) / vs + vol_y / 2.0))
        j1 = min(vol_y, int((py + reach) / vs + vol_y / 2.0) + 1)
        k0 = max(0, min(int(z_bot_vox + blo / vs), vol_z))
        k1 = max(0, min(int(z_bot_vox + bhi / vs) + 1, vol_z))
        if i0 >= i1 or j0 >= j1 or k0 >= k1:
            return dict(void_pct=0.0)

        xl = (x1d[i0:i1] - px - off)[None, None, :]
        yl = (y1d[j0:j1] - py)[None, :, None]
        zc = blo + (bhi - blo) * 0.5
        zl = ((np.arange(k0, k1, dtype=np.float32) - z_bot_vox) * vs - zc)[:, None, None]
        sub = vol[k0:k1, j0:j1, i0:i1]
        sub[(xl ** 2 + yl ** 2 + zl ** 2) < r ** 2] = MU_SOLDER

        # a void of diameter d projects (d / bump dia)^2 of the bump
        if kind == "gross_void":
            pct = float(rng.uniform(25.0, 45.0))
        elif kind == "void":
            pct = float(rng.uniform(8.0, 20.0))
        else:
            return dict(void_pct=0.0)

        rv = r * float(np.sqrt(pct / 100.0))
        free = max(0.0, min(r, (bhi - blo) * 0.5) - rv - vs)
        u = rng.normal(size=3)
        norm = float(np.linalg.norm(u))
        u = u / norm if norm > 0 else np.zeros(3)
        vz, vy, vx = u * free * float(rng.random()) ** (1.0 / 3.0)
        sub[((xl - vx) ** 2 + (yl - vy) ** 2 + (zl - vz) ** 2) < rv ** 2] = 0.0
        return dict(void_pct=pct, void_dia_um=float(rv * 2.0))
