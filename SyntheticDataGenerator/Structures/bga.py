import numpy as np


class BgaStructure:
    """
    Ball-grid-array cross-section phantom: a silicon die carried on solder
    balls over a board, with the void / open / bridge defect population an
    X-ray or CL inspection is actually looking for.

    Two builds, selected by `mode`:

      FCBGA  die -> die attach -> package substrate (BT/ABF, laser microvias)
             -> substrate pad + solder mask -> SAC305 ball -> PCB pad + mask
             -> FR4 laminate.  Ball 450 um on a 900 um pitch.
      WLCSP  the wafer-level case: no substrate at all.  Balls are grown
             straight onto the die through a Cu UBM, 300 um on a 500 um pitch
             (250 um on 400 um is the other common stop).

    Dimensions (um), midpoints of the published ranges:

      ball dia / pitch     450 / 900 (FCBGA), 300 / 500 (WLCSP)
      standoff after reflow  ~0.78 x ball dia - the ball collapses, so the
                             joint is a sphere truncated at both pads, not a
                             sphere sitting in a gap
      pad dia              0.78 x ball dia (NSMD), 10-20% under the ball dia
      UBM (WLCSP)          ball dia - 20 um
      Cu pad / plane       25-35 um
      solder mask          25 um
      laser microvia       90 um dia, 150 um capture pad
      substrate (BT)       260 um
      die (thinned wafer)  300 um

    Attenuation values are linear attenuation coefficients in mm^-1 at 60 keV
    (NIST XCOM mu/rho x density), not arbitrary labels, so a forward projection
    through a volume gridded in mm is a physical line integral:

      FR4                 1.85 g/cc          0.055
      solder mask / epoxy 1.30               0.042
      BT resin            1.90               0.060
      silicon             2.33 x 0.3207      0.075
      copper              8.96 x 1.593       1.427
      SAC305              7.37 x ~6.51       4.790   (96.5Sn/3.0Ag/0.5Cu)

    Solder is ~64x silicon and ~3.4x copper here, which is why the ball reads
    as opaque and the void is the only thing visible through it.

    Defects are keyed to the inspection standard: IPC-7095 calls a ball a
    defect when the summed void area in projection passes 25% (Class 2);
    Class 3 practice is nearer 10%.  Void diameter follows from that directly,
    since a single void of diameter d in a ball of diameter D projects
    (d/D)^2 -- 25% => d = 0.50 D, 10% => d = 0.32 D.

      void            one void, 8-20% projected
      gross_void      one void, 25-45% projected - fails Class 2
      void_cluster    5-9 small voids, 2-6% each
      head_in_pillow  ball never merged with the paste: a thin air seam
                      across the lower third of the joint
      open            no wetting on the board pad: the ball stays balled up
                      with a ~40 um gap under it
      bridge          solder column to the +x neighbour at mid standoff
      misaligned      ball centre off its pad by ~22% of pitch
      missing         no ball

    Build returns the volume and a manifest listing every ball, its defect and
    its summed projected void percentage, so a detector can be scored against
    ground truth instead of eyeballed.
    """

    # geometry, micrometres
    _FCBGA = dict(
        ball_dia=450.0, pitch=900.0, standoff=350.0,
        pcb_t=250.0, pcb_pad_t=30.0, mask_t=25.0,
        pkg_pad_t=25.0, substrate_t=260.0, die_attach_t=25.0, die_t=300.0,
    )
    _WLCSP = dict(
        ball_dia=300.0, pitch=500.0, standoff=235.0,
        pcb_t=250.0, pcb_pad_t=30.0, mask_t=25.0,
        pkg_pad_t=10.0, substrate_t=0.0, die_attach_t=0.0, die_t=300.0,
    )

    _PAD_RATIO       = 0.78   # pad dia / ball dia, NSMD
    _MICROVIA_DIA_UM = 90.0
    _CAPTURE_DIA_UM  = 150.0
    _PLANE_T_UM      = 15.0

    # linear attenuation coefficient, mm^-1 @ 60 keV
    _MU_FR4    = 0.055
    _MU_MASK   = 0.042
    _MU_BT     = 0.060
    _MU_SI     = 0.075
    _MU_CU     = 1.427
    _MU_SOLDER = 4.790

    _DEFECTS = ("void", "gross_void", "void_cluster", "head_in_pillow",
                "open", "bridge", "misaligned", "missing")

    @staticmethod
    def GetStructure(vol_shape: tuple[int, int, int], voxel_size_mm: float) -> np.ndarray:
        """Phantom only, in ASTRA (z, y, x) order - the Structures interface."""
        vol, _ = BgaStructure.Build(vol_shape, voxel_size_mm)
        return vol

    @staticmethod
    def Build(
        vol_shape: tuple[int, int, int],
        voxel_size_mm: float,
        mode: str = "FCBGA",
        seed: int = 0,
        defect_rate: float = 0.4,
        background_void_pct: tuple[float, float] = (1.0, 7.0),
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Build the phantom and the ground-truth manifest.

        Args:
            vol_shape:           (vol_z, vol_y, vol_x) in voxels.
            voxel_size_mm:       isotropic voxel edge, mm.
            mode:                "FCBGA" or "WLCSP".
            seed:                RNG seed - the same seed gives the same defects.
            defect_rate:         fraction of balls carrying a named defect; the
                                 rest get only background voiding.
            background_void_pct: projected-area range for the void every
                                 healthy ball carries, because reflowed joints
                                 are never wholly void free.

        Returns:
            (volume float32 (z, y, x), manifest list of per-ball dicts).
        """
        H = BgaStructure
        g = dict(H._FCBGA if str(mode).upper() == "FCBGA" else H._WLCSP)

        vol_z, vol_y, vol_x = vol_shape
        vs = voxel_size_mm * 1_000.0                  # um per voxel
        vol = np.zeros((vol_z, vol_y, vol_x), dtype=np.float32)
        rng = np.random.default_rng(seed)

        ball_d = g["ball_dia"]
        pitch = g["pitch"]
        pad_r = ball_d * H._PAD_RATIO * 0.5

        # z plan, um from the bottom of the board, built bottom-up
        layers = []
        z = 0.0

        def layer(name, thickness):
            nonlocal z
            layers.append((name, z, z + thickness))
            z += thickness

        layer("pcb", g["pcb_t"])
        layer("pcb_pad", g["pcb_pad_t"])
        layer("joint", g["standoff"])
        layer("pkg_pad", g["pkg_pad_t"])
        layer("substrate", g["substrate_t"])
        layer("die_attach", g["die_attach_t"])
        layer("die", g["die_t"])
        span = {n: (lo, hi) for n, lo, hi in layers}
        stack_um = z

        z_bot_vox = vol_z / 2.0 - stack_um / vs / 2.0

        def zrange(name):
            """um layer bounds -> clamped voxel slice."""
            lo, hi = span[name]
            k0 = max(0, min(int(z_bot_vox + lo / vs), vol_z))
            k1 = max(0, min(int(z_bot_vox + hi / vs), vol_z))
            return k0, k1

        # 1-D coordinates in um, centred on the volume midpoint
        y1d = (np.arange(vol_y, dtype=np.float32) - vol_y / 2.0) * vs
        x1d = (np.arange(vol_x, dtype=np.float32) - vol_x / 2.0) * vs

        def grid_disc(radius_um):
            """(vol_y, vol_x) mask, True inside any disc of the ball lattice."""
            xm = (x1d % pitch + pitch * 0.5) % pitch - pitch * 0.5
            ym = (y1d % pitch + pitch * 0.5) % pitch - pitch * 0.5
            return xm[None, :] ** 2 + ym[:, None] ** 2 < radius_um ** 2

        pads = grid_disc(pad_r)

        # board
        k0, k1 = zrange("pcb")
        vol[k0:k1, :, :] = H._MU_FR4

        # board pad layer: Cu inside the pad, solder mask everywhere else
        k0, k1 = zrange("pcb_pad")
        vol[k0:k1, :, :] = H._MU_MASK
        vol[k0:k1, pads] = H._MU_CU

        # package side of the joint: UBM disc for WLCSP, pad + mask for FCBGA
        k0, k1 = zrange("pkg_pad")
        if g["substrate_t"] > 0:
            vol[k0:k1, :, :] = H._MU_MASK
            vol[k0:k1, pads] = H._MU_CU
        else:
            vol[k0:k1, grid_disc((ball_d - 20.0) * 0.5)] = H._MU_CU

        # package substrate: BT with a Cu plane and a Cu-filled laser microvia
        # plus capture pad over every ball
        if g["substrate_t"] > 0:
            lo, hi = span["substrate"]
            k0, k1 = zrange("substrate")
            vol[k0:k1, :, :] = H._MU_BT
            vol[k0:k1, grid_disc(H._MICROVIA_DIA_UM * 0.5)] = H._MU_CU

            antipad = grid_disc(H._CAPTURE_DIA_UM * 0.5 + 60.0)
            p_lo = lo + 0.45 * (hi - lo)
            p0 = max(0, min(int(z_bot_vox + p_lo / vs), vol_z))
            p1 = max(0, min(int(z_bot_vox + (p_lo + H._PLANE_T_UM) / vs), vol_z))
            if p0 < p1:
                plane = np.full((vol_y, vol_x), H._MU_CU, dtype=np.float32)
                plane[antipad] = H._MU_BT
                vol[p0:p1, :, :] = plane
                vol[p0:p1, grid_disc(H._MICROVIA_DIA_UM * 0.5)] = H._MU_CU

            c0 = max(0, min(int(z_bot_vox + (hi - H._PLANE_T_UM) / vs), vol_z))
            c1 = max(0, min(int(z_bot_vox + hi / vs), vol_z))
            vol[c0:c1, grid_disc(H._CAPTURE_DIA_UM * 0.5)] = H._MU_CU

        if g["die_attach_t"] > 0:
            k0, k1 = zrange("die_attach")
            vol[k0:k1, :, :] = H._MU_MASK

        # silicon die - the wafer the balls hang off
        k0, k1 = zrange("die")
        vol[k0:k1, :, :] = H._MU_SI

        # solder joints
        jlo, jhi = span["joint"]
        z_mid = (jlo + jhi) * 0.5
        R = ball_d * 0.5
        half_h = g["standoff"] * 0.5

        n_x = int(np.ceil(vol_x * vs / (2.0 * pitch)))
        n_y = int(np.ceil(vol_y * vs / (2.0 * pitch)))
        manifest = []

        for iy in range(-n_y, n_y + 1):
            for ix in range(-n_x, n_x + 1):
                cx, cy = ix * pitch, iy * pitch
                kind = (str(rng.choice(H._DEFECTS))
                        if rng.random() < defect_rate else "none")
                entry = BgaStructure.paintBall_Internal(
                    vol, rng, kind, cx, cy, z_mid, R, half_h, pitch,
                    vs, z_bot_vox, x1d, y1d, jlo, jhi, background_void_pct,
                )
                entry.update(index=(ix, iy), centre_um=(cx, cy), defect=kind)
                manifest.append(entry)

        # ASTRA index 0 is the source-facing slice; the stack was built
        # bottom-up, so flip z to put the die at the top.
        return np.ascontiguousarray(vol[::-1]), manifest

    # private / internal
    @staticmethod
    def paintBall_Internal(vol, rng, kind, cx, cy, z_mid, R, half_h, pitch,
                           vs, z_bot_vox, x1d, y1d, jlo, jhi,
                           background_void_pct) -> dict:
        """
        Paint one joint into the joint layer and return its ground truth.

        The nominal joint is a sphere of radius R truncated by the standoff at
        both pads, which is what a collapsed ball actually is. Defects change
        that shape or subtract air spheres from it.
        """
        H = BgaStructure
        vol_z, vol_y, vol_x = vol.shape
        if kind == "missing":
            return dict(void_pct=0.0, voids=[])

        reach = R * 1.2 + (pitch if kind == "bridge" else 0.0)
        off_x = pitch * 0.22 if kind == "misaligned" else 0.0

        i0 = max(0, int((cx + off_x - reach) / vs + vol_x / 2.0))
        i1 = min(vol_x, int((cx + off_x + reach) / vs + vol_x / 2.0) + 1)
        j0 = max(0, int((cy - reach) / vs + vol_y / 2.0))
        j1 = min(vol_y, int((cy + reach) / vs + vol_y / 2.0) + 1)
        k0 = max(0, min(int(z_bot_vox + jlo / vs), vol_z))
        k1 = max(0, min(int(z_bot_vox + jhi / vs) + 1, vol_z))
        if i0 >= i1 or j0 >= j1 or k0 >= k1:
            return dict(void_pct=0.0, voids=[])

        # local coordinates in um, relative to the ball centre
        xl = (x1d[i0:i1] - cx - off_x)[None, None, :]
        yl = (y1d[j0:j1] - cy)[None, :, None]
        zl = ((np.arange(k0, k1, dtype=np.float32) - z_bot_vox) * vs - z_mid)[:, None, None]
        sub = vol[k0:k1, j0:j1, i0:i1]

        if kind == "open":
            # never wetted the board pad: stays a full sphere, sitting high
            gap = 40.0
            eff_r = (half_h * 2.0 - gap) * 0.5
            shift = (jlo + gap + eff_r) - z_mid
            sub[(xl ** 2 + yl ** 2 + (zl - shift) ** 2) < eff_r ** 2] = H._MU_SOLDER
        else:
            eff_r = R
            sub[(xl ** 2 + yl ** 2 + zl ** 2) < R ** 2] = H._MU_SOLDER

        if kind == "bridge":
            # solder column across to the +x neighbour, at mid standoff
            xb = xl - pitch * 0.5
            sub[((yl ** 2 + zl ** 2) < (R * 0.16) ** 2)
                & (np.abs(xb) < pitch * 0.5)] = H._MU_SOLDER

        if kind == "head_in_pillow":
            # air seam where the ball failed to merge with the paste
            seam_z = jlo + (jhi - jlo) * 0.33 - z_mid
            sub[(np.abs(zl - seam_z) < max(vs, 8.0) * 0.5)
                & ((xl ** 2 + yl ** 2) < (R * 0.42) ** 2)] = 0.0

        # voids: a void of diameter d projects (d / ball dia)^2 of the ball
        if kind == "gross_void":
            pcts = [float(rng.uniform(25.0, 45.0))]
        elif kind == "void":
            pcts = [float(rng.uniform(8.0, 20.0))]
        elif kind == "void_cluster":
            pcts = [float(rng.uniform(2.0, 6.0)) for _ in range(int(rng.integers(5, 10)))]
        else:
            pcts = [float(rng.uniform(*background_void_pct))]

        voids = []
        for pct in pcts:
            rv = eff_r * float(np.sqrt(pct / 100.0))
            free = max(0.0, min(eff_r, half_h) - rv - vs)
            u = rng.normal(size=3)
            norm = float(np.linalg.norm(u))
            u = u / norm if norm > 0 else np.array([0.0, 0.0, 0.0])
            vz, vy, vx = u * free * float(rng.random()) ** (1.0 / 3.0)
            sub[((xl - vx) ** 2 + (yl - vy) ** 2 + (zl - vz) ** 2) < rv ** 2] = 0.0
            voids.append(dict(pct=pct, dia_um=float(rv * 2.0),
                              offset_um=(float(vx), float(vy), float(vz))))

        return dict(void_pct=float(sum(pcts)), voids=voids)
