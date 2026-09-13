import numpy as np


class VxNoiseModel:
    """
    Adds detector-realistic noise to a line-integral (attenuation)
    projection stack - the representation VxSyntheticDataGenerator stores
    its Sinogram in, where 0 is pure air and higher values are more
    attenuating material along the ray.

    A real detector never reads attenuation directly - it counts
    transmitted photons, I = I0 * exp(-line_integral), and that count is
    what actually carries the physical noise: Poisson (shot) noise from
    the finite number of photons, plus Gaussian electronic/read noise and
    a small residual dark-current offset on top. Noise is therefore added
    in the transmission/count domain and converted back - adding it
    directly to the line integral (`sinogram + noise`) would not
    correspond to any real physical process, and would not reproduce the
    property this model exists to reproduce: air reading as a small
    residual value above 0 rather than an exact 0, the way a real
    detector's imperfect dark/flat-field correction leaves behind.

    Defaults are tuned to be visible but not overwhelming at this
    generator's typical line-integral magnitudes (a few hundredths to a
    few tenths for the phantoms in Structures/) - lower photon_count gives
    proportionally more Poisson shot noise, the way a faster or
    lower-dose acquisition would.
    """

    def __init__(self, photon_count: float = 20_000.0,
                 read_noise_std: float = 5.0, dark_count: float = 20.0,
                 seed: int = 0):
        """
        Args:
            photon_count:    I0, incident photons per pixel per projection -
                             the dominant noise dial. Lower means
                             proportionally more Poisson shot noise.
            read_noise_std:  Gaussian electronic/read noise, in raw counts.
            dark_count:      constant residual dark-current offset, in raw
                             counts - what keeps air from reading as an
                             exact 0 even before any read noise is added.
            seed:            RNG seed. 0 (not None) by default, matching
                             this generator's other seeded-defect classes
                             (BgaStructure, PcbPanelStructure,
                             MicroJigStructure all default to seed=0), so a
                             dataset is reproducible unless asked otherwise.
        """
        self.PhotonCount = float(photon_count)
        self.ReadNoiseStd = float(read_noise_std)
        self.DarkCount = float(dark_count)
        self.Seed = seed

    def Apply(self, sinogram: np.ndarray) -> np.ndarray:
        """
        Return a noisy copy of `sinogram` (any shape - this generator's own
        ASTRA (det_v, angles, det_u) order included, since noise is applied
        per element and does not care about axis order).
        """
        rng = np.random.default_rng(self.Seed)
        transmission = np.exp(-np.asarray(sinogram, dtype=np.float64))
        counts = rng.poisson(self.PhotonCount * transmission).astype(np.float64)
        counts += self.DarkCount
        counts += rng.normal(0.0, self.ReadNoiseStd, size=counts.shape)
        counts = np.clip(counts, 1e-6, None)   # keep the log finite
        noisy = -np.log(counts / self.PhotonCount)
        # a real detector's reading never goes below 0 attenuation either -
        # noise just pushes some pixels' apparent transmission above 1
        return np.clip(noisy, 0.0, None).astype(np.float32)
