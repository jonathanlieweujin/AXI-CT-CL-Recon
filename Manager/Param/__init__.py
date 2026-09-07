import numpy as np
from Manager.Constants.recon_method import ReconMethodConstants
from Manager.Constants.laminography_method import LaminographyMethodConstants

class VxParam:
    def __init__(
        self,
        sod: float = 0.0,
        sdd: float = 0.0,
        angles: np.ndarray = None,
        tilt_x: float = 0.0,
        tilt_y: float = 0.0,
        det_width: int = 0,
        det_height: int = 0,
        det_pitch: float = 1.0,
        offset_u: float = 0.0,
        offset_v: float = 0.0,
        binning: int = 1,
        volume_mid_x: float = 0.0,
        volume_mid_y: float = 0.0,
        volume_mid_z: float = 0.0,
        dst_x: float = 0.0,
        dst_y: float = 0.0,
        dst_z: float = 0.0,
        dst_pixel_format: str = "u8", #u8, u16
        recon_method: ReconMethodConstants = ReconMethodConstants.FDK,
        laminography_method: LaminographyMethodConstants = LaminographyMethodConstants.INCLINED,
        iterations: int = 1,
    ):
        self.sod = sod
        self.sdd = sdd
        self.angles = angles if angles is not None else np.array([], dtype=np.float64)
        self.tilt_x = tilt_x
        self.tilt_y = tilt_y
        b = max(1, int(binning))
        self.binning = b
        self.det_pitch = det_pitch * b
        self.offset_u = offset_u / b
        self.offset_v = offset_v / b
        self.det_width = det_width // b
        self.det_height = det_height // b
        self.volume_mid_x = volume_mid_x
        self.volume_mid_y = volume_mid_y
        self.volume_mid_z = volume_mid_z
        self.dst_x = dst_x
        self.dst_y = dst_y
        self.dst_z = dst_z
        self.dst_pixel_format = dst_pixel_format
        self.iterations = iterations
        self.recon_method = recon_method
        self.laminography_method = laminography_method

    @property
    def num_of_imgs(self) -> int:
        return len(self.angles)
