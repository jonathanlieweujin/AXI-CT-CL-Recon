import numpy as np

from Manager.Param import VxParam
from Manager.Constants.flow_method import VxFlowMethod
from Manager.Constants.laminography_method import LaminographyMethodConstants
from Manager.Util.compute_geometry import VxComputeGeometry
from Manager.Util.load_images_internal import FileLoader
from Manager.Tool.Performance import VxTool as VxPerformanceTool
from Manager.Tool.Quality import VxTool as VxQualityTool


class VxManager:
    """
    Single entry point for a reconstruction: holds the parameters, both backend
    tools, and the resulting volume.

    The flow method chooses the backend (Quality = TIGRE, Performance = ASTRA);
    the recon method on VxParam chooses the algorithm (FDK, SIRT, CGLS, ...).
    """

    def __init__(
        self,
        param: VxParam,
        flow_method: VxFlowMethod = VxFlowMethod.QUALITY,
        left_pad: int = 0,
        right_pad: int = 0,
    ):
        self.Param = param
        self.FlowMethod = flow_method
        self.ReconMethod = param.recon_method
        self.Images = None
        self.Result = np.ndarray(0, dtype=np.float32)

        self.QualityTool = VxQualityTool(
            param, laminography_method=param.laminography_method,
            left_pad=left_pad, right_pad=right_pad)
        self.PerformanceTool = VxPerformanceTool(
            param, laminography_method=param.laminography_method,
            left_pad=left_pad, right_pad=right_pad)

    # ------------------------------------------------------------------ public

    def LoadImages(self, input_folder: str, interval: int = 1) -> np.ndarray:
        """
        Load the projection stack described by Param from input_folder.

        Param already holds post-binning detector dimensions, so the source
        dimensions are restored here before handing them to the loader.
        """
        p = self.Param
        src_width = int(p.det_width) * p.binning
        src_height = int(p.det_height) * p.binning

        self.Images = FileLoader.loadImages(
            input_folder,
            src_width,
            src_height,
            p.num_of_imgs,
            interval=interval,
            binning=p.binning,
        )
        return self.Images

    def Run(self, projections: np.ndarray = None, **kwargs) -> np.ndarray:
        """Reconstruct with the backend selected by FlowMethod."""
        if projections is None:
            if self.Images is None:
                raise ValueError("No projections: call LoadImages first or pass them to Run.")
            projections = self.Images

        tool = self._active_tool()
        self.Result = tool.run_internal(projections, algo=self.ReconMethod, **kwargs)
        return self.Result

    def GetScanGeometry(self) -> np.ndarray:
        """
        (n, 12) ASTRA cone_vec rows [Sx Sy Sz  Dx Dy Dz  Ux Uy Uz  Vx Vy Vz]
        for the current parameters - the same vectors the Performance flow
        reconstructs from, and what synthetic data must be projected through.
        """
        p = self.Param
        if p.laminography_method == LaminographyMethodConstants.COPLANAR:
            vecs = VxComputeGeometry.computeCoplanarTranslationalLaminographyGeometry(p)
        else:
            vecs = VxComputeGeometry.computeDefaultInclinedLaminographyGeometry(p)
        return vecs.reshape(-1, 12)

    # ----------------------------------------------------------------- private

    def _active_tool(self):
        if self.FlowMethod == VxFlowMethod.PERFORMANCE:
            return self.PerformanceTool
        return self.QualityTool
