import configparser
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

from Manager.Param import VxParam
from Manager.Constants.flow_method import VxFlowMethod
from Manager.Constants.laminography_method import LaminographyMethodConstants
from Manager.Constants.recon_method import ReconMethodConstants
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

    # Constructor
    def __init__(self):
        self.Param = None
        self.FlowMethod = VxFlowMethod.QUALITY
        self.ReconMethod = None
        self.LeftPad = 0
        self.RightPad = 0
        self.Filter = None
        self.Interval = 1
        self.Images = None
        self.Result = np.ndarray(0, dtype=np.float32)

        self.QualityTool = VxQualityTool()
        self.PerformanceTool = VxPerformanceTool()

    # Public
    def SetParam(self, param: VxParam):
        """
        Set the geometry and push it to both backends. ReconMethod is taken
        from it, so switching parameters switches the algorithm with them.
        """
        self.Param = param
        self.ReconMethod = param.recon_method
        self.QualityTool.SetParam(param)
        self.PerformanceTool.SetParam(param)

    def SetFlowMethod(self, flow_method: VxFlowMethod):
        """Choose the backend: QUALITY (TIGRE) or PERFORMANCE (ASTRA)."""
        self.FlowMethod = flow_method

    def SetLeftPad(self, left_pad: int):
        """Detector padding, applied to both backends."""
        self.LeftPad = int(left_pad)
        self.QualityTool.SetLeftPad(self.LeftPad)
        self.PerformanceTool.SetLeftPad(self.LeftPad)

    def SetRightPad(self, right_pad: int):
        """Detector padding, applied to both backends."""
        self.RightPad = int(right_pad)
        self.QualityTool.SetRightPad(self.RightPad)
        self.PerformanceTool.SetRightPad(self.RightPad)

    def LoadConfig(self,
                   configPath: str,
                   inputFolder: str = None,
                   toLoadImages: bool = True) -> VxParam:
        """
        Build the parameters from a VXMPR geometry.config and apply them.

        Sets Param (and with it ReconMethod), FlowMethod and the padding from
        the file, then loads the projection stack so Run can follow directly.
        inputFolder defaults to the Corrected/ folder beside the config's
        Config/ directory, the layout the generator and a real acquisition both
        write. Pass toLoadImages=False to read the geometry only.

        The detector fields in the file are pre-binning, as VxParam expects,
        and Interval sub-samples the angles and the stack together.
        """
        cfg = self.parseConfig_Internal(configPath)

        self.SetParam(cfg["param"])
        if cfg["flow_method"] is not None:
            self.SetFlowMethod(cfg["flow_method"])
        self.SetLeftPad(cfg["left_pad"])
        self.SetRightPad(cfg["right_pad"])
        self.Filter = cfg["filter"]
        self.Interval = cfg["interval"]

        if toLoadImages:
            if inputFolder is None:
                # <root>/Config/geometry.config -> <root>/Corrected
                root = os.path.dirname(os.path.dirname(os.path.abspath(configPath)))
                inputFolder = os.path.join(root, "Corrected")
            if not os.path.isdir(inputFolder):
                raise NotADirectoryError(
                    f"Projection folder not found: {inputFolder}. "
                    f"Pass inputFolder=... or toLoadImages=False.")
            self.LoadImages(inputFolder, interval=cfg["interval"])

        return self.Param

    def LoadImages(self, input_folder: str, interval: int = 1) -> np.ndarray:
        """
        Load the projection stack described by Param from input_folder.

        Param already holds post-binning detector dimensions, so the source
        dimensions are restored here before handing them to the loader.
        """
        p = self.param_Internal()
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

    def SaveImages(self,
                   output_folder: str,
                   images: np.ndarray = None,
                   savePad: str = None) -> str:
        """
        Write a projection stack to output_folder as slice_XXXX.tif files, one
        per frame, in stack order and at the stack's own dtype. Defaults to the
        cached Images. Returns output_folder.

        Writes are threaded; the index in the name carries the order, not the
        completion order. savePad is the format spec for that index - left as
        None it is derived from the stack length (at least "04d", wider for
        stacks past 9999) so the names always sort in stack order.
        """
        if images is None:
            if self.Images is None:
                raise ValueError("No projections: call LoadImages first or pass them to SaveImages.")
            images = self.Images

        os.makedirs(output_folder, exist_ok=True)
        if savePad is None:
            savePad = f"0{max(4, len(str(max(1, len(images)) - 1)))}d"

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.save_Internal, output_folder, savePad, i, img)
                       for i, img in enumerate(images)]
            for future in as_completed(futures):
                future.result()

        return output_folder

    def Run(self, projections: np.ndarray = None, **kwargs):
        """
        Reconstruct with the backend selected by FlowMethod.

        The volume is left on Result rather than returned.
        """
        if projections is None:
            if self.Images is None:
                raise ValueError("No projections: call LoadImages first or pass them to Run.")
            projections = self.Images

        p = self.param_Internal()

        if p.dst_pixel_format not in ("u8", "u16"):
            raise ValueError("Invalid result pixel format")

        # source stack: at least two projections to reconstruct from
        if p.num_of_imgs <= 1:
            raise ValueError("Invalid number of projections")

        # destination volume: at least one slice
        if p.dst_z < 1:
            raise ValueError("Invalid number of volume slices")

        # detector and volume must have a real footprint
        if p.det_width < 1 or p.det_height < 1 or p.dst_x < 1 or p.dst_y < 1:
            raise ValueError("Invalid image dimension input")

        # geometry: magnification and voxel size divide by these
        if p.sod <= 0 or p.sdd <= p.sod:
            raise ValueError("Invalid source/detector distances")
        if p.det_pitch <= 0:
            raise ValueError("Invalid detector pitch")

        # the stack has to match the detector and angles it is reconstructed with
        if projections.ndim != 3:
            raise ValueError("Projections must be a 3D (angles, height, width) stack")
        n, h, w = projections.shape
        if (h, w) != (p.det_height, p.det_width):
            raise ValueError(
                f"Projection size {w}x{h} does not match detector {p.det_width}x{p.det_height}")
        if n != p.num_of_imgs:
            raise ValueError(
                f"Projection count {n} does not match {p.num_of_imgs} angles")

        # iterative algorithms need a positive iteration count
        if self.ReconMethod != ReconMethodConstants.FDK and p.iterations < 1:
            raise ValueError("Invalid iteration count")

        tool = self.activeTool_Internal()
        self.Result = tool.run_internal(projections, algo=self.ReconMethod, **kwargs)

    def GetScanGeometry(self) -> np.ndarray:
        """
        (n, 12) ASTRA cone_vec rows [Sx Sy Sz  Dx Dy Dz  Ux Uy Uz  Vx Vy Vz]
        for the current parameters - the same vectors the Performance flow
        reconstructs from, and what synthetic data must be projected through.
        """
        p = self.param_Internal()
        if p.laminography_method == LaminographyMethodConstants.COPLANAR:
            vecs = VxComputeGeometry.computeCoplanarTranslationalLaminographyGeometry(p)
        else:
            vecs = VxComputeGeometry.computeDefaultInclinedLaminographyGeometry(p)
        return vecs.reshape(-1, 12)

    def NormaliseProjections(self,
                             srcPath: str,
                             dstPath: str,
                             toApplyLogTransform : bool = True,
                             savePad: str = None) -> np.ndarray:
        """
        Normalise the projection stack in srcPath and write it to dstPath.

        The whole stack is scaled to [0, 1] with a single global min/max so the
        projections stay comparable to each other, then optionally turned into
        attenuation values with -log(x + eps). Saved as float32 .tif files named
        slice_0000.tif ... in stack order, and the stack is returned.

        savePad is passed through to SaveImages, which derives it from the stack
        length when left as None.
        """

        # epsilon constant
        eps: float = 1e-5
        
        # guard srcPath and dstPath
        if not os.path.isdir(srcPath):
            raise NotADirectoryError(f"Source folder does not exist: {srcPath}")
        if os.path.abspath(srcPath) == os.path.abspath(dstPath):
            raise ValueError("dstPath must differ from srcPath.")

        # load images LoadImages
        stack = self.LoadImages(srcPath)
        if stack.size == 0:
            raise RuntimeError(f"No projections loaded from: {srcPath}")

        # normalise based on global min max to [0,1]
        g_min = float(stack.min())
        g_max = float(stack.max())
        denom = g_max - g_min if g_max != g_min else 1.0
        out = (stack - g_min) / denom

        # log tranform (Beer-Lambert)
        if toApplyLogTransform:
            out = -np.log(out + eps)
        out = out.astype(np.float32)

        # save
        self.SaveImages(dstPath, out, savePad=savePad)

        self.Images = out
        return out

    # private / internal

    def parseConfig_Internal(self, configPath: str) -> dict:
        """
        Read a VXMPR geometry.config into a VxParam plus the settings that live
        on the manager rather than on the parameters.

        Vocabulary in the file that does not map one-to-one:
          ReconType                "<ALGO>" or "<ALGO>_<Flow>", e.g. FDK_Performance
          DstPixelFormat           U8C1 / U16C1 -> u8 / u16
          IsDetectorParallelToFOV  True -> COPLANAR, False -> INCLINED
        """
        if not os.path.isfile(configPath):
            raise FileNotFoundError(f"Config not found: {configPath}")

        parser = configparser.ConfigParser()
        parser.read(configPath, encoding="utf-8")
        if "VXMPR CONFIG" not in parser:
            raise ValueError(f"Missing [VXMPR CONFIG] section: {configPath}")
        raw = parser["VXMPR CONFIG"]

        def need(key):
            if key not in raw:
                raise ValueError(f"Config is missing '{key}': {configPath}")
            return raw[key].strip()

        # unsupported geometry, rejected rather than silently reinterpreted
        if raw.getboolean("islaminostaticsourcegeometry", fallback=False):
            raise NotImplementedError(
                "IsLaminoStaticSourceGeometry is not implemented in this module.")

        # VolScale never reaches the voxel size here, so honouring it silently
        # would reconstruct at the wrong scale
        vol_scale = float(raw.get("volscale", "1"))
        if vol_scale != 1:
            raise NotImplementedError(
                f"VolScale={vol_scale:g} is not implemented; only VolScale = 1 is supported.")

        angles = np.array(
            [float(v) for v in need("projectionangles").strip("[]").split(",") if v.strip()],
            dtype=np.float64)
        if angles.size == 0:
            raise ValueError(f"Config has no ProjectionAngles: {configPath}")

        interval = max(1, int(float(raw.get("interval", "1"))))
        angles = angles[::interval]

        # "FDK_Performance" -> algorithm + backend. A bare algorithm leaves the
        # backend as whatever the manager already has.
        algo_text, _, flow_text = need("recontype").partition("_")
        try:
            recon_method = ReconMethodConstants(algo_text.strip().upper())
        except ValueError:
            raise ValueError(
                f"Unknown ReconType algorithm '{algo_text}' in {configPath}. "
                f"Known: {', '.join(m.value for m in ReconMethodConstants)}")

        flow_method = None
        if flow_text.strip():
            try:
                flow_method = VxFlowMethod(flow_text.strip().upper())
            except ValueError:
                raise ValueError(
                    f"Unknown ReconType flow '{flow_text}' in {configPath}. "
                    f"Known: {', '.join(m.value for m in VxFlowMethod)}")

        pixel_text = raw.get("dstpixelformat", "U8C1").strip().upper()
        if pixel_text.startswith("U8"):
            dst_pixel_format = "u8"
        elif pixel_text.startswith("U16"):
            dst_pixel_format = "u16"
        else:
            raise ValueError(
                f"Unknown DstPixelFormat '{pixel_text}' in {configPath}; expected U8C1 or U16C1.")

        # IsDetectorParallelToFOV is the current key: true is COPLANAR, anything
        # else INCLINED. LaminographyMethod is the older spelling and is still on
        # datasets already written, so a file carrying only that one must not
        # fall through to the INCLINED default.
        if "isdetectorparalleltofov" in raw:
            laminography_method = (
                LaminographyMethodConstants.COPLANAR
                if raw.getboolean("isdetectorparalleltofov")
                else LaminographyMethodConstants.INCLINED)
        elif "laminographymethod" in raw:
            text = raw["laminographymethod"].strip().upper()
            try:
                laminography_method = LaminographyMethodConstants(text)
            except ValueError:
                raise ValueError(
                    f"Unknown LaminographyMethod '{text}' in {configPath}. "
                    f"Known: {', '.join(m.value for m in LaminographyMethodConstants)}")
        else:
            laminography_method = LaminographyMethodConstants.INCLINED

        param = VxParam(
            sod=float(need("sod")),
            sdd=float(need("sdd")),
            angles=angles,
            tilt_x=float(raw.get("dettiltx", "0")),
            tilt_y=float(raw.get("dettilty", "0")),
            det_width=int(float(need("detu"))),
            det_height=int(float(need("detv"))),
            det_pitch=float(need("detpitch")),
            offset_u=float(raw.get("detoffsetu", "0")),
            offset_v=float(raw.get("detoffsetv", "0")),
            binning=max(1, int(float(raw.get("binning", "1")))),
            volume_mid_x=float(raw.get("volmidx", "0")),
            volume_mid_y=float(raw.get("volmidy", "0")),
            volume_mid_z=float(raw.get("volmidz", "0")),
            dst_x=int(float(need("volx"))),
            dst_y=int(float(need("voly"))),
            dst_z=int(float(need("volz"))),
            dst_pixel_format=dst_pixel_format,
            recon_method=recon_method,
            laminography_method=laminography_method,
            iterations=max(1, int(float(raw.get("iterations", "1")))),
        )

        return {
            "param": param,
            "flow_method": flow_method,
            "left_pad": int(float(raw.get("leftpad", "0"))),
            "right_pad": int(float(raw.get("rightpad", "0"))),
            "filter": raw.get("filter", "SheppLogan").strip(),
            "interval": interval,
        }

    def param_Internal(self) -> VxParam:
        """Param, or a readable error when it was never set."""
        if self.Param is None:
            raise ValueError("No parameters: call SetParam first.")
        return self.Param

    def activeTool_Internal(self):
        """Pick the backend that matches FlowMethod."""
        if self.FlowMethod == VxFlowMethod.PERFORMANCE:
            return self.PerformanceTool
        return self.QualityTool

    def save_Internal(self, dstPath: str, savePad: str, idx: int, img: np.ndarray):
        """Write one projection as slice_<idx>.tif under dstPath."""
        out_path = os.path.join(dstPath, f"slice_{idx:{savePad}}.tif")
        if not cv2.imwrite(out_path, img):
            raise RuntimeError(f"Failed to save: {out_path}")
