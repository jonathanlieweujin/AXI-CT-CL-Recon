import os
import warnings

import numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy

from SyntheticDataGenerator.Structures.materials import MU_CU


class MeshStructure:
    """
    Phantom voxelised from a 3D model file rather than built from constants.

    Supported: .stl and .obj. Both are read with VTK, which is already a
    dependency of nothing else here but ships with the environment; no FBX,
    because VTK has no FBX importer (it is a proprietary Autodesk format).
    Convert to OBJ or glTF first if that is what you have.

    The fill is SOLID, not a shell: vtkPolyDataToImageStencil marks every voxel
    inside the closed surface and vtkImageStencil writes them. A shell would
    give wrong line integrals - an X-ray sees the material a ray passes
    through, not the skin it crosses. Validated against an analytic volume: a
    2 mm sphere with a 0.6 mm hole drilled through (29.1948 mm^3) voxelises to
    29.0960 mm^3 at 20 um, -0.34%.

    Two things to know about accuracy and validity:

    - The result converges to the TESSELLATED mesh, not to the shape the CAD
      model represents. Past a certain voxel size the error stops improving
      and sits at whatever the export tolerance baked in, so set the export
      tolerance below the voxel size rather than the other way round.
    - The fill needs a closed manifold. CAD exports frequently are not, and a
      leaking fill produces silent nonsense rather than an error, so the
      boundary-edge count is checked and warned about before voxelising.

    Unlike the board phantoms, a mesh is a genuinely finite object: its edge in
    the projections is real, not an artefact of the phantom box stopping early.
    The coverage solve in VxSyntheticDataGenerator therefore does not apply -
    the box is the mesh bounding box plus a margin, which the generator sets as
    a phantom_size_mm override.
    """

    SUPPORTED = (".stl", ".obj")

    @staticmethod
    def Load(path: str, scale: float = 1.0, centre: bool = True,
             clean: bool = True):
        """
        Read a model file and return its vtkPolyData, cleaned, scaled, centred.

        Args:
            path:   .stl or .obj file.
            scale:  model units -> mm. STL and OBJ carry no unit information
                    whatsoever, so this is the caller's responsibility; getting
                    it wrong silently produces a part 1000x the intended size.
            centre: move the bounding-box centre to the origin, which is where
                    the projection geometry expects the object to be.
            clean:  merge coincident points and triangulate. Almost always
                    needed: OBJ (and STL) store a separate vertex per face, so
                    a closed model arrives with nothing shared between
                    neighbours and every edge counts as a boundary - the
                    watertight test fails and the solid fill leaks. One real
                    example: a 238k-quad model read as 952,478 points and
                    952,478 boundary edges; merging drops it to 238,153 points
                    and 0 boundary edges. Triangulating is separately required
                    because the stencil needs triangles, not quads.
        """
        if not path:
            raise ValueError("No model path given.")

        # format first: "this is an FBX" is more use than "not found" when a
        # wrong-format path is also a wrong path
        ext = os.path.splitext(path)[1].lower()
        if ext not in MeshStructure.SUPPORTED:
            raise ValueError(
                f"Unsupported model format {ext!r}: only "
                f"{', '.join(MeshStructure.SUPPORTED)} are read. VTK has no FBX "
                f"importer - convert to OBJ or glTF first.")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        reader = vtk.vtkSTLReader() if ext == ".stl" else vtk.vtkOBJReader()
        reader.SetFileName(path)
        reader.Update()
        poly = reader.GetOutput()
        if poly is None or poly.GetNumberOfCells() == 0:
            raise ValueError(f"Model file read but contains no geometry: {path}")

        if clean:
            merge = vtk.vtkCleanPolyData()
            merge.SetInputData(poly)
            merge.PointMergingOn()
            tri = vtk.vtkTriangleFilter()
            tri.SetInputConnection(merge.GetOutputPort())
            tri.Update()
            poly = tri.GetOutput()

        if scale != 1.0 or centre:
            b = poly.GetBounds()
            xf = vtk.vtkTransform()
            xf.PostMultiply()
            if centre:
                xf.Translate(-(b[0] + b[1]) / 2.0,
                             -(b[2] + b[3]) / 2.0,
                             -(b[4] + b[5]) / 2.0)
            if scale != 1.0:
                xf.Scale(scale, scale, scale)
            tf = vtk.vtkTransformPolyDataFilter()
            tf.SetInputData(poly)
            tf.SetTransform(xf)
            tf.Update()
            poly = tf.GetOutput()

        return poly

    @staticmethod
    def BoundsMm(poly) -> tuple[float, float, float]:
        """Bounding box of a loaded model, (width, length, thickness) in mm."""
        b = poly.GetBounds()
        return (b[1] - b[0], b[3] - b[2], b[5] - b[4])

    @staticmethod
    def IsWatertight(poly) -> int:
        """
        Boundary-edge count: 0 means the surface is closed and the solid fill
        is trustworthy. Anything else and the fill leaks into the background.
        """
        fe = vtk.vtkFeatureEdges()
        fe.SetInputData(poly)
        fe.BoundaryEdgesOn()
        fe.FeatureEdgesOff()
        fe.ManifoldEdgesOff()
        fe.NonManifoldEdgesOff()
        fe.Update()
        return int(fe.GetOutput().GetNumberOfCells())

    @staticmethod
    def GetStructure(vol_shape: tuple[int, int, int], voxel_size_mm: float,
                     path: str = None, **kwargs) -> np.ndarray:
        """Phantom only, in ASTRA (z, y, x) order - the Structures interface."""
        vol, _ = MeshStructure.Build(vol_shape, voxel_size_mm, path=path, **kwargs)
        return vol

    @staticmethod
    def Build(
        vol_shape: tuple[int, int, int],
        voxel_size_mm: float,
        path: str = None,
        scale: float = 1.0,
        mu: float = MU_CU,
        poly=None,
    ) -> tuple[np.ndarray, dict]:
        """
        Voxelise a model into a phantom.

        Args:
            vol_shape:     (vol_z, vol_y, vol_x) in voxels - the box to fill.
            voxel_size_mm: isotropic voxel edge, mm.
            path:          .stl or .obj file; ignored when `poly` is given.
            scale:         model units -> mm.
            mu:            linear attenuation coefficient written into every
                           voxel inside the surface, mm^-1. Defaults to copper.
                           STL carries no material information at all, so one
                           file is one material; see materials.py for the table.
            poly:          an already-loaded vtkPolyData, to avoid re-reading.

        Returns:
            (volume float32 (z, y, x), info dict describing what was loaded).
        """
        if poly is None:
            poly = MeshStructure.Load(path, scale=scale, centre=True)

        open_edges = MeshStructure.IsWatertight(poly)
        if open_edges:
            warnings.warn(
                f"Model surface is not closed ({open_edges} boundary edges): "
                f"the solid fill will leak and the phantom will be wrong. "
                f"Repair the mesh, or re-export it as a watertight solid.",
                stacklevel=2)

        vol_z, vol_y, vol_x = (int(v) for v in vol_shape)
        dims = (vol_x, vol_y, vol_z)
        # the box is centred on the origin, matching the projection geometry
        origin = [-d * voxel_size_mm / 2.0 for d in dims]

        img = vtk.vtkImageData()
        img.SetSpacing(voxel_size_mm, voxel_size_mm, voxel_size_mm)
        img.SetDimensions(*dims)
        img.SetOrigin(*origin)
        img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        img.GetPointData().GetScalars().Fill(1)

        stencil = vtk.vtkPolyDataToImageStencil()
        stencil.SetInputData(poly)
        stencil.SetOutputOrigin(*origin)
        stencil.SetOutputSpacing(voxel_size_mm, voxel_size_mm, voxel_size_mm)
        stencil.SetOutputWholeExtent(img.GetExtent())
        stencil.Update()

        cut = vtk.vtkImageStencil()
        cut.SetInputData(img)
        cut.SetStencilConnection(stencil.GetOutputPort())
        cut.ReverseStencilOff()
        cut.SetBackgroundValue(0)
        cut.Update()

        mask = vtk_to_numpy(cut.GetOutput().GetPointData().GetScalars())
        vol = mask.reshape(vol_z, vol_y, vol_x).astype(np.float32) * float(mu)

        w, l, t = MeshStructure.BoundsMm(poly)
        info = dict(path=path, scale=scale, mu=float(mu),
                    open_edges=open_edges,
                    triangles=int(poly.GetNumberOfCells()),
                    bounds_mm=(w, l, t),
                    filled_voxels=int(mask.sum()),
                    filled_mm3=float(mask.sum()) * voxel_size_mm ** 3)

        # ASTRA index 0 is the source-facing slice; VTK hands back z ascending,
        # so flip to match the convention the other Structures use.
        return np.ascontiguousarray(vol[::-1]), info
