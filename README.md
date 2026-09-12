# Overview

Cone-beam CT (computed tomography) and CL (computed laminography) reconstruction
driven from a single parameter object, with two interchangeable GPU backends, Quality that utilises TIGRE framework and Performance that utilises ASTRA Toolbox framework.

*ASTRA Toolbox: <https://github.com/astra-toolbox/astra-toolbox>*  
*TIGRE Toolbox: <https://github.com/CERN/TIGRE>*

The two flows are held to the same result: `tests/test_flow_equivalence.py`
reconstructs synthetic datasets of known phantoms through both and requires the
volumes to agree. Everything the module needs beyond reconstruction: projection
loading, normalisation, geometry vectors, synthetic data generation: lives in
the package rather than in separate scripts.

## Content

- [CT vs CL](#ct-vs-cl)
- [Layout](#layout)
- [Manager](#manager)
  - [Capabilities](#capabilities)
  - [State](#state)
  - [Geometry](#geometry)
- [Requirements](#requirements)
- [Pre-Requisites](#pre-requisites)
  - [1. Create the environment](#1-create-the-environment)
  - [2. Install the packages](#2-install-the-packages)
  - [3. Install TIGRE](#3-install-tigre)
  - [4. Verify](#4-verify)
- [Running](#running)
- [Synthetic data](#synthetic-data)
  - [Phantom sizing](#phantom-sizing)
  - [BGA phantom](#bga-phantom)
  - [PCB panel phantom](#pcb-panel-phantom)
  - [3D model phantoms (STL / OBJ)](#3d-model-phantoms-stl--obj)
  - [Memory](#memory)
- [Sample dataset](#sample-dataset)
- [Tests](#tests)
- [Future Implementations](#future-implementations)
- [Notes](#notes)

## CT vs CL

Both modes share the same code path; the geometry decides which one you get:

- **CT**: detector upright (`DetTiltX = 90`), full circular scan.
- **CL**: detector tilted away from upright (`DetTiltX` e.g. `51`), laminographic
  geometry. Set `LaminographyMethod = LaminographyMethodConstants.INCLINED` for a
  genuinely tilted detector, or `.COPLANAR` to keep a flat detector while the
  source/detector pair traverses the tilted orbit.

Switch backend with `Mode = VxFlowMethod.QUALITY` (TIGRE) or
`Mode = VxFlowMethod.PERFORMANCE` (ASTRA) in `tests/reconTest.py`.

*Fisher, S., Mavrogordato, M. N., Blumensath, T., & Boardman, R. P. (2019).
Laminography in the lab: Imaging planar objects using a conventional x-ray CT
scanner. Measurement Science and Technology, 30(3), Article 035401.
<https://iopscience.iop.org/article/10.1088/1361-6501/aafcae/pdf>*

## Layout

| Path | Purpose |
| --- | --- |
| `tests/reconTest.py` | Entry point: sets geometry/volume parameters, loads projections, reconstructs, and displays slices with a slider. Every parameter is also a CLI flag (`--src`, `--tilt-x`, `--sod`, …) - run with `--help` for the full list. |
| `tests/reconWithConfigTest.py` | Entry point: reconstructs a dataset straight from its own `Config/geometry.config` rather than hand-built params - the counterpart to `manager.LoadConfig`. `--dataset <root>` points it at a dataset's root folder (`Corrected/`, `Config/geometry.config`, output `Recon/`); `--src`/`--src-config`/`--dst` override the three individually. |
| `Manager/__init__.py` | `VxManager`: the single entry point: owns the parameters, both backends, the projections and the result. `LoadImages` / `SaveImages` read and write a projection stack, `NormaliseProjections` conditions one, `Run` reconstructs. See [Manager](#manager). |
| `Manager/Param/__init__.py` | `VxParam`: the shared geometry/acquisition parameter container. Applies binning to the detector size, pitch and offsets on construction. |
| `Manager/Constants/flow_method.py` | `VxFlowMethod`: backend selector enum (`QUALITY` / `PERFORMANCE`). |
| `Manager/Constants/laminography_method.py` | `LaminographyMethodConstants`: geometry selector enum (`INCLINED` / `COPLANAR`). |
| `Manager/Constants/recon_method.py` | `ReconMethodConstants`: reconstruction algorithm identifiers (`FDK`, `CGLS`, `SIRT`, …). |
| `Manager/Util/compute_geometry.py` | `VxComputeGeometry`: the shared coplanar/inclined ASTRA `cone_vec` geometry builders used by both backends. |
| `Manager/Util/load_images_internal.py` | `FileLoader`: reads raw projection stacks with interval and binning, threaded across the stack. Backs `VxManager.LoadImages`; the matching writes live in `VxManager.SaveImages`. |
| `Manager/Util/redundancy_weighting.py` | `to_apply_redundancy_weighting`: decides when Wang redundancy weights apply (offset detector, upright CT geometry only). |
| `Manager/Tool/Quality/__init__.py` | TIGRE backend (`VxTool`, `VxGeom`): builds the TIGRE geometry and runs reconstruction. |
| `Manager/Tool/Performance/__init__.py` | ASTRA backend (`VxTool`): same interface, ASTRA toolbox, plus `plot_geometry` for inspecting the scan vectors. |
| `tests/syntheticTest_inclined.py` | Entry point: writes an inclined-laminography `MICRO_JIG` dataset. `--tilt`, `--offset-u`/`--offset-v` are CLI flags; each combination gets its own output subfolder so runs don't overwrite each other. |
| `tests/syntheticTest_coplanar.py` | Entry point: writes a coplanar-laminography `PCB_PANEL` dataset (with a commented-out projection viewer). Same CLI-flag convention as `syntheticTest_inclined.py`, plus detector/volume/geometry overrides. |
| `tests/syntheticTest_3dModel.py` | Entry point: generates a dataset from a `.obj` / `.stl` model instead of a built-in phantom (`--model`, `--model-scale`, plus the same tilt/offset/geometry flags). See [Sample dataset](#sample-dataset). |
| `SyntheticDataGenerator/__init__.py` | `VxSyntheticDataGenerator`: the generator entry point. Derives the binned/unbinned parameter pair, builds a phantom, forward-projects it through the declared geometry, and writes a `Corrected/` + `Config/` dataset matching the layout of a real acquisition. See [Synthetic data](#synthetic-data). |
| `SyntheticDataGenerator/Structures/bga.py` | `BgaStructure`: BGA / WLCSP joint phantom at physical scale, with seeded IPC-7095 defects and a ground-truth manifest. See [BGA phantom](#bga-phantom). |
| `SyntheticDataGenerator/Structures/pcb_panel.py` | `PcbPanelStructure`: PCB panel phantom measured off the FID_2 reference acquisition. See [PCB panel phantom](#pcb-panel-phantom). |
| `SyntheticDataGenerator/Structures/jig.py` | `JigStructure`: full-size (~76 mm) 22-sphere VDI/VDE 2630 accuracy check piece, for a general industrial CT. |
| `SyntheticDataGenerator/Structures/micro_jig.py` | `MicroJigStructure`: small-FOV (~4 mm) 22-ball check piece fitted to published multisphere-standard metrology, for a micro-CT / laminography-scale rig. |
| `SyntheticDataGenerator/Structures/mesh.py` | `MeshStructure`: voxelises an `.stl` or `.obj` model into a phantom with VTK. See [3D model phantoms](#3d-model-phantoms-stl--obj). |
| `SyntheticDataGenerator/Structures/materials.py` | Linear attenuation coefficients (mm⁻¹, 60 keV) shared by the phantoms. |
| `tests/tilt_optimizer.py` | Reprojection-matching optimiser that recovers a scanner's tilt (`DetTiltX`) and detector offset (`offset_u`/`offset_v`) purely from a `MICRO_JIG` dataset's projections - see its module docstring for the method and references. |
| `tests/` | Pytest suite plus the standalone entry-point scripts above and the synthetic dataset generator they run against. |

## Manager

`VxManager` (`Manager/__init__.py`) is the single entry point for a
reconstruction. It owns the parameters, both backends, the loaded projections and
the resulting volume:

```python
from Manager import VxManager
from Manager.Param import VxParam
from Manager.Constants.flow_method import VxFlowMethod

manager = VxManager(param, flow_method=VxFlowMethod.QUALITY, left_pad=0, right_pad=0)
manager.LoadImages(SRC, interval=1)
manager.Run()
volume = manager.Result
```

### Capabilities

| Method | What it does |
| --- | --- |
| `LoadImages(input_folder, interval=1)` | Loads the `.tif` projection stack described by `Param`: restores the pre-binning detector size, applies `interval` sub-sampling and `binning`, and resamples to exactly `Param.num_of_imgs` frames. Returns and caches the stack as `Images`. |
| `SaveImages(output_folder, images=None, savePad=None)` | Writes a stack to `output_folder` as `slice_XXXX.tif`, one file per frame, threaded. Defaults to the cached `Images`. The index width is derived from the stack length unless `savePad` forces one, so the names always sort in stack order. |
| `Run(projections=None, **kwargs)` | Reconstructs with the backend selected by `FlowMethod` and the algorithm on `Param.recon_method`. Falls back to the cached `Images` when no stack is passed; extra `kwargs` (`filter`, `niter`, …) go straight to the backend algorithm. |
| `GetScanGeometry()` | Returns the `(n, 12)` ASTRA `cone_vec` rows `[Sx Sy Sz  Dx Dy Dz  Ux Uy Uz  Vx Vy Vz]` for the current parameters. |
| `NormaliseProjections(srcPath, dstPath, toApplyLogTransform=True, savePad=None)` | Loads a raw stack, scales it to `[0, 1]` on a single global min/max, optionally applies the Beer–Lambert `-log(x + eps)` transform, and hands the result to `SaveImages` for `dstPath`. Returns the normalised stack and caches it as `Images`. |

### State

| Attribute | Contents |
| --- | --- |
| `Param` | The `VxParam` geometry/acquisition container. |
| `FlowMethod` | `QUALITY` (TIGRE) or `PERFORMANCE` (ASTRA): chooses the backend. |
| `ReconMethod` | Algorithm identifier taken from `Param.recon_method` at construction. |
| `Images` | Last loaded/normalised projection stack, `(n, v, u)` float32. |
| `Result` | Last reconstructed volume. |
| `QualityTool` / `PerformanceTool` | Both backends, constructed up front with the same `Param`, laminography method and padding: switching `FlowMethod` needs no rebuild. |

Padding (`left_pad` / `right_pad`) is edge-extrapolated onto the detector by the
backend and the detector centre is shifted to match, which suppresses truncation
artefacts without changing the caller's geometry.

### Geometry

`Param.laminography_method` picks which builder in `VxComputeGeometry` produces
the scan vectors. Both return the same `(n, 12)` ASTRA `cone_vec` rows, so
`GetScanGeometry()` and both backends stay geometry-agnostic: only the vectors
differ.

| | `INCLINED` (default) | `COPLANAR` |
| --- | --- | --- |
| Detector | Tilted, rotates with the rig: pixel axes differ per projection | Stays flat in the world frame: `U = (pitch, 0, 0)`, `V = (0, pitch, 0)` for every projection |
| Motion | Source and detector orbit on a rotation `R = Rx(-θ)·Ry(β)·Rz(-φ)`, θ from `tilt_x`, β from `tilt_y` | Source and detector translate on an XY orbit at fixed axial Z; the tilt only shifts them in-plane by `z·tan(tilt_x)` |
| `tilt_y` | Honoured | Ignored: `tilt_x` alone defines the orbit |
| Covers CT | Yes: CT is the `tilt_x = 90` case of this builder | No |
| Wang redundancy weighting | Available (offset detector, `tilt_x ≈ 90` only) | Never applied |

**Inclined** is the general case: a genuinely tilted detector carried around the
rotation, with the source and detector placed along opposite ends of the rig axis
and the pixel axes taken from the same rotation. Because a `tilt_x` of 90° is
just an untilted detector, ordinary circular CT falls out of this builder rather
than needing a path of its own.

*Van Aarle, W., Palenstijn, W. J., Cant, J., Janssens, E., Bleichrodt, F.,
Dabravolski, A., De Beenhouwer, J., Batenburg, K. J., & Sijbers, J. (2016). Fast
and flexible X-ray tomography using the ASTRA toolbox. Optics Express, 24(22),
25129–25145. <https://doi.org/10.1364/OE.24.025129>*

**Coplanar** models the other way to get laminographic sampling: keep the
detector flat and physically traverse the source/detector pair across the tilted
orbit. `SOD`/`SDD` are read as axial distances along Z, and detector offsets move the centre along the fixed `U`/`V` axes.

*Porsch, F. (2010). Computed Laminography for X-ray Inspection of
Lightweight Constructions.
<https://www.academia.edu/download/82596285/mo3a3.pdf>*

## Requirements

- Python 3.13 (required: the TIGRE build is a `cp313` extension)
- An NVIDIA GPU + driver. No system CUDA Toolkit is needed at runtime: `astra-toolbox`
  ships its own CUDA runtime via pip, and the TIGRE wheel is self-contained. The
  Toolkit (`nvcc`) is only required to *build* TIGRE from source.
- `numpy`, `matplotlib`, `opencv-python`, `scipy`, `vtk`, and `pytest` for the
  test suite. `astra`, `vtk` and `scipy` are imported at the top of
  `SyntheticDataGenerator`, so all three must be present to import it at all -
  not only to project.
- One or both reconstruction backends:
  - [TIGRE](https://github.com/CERN/TIGRE): built from source, see Installation
  - [ASTRA Toolbox](https://astra-toolbox.com/): `astra-toolbox` on PyPI

## Pre-Requisites

The environment lives at `.venv` in the repository root and is created with conda
(Miniconda), because ASTRA and TIGRE both need a CUDA-aware Python.

**Python must be 3.13**: the TIGRE build is a compiled `cp313` extension and will
not load on another minor version.

### 1. Create the environment

```powershell
conda create -p ".venv" python=3.13 pip -y
```

### 2. Install the packages

```powershell
.venv\python.exe -m pip install numpy matplotlib opencv-python scipy vtk pytest astra-toolbox
```

This covers everything except TIGRE:

| Package | Used by |
| --- | --- |
| `numpy` | everywhere |
| `matplotlib` | `tests/reconTest.py` slice viewer, `plot_geometry` |
| `opencv-python` (`cv2`) | projection loading and normalisation |
| `scipy` | TIGRE / ASTRA dependency |
| `pytest` | `tests/` |
| `astra-toolbox` | Performance backend (bundles its own CUDA runtime + cuFFT) |
| `vtk` | Reading and voxelising `.stl` / `.obj` model phantoms |

### 3. Install TIGRE

TIGRE is **not** on PyPI at the version used here (3.1.3): the `tigre` package on
PyPI is a different project. It has to be built from the TIGRE source tree, which
requires the CUDA Toolkit (`nvcc`) and MSVC build tools:

```powershell
.venv\python.exe -m pip install "<path-to>\TIGRE-master"
```

If a matching wheel has already been built on this machine, pip caches it and the
rebuild is skipped. To install it directly:

```powershell
.venv\python.exe -m pip install "$env:LOCALAPPDATA\pip\Cache\wheels\...\tigre-3.1.3-cp313-cp313-win_amd64.whl"
```

TIGRE pulls in `h5py` and `tqdm` as dependencies.

### 4. Verify

```powershell
.venv\python.exe -c "import numpy, matplotlib, cv2, scipy, vtk, astra, tigre.algorithms; print('ok', astra.use_cuda())"
```

`astra.use_cuda()` must print `True`: if it prints `False`, the GPU is not visible
to ASTRA and reconstruction will fail.

## Running

```powershell
.venv\python.exe tests\reconTest.py
```

Every parameter below is also a CLI flag (`--src`, `--tilt-x`, `--sod`, …) -
run `.venv\python.exe tests\reconTest.py --help` for the full list. Check the
defaults at the top of `tests/reconTest.py`:

- `SRC`: folder of normalised raw projections
- `DetU` / `DetV` / `DetZ`: detector width, height, and number of projections
- `DetPitch`, `Binning`, `Interval`
- `SOD` / `SDD`: source-to-object and source-to-detector distance (mm)
- `DetTiltX` / `DetTiltY`, `DetOffsetU` / `DetOffsetV`
- `VolX` / `VolY` / `VolZ` and `VolMid*`: output volume size and centre
- `LEFT_PAD` / `RIGHT_PAD`: projection padding, used to suppress truncation artefacts
- `Iterations`: iteration count for the iterative algorithms (ignored by `FDK`)

## Synthetic data

`VxSyntheticDataGenerator` (`SyntheticDataGenerator/__init__.py`) builds a
dataset the recon flows can be checked against: a phantom, the projections that
geometry would produce, and a `Corrected/` + `Config/` pair matching the layout
of a real acquisition. Because the projections are forward-projected through the
same geometry the backends rebuild, a correct backend reconstructs the phantom
that produced them.

It mirrors `VxManager`: construct, set the params, generate.

```powershell
.venv\python.exe tests\syntheticTest_inclined.py
```

```python
from SyntheticDataGenerator import VxSyntheticDataGenerator, VxPhantomConstants

generator = VxSyntheticDataGenerator(phantom=VxPhantomConstants.SLAB)
generator.SetParams(angles=ProjectionAngles, sod=SOD, sdd=SDD,
                    det_width=DetU, det_height=DetV, det_pitch=DetPitch,
                    volume=(VolX, VolY, VolZ), tilt_x=DetTiltX, binning=Binning,
                    laminography_method=LaminographyMethod)
generator.Generate(DST)
print(generator.Describe())
```

| Method | What it does |
| --- | --- |
| `SetParams(...)` | Derives the geometries a dataset needs from one unbinned detector description: the reconstruction geometry (binned) and the acquisition geometry (unbinned, the detector the projections are stored at). `volume` is the *reconstruction* volume only: it does not size the phantom. Returns the reconstruction param. |
| `PhantomSizeMm` / `PhantomVolume` | The box the phantom is built in, mm and voxels. Thickness comes from the phantom class, lateral extent is solved from the geometry so the object covers the detector at every angle. See [Phantom sizing](#phantom-sizing). |
| `MinimumPhantomVolume()` | Smallest box that keeps the phantom's own edge out of every projection, or `None` when no size achieves it. |
| `Run(model3DPath=None, model_scale=1.0, model_mu=None)` | Builds the phantom and forward-projects it. Leaves the phantom on `Volume` and the projections on `Sinogram`, in ASTRA `(det_v, angles, det_u)` order. Given a `.stl` or `.obj` path it voxelises that instead of `PhantomKind`. See [3D model phantoms](#3d-model-phantoms-stl--obj). |
| `Save(root)` | Writes `Corrected/`, `Config/geometry.config` and `phantom.npy` under `root`. Projects first if `Run` has not been called. |
| `Generate(root)` | `Run` then `Save`, in one call. |
| `VoxelSize` / `NumProjections` | Derived geometry values. |
| `Describe()` | One-line summary of what was generated, for progress output. |

### Phantom sizing

The reconstruction volume and the object are different things, and the
generator keeps them apart. A real board extends well past the reconstructed
field of view; a phantom that stops at it puts its own box wall into the
projections, which shows up as a straight edge sweeping across the frame and
then as an artefact in the reconstruction.

So `volume=(VolX, VolY, VolZ)` is written to the config and used to crop the
saved phantom, but it does not size the object. The phantom box is derived:

- **thickness** from the phantom class (`STACK_THICKNESS_MM` / `DEFAULT_SIZE_MM`)
  — it is a property of the part, not of the scan.
- **lateral extent** solved by bisection on the scan vectors: the smallest box
  whose silhouette covers the whole detector at every angle.

Every geometry input moves that solve. At the FID_2 geometry with `PCB_PANEL`:

| Geometry | phantom |
| --- | --- |
| tilt 30, INCLINED | 7.02 mm |
| tilt 51, INCLINED | 9.03 mm |
| COPLANAR | 4.61 mm — exactly `DetU` |
| `DetU/DetV` 1024 | 4.42 mm |
| `DetOffsetU` 200 px | 7.94 mm |

COPLANAR needs no margin at all because the detector stays parallel to the FOV,
so the silhouette never rotates. Inclined costs 1.5–2× on top of that.

Two cases the solve cannot serve, both of which warn rather than fail silently:
an upright detector (CT geometry, `tilt_x = 90`) where a flat object turns
edge-on and no finite board fills the frame; and a box too large for
`max_phantom_voxels`, which is clamped. `phantom_size_mm=(w, l, t)` overrides
the solve outright when truncation is what you want.

The phantom is built at the **acquisition** (unbinned) voxel, not the binned
reconstruction voxel, so binning does not throw away detail before the
projection ever happens. `phantom.npy` is saved cropped to the reconstruction
volume — block-averaged back down when binning applies — so it still compares
directly against a reconstruction.

Two exemptions, for opposite reasons:

- `SOLID` and `SLAB` have no physical size at all - their features are
  fractions of the voxel array - so there is nothing to solve for and they are
  still built on the reconstruction grid.
- A **model phantom** is a genuinely finite object. Its edge in the projections
  is real, not an artefact of the box stopping early, so the coverage solve is
  skipped and the box is the model bounding box plus a margin.

`VxPhantomConstants` selects the shape: `SOLID` (sphere plus an off-centre rod,
so a mirrored reconstruction cannot score as a match), `SLAB` (a thin plate with
in-plane structure, the laminography case), `HBM` (a stacked-die cross-section:
μbumps, C4 bumps, DRAM layers), or `BGA` / `WLCSP` (see below).

### BGA phantom

`BgaStructure` (`SyntheticDataGenerator/Structures/bga.py`) is the inspection
case rather than a geometry check: a die on solder balls over a board, built at
physical scale with the defect population an X-ray or CL inspection is looking
for. `BGA` is the die-on-substrate build (450 μm ball on a 900 μm pitch, BT
substrate with 90 μm laser microvias); `WLCSP` is the wafer-level build, balls
grown straight onto the die through a Cu UBM at 300 μm on a 500 μm pitch.

Voxel values are linear attenuation coefficients in mm⁻¹ at 60 keV (NIST
μ/ρ × density), so the forward projection through a volume gridded in mm is a
physical line integral rather than an arbitrary label map:

| Material | μ (mm⁻¹) |
| --- | --- |
| SAC305 solder (96.5Sn/3.0Ag/0.5Cu) | 4.790 |
| copper (pad, plane, microvia) | 1.427 |
| silicon (die) | 0.075 |
| BT resin (substrate) | 0.060 |
| FR4 (board) | 0.055 |
| solder mask / epoxy | 0.042 |

Solder is ~64× silicon and ~3.4× copper, which is why the ball reads as opaque
and the void is the only thing visible through it.

Defects are seeded per ball from a fixed RNG seed and sized against IPC-7095,
which calls a ball a defect when the summed void area in projection passes 25 %
(Class 2; Class 3 practice is nearer 10 %). A single void of diameter `d` in a
ball of diameter `D` projects `(d/D)²`, so the void diameter follows from the
target percentage directly: `void`, `gross_void`, `void_cluster`,
`head_in_pillow`, `open`, `bridge`, `misaligned`, `missing`.

`Save` writes `ground_truth.json` alongside `phantom.npy`, listing every ball,
its defect and its summed projected void percentage, so a detector can be scored
against what was seeded instead of eyeballed. Use `BgaStructure.Build` directly
to override `mode`, `seed` or `defect_rate`.

### PCB panel phantom

`PcbPanelStructure` (`SyntheticDataGenerator/Structures/pcb_panel.py`), selected
with `VxPhantomConstants.PCB_PANEL`, reproduces the FID_2 reference acquisition
(1536 × 1536 × 300 at 3.0 μm = 4.608 × 4.608 × 0.900 mm). Every in-plane
dimension was measured off that reconstruction, not taken from a datasheet:

| Feature | Measured |
| --- | --- |
| bump pitch | 200 μm |
| bump diameter | 83 μm (p10 80, p90 86, over 103 objects) |
| main site | 7 × 8 bumps, x 1710–2910, y 1610–3010 μm |
| second site | 7 × 3 bumps, x 1710–2910, y 4104–4504 μm |
| plated through hole | 110 μm drill, 165 μm outer → 27 μm barrel wall |
| via layout | 28 measured centres, mostly perimeter |

The z stack is *not* measured. A 30° laminography reconstruction smears depth
badly — a 60 μm bump reads as a ~190 μm streak in the reference — so copying
its z profile would bake the point-spread into the phantom. Ordinary board
values are used instead, summing to the ~570 μm of metal the reference does
show: 25 μm mask / 30 μm Cu / 400 μm FR4 / 30 μm Cu / 25 μm mask / 60 μm bump.

Feature coordinates are held in μm relative to the volume centre, so the panel
renders at any voxel size or volume shape and clips at the box — only the crop
changes, never the part. Defects are off by default (`defect_rate=0.0`) so the
panel is a faithful copy; raise it to seed `void` / `gross_void` / `missing` /
`misaligned` with a ground-truth manifest, as `BgaStructure` does.

Storing at full detector resolution and binning on load is what a real
acquisition does, so `binning` is applied to the reconstruction geometry only.
With `binning=1` the two geometries are the same object, which is how the written
config knows binning has not already been applied.

### 3D model phantoms (STL / OBJ)

`Run` takes an optional model path, which replaces the built-in phantom:

```python
generator.SetParams(...)
generator.Run(model3DPath=r"part.stl", model_scale=1.0, model_mu=MU_CU)
generator.Save(DST)
```

`MeshStructure` (`SyntheticDataGenerator/Structures/mesh.py`) reads the file
with VTK, centres it on the origin, and voxelises it. The box is the model
bounding box plus four voxels on each side, applied through the same
`phantom_size_mm` override described in [Phantom sizing](#phantom-sizing), so
the projection path is unchanged.

| Format | Read by |
| --- | --- |
| `.stl` | `vtkSTLReader` |
| `.obj` | `vtkOBJReader` |
| `.fbx` | **not supported** - VTK has no FBX importer (proprietary Autodesk format). Convert to OBJ or glTF first. |

**The fill is solid, not a shell.** `vtkPolyDataToImageStencil` marks every
voxel inside the closed surface; a shell would give the wrong line integrals,
since an X-ray sees the material a ray passes through rather than the skin it
crosses. Checked against an analytic volume - a 2 mm sphere with a 0.6 mm hole
drilled through, 29.1948 mm³:

| voxel | grid | time | measured | error |
| --- | --- | --- | --- | --- |
| 0.050 mm | 80 x 76 x 80 | 0.18 s | 29.1945 mm³ | -0.00 % |
| 0.020 mm | 200 x 191 x 200 | 0.25 s | 29.0960 mm³ | -0.34 % |
| 0.010 mm | 400 x 382 x 400 | 0.40 s | 29.0816 mm³ | -0.39 % |

`model_mu` is the linear attenuation coefficient written into every voxel
inside the surface, defaulting to copper. STL has no material information at
all, so one file is one material; `.obj` group and glTF per-node materials
would be needed for a multi-material assembly, which is not implemented.

`MeshStructure.Build` also returns an info dict - triangle count, bounding box,
open-edge count, filled voxels and filled mm³ - which `Run` leaves on
`Manifest` and `Save` writes to `ground_truth.json`.

### Memory

`create_sino3d_gpu` needs the phantom **and** the sinogram resident on the GPU
at once. When that allocation fails ASTRA reports
`Cannot create cython.array from NULL pointer`, which says nothing about size,
so the sizes are checked and reported here instead:

- `max_phantom_voxels` on `SetParams` caps the phantom. Past it the box is
  clamped and a warning names the phantom size, the sinogram size and their
  total in GB. The default is 3e9 voxels, i.e. ~12 GB of float32 - **higher
  than most cards**, so set it to suit yours.
- A failed `create_sino3d_gpu` is re-raised as a `MemoryError` carrying both
  shapes and their total, rather than the bare NULL-pointer `ValueError`.

The sinogram is fixed by the detector and angle count, so when memory is tight
reducing `DetU`/`DetV` often buys more than shrinking the volume: at
3072 x 3072 x 180 the sinogram alone is 6.8 GB, and halving the detector cuts
it to 1.7 GB.

## Sample dataset

Greymon: <https://free3d.com/3d-model/greymon-39969.html>

## Tests

```powershell
.venv\python.exe -m pytest tests
```

`test_flow_equivalence.py` reconstructs three synthetic datasets: CT (inclined,
`DetTiltX = 90`), CL (inclined, `51`) and Coplanar (`51`): through both backends,
then asserts the two volumes correlate and that each recovers the phantom the
projections came from. It needs the datasets on disk first:

```powershell
.venv\python.exe tests\make_synthetic_data.py
```

That writes roughly 4.5 GB of projections per dataset under `DATASET_ROOT`, an
absolute path set at the top of both files.

## Future Implementations

### Flat field Correction

Beer Lambert absorbtion law implementation with input dark, flat and source projection image(s). Support single, average and index application, where epsilon e = 1e-5.

### Geometry Alignment Optimisation / Jitter Correction

Recovering the geometry the data was actually taken with, rather than the one
that was declared: systematic setup error (centre of rotation, detector tilt,
magnification) and per-projection jitter (stage wobble, sample motion).

**Where the correction goes.** `GetScanGeometry()` is the single point every
backend takes its vectors from, so it is where a correction belongs: nominal
`(n, 12)` from `VxComputeGeometry`, compose a setup delta and an optional
per-view rigid delta on top, hand the corrected rows to the backend. This is
the structure TomoJAX uses (`pose_stack = setup_pose @ pose_delta`).

**What must not be optimised: the raw 12-vector.** It is where a correction is
applied, never how it is parametrised. Measured on this geometry, 256 angles:

- Rotating or translating every view together with the object changes the
  projections by ~1e-12 px. Alignment solves geometry and volume jointly, so
  those 6 directions are exactly unobservable and the normal equations are
  singular in them. Damping hides that; it does not fix it.
- Scaling `U` by 2 % (non-square pixels) moves points 8.6 px, shearing `V`
  toward `U` moves them 8.8 px. Both are large, easily fitted, and physically
  impossible. Given those degrees of freedom an optimiser will absorb beam
  hardening and gain drift into the geometry and report a lower loss.
- 12 x 256 = 3072 free numbers, against 1280 for a 5-DOF per-view pose, 60 for
  a spline pose (12 knots), and 6 for the setup vector alone.

**Parametrisation.** Shared setup vector (`tilt_x`, `tilt_y`, `offset_u`,
`offset_v`, `sod`, `sdd` - already on `VxParam`), plus an optional per-view
rigid delta constrained to a spline over angle rather than free per view.
Sizing matters more here than in a JAX toolkit: with no autodiff behind ASTRA
or TIGRE, finite differences cost ~2N forward projections per Jacobian. Twelve
projections for the setup vector is affordable; 2560 for a free per-view pose
is not.

**Gauge.** The 6 global rigid directions above have to be pinned before any
fit - anchor the mean pose delta to zero, as TomoJAX's `anchor_mean` policy
does. `offset_u` and `volume_mid_x` are near-degenerate for the same reason and
should not be fitted together.

**Validation.** The synthetic generator is the harness: perturb a known
geometry, generate, fit, and check the parameters come back. `phantom.npy` and
`ground_truth.json` give a scoring target that does not depend on eyeballing a
slice.

#### The three setups do not take a uniform treatment

Measured differences that change what is fittable:

| | CT (`INCLINED`, 90) | CL `INCLINED` (51) | CL `COPLANAR` (51) |
| --- | --- | --- | --- |
| `tilt_y` effect on the vectors (0 -> 5 deg) | 0.0073 | 25.18 | **0.0000** |
| views 180 deg apart conjugate | **yes, exact** | no | no |
| detector axes rotate per view | `U` only | `U` and `V` | **neither** |
| Wang redundancy weighting | yes | no | no |

**CT (`INCLINED`, `tilt_x = 90`)**

- Centre of rotation dominates; it is `offset_u`. Seed it before optimising -
  views 0 and 180 deg are exactly conjugate here, so the classic opposing-pair
  match and a centre-of-mass-along-`u` estimate both apply.
- Fit `offset_u`, `tilt_x` around 90, and magnification. **Fix `tilt_y`** - 5
  deg of it moves the vectors by 0.0073, so it is effectively unobservable and
  will only add a flat direction.
- `to_apply_redundancy_weighting` switches on hard thresholds
  (`offset_u != 0`, `abs(tilt_x - 90) <= 0.5`). An optimiser crossing either
  sees a step change in the objective, so freeze the weighting decision for the
  duration of a fit rather than re-evaluating it per iteration.

**CL `INCLINED`**

- No conjugate views, so the CT centre-of-rotation seeds do not transfer. Needs
  a different initialiser - a fiducial, or a coarse grid search on `offset_u`.
- `tilt_y` is fully live (25.18) and belongs in the fit set, unlike CT.
- `tilt_x` is strongly observable but couples to axial scale; fit it jointly
  with `sod`/`sdd` or hold magnification fixed.
- Both `U` and `V` rotate per view, so per-view orientation error is real and a
  full rigid pose delta is meaningful. The missing wedge enlarges the null
  space, so lean harder on the spline constraint.

**CL `COPLANAR`**

- **`tilt_y` must be excluded from the parameter set.** The builder ignores it
  entirely (0.0000 change), so including it contributes an exactly flat
  direction and a singular Hessian.
- `U` and `V` are constant across views, so per-view orientation error cannot
  be represented by construction. Per-view corrections reduce to translations -
  a smaller and better conditioned problem than the inclined case.
- The detector tracks the object centre exactly (the shadow of the origin lands
  at `u = v = 0` at every angle), so detector offsets are the clean handle -
  and note an offset is what breaks that tracking.
- No redundancy weighting, so no threshold discontinuity in the objective.
- Cheapest setup to validate against: a coplanar phantom needs no lateral
  margin at all (see [Phantom sizing](#phantom-sizing)), so the test datasets
  are a fraction of the size the inclined ones need.

#### Order of work

1. Correction layer on `GetScanGeometry()` - compose deltas onto the nominal
   vectors, no solver yet, verified by round-tripping a known perturbation.
2. Setup-only fit (6 parameters, finite differences), per-setup parameter masks
   as above, validated against synthetic data with a known offset.
3. Gauge anchoring and the seeds, CT first since it has real initialisers.
4. Smooth per-view delta (spline) for jitter, coarse-to-fine, inclined last
   since it is the worst conditioned.

## Notes

- Raw projections, reconstructed volumes, and other imaging data are excluded by
  `.gitignore`: keep them outside the repository.
- `SRC` in `tests/reconTest.py` and `DATASET_ROOT` in `tests/` are absolute,
  machine-specific paths and need adjusting per setup. There are no hardcoded CUDA
  or TIGRE paths: both are resolved from the installed packages in `.venv`.