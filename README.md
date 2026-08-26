# Latest CT / CL Module

Working code for the latest CT (computed tomography) and CL (computed laminography)
reconstruction module, plus supporting detector pre-processing.

## Layout

| Path | Purpose |
| --- | --- |
| `CT/New/reconTest.py` | Entry point — sets geometry/volume parameters, loads projections, reconstructs, and displays slices with a slider. |
| `CT/New/Util/param.py` | `VxParam` — the shared geometry/acquisition parameter container. |
| `CT/New/Util/load_files.py` | `FileLoader` — reads raw projection stacks with interval and binning. |
| `CT/New/TigreLib/get_structure.py` | TIGRE backend (`VxTool`) — builds the geometry and runs reconstruction. |
| `CT/New/AstraLib/get_structure.py` | ASTRA backend (`VxTool`) — same interface, ASTRA toolbox. |
| `CT/normalise.py` | Projection normalisation (flat/dark handling) prior to reconstruction. |
| `Bad Pixel Correction/` | Detector bad-pixel mask generation (`MaskGenerator`) and correction (`BadPixelCorrector`). |

## CT vs CL

Both modes share the same code path; the geometry decides which one you get:

- **CT** — detector untilted (`DetTiltX = 0`), full circular scan.
- **CL** — detector tilted (`DetTiltX` non-zero, e.g. `30`), laminographic geometry.
  Set `DetectorParallel = False` for a genuinely tilted detector, `True` to keep a
  planar (flat) detector while the source/detector pair traverses the tilted orbit.

Switch backend with `Mode = "tigre"` or `Mode = "astra"` in `reconTest.py`.

## Requirements

- Python 3.12
- NVIDIA CUDA Toolkit (12.8 by default — `cuda_path` at the top of `reconTest.py`)
- `numpy`, `matplotlib`
- One or both reconstruction backends:
  - [TIGRE](https://github.com/CERN/TIGRE) — path added via `sys.path` in `reconTest.py`
  - [ASTRA Toolbox](https://astra-toolbox.com/)

## Running

```powershell
python "CT/New/reconTest.py"
```

Before running, check the parameters at the top of `reconTest.py`:

- `SRC` — folder of normalised raw projections
- `DetU` / `DetV` / `DetZ` — detector width, height, and number of projections
- `DetPitch`, `Binning`, `Interval`
- `SOD` / `SDD` — source-to-object and source-to-detector distance (mm)
- `DetTiltX` / `DetTiltY`, `DetOffsetU` / `DetOffsetV`
- `VolX` / `VolY` / `VolZ` and `VolMid*` — output volume size and centre
- `LEFT_PAD` / `RIGHT_PAD` — projection padding, used to suppress truncation artefacts

## Notes

- Raw projections, reconstructed volumes, and other imaging data are excluded by
  `.gitignore` — keep them outside the repository.
- Absolute paths (CUDA, TIGRE, `SRC`) are machine-specific and need adjusting per setup.
