# Claude Code Configuration per Step — v2

## Model & Effort Strategy for CUDA 3D Reconstruction Pipeline

> **Principle:** Use Sonnet for straightforward implementation, Opus for algorithmic
> complexity and architectural decisions. Scale effort to match cognitive demand.
> This saves 60-80% cost while maintaining quality where it matters.
>
> **v2 changes:** Step 0 added. Step 3 now has two paths (wrapper vs custom).
> Step 9 downgraded from `max` to `high` because Poisson is now a binary wrapper,
> not a from-scratch GPU solver.

---

## Quick Reference

| Step | Description | Model | Effort | Thinking | Est. Time |
|------|-------------|-------|--------|----------|-----------|
| 0 | Environment verification | `sonnet` | `low` | adaptive (off at low) | 3 min |
| 1 | Scaffold + types.h + cuda_utils | `sonnet` | `medium` | adaptive (light) | 10 min |
| 2 | Image loader + synthetic test data | `sonnet` | `medium` | adaptive (light) | 10 min |
| 3a | CUDA SIFT (OpenCV wrapper) | `sonnet` | `medium` | adaptive (light) | 10 min |
| 3b | CUDA SIFT (custom kernels) | `opus` | `high` | adaptive (deep) | 25 min |
| 4 | CUDA feature matching | `opus` | `medium` | adaptive (moderate) | 15 min |
| 5 | SfM + scale calibration | `opus` | `high` | adaptive (deep) | 25 min |
| 6 | Bundle adjustment (Ceres) | `opus` | `high` | adaptive (deep) | 20 min |
| 7 | CUDA PatchMatch MVS | `opus` | `max` | adaptive (deepest) | 40 min |
| 8 | Point cloud fusion | `opus` | `medium` | adaptive (moderate) | 15 min |
| 9 | Meshing (PoissonRecon wrap + MC) | `opus` | `high` | adaptive (deep) | 25 min |
| 10 | Mesh export OBJ/STL/PLY | `sonnet` | `low` | adaptive (off at low) | 5 min |
| 11 | CLI + orchestrator + resume | `sonnet` | `medium` | adaptive (light) | 15 min |
| 12 | Capture guide + preprocessing | `sonnet` | `low` | adaptive (off at low) | 5 min |
| 13 | Testing + integration + benchmark | `opus` | `medium` | adaptive (moderate) | 20 min |
| 14 | Optimization + hardening | `opus` | `high` | adaptive (deep) | 25 min |

---

## How Effort + Thinking Work in Claude Code (4.6 models)

Claude Code 4.6 uses **adaptive thinking** by default. You don't toggle
thinking on/off manually — instead, you control **effort level**, and the
model decides how much to think:

- **`low`** → Skips thinking for simple tasks. Fast, cheap.
- **`medium`** → Moderate thinking. Good balance for standard code.
- **`high`** → Almost always thinks deeply. Best for complex algorithms.
- **`max`** → No constraint on thinking tokens. Opus only. Slowest, most expensive.

Set effort in-session:
```
/effort low
/effort medium
/effort high
/effort max          # Opus only, current session only
```

Switch models in-session:
```
/model sonnet
/model opus
/model opusplan      # Opus for planning, Sonnet for execution
```

---

## Detailed Step Configuration

---

### Step 0 — Environment Verification

```
/model sonnet
/effort low
```

**Why:** Running shell commands to check `nvcc`, `nvidia-smi`, `pkg-config`,
etc. No code generation. Sonnet handles this trivially.

**Gate:** All dependencies present and GPU visible.

---

### Step 1 — Project Scaffolding + Core Types

```
/model sonnet
/effort medium
```

**Why:** Bumped from `low` to `medium` vs v1. This step now creates
substantial real code (not just directory stubs): `types.h` with 10+ structs,
`cuda_utils.cuh` with RAII DeviceBuffer, `math_utils.h` with device-compatible
float3 operations. Sonnet at medium handles struct design and template RAII fine.

**What to watch for:**
- types.h uses the correct world-to-camera convention in CameraPose
- DeviceBuffer disables copy, enables move
- math_utils.h functions are marked `__host__ __device__`
- CMake flags include `--expt-relaxed-constexpr` and the cudafe suppression

**Gate:** `cmake .. && make -j$(nproc)` with zero errors. `./recon3d --help` prints usage.

---

### Step 2 — Image Loading + Synthetic Test Data

```
/model sonnet
/effort medium
```

**Why:** Bumped from `low` to `medium` vs v1. Now includes a synthetic
test data generator (render a textured sphere from 36 known camera poses),
which requires basic 3D math (pinhole projection, z-buffer). Sonnet at
medium handles this.

**What to watch for:**
- EXIF orientation handling (8 cases)
- Focal length → pixel conversion formula: fx = f_mm × width_px / sensor_mm
- Synthetic cameras form a correct ring, all pointing at center
- Ground-truth poses saved as JSON for later validation

**Gate:** Synthetic images generated. Loader reads them. Intrinsics correct. Test passes.

---

### Step 3 — Feature Detection (CUDA SIFT)

**Two paths with different configurations:**

#### Step 3a — OpenCV Wrapper (primary, do this first)

```
/model sonnet
/effort medium
```

**Why:** Wrapping `cv::cuda::SIFT::create()` is 50 lines of code. The hard
part (scale-space, DoG, descriptors) is handled by OpenCV. Medium effort
is plenty for API wiring + testing.

#### Step 3b — Custom CUDA Kernels (stretch goal)

```
/model opus
/effort high
```

**Why:** Writing SIFT kernels from scratch is the first serious algorithmic
challenge: separable Gaussian convolution with shared memory tiling,
3D extrema detection, orientation histograms with gradient weighting,
128-D descriptor with normalization. Opus at high effort reasons through
the memory access patterns and numerical correctness.

**What to watch for (3b only):** Shared memory bank conflicts in histogram
kernels, boundary handling in scale-space, descriptor normalization order
(normalize → clamp 0.2 → re-normalize).

**Gate (both paths):** Features detected on all 36 synthetic images.
Keypoint count in [500, 10000]. Descriptors are 128-D. Test passes.

---

### Step 4 — CUDA Feature Matching

```
/model opus
/effort medium
```

**Why:** KNN matching on GPU needs tiled shared-memory dot products and
warp reductions, but the algorithm is well-studied and mechanical. Medium
effort is enough. The geometric verification (RANSAC fundamental matrix)
uses OpenCV, not custom CUDA.

**What to watch for:** L2 distance precision (squared distances are fine for
comparison, sqrt only the final result), ratio test at 0.75, cross-check
logic, match graph connectivity check.

**Gate:** Match graph fully connected for 36 synthetic views. ≥100 inliers
per adjacent pair. Test passes.

---

### Step 5 — Structure from Motion

```
/model opus
/effort high
```

**Why:** Incremental SfM is architecturally the most complex step on the
CPU side. v2 adds scale calibration (known scale bar) and turntable
constraints (fit circle to camera centers), making it more involved:
- Essential matrix decomposition with chirality check
- PnP with RANSAC (iterative registration loop)
- Triangulation with angle and reprojection validation
- Image ordering strategy
- Scale calibration from triangulated scale bar endpoints
- Turntable: circle fitting + axis estimation

Opus at high effort reasons about degenerate cases (collinear cameras,
planar scenes, insufficient baseline, failed registration).

**What to watch for:** Coordinate convention (world-to-camera consistently),
that PnP returns R,t in the correct frame, scale_factor applied correctly
to all points and translations.

**Gate:** All 36 cameras registered. Mean reproj < 1.0px. Pose error < 2°
rotation, < 5% translation (after Umeyama alignment to ground truth).

---

### Step 6 — Bundle Adjustment

```
/model opus
/effort high
```

**Why:** Ceres Solver integration requires correct Jacobian structure:
- Angle-axis parameterization with `ceres::AngleAxisRotatePoint`
- Schur complement exploit (`SPARSE_SCHUR`)
- Huber loss (delta=1.0)
- Gauge freedom (fix first camera)
- API design: must support being called multiple times during incremental SfM

**What to watch for:** That angle-axis ↔ rotation matrix conversion is
correct, that the residual sign convention matches the projection direction,
and that Ceres actually converges (check `summary.IsSolutionUsable()`).

**Gate:** BA converges. Mean reproj drops below 0.5px. Step 5 test still passes.

---

### Step 7 — CUDA PatchMatch MVS

```
/model opus
/effort max
```

**Why:** **Hardest step in the pipeline.** v2 adds more detail (source view
selection, cuRAND state management, plane-induced homography formula,
checkerboard scheduling), making it even more demanding:
- cuRAND initialization (Philox counter-based, no per-pixel state)
- Plane-induced homography: H = K_s(R_s R_ref^T − ...) K_ref^{-1}
- NCC with multi-view top-K aggregation
- Red-black checkerboard propagation (race condition avoidance)
- Random refinement with shrinking perturbation
- Geometric consistency filtering across all depth maps

Each is a separate CUDA kernel with complex memory access patterns.
`max` effort lets Opus think without constraint.

**MUST break into sub-prompts** (use `/sub-step 7`):
```
> 7a: Source view selection + init_depth_normal_kernel (cuRAND)
> 7b: compute_ncc_cost_kernel (homography + NCC + multi-view aggregation)
> 7c: propagate_and_refine_kernel (checkerboard + random perturbation)
> 7d: geometric_consistency_filter_kernel
> 7e: Main loop + memory management + test on synthetic data
```

**What to watch for:** NCC numerical stability (zero-variance patch → NaN),
homography singularity (depth → 0 or normal perpendicular to view),
depth range from SfM (percentiles, not min/max), cuRAND Philox vs state-based.

**Gate:** Depth maps for all 36 views. Geometric filter retains ≥60%.
Depth error < 5% on retained pixels vs ground truth. Time < 5 min at 1280×960.

---

### Step 8 — Dense Point Cloud Generation

```
/model opus
/effort medium
```

**Why:** Depth-to-3D backprojection is straightforward linear algebra.
Voxel downsampling with hash tables is more involved but pattern-matchable.
Statistical outlier removal via grid-based neighbor counting is simpler than
KNN. Medium effort handles this.

If implementing **voxel hashing for TSDF** (Option B), bump to `high` —
open-addressing hash tables on GPU with dynamic block allocation are tricky.

**What to watch for:**
- Backprojection uses R^T (transpose) not R^{-1} (for orthogonal R, same thing, but be explicit)
- Color sampled from the BGR original image, not grayscale
- Normals transformed to world frame: N_world = R^T * N_cam
- Voxel resolution 0.1mm for small specimens

**Gate:** Dense cloud forms a sphere. Normals point outward. Outlier rate < 10%.

---

### Step 9 — Surface Reconstruction (Meshing)

```
/model opus
/effort high
```

**Why:** Downgraded from `max` (v1) to `high` (v2). The Poisson solver is
now a **binary wrapper** (call PoissonRecon), not a from-scratch GPU
implementation. The complexity comes from:
- Wrapping PoissonRecon with error handling and fallback
- Density-based face trimming
- Color transfer from point cloud to mesh vertices
- CUDA Marching Cubes (for TSDF path): lookup tables, prefix sum, interpolation
- Post-processing: Taubin smoothing, QEM decimation, connected components, hole filling

This is still substantial, but no longer requires the unconstrained thinking
of implementing a Poisson solver from scratch.

**Consider breaking into sub-prompts:**
```
> 9a: PoissonRecon wrapper + PLY export/import + density trimming
> 9b: CUDA Marching Cubes (edgeTable, triTable, prefix sum, kernel)
> 9c: Post-processing (Taubin smooth, decimate, connected components, hole fill)
> 9d: Color transfer + watertight check + test on synthetic sphere
```

**What to watch for:** PoissonRecon binary path detection, fallback if
binary missing, marching cubes ambiguity cases (ensure __constant__ tables
are correct), Taubin smoothing lambda/mu balance to prevent shrinkage,
consistent face winding (counterclockwise = outward normal).

**Gate:** Watertight mesh from synthetic sphere. Hausdorff < 1% of radius. Test passes.

---

### Step 10 — Mesh Export (OBJ/STL/PLY)

```
/model sonnet
/effort low
```

**Why:** File format writers are mechanical. OBJ is text, STL is binary
with a fixed 50-byte-per-triangle layout, PLY has a simple header. No
algorithmic complexity.

**What to watch for:** OBJ faces are 1-indexed (add 1 to all indices!),
STL binary is little-endian, PLY color values are uchar [0-255] not float.
Buffered I/O for large meshes.

**Gate:** All three formats export and re-import correctly. STL file size matches expected.

---

### Step 11 — CLI + Pipeline Orchestrator + Resume

```
/model sonnet
/effort medium
```

**Why:** Argument parsing, quality presets, progress reporting, error handling,
and the new checkpoint/resume system. Medium effort because the orchestrator
needs to handle failure modes at every stage and the resume logic (check
for cached outputs before running each stage), but the code is control flow.

**What to watch for:**
- Memory management between stages (free GPU buffers after feature detection before MVS)
- Resume logic: if `sfm_cameras.json` exists, skip SfM
- Error messages are diagnostic ("No pairs matched — check image overlap" not just "Error")
- Quality presets set consistent values across all parameters

**Gate:** Full pipeline runs end-to-end on synthetic data. `--resume` correctly
skips cached stages. All error paths print useful messages.

---

### Step 12 — Capture Guide + Preprocessing Script

```
/model sonnet
/effort low
```

**Why:** Documentation writing and a Python preprocessing script. No CUDA,
no complex algorithms. v2 adds more detail (three elevation rings, blur
detection, quality report) but it's still straightforward.

**What to watch for:** Photography recommendations are physically accurate
(overlap percentages, angle suggestions), Python script handles EXIF
orientation correctly (8 cases), blur detection threshold is reasonable.

**Gate:** Script runs. Quality report prints. Preprocessed images load in pipeline.

---

### Step 13 — Testing + Integration + Benchmarks

```
/model opus
/effort medium
```

**Why:** Writing good tests requires end-to-end pipeline understanding.
v2 adds: integration test (full pipeline on synthetic → compare to ground
truth via Chamfer/Hausdorff), export round-trip tests, watertight check,
and a separate benchmark target. Opus at medium reasons about what
tolerances are correct without overthinking the test harness code itself.

**What to watch for:**
- Synthetic camera setup has enough baseline for reconstruction
- Chamfer/Hausdorff computed correctly (bidirectional nearest neighbor)
- Surface area/volume compared to analytic sphere values
- Benchmark targets are realistic for the synthetic dataset size

**Gate:** `ctest --output-on-failure` passes all tests. Integration test produces
correct sphere. Benchmarks are within targets.

---

### Step 14 — Optimization + Hardening

```
/model opus
/effort high
```

**Why:** Performance optimization requires deep reasoning about CUDA
occupancy, register pressure, memory coalescing, shared memory bank
conflicts, stream overlap, and tiled processing for memory-limited GPUs.
v2 adds: compute-sanitizer validation, signal handling (SIGINT cleanup),
edge case testing (10 images, no EXIF, mixed resolutions), and OOM
graceful fallback.

**What to watch for:** That optimizations don't break correctness (run
ctest after each change), compute-sanitizer reports zero errors, all edge
case tests pass.

**Gate:** All checklist items addressed. `ctest` passes. `compute-sanitizer`
clean. Pipeline runs at all 4 quality levels. Benchmarks meet targets.

---

## Session Plan with Model Switching

### Session 1 (Steps 0-2): Environment + Scaffolding + Test Data
```
claude --model sonnet
> /effort low
> /step 0          # verify dependencies
> /effort medium
> /step 1          # scaffold + types.h + cuda_utils
> /step 2          # image loader + synthetic data
> /gate 1 && /gate 2
```

### Session 2 (Steps 3-4): Feature Pipeline
```
claude --model sonnet
> /effort medium
> /step 3          # OpenCV SIFT wrapper (path 3a)
> /gate 3
/model opus
> /effort medium
> /step 4          # CUDA matching
> /gate 4

# Optional: if you want custom SIFT kernels (path 3b):
> /effort high
> Implement Step 3b from PLAN.md (custom CUDA SIFT kernels)
```

### Session 3 (Steps 5-6): SfM + Bundle Adjustment
```
claude --model opus
> /effort high
> /step 5          # incremental SfM + scale calibration
> /gate 5
> /step 6          # bundle adjustment
> /gate 6
```

### Session 4 (Step 7): Dense MVS — the big one
```
claude --model opus
> /effort max
> /sub-step 7      # get the sub-task breakdown
> Implement 7a     # source view selection + init kernel
> Implement 7b     # NCC cost kernel
> Implement 7c     # propagation + refinement kernel
> Implement 7d     # geometric consistency filter
> Implement 7e     # wire together + test
> /gate 7

# If context runs low mid-step:
> Write progress to PROGRESS.md
> /clear
> Read PLAN.md and PROGRESS.md, continue Step 7
```

### Session 5 (Steps 8-9): Point Cloud + Meshing
```
claude --model opus
> /effort medium
> /step 8          # point cloud fusion
> /gate 8
> /effort high
> /sub-step 9      # get sub-task breakdown
> Implement 9a     # PoissonRecon wrapper
> Implement 9b     # Marching Cubes
> Implement 9c     # post-processing
> Implement 9d     # color transfer + test
> /gate 9
```

### Session 6 (Steps 10-11): Export + CLI
```
claude --model sonnet
> /effort low
> /step 10         # OBJ/STL/PLY exporters
> /gate 10
> /effort medium
> /step 11         # CLI + orchestrator + resume
> Run: ./recon3d -i data/synthetic/ -o output/sphere.obj --quality medium --verbose
```

### Session 7 (Steps 12-14): Polish
```
claude --model sonnet
> /effort low
> /step 12         # capture guide + preprocessing script
> /gate 12

/model opus
> /effort medium
> /step 13         # full test suite + integration + benchmarks
> /gate 13
> /effort high
> /step 14         # optimization + hardening
> /gate 14

# Final validation:
> /test-all
> Run compute-sanitizer on the pipeline
> Run all 4 quality presets on synthetic data
```

---

## Cost Estimate (Approximate)

Assuming API usage (not subscription), with v2 scope:

| Model | Effort | Steps | Est. Tokens | Est. Cost |
|-------|--------|-------|-------------|-----------|
| Sonnet | low | 0, 10, 12 | ~40K out | ~$0.60 |
| Sonnet | medium | 1, 2, 3a, 11 | ~120K out | ~$1.80 |
| Opus | medium | 4, 8, 13 | ~130K out | ~$3.25 |
| Opus | high | 3b*, 5, 6, 9, 14 | ~250K out | ~$6.25 |
| Opus | max | 7 | ~200K out | ~$5.00 |
| **Total (with 3a)** | | **15 steps** | **~740K** | **~$17** |
| **Total (with 3a+3b)** | | **16 steps** | **~850K** | **~$21** |

\* Step 3b is optional (custom SIFT kernels). Skip it to save ~$2.50 and 25 min.

With a **Max subscription**, all of this is included in your plan.
Using Opus for everything would cost ~$50+ on API — this config saves ~65%.

---

## Environment Variables (Optional)

```bash
# In your shell profile (~/.bashrc or ~/.zshrc):

# Default to Sonnet for most work
export ANTHROPIC_DEFAULT_MODEL=sonnet

# Or start every session on Opus:
# export ANTHROPIC_DEFAULT_MODEL=opus

# Disable auto-compaction if you prefer manual /clear + /resume
# export CLAUDE_CODE_DISABLE_AUTO_COMPACTION=1

# Disable 1M context if you want faster responses on shorter sessions
# export CLAUDE_CODE_DISABLE_1M_CONTEXT=1
```
