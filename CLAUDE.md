# CUDA 3D Reconstruction from Smartphone Images
 
## Project Goal
CUDA-accelerated photogrammetry pipeline: smartphone photos → watertight 3D mesh.
Target: sub-millimeter resolution for small biological specimens (2-10cm).
 
## Master Plan
Read PLAN.md for the full 15-step (0-14) development plan.
Read STEP_CONFIG.md for which model/effort to use per step.
Every step has a **verification gate** — do not proceed until it passes.
 
## Coordinate Convention (CRITICAL)
[R|t] is world-to-camera: P_cam = R * P_world + t
Inverse: P_world = R^T * (P_cam - t)
Camera center in world: C = -R^T * t
ALL code must follow this. If you see the inverse convention, it is a bug.
 
## Tech Stack
- C++17, CUDA 12.x (cuBLAS, cuSPARSE, cuRAND, CUB)
- OpenCV 4.x with CUDA support + opencv_contrib (cv::cuda::SIFT)
- Eigen3 ≥ 3.4 (CPU-only — use float3 in device code)
- Ceres Solver ≥ 2.1
- PoissonRecon binary (for surface reconstruction)
- Google Test (unit/integration tests)
- CMake ≥ 3.24
 
## Build
```sh
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
```
 
## Test
```sh
cd build && ctest --output-on-failure
```
 
## Verification Pattern
After every step:
1. `make -j$(nproc)` — must compile with zero errors
2. `ctest --output-on-failure` — all existing tests pass
3. Check the step's specific verification gate in PLAN.md
 
## Key Rules
- CUDA_CHECK() after every CUDA API call (defined in cuda_utils.cuh)
- DeviceBuffer<T> RAII wrapper for all GPU allocations (no raw cudaMalloc)
- Eigen in __host__ functions only; math_utils.h float3 ops for device code
- --use_fast_math for non-precision-critical kernels
- Save intermediate outputs when --save-intermediate is set (checkpoint/resume)
- All shared types are in include/types.h — do not redefine structs elsewhere
 
## CUDA Compilation Flags
```
-O3 --use_fast_math --expt-relaxed-constexpr
-Xcudafe "--diag_suppress=esa_on_defaulted_function_ignored"
```
