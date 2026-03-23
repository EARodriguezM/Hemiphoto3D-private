# CUDA-Accelerated 3D Reconstruction from Smartphone Images

## Claude Code Step-by-Step Development Guide — v2

> **Goal:** Build a CUDA program that takes a set of 2D smartphone photos (taken
> from multiple angles around a turntable or via camera movement) and generates a
> watertight 3D mesh (.obj and/or .stl) of a complex organic object (hemipenis
> specimen for herpetological morphology research).

---

## Pipeline Architecture

```
                            ┌──────────────────────────────────────────────────────┐
                            │              CUDA 3D Reconstruction Pipeline         │
                            ├──────────────────────────────────────────────────────┤
 Input Images ─┐            │                                                      │
               ├─► [0] Deps │  [1] Scaffold ► [2] Image Load ► [3] Features(CUDA) │
 EXIF metadata ┘   Verify   │       │                                │             │
                            │       ▼                                ▼             │
                            │  types.h ◄──── shared structs ──► [4] Matching(CUDA) │
                            │  cuda_utils                            │             │
                            │       │                                ▼             │
                            │       │         [6] Bundle ◄──── [5] SfM             │
                            │       │         Adjustment        (incremental)      │
                            │       │              │                               │
                            │       │              ▼                               │
                            │       │  [7] Dense MVS (CUDA) ── PatchMatch          │
                            │       │              │                               │
                            │       │              ▼                               │
                            │       │  [8] Point Cloud Fusion (CUDA)               │
                            │       │              │                               │
                            │       │              ▼                               │
                            │       │  [9] Meshing ── Poisson + Marching Cubes     │
                            │       │              │                               │
                            │       │              ▼                               │
                            │       └──► [10] Export (.obj/.stl/.ply)              │
                            │                      │                               │
                            │              [11] CLI Orchestrator                    │
                            │              [12] Capture Guide                      │
                            │              [13] Tests                              │
                            │              [14] Optimize                           │
                            └──────────────────────────────────────────────────────┘
```

## Key Design Decisions (read before coding)

1. **Scale ambiguity:** SfM produces geometry up to an unknown scale factor.
   For scientific measurement, the pipeline must support scale calibration via
   a known-length reference (scale bar in the scene). This is handled in Step 5.

2. **Poisson reconstruction:** Implementing a full Screened Poisson solver on GPU
   is a research-grade effort. The **primary** approach is wrapping the
   PoissonRecon binary; a custom CUDA solver is a stretch goal.

3. **SIFT strategy:** Use OpenCV `cv::cuda::SIFT` as the primary detector.
   Custom CUDA SIFT kernels are a secondary path for maximum control, but
   should not block progress on the rest of the pipeline.

4. **Coordinate convention:** Throughout the pipeline, camera extrinsics `[R|t]`
   represent the **world-to-camera** transform: `P_cam = R * P_world + t`.
   To go back: `P_world = R^T * (P_cam - t)`. All steps must use this consistently.

5. **Checkpoint/resume:** Every stage writes its output to disk when
   `--save-intermediate` is set. If a later stage fails, earlier stages
   can be skipped on re-run by loading cached results.

6. **Eigen in CUDA files:** Eigen is CPU-only. In `.cu` files, use Eigen only in
   `__host__` functions, or use `float3`/`float4` and manual math in device code.
   Compile with `-Xcudafe "--diag_suppress=esa_on_defaulted_function_ignored"`
   if Eigen headers leak into device compilation paths.

---

## Prerequisites & Dependencies

- NVIDIA GPU with Compute Capability ≥ 6.0 (Pascal+), ≥ 6 GB VRAM
- CUDA Toolkit 12.x (includes cuBLAS, cuSPARSE, cuRAND, CUB)
- OpenCV 4.x compiled with `WITH_CUDA=ON` and `opencv_contrib` (for cv::cuda::SIFT)
- Eigen3 ≥ 3.4
- Ceres Solver ≥ 2.1 (for bundle adjustment)
- Google Test (for unit tests)
- CMake ≥ 3.24
- PoissonRecon binary (for surface reconstruction; install via apt or build from source)
- Python 3.10+ with OpenCV, Pillow, numpy (for preprocessing)

---

## STEP 0 — Environment Verification

**Prompt for Claude Code:**

```
Before writing any code, verify the development environment:

1. Check CUDA toolkit:
   nvcc --version          # must be 12.x
   nvidia-smi              # verify GPU is visible and driver is compatible

2. Check GPU compute capability:
   nvidia-smi --query-gpu=compute_cap,name,memory.total --format=csv
   # Record the compute capability (e.g., 8.6) and VRAM

3. Check OpenCV with CUDA:
   pkg-config --modversion opencv4   # must be 4.x
   python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -A5 "CUDA"
   # Must show "CUDA: YES" and list cuda modules including cudafeatures2d

4. Check Eigen3:
   pkg-config --modversion eigen3    # must be 3.4+

5. Check Ceres Solver:
   pkg-config --modversion ceres     # or find in /usr/local/lib/cmake/Ceres

6. Check PoissonRecon:
   which PoissonRecon || which poissonrecon
   # If not found, install:
   # sudo apt install poissonrecon
   # Or build from https://github.com/mkazhdan/PoissonRecon

7. Check cuRAND (needed for PatchMatch MVS random init):
   ls /usr/local/cuda/lib64/libcurand*   # should exist

8. Check Google Test:
   pkg-config --modversion gtest || dpkg -l libgtest-dev

Report which dependencies are present and which are missing.
Do NOT proceed until all critical dependencies are verified.
If OpenCV lacks CUDA support, provide the build commands to recompile it.
```

**Verification gate:** All dependencies present and GPU visible via nvidia-smi.

---

## STEP 1 — Project Scaffolding and Build System

**Prompt for Claude Code:**

```
Create the project structure. Every file listed below must exist (stubs are fine
for files implemented in later steps, but they must compile):

cuda_3d_recon/
├── CMakeLists.txt
├── cmake/
│   └── FindDependencies.cmake
├── src/
│   ├── main.cpp                    # CLI entry point (stub: parse args, print help)
│   ├── pipeline.h                  # Pipeline class declaration
│   ├── pipeline.cu                 # Pipeline orchestrator (stub)
│   ├── feature_detection/
│   │   ├── sift_cuda.h
│   │   ├── sift_cuda.cu            # stub
│   │   ├── feature_matching.h
│   │   └── feature_matching.cu     # stub
│   ├── sfm/
│   │   ├── camera_pose.h
│   │   ├── camera_pose.cpp         # stub
│   │   ├── bundle_adjust.h
│   │   └── bundle_adjust.cpp       # stub
│   ├── dense_recon/
│   │   ├── patchmatch_mvs.h        # renamed from stereo_cuda for clarity
│   │   ├── patchmatch_mvs.cu       # stub
│   │   ├── point_cloud.h
│   │   └── point_cloud.cu          # stub
│   ├── meshing/
│   │   ├── poisson_recon.h
│   │   ├── poisson_recon.cpp       # NOTE: .cpp not .cu — wraps PoissonRecon binary
│   │   ├── marching_cubes.h
│   │   └── marching_cubes.cu       # stub
│   ├── export/
│   │   ├── mesh_exporter.h
│   │   ├── obj_exporter.cpp        # stub
│   │   ├── stl_exporter.cpp        # stub
│   │   └── ply_exporter.cpp        # stub
│   └── utils/
│       ├── image_loader.h
│       ├── image_loader.cpp        # stub
│       ├── cuda_utils.cuh          # IMPLEMENT NOW (see below)
│       └── math_utils.h            # IMPLEMENT NOW (see below)
├── include/
│   └── types.h                     # IMPLEMENT NOW (see below)
├── tests/
│   ├── CMakeLists.txt              # Google Test integration
│   ├── test_sift.cpp               # stub
│   ├── test_matching.cpp           # stub
│   ├── test_sfm.cpp                # stub
│   └── test_meshing.cpp            # stub
├── scripts/
│   ├── capture_guide.md            # placeholder
│   └── preprocess_images.py        # placeholder
└── data/
    └── sample/                     # empty, for test images

Implement these files NOW (not as stubs):

### include/types.h — All shared data structures for the entire pipeline:

struct ImageData {
    cv::Mat image;              // BGR uint8
    cv::Mat gray;               // Grayscale float32 [0,1]
    cv::Mat K;                  // 3x3 double intrinsic matrix
    cv::Mat dist_coeffs;        // distortion coefficients (k1,k2,p1,p2,k3)
    double focal_length_px;     // focal length in pixels
    std::string filename;
    int width, height;
    int id;                     // sequential index
};

struct Keypoint {
    float x, y;                 // pixel coordinates
    float scale;                // detection scale
    float orientation;          // radians
    int octave;
};

struct SIFTFeatures {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;        // Nx128 float32 (CPU)
    float* d_descriptors;       // GPU pointer (may be nullptr if transferred back)
    int count;
    int image_id;
};

struct FeatureMatch {
    int idx_i, idx_j;           // keypoint indices in respective images
    float distance;             // L2 descriptor distance
};

struct ImagePairMatches {
    int image_i, image_j;
    std::vector<FeatureMatch> matches;
    cv::Mat F;                  // 3x3 fundamental matrix
    cv::Mat E;                  // 3x3 essential matrix
    int num_inliers;
    double median_triangulation_angle;  // degrees
};

struct CameraPose {
    int image_id;
    Eigen::Matrix3d R;          // world-to-camera rotation
    Eigen::Vector3d t;          // world-to-camera translation
    Eigen::Matrix3d K;          // intrinsics
    bool is_registered = false;

    // P_cam = R * P_world + t
    Eigen::Matrix<double,3,4> projection() const {
        Eigen::Matrix<double,3,4> Rt;
        Rt.block<3,3>(0,0) = R;
        Rt.col(3) = t;
        return K * Rt;
    }

    // P_world = R^T * (P_cam - t)
    Eigen::Vector3d cam_center() const {
        return -R.transpose() * t;
    }
};

struct SparsePoint {
    Eigen::Vector3d position;
    Eigen::Vector3f color;
    std::vector<int> image_ids;
    std::vector<int> keypoint_ids;
    float mean_reprojection_error;
    int track_length;           // number of images that see this point
};

struct DepthMap {
    cv::Mat depth;              // float32, world units (0 = invalid)
    cv::Mat normal;             // float32x3, world-frame surface normal
    cv::Mat confidence;         // float32, [0, 1]
    int image_id;
    int width, height;
};

struct DensePointCloud {
    std::vector<Eigen::Vector3f> positions;
    std::vector<Eigen::Vector3f> normals;
    std::vector<Eigen::Vector3f> colors;
    int size() const { return positions.size(); }

    // GPU mirrors (allocated on demand)
    float3* d_positions = nullptr;
    float3* d_normals   = nullptr;
    float3* d_colors    = nullptr;
    bool on_gpu = false;

    void freeGPU();             // release device memory
};

struct Mesh {
    std::vector<Eigen::Vector3f> vertices;
    std::vector<Eigen::Vector3f> normals;       // per-vertex
    std::vector<Eigen::Vector3f> colors;        // per-vertex
    std::vector<Eigen::Vector2f> texcoords;     // optional UV
    std::vector<Eigen::Vector3i> faces;         // triangle indices (0-indexed)

    int num_vertices() const { return vertices.size(); }
    int num_faces()    const { return faces.size(); }
    bool is_watertight() const;                 // check if all edges shared by exactly 2 faces
};

struct PipelineConfig {
    // Input / Output
    std::string input_dir;
    std::string output_path;            // .obj, .stl, or .ply
    std::string pointcloud_path;        // optional .ply export
    std::string log_path;
    bool save_intermediate = false;
    bool verbose = false;

    // Quality preset: "low", "medium", "high", "ultra"
    std::string quality = "high";

    // Image
    int max_image_size = 3200;

    // Feature detection
    float match_ratio = 0.75f;          // Lowe's ratio
    int min_matches = 30;               // per pair

    // SfM
    double min_triangulation_angle = 2.0; // degrees
    double max_reprojection_error = 4.0;  // pixels
    bool turntable = false;

    // Camera overrides
    double focal_length_mm = 0.0;       // 0 = use EXIF
    double sensor_width_mm = 0.0;       // 0 = estimate

    // Dense reconstruction
    float mvs_resolution = 1.0f;        // downscale factor for MVS
    int mvs_iterations = 8;
    int mvs_patch_size = 11;

    // Meshing
    int poisson_depth = 10;
    float poisson_screening = 4.0f;
    int smooth_iterations = 3;
    float smooth_lambda = 0.5f;
    int decimate_target = 0;            // 0 = no decimation

    // Scale calibration
    double scale_bar_length_mm = 0.0;   // 0 = uncalibrated

    // GPU
    int gpu_id = 0;
    size_t gpu_memory_limit = 0;        // 0 = auto

    // Derived from quality preset (call applyPreset() after parsing)
    void applyPreset();
};

### src/utils/cuda_utils.cuh — CUDA error checking and memory RAII:

#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Fatal error check — use after every CUDA call
#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Check last kernel launch
#define CUDA_CHECK_KERNEL()                                                    \
    do {                                                                       \
        CUDA_CHECK(cudaGetLastError());                                        \
        CUDA_CHECK(cudaDeviceSynchronize());                                   \
    } while (0)

// RAII wrapper for device memory
template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    size_t count = 0;

    void alloc(size_t n) {
        free();
        count = n;
        CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
    }

    void free() {
        if (ptr) { cudaFree(ptr); ptr = nullptr; count = 0; }
    }

    void upload(const T* host_data, size_t n) {
        if (n > count) alloc(n);
        CUDA_CHECK(cudaMemcpy(ptr, host_data, n * sizeof(T), cudaMemcpyHostToDevice));
    }

    void download(T* host_data, size_t n) const {
        CUDA_CHECK(cudaMemcpy(host_data, ptr, n * sizeof(T), cudaMemcpyDeviceToHost));
    }

    ~DeviceBuffer() { free(); }

    // No copy, allow move
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr(o.ptr), count(o.count) { o.ptr = nullptr; o.count = 0; }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept { free(); ptr = o.ptr; count = o.count; o.ptr = nullptr; o.count = 0; return *this; }
};

// Query and print GPU info
void printGPUInfo(int device_id = 0);

// Get available GPU memory in bytes
size_t getAvailableGPUMemory(int device_id = 0);

### src/utils/math_utils.h — Lightweight math for device code:

Implement small inline functions (host+device where possible):
- float3 cross(float3 a, float3 b)
- float  dot(float3 a, float3 b)
- float3 normalize(float3 v)
- float3 operator+, -, * for float3
- Matrix3x3 struct (float[9]) with multiply, transpose, inverse
- Angle-axis to rotation matrix and back
- Bilinear interpolation helper

These must be usable in CUDA device code (__host__ __device__).
Do NOT use Eigen in this file.

### CMakeLists.txt requirements:

- project(cuda_3d_recon LANGUAGES CXX CUDA)
- CMAKE_CUDA_ARCHITECTURES: query the actual GPU and set accordingly,
  plus common targets: "60;70;75;80;86;89;90"
- C++17 minimum, CUDA standard 17
- Find: OpenCV (with cuda modules), Eigen3, Ceres, GTest
- Compile flags for .cu: -O3 --use_fast_math
  --expt-relaxed-constexpr
  -Xcudafe "--diag_suppress=esa_on_defaulted_function_ignored"
  (the last two for Eigen header compatibility)
- Two targets:
  - recon3d (main executable)
  - recon3d_tests (test executable, linked with GTest)
- Enable CTest

After creating all files, run:
  mkdir build && cd build && cmake .. && make -j$(nproc)
  The project MUST compile with zero errors before proceeding.
```

**Verification gate:** `cmake .. && make -j$(nproc)` succeeds. `./recon3d --help` prints usage. `ctest` runs (0 tests is ok at this stage).

---

## STEP 2 — Image Loading, EXIF Extraction, and Test Data

**Prompt for Claude Code:**

```
Implement src/utils/image_loader.cpp and image_loader.h.

Also create a synthetic test image set so we can test the pipeline
incrementally without needing real photographs.

### Image Loader:

std::vector<ImageData> loadImages(const std::string& dir, int max_size);

1. Scan directory for .jpg, .jpeg, .png, .tiff files (case-insensitive)
2. Sort by filename for deterministic ordering
3. For each image:
   a. Load with cv::imread (IMREAD_COLOR)
   b. Read EXIF orientation tag, auto-rotate if needed
      (use cv::ROTATE_90_CLOCKWISE etc. based on EXIF orientation 1-8)
   c. Resize if largest dimension > max_size (preserve aspect ratio)
   d. Convert to grayscale float32 [0,1] for feature detection
   e. Extract EXIF focal length (mm) using a lightweight parser
      - Try TinyEXIF or implement minimal EXIF reader for JPEG APP1 segment
      - Common smartphone focal lengths: 2.5-6mm physical, 24-28mm equivalent
   f. Estimate sensor width from EXIF camera make/model
      - Build a lookup table of common smartphones:
        iPhone 15/14/13: 6.17mm, Pixel 8/7: 6.17mm, Samsung Galaxy S24/23: 6.40mm
      - Fallback: assume 6.0mm sensor width (reasonable for most phones)
   g. Compute intrinsics:
        fx = fy = focal_mm * image_width_px / sensor_width_mm
        cx = image_width_px / 2.0
        cy = image_height_px / 2.0
   h. If focal_length_mm and sensor_width_mm are overridden in config, use those
   i. Set dist_coeffs to zeros (assume phone images are pre-corrected by ISP)
4. Validate: require minimum 10 images, warn if < 30
5. Log: image count, resolution, estimated focal length

### Synthetic Test Data Generator:

Create scripts/generate_synthetic.py (or a C++ utility):

1. Define a 3D model: use a textured sphere or torus (parametric equations)
   - Sphere: radius 5cm, with checkerboard + noise texture
   - This gives known ground-truth geometry for validation
2. Generate 36 synthetic views around the object:
   - Cameras on a circle at radius 25cm, every 10 degrees
   - All cameras point toward center
   - Known intrinsics: fx=fy=2000, cx=640, cy=480, image 1280x960
3. Render using OpenCV (simple pinhole projection + z-buffer)
   or Python with numpy raycasting
4. Save as JPEG images in data/synthetic/
5. Save ground-truth camera poses as data/synthetic/cameras_gt.json
6. Save ground-truth mesh as data/synthetic/ground_truth.obj

This synthetic set will be used to test Steps 3-13 without real photos.

### Unit test — test_image_loader.cpp:

- Test that synthetic images load correctly
- Test that intrinsics K matrix has expected values
- Test auto-rotation on a test image with EXIF orientation=6
- Test that grayscale conversion produces float32 in [0,1]
```

**Verification gate:** `./recon3d -i data/synthetic/ -o /dev/null --verbose` loads all 36 images and prints intrinsics. `ctest -R test_image` passes.

---

## STEP 3 — Feature Detection (CUDA SIFT)

**Prompt for Claude Code:**

```
Implement src/feature_detection/sift_cuda.cu and sift_cuda.h.

### Primary approach: Wrap OpenCV CUDA SIFT

This gets us a working pipeline fastest:

SIFTFeatures detectFeatures(const ImageData& image, int max_features = 8000);

1. Create cv::cuda::SIFT_CUDA (or cv::cuda::SIFT::create() depending on
   OpenCV version) with:
   - nfeatures = max_features
   - nOctaveLayers = 3
   - contrastThreshold = 0.04
   - edgeThreshold = 10.0
   - sigma = 1.6
2. Upload grayscale image to GPU: cv::cuda::GpuMat
3. Detect and compute: sift->detectAndCompute(gpu_gray, noArray(), keypoints, descriptors)
4. Download descriptors to CPU (keep GPU copy if memory allows)
5. Populate SIFTFeatures struct

### Secondary approach: Custom CUDA SIFT kernels (stretch goal)

If OpenCV's cv::cuda::SIFT is not available in the installed OpenCV version,
OR if we need more control, implement from scratch:

Kernels needed (in execution order):
a. gaussian_blur_separable_kernel — 1D horizontal + vertical passes
   - Use shared memory tiling (tile_width + 2*radius)
   - Build scale space: 4 octaves × 5 scales each
   - Sigma progression: sigma_0 * 2^(s/S) where S=3 scales per octave

b. dog_subtract_kernel — element-wise subtraction of adjacent scales
   - Simple: one thread per pixel, no shared memory needed

c. detect_extrema_kernel — 3×3×3 neighborhood max/min in DoG volume
   - Contrast threshold: |DoG(x,y,s)| > 0.04 * 255
   - Edge ratio: (trace²/det) < (r+1)²/r with r=10
   - Sub-pixel via Taylor expansion (3x3 Hessian at extremum)
   - Output: compact list using atomicAdd for count

d. compute_orientation_kernel — one thread-block per keypoint
   - 36-bin histogram over 1.5σ radius patch
   - Gaussian weighting with σ = 1.5 * keypoint_scale
   - Find peak(s) > 80% of max → assign orientation(s)

e. compute_descriptor_kernel — one thread-block per keypoint
   - Rotate 16×16 patch to dominant orientation
   - 4×4 spatial bins × 8 orientation bins = 128 floats
   - Gaussian window weighting (σ = 8.0)
   - Normalize → clamp to 0.2 → re-normalize

Memory layout:
- Scale space images: use cudaArray + texture objects for 2D interpolation
- Keypoint list: SoA (Structure of Arrays) for coalesced access
  - float* d_x, d_y, d_scale, d_orientation (separate arrays)
- Descriptors: Nx128 row-major float array

### Implement BOTH paths and select at compile time or runtime:

#ifdef USE_OPENCV_CUDA_SIFT
    // OpenCV path
#else
    // Custom CUDA path
#endif

### Unit test — test_sift.cpp:

- Detect features on one synthetic image
  - Verify count is in range [500, 10000]
  - Verify all keypoints are within image bounds
  - Verify descriptors are 128-D, L2-normalized (norm ≈ 512 for SIFT)
- Detect features on a rotated copy of the same image
  - Verify ≥ 50% of keypoints are repeatable (found in both)
- If both paths compiled, compare OpenCV vs custom:
  - Keypoint count within 20% of each other
```

**Verification gate:** Feature detection runs on all 36 synthetic images. Unit test passes. Report average keypoints per image and time per image.

---

## STEP 4 — CUDA Feature Matching

**Prompt for Claude Code:**

```
Implement src/feature_detection/feature_matching.cu and feature_matching.h.

### API:

std::vector<ImagePairMatches> matchAllPairs(
    const std::vector<SIFTFeatures>& features,
    float ratio_threshold = 0.75f,
    int min_inliers = 30
);

### 1. GPU Brute-Force KNN (K=2) Matcher:

CUDA kernel: bf_knn_match_kernel<<<grid, block>>>

Algorithm (tiled for shared memory):
- Grid: one block per tile of query descriptors
- Block: 128 threads
- For each tile of query descriptors (loaded into shared memory):
  - For each tile of reference descriptors (loaded into shared memory):
    - Each thread computes L2 distances between its query and all refs in tile
    - Maintain running top-2 (nearest + second nearest) per query
- After all tiles: each thread writes its best 2 matches

Shared memory layout:
  __shared__ float s_query[TILE_SIZE][128];    // query tile
  __shared__ float s_ref[TILE_SIZE][128];      // reference tile

Distance: L2 = sqrt(sum((q_i - r_i)^2)) — can skip sqrt and compare squared
distances, then sqrt only the final results.

### 2. Lowe's Ratio Test (GPU-side):

CUDA kernel: ratio_test_kernel<<<grid, block>>>
- Input: Kx2 match distances per query
- Keep if: dist_1st / dist_2nd < ratio_threshold
- Output: filtered match list (use stream compaction with prefix sum)

### 3. Cross-Check (GPU-side):

- Run matching in both directions: A→B and B→A
- CUDA kernel: cross_check_kernel<<<grid, block>>>
- Keep only mutual nearest neighbors

### 4. Geometric Verification (CPU):

For each surviving image pair:
- cv::findFundamentalMat(pts_i, pts_j, cv::FM_RANSAC, 3.0, 0.99, inlier_mask)
- Count inliers; reject pair if < min_inliers
- Compute essential matrix: E = K_j^T * F * K_i
  (If shared intrinsics: E = K^T * F * K)
- Store F, E, inlier count, and median triangulation angle

### 5. Pair Selection Strategy:

For N images, N*(N-1)/2 pairs is expensive for large N:
- N ≤ 50: evaluate all pairs (exhaustive)
- N > 50: use sequential matching (each image vs its K=10 nearest neighbors
  by image index, assuming roughly sequential capture order)
- Process pairs in GPU batches of 50-100 to manage memory

### 6. Match Graph:

Build adjacency structure: each node = image, each edge = verified pair.
Reject isolated images (not connected to any other image).
Warn if graph is disconnected (reconstruction will be partial).

### Unit test — test_matching.cpp:

- Match two adjacent synthetic views (10° apart)
  - Verify ≥ 100 inlier matches
  - Verify fundamental matrix satisfies x'^T F x ≈ 0 for inliers
- Match two distant views (180° apart)
  - Verify fewer matches (more occlusion)
- Match identical image to itself
  - Verify all keypoints match with distance ≈ 0
- Verify ratio test reduces match count by ≥ 30%
```

**Verification gate:** All 36 synthetic images matched. Match graph is fully connected. Report total pairs, average inliers per pair, matching time.

---

## STEP 5 — Structure from Motion (Camera Pose Estimation)

**Prompt for Claude Code:**

```
Implement src/sfm/camera_pose.cpp and camera_pose.h.

### API:

struct SfMResult {
    std::vector<CameraPose> cameras;
    std::vector<SparsePoint> points;
    int num_registered;
    double mean_reprojection_error;
    double scale_factor;        // world-units per reconstruction-unit (1.0 if uncalibrated)
};

SfMResult runIncrementalSfM(
    const std::vector<SIFTFeatures>& features,
    const std::vector<ImagePairMatches>& matches,
    const std::vector<ImageData>& images,
    const PipelineConfig& config
);

### Phase 1: Initialization — Best Initial Pair

Score each verified pair by: num_inliers × sin(median_triangulation_angle)
(sin because baseline quality matters more than raw angle for small angles)

For the top-scoring pair:
1. Decompose E into [R|t] using cv::recoverPose
   - recoverPose returns 4 solutions; it picks the one with most points
     in front of both cameras
2. Set camera 0 as identity: R=I, t=0 (world frame origin)
3. Set camera 1 as: R_1, t_1 from essential matrix decomposition
4. Triangulate initial 3D points:
   - cv::triangulatePoints(P_0, P_1, pts_0, pts_1)
   - Output is 4×N homogeneous; divide by w to get 3D
   - Filter: reject points behind either camera (negative depth)
   - Filter: reject points with reprojection error > max_reprojection_error
   - Filter: reject points with triangulation angle < min_triangulation_angle

### Phase 2: Incremental Registration Loop

Repeat until no more images can be registered:
1. Score unregistered images by: count of 2D keypoints that correspond
   to already-triangulated 3D points
2. Pick image with highest count (need ≥ 6 correspondences, prefer ≥ 20)
3. Solve PnP:
   - cv::solvePnPRansac(object_pts_3d, image_pts_2d, K, dist_coeffs,
       rvec, tvec, false, 10000, 8.0, 0.99, inliers, cv::SOLVEPNP_EPNP)
   - Convert rvec to R via cv::Rodrigues
   - This gives [R|t] in world-to-camera convention directly
4. Triangulate new points:
   - For every match between this image and any registered image where
     the 3D point doesn't exist yet:
     - Triangulate and apply the same filters as Phase 1
5. Run bundle adjustment every 5 newly registered images (see Step 6)
6. After BA, remove points with reprojection error > 2× threshold
   and points seen by < 2 cameras

### Phase 3: Scale Calibration (if scale_bar_length_mm > 0)

If the user provides a known scale bar length:
1. Detect the scale bar in images (two endpoints)
   - This could be manual (user provides pixel coordinates) or automatic
     (detect parallel lines with known separation)
2. Triangulate both endpoints
3. Compute 3D distance between them
4. scale_factor = scale_bar_length_mm / triangulated_distance
5. Apply: multiply all translations and point positions by scale_factor
6. Store scale_factor in SfMResult

### Phase 4: Turntable Constraint (if --turntable)

If turntable capture assumed:
1. After initial SfM, fit a circle to camera centers (least-squares)
2. Compute turntable axis (normal to the fitted circle plane)
3. Refine camera positions to lie on the circle
4. Pass turntable axis as a prior to bundle adjustment
   (soft constraint: regularize cameras toward circle)

### Checkpoint:

If config.save_intermediate:
- Save cameras to <output_dir>/sfm_cameras.json
  (image_id, R as 3x3, t as 3x1, K as 3x3, is_registered)
- Save sparse point cloud to <output_dir>/sparse.ply
- On re-run, if sfm_cameras.json exists, load and skip SfM

### Unit test — test_sfm.cpp:

- Run SfM on 36 synthetic images with known ground-truth poses
- Verify: all 36 cameras registered
- Verify: mean reprojection error < 1.0 pixel
- Compare recovered poses to ground truth:
  - Align via similarity transform (Umeyama)
  - Rotation error < 2° per camera
  - Translation error < 5% of scene diameter
- Verify sparse point cloud has > 1000 points
```

**Verification gate:** All 36 synthetic cameras registered. Mean reprojection < 1.0px. Sparse cloud is visually correct (save as .ply and inspect). Test passes.

---

## STEP 6 — Bundle Adjustment

**Prompt for Claude Code:**

```
Implement src/sfm/bundle_adjust.cpp and bundle_adjust.h.

### API:

void bundleAdjust(
    std::vector<CameraPose>& cameras,
    std::vector<SparsePoint>& points,
    const std::vector<SIFTFeatures>& features,
    const PipelineConfig& config,
    bool fix_first_camera = true,
    bool optimize_intrinsics = false  // only if enough images (>20)
);

### Implementation using Ceres Solver:

1. Reprojection error functor:

struct ReprojectionCost {
    double obs_x, obs_y;
    double fx, fy, cx, cy;  // fixed intrinsics (when not optimizing them)

    template <typename T>
    bool operator()(const T* const cam_angle_axis,  // 3 params
                    const T* const cam_translation,  // 3 params
                    const T* const point_3d,          // 3 params
                    T* residuals) const {
        // 1. Rotate point: P_cam = AngleAxis(aa) * P_world
        T p[3];
        ceres::AngleAxisRotatePoint(cam_angle_axis, point_3d, p);

        // 2. Translate: P_cam += t
        p[0] += cam_translation[0];
        p[1] += cam_translation[1];
        p[2] += cam_translation[2];

        // 3. Project: pixel = K * [P_cam.x/P_cam.z, P_cam.y/P_cam.z]
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];
        T predicted_x = T(fx) * xp + T(cx);
        T predicted_y = T(fy) * yp + T(cy);

        // 4. Residual
        residuals[0] = predicted_x - T(obs_x);
        residuals[1] = predicted_y - T(obs_y);
        return true;
    }
};

2. Build the problem:
   - Convert each camera R to angle-axis (3 doubles) via ceres::RotationMatrixToAngleAxis
   - Parameter blocks: camera_aa[3], camera_t[3] per camera; point[3] per 3D point
   - Loss function: new ceres::HuberLoss(1.0) on each residual
   - If fix_first_camera: SetParameterBlockConstant for camera 0's aa and t
   - Optional: if optimize_intrinsics, add fx/fy/cx/cy as shared parameter block

3. Solver options:
   options.linear_solver_type = ceres::SPARSE_SCHUR;
   options.preconditioner_type = ceres::SCHUR_JACOBI;
   options.max_num_iterations = 100;
   options.function_tolerance = 1e-6;
   options.gradient_tolerance = 1e-10;
   options.parameter_tolerance = 1e-8;
   options.num_threads = std::thread::hardware_concurrency();
   options.minimizer_progress_to_stdout = config.verbose;

4. After solving:
   - Convert angle-axis back to rotation matrices
   - Update cameras and points in-place
   - Log: initial cost, final cost, iterations, termination reason

5. Post-BA cleanup (call separately after BA):

void filterPointCloud(
    std::vector<CameraPose>& cameras,
    std::vector<SparsePoint>& points,
    double max_reproj_error,
    int min_track_length
);
   - Remove points with mean_reprojection_error > max_reproj_error
   - Remove points with track_length < min_track_length
   - Log: how many points removed

NOTE: Steps 5 and 6 are tightly coupled. Step 5 calls bundleAdjust()
periodically during registration. Make sure the API supports being called
multiple times with growing camera/point sets.
```

**Verification gate:** BA converges (final cost < initial cost). Mean reprojection error drops below 0.5px on synthetic data. Step 5 test still passes after integrating BA.

---

## STEP 7 — CUDA Dense Multi-View Stereo (PatchMatch MVS)

**Prompt for Claude Code:**

```
Implement src/dense_recon/patchmatch_mvs.cu and patchmatch_mvs.h.

This is the most complex step. Break it into sub-kernels and test each.

### API:

std::vector<DepthMap> runPatchMatchMVS(
    const std::vector<ImageData>& images,
    const std::vector<CameraPose>& cameras,
    const SfMResult& sfm,
    const PipelineConfig& config
);

### Overview:

For each image as "reference view", estimate per-pixel depth and normal
by finding photometric consistency with neighbor "source views".
Uses PatchMatch random search: initialize → propagate → evaluate → refine → repeat.

### Source View Selection:

For each reference image, select N_src = 5-8 source views:
- Sort other images by: number of shared SfM points × baseline angle score
- Baseline angle score: sin(angle) clamped to [sin(5°), sin(30°)]
  (too small = degenerate, too large = too different)
- Select top N_src views

### Kernel 1: init_depth_normal_kernel

__global__ void init_depth_normal_kernel(
    float* depth,       // output: H×W
    float* normal,      // output: H×W×3
    float depth_min,
    float depth_max,
    int width, int height,
    unsigned long long seed
)

For each pixel:
- depth = random uniform in [depth_min, depth_max]
  - Use cuRAND: curand_uniform() * (depth_max - depth_min) + depth_min
  - Initialize cuRAND states: one per pixel, seed = global_seed + pixel_index
- normal = random unit vector with z-component toward camera
  - Generate random theta, phi; ensure dot(normal, view_ray) < 0

depth_min/depth_max: compute from SfM sparse points projected into this view
- depth_min = percentile_5(depths) * 0.5
- depth_max = percentile_95(depths) * 1.5

### Kernel 2: compute_ncc_cost_kernel

__global__ void compute_ncc_cost_kernel(
    const cudaTextureObject_t ref_image,     // reference grayscale
    const cudaTextureObject_t* src_images,   // source grayscales (N_src)
    const float* depth,                      // current depth map
    const float* normal,                     // current normal map
    const float* ref_K, const float* ref_Rt, // reference camera (3x3, 3x4)
    const float* src_K, const float* src_Rt, // source cameras (N_src × ...)
    float* cost,                             // output: per-pixel cost
    int patch_radius,                        // e.g., 5 → 11×11 patch
    int width, int height,
    int num_sources
)

For each pixel (x, y):
1. Get depth d and normal n
2. Compute 3D point: P = backproject(x, y, d, ref_K_inv, ref_R, ref_t)
3. For each source view s:
   a. Compute homography H_s from depth + normal:
      H_s = K_s * (R_s * R_ref^T - (R_s * R_ref^T * t_ref - t_s) * n^T / (n^T * P)) * K_ref^{-1}
      (plane-induced homography for a local tangent plane at P with normal n)
   b. For each pixel (dx, dy) in the patch around (x, y):
      - Warp to source view: [u_s, v_s] = H_s * [x+dx, y+dy, 1]
      - Sample source image at (u_s, v_s) using texture interpolation
   c. Compute NCC between reference patch and warped source patch:
      NCC = (sum(ref*src) - n*mean_ref*mean_src) /
            (sqrt(sum(ref²)-n*mean_ref²) * sqrt(sum(src²)-n*mean_src²))
   d. cost_s = 1.0 - NCC (range [0, 2], lower = better)

4. Aggregate multi-view costs:
   - Sort costs from all source views
   - Use top K=3 lowest costs (robust to occlusion)
   - Final cost = mean of top-K costs

Shared memory strategy:
- Load reference patch into shared memory (reused for all source views)
- Each thread handles one pixel; thread-block is a 2D tile (e.g., 16×16)

### Kernel 3: propagate_and_refine_kernel

__global__ void propagate_and_refine_kernel(
    float* depth, float* normal, float* cost,
    // ... same camera params as cost kernel
    int iteration,
    int checkerboard_pass,  // 0 = even pixels, 1 = odd pixels
    int width, int height
)

Red-black checkerboard pattern: process even pixels in pass 0, odd in pass 1.
This avoids read-write races on neighbor data.

For each pixel in current checkerboard color:
1. PROPAGATION: Test hypotheses from 4 neighbors (left, right, up, down):
   - Read neighbor's (depth, normal)
   - Compute cost with that hypothesis
   - If better than current cost → adopt it

2. RANDOM REFINEMENT:
   - Perturb depth: d' = d + uniform(-range, +range)
     where range = (depth_max - depth_min) * 0.5^iteration
   - Perturb normal: rotate by random angle ≤ max_angle * 0.5^iteration
     where max_angle starts at 30°
   - Compute cost of (d', n')
   - If better → adopt

### Kernel 4: geometric_consistency_filter_kernel

After all reference views have depth maps:

__global__ void geometric_consistency_filter_kernel(
    float* ref_depth, float* ref_normal, float* ref_confidence,
    const float* src_depths[],          // depth maps of source views
    // ... camera params
    float depth_tolerance,              // relative: 0.01 = 1%
    float normal_tolerance,             // degrees: 10.0
    int min_consistent_views,           // e.g., 2
    int width, int height
)

For each pixel with valid depth:
1. Backproject to 3D
2. Project into each source view
3. Read source's depth at that pixel
4. Check consistency:
   - |depth_src - expected_depth| / expected_depth < depth_tolerance
   - angle(normal_ref, normal_src) < normal_tolerance
5. Count consistent views
6. If consistent_count < min_consistent_views → set depth to 0 (invalid)
7. confidence = consistent_count / num_sources

### Main loop:

for each reference_image:
    init_depth_normal_kernel<<<...>>>
    for iter in 0..mvs_iterations:
        propagate_and_refine_kernel<<<...>>> (checkerboard_pass=0)
        propagate_and_refine_kernel<<<...>>> (checkerboard_pass=1)
    store depth map

// After ALL depth maps are computed:
for each reference_image:
    geometric_consistency_filter_kernel<<<...>>>

### Memory management:

- Reference image: 1× cudaArray + texture object
- Source images: N_src × cudaArray + texture objects (bind before each ref view)
- Depth/normal/cost maps: 3 × H × W × float = ~48MB per map at 4000×3000
  (process one reference at a time, reuse buffers)
- cuRAND states: H × W × sizeof(curandState) ≈ 200MB at full res
  → Reduce: use per-block states (256 states, one per thread in block)
  → Or use Philox counter-based generator (no state needed)
- Total GPU memory: ~2-4 GB for one reference view

### Checkpoint:

If save_intermediate:
- Save each depth map as <output_dir>/depth_<image_id>.pfm (Portable FloatMap)
- Save confidence map as <output_dir>/confidence_<image_id>.pfm
- On re-run, if depth files exist, load and skip MVS

### Testing:

- Run on synthetic data (sphere)
- Verify: depth maps have valid values in expected range
- Verify: filtered depth maps show clean sphere surface
- Verify: at least 80% of pixels within 5% of ground-truth depth
```

**Verification gate:** Depth maps generated for all synthetic views. Geometric consistency filter retains ≥ 60% of pixels. Depth error vs ground truth < 5% for retained pixels. Total MVS time < 5 minutes for 36 images at 1280×960.

---

## STEP 8 — CUDA Dense Point Cloud Generation

**Prompt for Claude Code:**

```
Implement src/dense_recon/point_cloud.cu and point_cloud.h.

### API:

DensePointCloud fuseDepthMaps(
    const std::vector<DepthMap>& depth_maps,
    const std::vector<CameraPose>& cameras,
    const std::vector<ImageData>& images,
    const PipelineConfig& config
);

### Kernel 1: depth_to_points_kernel

For each valid pixel (depth > 0 AND confidence ≥ 0.5):
1. Backproject to camera frame:
   P_cam = depth * K_inv * [x, y, 1]^T
2. Transform to world frame:
   P_world = R^T * (P_cam - t)
   (recall: [R|t] is world-to-camera, so inverse is R^T and -R^T*t)
3. Transform normal to world frame:
   N_world = R^T * N_cam
4. Sample color from original BGR image at (x, y):
   color = image(y, x) / 255.0  (normalize to [0, 1])
5. Write to output arrays using atomicAdd on a global counter
   for compacted output (or use prefix sum for exact sizing)

### Fusion Strategy — Implement Both Options:

Option A: Voxel Grid Downsampling (default, lower memory)
   CUDA kernel: voxel_downsample_kernel
   - Voxel size: 0.1mm for small specimens (config.voxel_resolution)
   - Hash each point to voxel: morton_code(floor(pos / voxel_size))
   - Use a hash table (GPU open-addressing with linear probing):
     - Key: morton code (uint64)
     - Value: accumulated position, normal, color, count
   - After all points inserted: divide accumulated values by count
   - Extract non-empty voxels as final points

Option B: TSDF Volume (better quality, higher memory)
   Only use when GPU has ≥ 8 GB VRAM.
   CUDA kernel: tsdf_integrate_kernel
   - Volume bounds: from SfM sparse point cloud bounding box + 10% margin
   - Resolution: 0.1mm voxel → for 10cm specimen: 1000³ = 4GB for float32+weight
   - Use voxel hashing (allocate only near surface) to reduce memory:
     - 8×8×8 voxel blocks
     - Hash table of block positions
     - Only allocate blocks within truncation distance of any depth map
   - For each depth map:
     - For each allocated voxel:
       - Project to image, read depth
       - SDF = depth_image(proj) - depth_voxel_to_cam
       - Truncate: clamp to [-trunc, +trunc] where trunc = 5 × voxel_size
       - Update running weighted average

### Outlier Removal:

CUDA kernel: statistical_outlier_removal_kernel
- Build a uniform 3D grid (cell size = 2mm)
- For each point, count neighbors in 3×3×3 neighborhood of cells
- Remove points with neighbor_count < threshold (e.g., < 5)
- This is faster than KNN-based outlier removal on GPU

### Normal Estimation (refine after fusion):

CUDA kernel: refine_normals_kernel
- For each point, gather neighbors from the same grid
- Fit plane via covariance matrix (PCA):
  - C = (1/N) * Σ (p_i - centroid)(p_i - centroid)^T
  - Normal = eigenvector of smallest eigenvalue of C
  - Use closed-form 3×3 eigendecomposition (Cardano's method)
- Orient normals toward nearest camera center:
  - If dot(normal, cam_center - point) < 0: flip normal

### Checkpoint:

If save_intermediate:
- Save dense point cloud as <output_dir>/dense.ply
- On re-run, if dense.ply exists, load and skip fusion

### Output expectations (synthetic sphere, 36 views):
- Total raw points: ~500K-2M
- After voxel downsampling: ~100K-500K
- After outlier removal: ~90K-450K
```

**Verification gate:** Dense point cloud generated. Visually forms a sphere (inspect in MeshLab). Point count in expected range. Normals point outward. Outlier rate < 10%.

---

## STEP 9 — Surface Reconstruction (Meshing)

**Prompt for Claude Code:**

```
Implement BOTH meshing approaches. Poisson is primary; Marching Cubes
is used only when TSDF volume is available from Step 8.

### src/meshing/poisson_recon.cpp — Wrapper for PoissonRecon binary

This is the PRIMARY approach. Do not attempt to implement Poisson from scratch.

Mesh poissonReconstruct(
    const DensePointCloud& cloud,
    const PipelineConfig& config
);

Implementation:
1. Export oriented point cloud to temporary .ply file:
   - Must include: x, y, z, nx, ny, nz (positions + normals)
   - Binary PLY for speed
2. Call PoissonRecon:
   std::string cmd = fmt::format(
       "PoissonRecon --in {} --out {} --depth {} --pointWeight {} "
       "--samplesPerNode 1.5 --threads {}",
       input_ply, output_ply,
       config.poisson_depth,       // 10-11 for fine detail
       config.poisson_screening,   // 4.0
       std::thread::hardware_concurrency()
   );
   int ret = system(cmd.c_str());
   if (ret != 0) {
       // Fallback: try with --depth reduced by 1
       // If still fails: try Open3D Python or CGAL
   }
3. Load output .ply mesh:
   - Parse PLY vertices + faces
   - Populate Mesh struct
4. Post-processing:
   a. Remove low-density faces:
      - PoissonRecon can output a density field
      - Use --density flag, then trim faces below 10th percentile
   b. Keep largest connected component (remove floating fragments)
   c. Transfer colors from point cloud to mesh vertices:
      - For each vertex, find nearest point in cloud (use grid or KD-tree)
      - Assign that point's color

### src/meshing/marching_cubes.cu — CUDA Marching Cubes (for TSDF path)

Mesh marchingCubes(
    const TSDFVolume& volume,  // from Step 8 Option B
    float isovalue = 0.0f
);

Implementation:

Kernel 1: classify_voxels_kernel
- For each voxel, sample TSDF at 8 corners
- Compute 8-bit cube index: bit_i = (tsdf_corner_i < isovalue) ? 1 : 0
- Look up triangle count from edgeTable[cube_index]
- Write count to per-voxel buffer (for prefix sum)

Prefix sum: use cub::DeviceScan::ExclusiveSum to compute write offsets

Kernel 2: generate_triangles_kernel
- For each active voxel (triangle_count > 0):
  - For each edge that has a zero-crossing:
    - Interpolate vertex position: lerp based on TSDF values at endpoints
    - Compute vertex normal: gradient of TSDF field (central differences)
  - Write triangles to output buffer at the prefix-sum offset
  - Use triTable[cube_index][...] for triangle connectivity

The 256-entry edgeTable and triTable:
- Store as __constant__ device memory (fast broadcast reads)
- Use the standard Paul Bourke marching cubes tables

### Post-processing (shared by both approaches):

void postProcessMesh(Mesh& mesh, const PipelineConfig& config);

1. Laplacian smoothing:
   - config.smooth_iterations iterations (default 3)
   - lambda = config.smooth_lambda (default 0.5)
   - For each vertex: v_new = v + lambda * (mean_neighbor - v)
   - Use Taubin smoothing (alternating +lambda/-mu) to avoid shrinkage

2. Mesh decimation (if config.decimate_target > 0):
   - Implement edge-collapse based decimation (Garland-Heckbert QEM)
   - Or shell out to: meshlab/meshlabserver with a decimation filter
   - Target: reduce to decimate_target faces

3. Remove small components:
   - BFS/DFS connected component analysis on face adjacency
   - Keep only the largest component
   - Log how many vertices/faces removed

4. Hole filling (optional):
   - Detect boundary edges (edges with only one incident face)
   - For small holes (< 20 boundary edges): triangulate the hole
   - Leave large holes open (likely legitimate concavities)

5. Watertight check:
   - Verify: every edge shared by exactly 2 faces
   - Log result (watertight: yes/no)

### Checkpoint:

If save_intermediate:
- Save raw mesh (before post-processing) as <output_dir>/mesh_raw.ply
- Save final mesh as <output_dir>/mesh_final.ply

### For hemipenis specimens specifically:

- Poisson depth 10 minimum, 11 for fine ornamentation (spines, ridges, calyces)
- Use Taubin smoothing (not pure Laplacian) to preserve sharp ridges
- Set decimation target based on specimen size:
  ~100K faces per cm of specimen length is a good baseline
- Anisotropic smoothing (smooth along ridges, not across them) is a stretch goal
```

**Verification gate:** Mesh generated from synthetic sphere. Watertight = yes. Vertex count in expected range. Hausdorff distance to ground-truth sphere < 1% of radius. MeshLab visualization looks correct.

---

## STEP 10 — Mesh Export (.obj / .stl / .ply)

**Prompt for Claude Code:**

```
Implement src/export/obj_exporter.cpp, stl_exporter.cpp, ply_exporter.cpp,
and the shared interface in mesh_exporter.h.

### Unified interface:

enum class ExportFormat { OBJ, STL, PLY, AUTO };

bool exportMesh(const Mesh& mesh, const std::string& filepath,
                ExportFormat format = ExportFormat::AUTO);
// AUTO: detect from file extension

bool exportPointCloud(const DensePointCloud& cloud, const std::string& filepath);
// Always PLY

### OBJ Exporter:

void exportOBJ(const Mesh& mesh, const std::string& filepath);

File format:
  # Generated by cuda_3d_recon
  # Vertices: N, Faces: M
  v x y z [r g b]          ← vertex position [+ optional color 0.0-1.0]
  vn nx ny nz              ← vertex normal
  vt u v                   ← texture coord (if available)
  f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3   ← face (1-INDEXED, not 0-indexed!)

CRITICAL: OBJ indices are 1-based. Add 1 to all indices when writing.

If mesh has colors but no texture, write colors as vertex extensions:
  v x y z r g b
(Non-standard but supported by MeshLab, Blender, CloudCompare)

If mesh has texcoords, also write companion .mtl file:
  mtllib model.mtl
in the OBJ header, and create model.mtl with basic material.

Use buffered I/O:
  std::ofstream out(filepath, std::ios::out);
  out.rdbuf()->pubsetbuf(buffer, 1 << 20);  // 1MB buffer
  out << std::fixed << std::setprecision(6);

### STL Exporter (Binary):

void exportSTL(const Mesh& mesh, const std::string& filepath);

Binary STL layout (little-endian):
  Bytes 0-79:    Header (80 chars, no "solid" prefix)
  Bytes 80-83:   uint32 triangle count
  Per triangle (50 bytes):
    float32[3]   face normal (nx, ny, nz)
    float32[3]   vertex 1
    float32[3]   vertex 2
    float32[3]   vertex 3
    uint16       attribute byte count (0)

Compute face normals from cross product:
  edge1 = v2 - v1
  edge2 = v3 - v1
  normal = normalize(cross(edge1, edge2))

Write entire buffer at once for speed:
  std::vector<char> buffer(84 + mesh.num_faces() * 50);
  // ... fill buffer ...
  out.write(buffer.data(), buffer.size());

### PLY Exporter (Binary):

For both meshes and point clouds.

void exportPLY(const Mesh& mesh, const std::string& filepath);
void exportPLY(const DensePointCloud& cloud, const std::string& filepath);

Header (ASCII):
  ply
  format binary_little_endian 1.0
  element vertex N
  property float x
  property float y
  property float z
  property float nx
  property float ny
  property float nz
  property uchar red
  property uchar green
  property uchar blue
  element face M           ← only for meshes
  property list uchar int vertex_indices
  end_header

Then binary data: vertices followed by faces.

### Unit tests:

- Export synthetic sphere mesh as OBJ → re-import with OpenCV/custom parser
  → verify vertex and face counts match
- Export as STL → verify file size = 84 + 50 * num_faces
- Export as PLY → re-import → verify vertex and face counts
- OBJ round-trip: export then manually verify first face line is "f 1/... 2/... 3/..."
  (1-indexed, not 0-indexed)
```

**Verification gate:** All three formats export correctly. Files open in MeshLab without errors. STL file size matches expected. OBJ re-import recovers correct geometry.

---

## STEP 11 — CLI Interface and Pipeline Orchestrator

**Prompt for Claude Code:**

```
Implement src/main.cpp and src/pipeline.cu/h — wire everything together.

### main.cpp:

Use a lightweight argument parser (implement manually or use a single-header
lib like cxxopts if you add it to CMake).

Usage: recon3d [options] -i <input_dir> -o <output_path>

Required:
  -i, --input <dir>             Input image directory
  -o, --output <path>           Output mesh (.obj, .stl, .ply)

Quality:
  --quality <low|medium|high|ultra>   Preset (default: high)

Presets:
  low:    max_image=1600, poisson_depth=8,  mvs_res=0.25, mvs_iter=4
  medium: max_image=2400, poisson_depth=9,  mvs_res=0.5,  mvs_iter=6
  high:   max_image=3200, poisson_depth=10, mvs_res=1.0,  mvs_iter=8
  ultra:  max_image=4800, poisson_depth=11, mvs_res=1.0,  mvs_iter=12

Full options (override presets):
  --max-image-size <px>         Max image dimension (default: from preset)
  --match-ratio <float>         Lowe's ratio threshold (default: 0.75)
  --mvs-resolution <float>      MVS resolution multiplier (default: from preset)
  --mvs-iterations <int>        PatchMatch iterations (default: from preset)
  --poisson-depth <int>         Octree depth (default: from preset)
  --smooth-iterations <int>     Mesh smoothing passes (default: 3)
  --decimate <int>              Target face count, 0=none (default: 0)
  --export-pointcloud <path>    Also save dense cloud as .ply

Camera:
  --focal-length <mm>           Override focal length
  --sensor-width <mm>           Override sensor width
  --turntable                   Assume turntable capture
  --scale-bar <mm>              Known scale bar length for calibration

GPU:
  --gpu <id>                    Device ID (default: 0)
  --gpu-memory <MB>             Max GPU memory (default: auto)

Output:
  --verbose                     Detailed progress
  --save-intermediate           Save depth maps, sparse cloud, etc.
  --log <path>                  Log to file
  --resume                      Skip stages whose outputs already exist

### pipeline.cu — Orchestrator:

class ReconstructionPipeline {
    PipelineConfig config;
    Timer timer;  // wall-clock timing per stage

public:
    bool run() {
        timer.start("total");

        // 0. GPU info
        printGPUInfo(config.gpu_id);
        CUDA_CHECK(cudaSetDevice(config.gpu_id));

        // 1. Load images
        timer.start("load");
        auto images = loadImages(config.input_dir, config.max_image_size);
        timer.stop("load");
        log("Loaded {} images ({}x{})", images.size(),
            images[0].width, images[0].height);
        if (images.size() < 10) {
            error("Need at least 10 images, got {}", images.size());
            return false;
        }

        // 2. Feature detection
        timer.start("features");
        auto features = detectAllFeatures(images);
        timer.stop("features");
        log("Detected avg {:.0f} features per image", avgCount(features));

        // 3. Feature matching
        timer.start("matching");
        auto matches = matchAllPairs(features, config.match_ratio, config.min_matches);
        timer.stop("matching");
        log("Matched {} image pairs", matches.size());
        if (matches.empty()) {
            error("No image pairs matched. Check image overlap.");
            return false;
        }

        // 4. Structure from Motion
        timer.start("sfm");
        auto sfm = runIncrementalSfM(features, matches, images, config);
        timer.stop("sfm");
        log("Registered {}/{} cameras, {} sparse points, reproj {:.2f}px",
            sfm.num_registered, images.size(),
            sfm.points.size(), sfm.mean_reprojection_error);
        if (sfm.num_registered < 3) {
            error("Only {} cameras registered. Need >= 3.", sfm.num_registered);
            return false;
        }

        // 5. Dense MVS
        timer.start("mvs");
        auto depth_maps = runPatchMatchMVS(images, sfm.cameras, sfm, config);
        timer.stop("mvs");
        log("Generated {} depth maps", depth_maps.size());

        // 6. Point cloud fusion
        timer.start("fusion");
        auto cloud = fuseDepthMaps(depth_maps, sfm.cameras, images, config);
        timer.stop("fusion");
        log("Dense cloud: {} points", cloud.size());
        if (config.pointcloud_path.size() > 0) {
            exportPointCloud(cloud, config.pointcloud_path);
        }

        // 7. Surface reconstruction
        timer.start("meshing");
        auto mesh = poissonReconstruct(cloud, config);
        postProcessMesh(mesh, config);
        timer.stop("meshing");
        log("Mesh: {} vertices, {} faces, watertight: {}",
            mesh.num_vertices(), mesh.num_faces(),
            mesh.is_watertight() ? "yes" : "no");

        // 8. Export
        timer.start("export");
        exportMesh(mesh, config.output_path);
        timer.stop("export");

        timer.stop("total");
        timer.printSummary();
        return true;
    }
};

### Error handling at every stage:

- If feature detection finds < 100 features in any image → warn
- If no pairs match → error with diagnostic (images too different? wrong focal?)
- If SfM registers < 50% of images → warn (partial reconstruction)
- If MVS produces empty depth maps → fallback: export sparse cloud
- If PoissonRecon binary not found → error with install instructions
- If mesh is empty after post-processing → error

### Resume logic (--resume):

Each stage checks for its checkpoint file before running:
  if (config.resume && fileExists(checkpoint_path)) {
      log("Loading cached {} ...", stage_name);
      loadFromDisk(...);
  } else {
      // run stage normally
      if (config.save_intermediate) saveToDisk(...);
  }
```

**Verification gate:** `./recon3d -i data/synthetic/ -o output/sphere.obj --quality medium --verbose --save-intermediate` runs end-to-end. All 8 stages complete. Output mesh opens in MeshLab.

---

## STEP 12 — Image Capture Protocol & Preprocessing

**Prompt for Claude Code:**

```
Create scripts/capture_guide.md and scripts/preprocess_images.py.

### capture_guide.md:

Write a practical field guide for capturing small biological specimens
(hemipenis, 2-10cm) for 3D reconstruction with a smartphone.

Sections:

1. Equipment (required / recommended):
   - Smartphone ≥ 12MP (iPhone 13+, Pixel 7+, Galaxy S22+)
   - Turntable (manual lazy susan, motorized, or DIY)
   - Tripod or phone clamp
   - Diffused lighting (LED light panel or lightbox)
   - Scale bar (printed ruler, 1cm grid card)
   - Matte background (not white — use gray/beige textured paper)

2. Camera Settings:
   - Lock focus (tap-and-hold on specimen in most camera apps)
   - Lock exposure (prevent flickering between shots)
   - Lock white balance (prevents color shift)
   - Shoot in highest quality JPEG (or RAW if supported)
   - Disable HDR (can cause ghosting with slight movement)
   - Use rear camera (not ultrawide — avoid barrel distortion)

3. Turntable Protocol (recommended):
   - Mount specimen at turntable center with clay/pins
   - Camera on tripod at 15-30cm distance
   - THREE elevation rings:
     * Low angle: ~15° above horizontal (36 shots, 10° increments)
     * Mid angle: ~45° above horizontal (24 shots, 15° increments)
     * High angle: ~75° (nearly top-down) (12 shots, 30° increments)
   - Total: 72 images
   - Include scale bar visible in at least 3 shots

4. Handheld Protocol (alternative):
   - Walk around the specimen in 3 rings at different heights
   - 60-80% overlap between adjacent shots
   - Minimum 50 images, recommended 80-100
   - Move the CAMERA, never the specimen

5. Common Mistakes:
   - White/reflective background → feature matching fails
   - Moving specimen between shots → reconstruction fails
   - Inconsistent lighting → color artifacts
   - Too few images → holes in reconstruction
   - All photos from same height → top/bottom missing
   - Shiny/wet specimen → specular reflections break MVS

6. Quick Validation:
   - Flip through images: does every surface appear in ≥ 3 shots?
   - Are there at least 2 elevation angles?
   - Is the scale bar visible?
   - Do images look consistent in exposure?

### preprocess_images.py:

#!/usr/bin/env python3
"""Preprocess smartphone images for CUDA 3D reconstruction pipeline."""

Features:
1. EXIF orientation auto-rotation (Pillow ExifTags)
2. Resize to target max dimension (preserve aspect ratio)
3. Exposure normalization: match histograms to median image
   (optional, --normalize-exposure)
4. Lens distortion correction: if camera model is in lookup table
   (optional, --undistort)
5. Sharpness check: flag blurry images (Laplacian variance < threshold)
6. Quality report:
   - Image count and resolution
   - Estimated focal length
   - Blur detection results
   - Brightness consistency score

Usage:
  python preprocess_images.py \
      --input raw_photos/ \
      --output processed/ \
      --max-size 3200 \
      --normalize-exposure \
      --check-quality

Dependencies: opencv-python, Pillow, numpy (all pip-installable)
```

**Verification gate:** Script runs on sample images. Quality report is printed. Preprocessed images load correctly in the pipeline.

---

## STEP 13 — Testing and Validation

**Prompt for Claude Code:**

```
Expand and finalize all tests in tests/.

### Unit Tests (Google Test):

test_sift.cpp — already written in Step 3, verify it still passes.
test_matching.cpp — already written in Step 4, verify it still passes.
test_sfm.cpp — already written in Step 5, verify it still passes.

test_meshing.cpp — add/verify:
- Marching cubes on analytic sphere SDF: f(x,y,z) = sqrt(x²+y²+z²) - R
  - Grid resolution 100³
  - Verify: genus 0, no boundary edges, vertex count ~ 4πR² / (voxel_size²/2)
  - Verify: all face normals point outward
- OBJ export → re-import round-trip
- STL export → verify file size
- PLY export → verify header and data integrity
- Mesh.is_watertight() returns true for closed sphere mesh

test_export.cpp — NEW:
- Export known mesh, re-import, compare vertex positions
- Verify OBJ uses 1-indexed faces
- Verify STL binary format correctness (parse back and compare)

test_pipeline.cpp — NEW (integration test):
- Run the full pipeline on synthetic data (36 sphere images)
- Verify: mesh is produced
- Compare to ground truth sphere:
  - Chamfer distance (mean nearest-neighbor distance between meshes)
  - Hausdorff distance (max nearest-neighbor distance)
  - Surface area: within 5% of 4πR²
  - Volume: within 10% of (4/3)πR³ (compute via divergence theorem)
  - Watertight: yes

### Performance Benchmark:

Create tests/benchmark.cpp (not run by ctest, separate target):

Measure and report for the synthetic dataset:
- Feature detection: ms/image (target: < 500ms at 3200px)
- Feature matching: ms/pair (target: < 100ms)
- SfM: total seconds (target: < 30s for 36 images)
- MVS per image: seconds (target: < 30s at 1280×960)
- Point cloud fusion: seconds (target: < 10s)
- Meshing: seconds (target: < 60s)
- Export: seconds (target: < 5s)
- TOTAL: minutes (target: < 10min for 36 images at medium quality)
- Peak GPU memory: MB

Output as a table to stdout and optionally to benchmark_results.json.

### CMakeLists.txt updates:

- Add all new test files
- Add benchmark target (separate from ctest):
  add_executable(recon3d_benchmark tests/benchmark.cpp ...)
  # Don't add to ctest — run manually with ./recon3d_benchmark
```

**Verification gate:** `ctest --output-on-failure` passes all tests. Integration test produces correct sphere reconstruction. Performance benchmarks are within targets.

---

## STEP 14 — Optimization and Hardening

**Prompt for Claude Code:**

```
Review and optimize the entire codebase. Work through each checklist item,
make concrete changes (not just comments), and verify with tests after each change.

### CUDA Performance Optimization:

1. Kernel launch configuration:
   - For each kernel, compute optimal block size using:
     cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, kernel_func, 0, 0);
   - Replace hardcoded block sizes with computed values
   - Log occupancy for each kernel when --verbose

2. Memory access patterns:
   - Audit every kernel: are global reads coalesced?
   - NCC kernel: ensure patch pixels are loaded in row-major order
   - Descriptor matching: verify tile loading is conflict-free
   - Point cloud: use SoA (separate x,y,z arrays) not AoS (xyz structs)

3. Shared memory optimization:
   - Check for bank conflicts in histogram kernels (pad if needed)
   - Verify shared memory doesn't exceed 48KB per block (or use dynamic)

4. Texture memory:
   - All image reads should go through texture objects (not global memory)
   - Verify bilinear interpolation is enabled for MVS warping

5. Streams and overlap:
   - Feature detection: process images in batches with multiple streams
   - Depth map computation: overlap kernel execution with H2D/D2H transfers
     for next/previous reference image

6. Reduce host-device synchronization:
   - Replace CUDA_CHECK_KERNEL() (which syncs) with async error checking
     in non-debug builds
   - Batch multiple kernel launches before any sync point

### Memory Management:

7. Pre-allocate all GPU buffers at pipeline start:
   - Compute max needed for each stage
   - Allocate once, reuse across stages
   - Use CUDA memory pools: cudaMemPool_t for async alloc/free

8. Handle GPU OOM gracefully:
   - Before each stage, check getAvailableGPUMemory()
   - If insufficient for full-res MVS: automatically reduce resolution
   - Log the fallback

### Robustness:

9. Edge cases:
   - Test with exactly 10 images (minimum)
   - Test with images of different resolutions in the same set
   - Test with images that have no EXIF data (should fall back to defaults)
   - Test with very few features (< 200 per image)
   - Test with mostly failed matches (< 5 valid pairs)

10. Error recovery:
    - If MVS produces mostly empty depth maps: skip fusion, export sparse cloud
    - If PoissonRecon crashes: try with --depth reduced by 2
    - If mesh post-processing produces empty mesh: export raw mesh

11. Signal handling:
    - Catch SIGINT/SIGTERM: free GPU memory, save partial results, exit cleanly

### Code Quality:

12. Documentation:
    - Every public function has a doxygen-style comment
    - Every CUDA kernel documents: grid size, block size, shared memory,
      expected inputs/outputs

13. Build verification:
    - cmake .. && make -j$(nproc) succeeds with no warnings (-Wall -Wextra)
    - CUDA compilation has no warnings

14. Memory leak check:
    - Run: compute-sanitizer --tool memcheck ./recon3d -i data/synthetic/ -o /tmp/out.obj --quality low
    - Fix any reported leaks or out-of-bounds accesses

15. Final validation:
    - Run full pipeline on synthetic data at all 4 quality presets
    - Verify all produce valid meshes
    - Run ctest — all tests pass
    - Run benchmark — all targets met
```

**Verification gate:** All checklist items addressed. `ctest` passes. `compute-sanitizer` reports no errors. Pipeline runs at all quality levels. Benchmark meets targets.

---

## Quick-Start Summary

```
Step  0: "Verify all dependencies are installed (CUDA, OpenCV+CUDA, Eigen, Ceres, PoissonRecon)"
Step  1: "Scaffold the project: CMake, types.h, cuda_utils.cuh, math_utils.h, all file stubs"
Step  2: "Implement image loader with EXIF extraction AND synthetic test data generator"
Step  3: "Implement CUDA SIFT (OpenCV wrapper primary, custom kernels secondary)"
Step  4: "Implement CUDA feature matching with ratio test, cross-check, RANSAC verification"
Step  5: "Implement incremental SfM with PnP, triangulation, scale calibration, turntable support"
Step  6: "Implement Ceres bundle adjustment with Huber loss and Schur complement"
Step  7: "Implement CUDA PatchMatch MVS: init, propagate, NCC cost, refine, filter (hardest step)"
Step  8: "Implement CUDA point cloud fusion: backproject, voxel downsample or TSDF, outlier removal"
Step  9: "Implement meshing: PoissonRecon wrapper (primary) + CUDA Marching Cubes + post-processing"
Step 10: "Implement OBJ, STL, PLY exporters with buffered I/O"
Step 11: "Wire CLI, pipeline orchestrator, error handling, resume logic"
Step 12: "Write capture guide and Python preprocessing script"
Step 13: "Write integration tests, round-trip tests, performance benchmarks"
Step 14: "CUDA optimization, memory management, robustness, compute-sanitizer clean"
```

Each step has a **verification gate** — do not proceed to the next step until the gate passes. After each step, run `make -j$(nproc) && ctest`.
