#include "dense_recon/patchmatch_mvs.h"
#include "utils/cuda_utils.cuh"
#include "utils/math_utils.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <fstream>

// ============================================================================
// Constants
// ============================================================================

static constexpr int MAX_SRC_VIEWS = 8;
static constexpr float BASELINE_MIN_DEG = 5.0f;
static constexpr float BASELINE_MAX_DEG = 30.0f;

// NCC cost kernel constants
static constexpr int NCC_BLOCK_DIM = 16;
static constexpr int NCC_PATCH_RADIUS = 5;  // 11×11 patch
static constexpr int NCC_TILE_DIM = NCC_BLOCK_DIM + 2 * NCC_PATCH_RADIUS; // 26
static constexpr int NCC_TOP_K = 3;

// Precomputed source view params for GPU (host-computed, uploaded to device)
// A = K_src * R_rel * K_ref_inv,  b = K_src * t_rel
// where R_rel = R_src * R_ref^T,  t_rel = t_src - R_rel * t_ref
struct GPUSourceView {
    float A[9];  // 3×3 row-major
    float b[3];  // 3-vector
};

// ============================================================================
// Source View Selection (CPU)
// ============================================================================

// Count shared SfM points between two camera views
static int countSharedPoints(const SfMResult& sfm, int cam_i, int cam_j) {
    int count = 0;
    for (const auto& pt : sfm.points) {
        bool has_i = false, has_j = false;
        for (int id : pt.image_ids) {
            if (id == cam_i) has_i = true;
            if (id == cam_j) has_j = true;
        }
        if (has_i && has_j) count++;
    }
    return count;
}

// Compute baseline angle between two cameras looking at the scene centroid
static float computeBaselineAngle(const CameraPose& cam_i, const CameraPose& cam_j,
                                   const Eigen::Vector3d& centroid) {
    Eigen::Vector3d ci = cam_i.cam_center();
    Eigen::Vector3d cj = cam_j.cam_center();
    Eigen::Vector3d vi = (centroid - ci).normalized();
    Eigen::Vector3d vj = (centroid - cj).normalized();
    double cos_angle = std::clamp(vi.dot(vj), -1.0, 1.0);
    return static_cast<float>(std::acos(cos_angle) * 180.0 / M_PI);
}

// Baseline angle score: sin(angle) clamped to [sin(5°), sin(30°)]
static float baselineScore(float angle_deg) {
    if (angle_deg < BASELINE_MIN_DEG || angle_deg > BASELINE_MAX_DEG * 3.0f) {
        return 0.0f; // reject very small or very large baselines
    }
    float angle_clamped = std::clamp(angle_deg, BASELINE_MIN_DEG, BASELINE_MAX_DEG);
    return std::sin(angle_clamped * static_cast<float>(M_PI) / 180.0f);
}

std::vector<MVSViewParams> selectSourceViews(
    const SfMResult& sfm,
    int num_sources)
{
    // Compute scene centroid from sparse points
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    if (!sfm.points.empty()) {
        for (const auto& pt : sfm.points) centroid += pt.position;
        centroid /= static_cast<double>(sfm.points.size());
    }

    // Collect registered camera indices
    std::vector<int> registered;
    for (const auto& cam : sfm.cameras) {
        if (cam.is_registered) registered.push_back(cam.image_id);
    }

    std::vector<MVSViewParams> result;
    result.reserve(registered.size());

    for (int ref_id : registered) {
        MVSViewParams params;
        params.ref_image_id = ref_id;

        // Score all other registered cameras
        std::vector<SourceViewInfo> candidates;
        for (int src_id : registered) {
            if (src_id == ref_id) continue;
            float angle = computeBaselineAngle(sfm.cameras[ref_id], sfm.cameras[src_id], centroid);
            float b_score = baselineScore(angle);
            if (b_score <= 0.0f) continue;
            int shared = countSharedPoints(sfm, ref_id, src_id);
            if (shared == 0) continue;
            SourceViewInfo info;
            info.image_id = src_id;
            info.baseline_deg = angle;
            info.score = static_cast<float>(shared) * b_score;
            candidates.push_back(info);
        }

        // Sort by score descending, take top N
        std::sort(candidates.begin(), candidates.end(),
                  [](const SourceViewInfo& a, const SourceViewInfo& b) {
                      return a.score > b.score;
                  });
        int n = std::min(num_sources, static_cast<int>(candidates.size()));
        params.source_views.assign(candidates.begin(), candidates.begin() + n);

        // Compute depth range from SfM points projected into this view
        const auto& cam = sfm.cameras[ref_id];
        std::vector<float> depths;
        depths.reserve(sfm.points.size());
        for (const auto& pt : sfm.points) {
            // Check if this point is visible in the reference view
            bool visible = false;
            for (int id : pt.image_ids) {
                if (id == ref_id) { visible = true; break; }
            }
            if (!visible) continue;

            // P_cam = R * P_world + t
            Eigen::Vector3d p_cam = cam.R * pt.position + cam.t;
            if (p_cam.z() > 0) depths.push_back(static_cast<float>(p_cam.z()));
        }

        if (depths.size() >= 2) {
            std::sort(depths.begin(), depths.end());
            size_t idx5  = static_cast<size_t>(depths.size() * 0.05);
            size_t idx95 = std::min(static_cast<size_t>(depths.size() * 0.95), depths.size() - 1);
            params.depth_min = depths[idx5] * 0.5f;
            params.depth_max = depths[idx95] * 1.5f;
        } else {
            // Fallback: estimate from camera distance to centroid
            Eigen::Vector3d p_cam = cam.R * centroid + cam.t;
            float d = static_cast<float>(p_cam.z());
            params.depth_min = d * 0.3f;
            params.depth_max = d * 3.0f;
        }

        result.push_back(std::move(params));
    }

    return result;
}

// ============================================================================
// CUDA Kernel 1: Initialize depth and normal maps
// ============================================================================

// Philox-based random initialization — no persistent cuRAND state needed
__global__ void init_depth_normal_kernel(
    float* depth,       // output: H × W
    float* normal,      // output: H × W × 3
    float depth_min,
    float depth_max,
    const float* ref_K_inv, // 3x3 row-major (for computing view rays)
    int width, int height,
    unsigned long long seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // Initialize Philox-based cuRAND state (no persistent state needed)
    curandStatePhilox4_32_10_t state;
    curand_init(seed, static_cast<unsigned long long>(idx), 0, &state);

    // Random depth in [depth_min, depth_max]
    float rand_d = curand_uniform(&state);
    depth[idx] = depth_min + rand_d * (depth_max - depth_min);

    // View ray: K_inv * [x, y, 1]
    float rx = ref_K_inv[0] * x + ref_K_inv[1] * y + ref_K_inv[2];
    float ry = ref_K_inv[3] * x + ref_K_inv[4] * y + ref_K_inv[5];
    float rz = ref_K_inv[6] * x + ref_K_inv[7] * y + ref_K_inv[8];
    float ray_len = sqrtf(rx * rx + ry * ry + rz * rz);
    if (ray_len > 1e-12f) { rx /= ray_len; ry /= ray_len; rz /= ray_len; }

    // Random normal: generate in spherical coords, ensure facing camera
    // (dot(normal, view_ray) < 0 in camera frame, i.e. normal points toward camera)
    float theta = curand_uniform(&state) * 2.0f * 3.14159265f;
    float cos_phi = curand_uniform(&state); // uniform on hemisphere
    float sin_phi = sqrtf(1.0f - cos_phi * cos_phi);

    float nx = sin_phi * cosf(theta);
    float ny = sin_phi * sinf(theta);
    float nz = cos_phi; // positive z = toward camera in camera frame

    // Ensure normal faces camera: dot(n, ray) < 0
    if (nx * rx + ny * ry + nz * rz > 0.0f) {
        nx = -nx; ny = -ny; nz = -nz;
    }

    // Normalize (should already be unit, but be safe)
    float nlen = sqrtf(nx * nx + ny * ny + nz * nz);
    if (nlen > 1e-12f) { nx /= nlen; ny /= nlen; nz /= nlen; }

    normal[idx * 3 + 0] = nx;
    normal[idx * 3 + 1] = ny;
    normal[idx * 3 + 2] = nz;
}

// ============================================================================
// Device helper: compute NCC cost for a single pixel at a candidate depth/normal
// ============================================================================

__device__ float computePixelCost(
    cudaTextureObject_t ref_image,
    const cudaTextureObject_t* __restrict__ src_images,
    float cand_depth, float cand_nx, float cand_ny, float cand_nz,
    const float* __restrict__ ref_K_inv,
    const GPUSourceView* __restrict__ src_views,
    int x, int y,
    int patch_radius,
    int width, int height,
    int num_sources)
{
    const float ki0 = ref_K_inv[0], ki1 = ref_K_inv[1], ki2 = ref_K_inv[2];
    const float ki3 = ref_K_inv[3], ki4 = ref_K_inv[4], ki5 = ref_K_inv[5];
    const float ki6 = ref_K_inv[6], ki7 = ref_K_inv[7], ki8 = ref_K_inv[8];

    const float fx = static_cast<float>(x);
    const float fy = static_cast<float>(y);

    // 3D point in camera frame: P = d * K_inv * [x, y, 1]
    const float Px = cand_depth * (ki0 * fx + ki1 * fy + ki2);
    const float Py = cand_depth * (ki3 * fx + ki4 * fy + ki5);
    const float Pz = cand_depth * (ki6 * fx + ki7 * fy + ki8);

    // n^T * P (plane offset)
    const float n_dot_P = cand_nx * Px + cand_ny * Py + cand_nz * Pz;
    if (fabsf(n_dot_P) < 1e-8f) return 2.0f;
    const float inv_n_dot_P = 1.0f / n_dot_P;

    // nK = K_inv^T * n
    const float nK0 = ki0 * cand_nx + ki3 * cand_ny + ki6 * cand_nz;
    const float nK1 = ki1 * cand_nx + ki4 * cand_ny + ki7 * cand_nz;
    const float nK2 = ki2 * cand_nx + ki5 * cand_ny + ki8 * cand_nz;

    const int patch_size = 2 * patch_radius + 1;
    const int min_count = (patch_size * patch_size) / 2;
    const float fwidth = static_cast<float>(width);
    const float fheight = static_cast<float>(height);

    // Top-K partial sort
    float top_costs[NCC_TOP_K];
    #pragma unroll
    for (int i = 0; i < NCC_TOP_K; i++) top_costs[i] = 2.0f;

    for (int s = 0; s < num_sources; s++) {
        const GPUSourceView& sv = src_views[s];

        float sum_ref = 0.0f, sum_src = 0.0f;
        float sum_ref2 = 0.0f, sum_src2 = 0.0f;
        float sum_rs = 0.0f;
        int count = 0;

        for (int dy = -patch_radius; dy <= patch_radius; dy++) {
            for (int dx = -patch_radius; dx <= patch_radius; dx++) {
                const float ref_val = tex2D<float>(ref_image,
                    fx + dx + 0.5f, fy + dy + 0.5f);

                const float wpx = fx + static_cast<float>(dx);
                const float wpy = fy + static_cast<float>(dy);

                const float qx = sv.A[0]*wpx + sv.A[1]*wpy + sv.A[2];
                const float qy = sv.A[3]*wpx + sv.A[4]*wpy + sv.A[5];
                const float qz = sv.A[6]*wpx + sv.A[7]*wpy + sv.A[8];

                const float sval = nK0 * wpx + nK1 * wpy + nK2;
                const float factor = sval * inv_n_dot_P;

                const float wz = qz + sv.b[2] * factor;
                if (fabsf(wz) < 1e-8f) continue;
                const float inv_wz = 1.0f / wz;
                const float src_u = (qx + sv.b[0] * factor) * inv_wz;
                const float src_v = (qy + sv.b[1] * factor) * inv_wz;

                if (src_u < 0.0f || src_u >= fwidth - 1.0f ||
                    src_v < 0.0f || src_v >= fheight - 1.0f) continue;

                const float src_val = tex2D<float>(src_images[s],
                    src_u + 0.5f, src_v + 0.5f);

                sum_ref += ref_val;
                sum_src += src_val;
                sum_ref2 += ref_val * ref_val;
                sum_src2 += src_val * src_val;
                sum_rs += ref_val * src_val;
                count++;
            }
        }

        float c;
        if (count < min_count) {
            c = 2.0f;
        } else {
            const float fc = static_cast<float>(count);
            const float num = fc * sum_rs - sum_ref * sum_src;
            const float den_r = fc * sum_ref2 - sum_ref * sum_ref;
            const float den_s = fc * sum_src2 - sum_src * sum_src;
            if (den_r < 1e-6f || den_s < 1e-6f) {
                c = 2.0f;
            } else {
                float ncc = num / (sqrtf(den_r) * sqrtf(den_s));
                ncc = fmaxf(-1.0f, fminf(1.0f, ncc));
                c = 1.0f - ncc;
            }
        }

        if (c < top_costs[NCC_TOP_K - 1]) {
            top_costs[NCC_TOP_K - 1] = c;
            #pragma unroll
            for (int i = NCC_TOP_K - 2; i >= 0; i--) {
                if (top_costs[i + 1] < top_costs[i]) {
                    float tmp = top_costs[i];
                    top_costs[i] = top_costs[i + 1];
                    top_costs[i + 1] = tmp;
                } else break;
            }
        }
    }

    int k = min(NCC_TOP_K, num_sources);
    float sum = 0.0f;
    for (int i = 0; i < k; i++) sum += top_costs[i];
    return sum / static_cast<float>(k);
}

// ============================================================================
// CUDA Kernel 2: Compute NCC cost with plane-induced homography warping
// ============================================================================

__global__ void compute_ncc_cost_kernel(
    cudaTextureObject_t ref_image,
    const cudaTextureObject_t* __restrict__ src_images,
    const float* __restrict__ depth,
    const float* __restrict__ normal,
    const float* __restrict__ ref_K_inv,  // 3×3 row-major
    const GPUSourceView* __restrict__ src_views,
    float* __restrict__ cost,
    int patch_radius,
    int width, int height,
    int num_sources)
{
    // Shared memory: reference image tile (block + apron)
    __shared__ float s_ref[NCC_TILE_DIM * NCC_TILE_DIM];

    const int bx = blockIdx.x * NCC_BLOCK_DIM;
    const int by = blockIdx.y * NCC_BLOCK_DIM;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * NCC_BLOCK_DIM + tx;

    // Cooperatively load reference tile into shared memory
    const int tile_size = NCC_TILE_DIM * NCC_TILE_DIM;
    for (int i = tid; i < tile_size; i += NCC_BLOCK_DIM * NCC_BLOCK_DIM) {
        int tile_y = i / NCC_TILE_DIM;
        int tile_x = i % NCC_TILE_DIM;
        int img_x = bx + tile_x - patch_radius;
        int img_y = by + tile_y - patch_radius;
        // tex2D with clamp handles OOB; +0.5 for texel center
        s_ref[i] = tex2D<float>(ref_image, img_x + 0.5f, img_y + 0.5f);
    }
    __syncthreads();

    const int x = bx + tx;
    const int y = by + ty;
    if (x >= width || y >= height) return;

    const int idx = y * width + x;
    const float d = depth[idx];
    const float nn0 = normal[idx * 3 + 0];
    const float nn1 = normal[idx * 3 + 1];
    const float nn2 = normal[idx * 3 + 2];

    // Load K_inv elements
    const float ki0 = ref_K_inv[0], ki1 = ref_K_inv[1], ki2 = ref_K_inv[2];
    const float ki3 = ref_K_inv[3], ki4 = ref_K_inv[4], ki5 = ref_K_inv[5];
    const float ki6 = ref_K_inv[6], ki7 = ref_K_inv[7], ki8 = ref_K_inv[8];

    // 3D point in camera frame: P = d * K_inv * [x, y, 1]
    const float fx = static_cast<float>(x);
    const float fy = static_cast<float>(y);
    const float Px = d * (ki0 * fx + ki1 * fy + ki2);
    const float Py = d * (ki3 * fx + ki4 * fy + ki5);
    const float Pz = d * (ki6 * fx + ki7 * fy + ki8);

    // n^T * P (plane offset in camera frame)
    const float n_dot_P = nn0 * Px + nn1 * Py + nn2 * Pz;
    if (fabsf(n_dot_P) < 1e-8f) {
        cost[idx] = 2.0f;
        return;
    }
    const float inv_n_dot_P = 1.0f / n_dot_P;

    // nK = K_inv^T * n  (for efficient warp: dot(nK, [px, py, 1]))
    const float nK0 = ki0 * nn0 + ki3 * nn1 + ki6 * nn2;
    const float nK1 = ki1 * nn0 + ki4 * nn1 + ki7 * nn2;
    const float nK2 = ki2 * nn0 + ki5 * nn1 + ki8 * nn2;

    const int patch_size = 2 * patch_radius + 1;
    const int min_count = (patch_size * patch_size) / 2;
    const float fwidth = static_cast<float>(width);
    const float fheight = static_cast<float>(height);

    // Partial sort: maintain top-K lowest costs
    float top_costs[NCC_TOP_K];
    #pragma unroll
    for (int i = 0; i < NCC_TOP_K; i++) top_costs[i] = 2.0f;

    for (int s = 0; s < num_sources; s++) {
        const GPUSourceView& sv = src_views[s];

        float sum_ref = 0.0f, sum_src = 0.0f;
        float sum_ref2 = 0.0f, sum_src2 = 0.0f;
        float sum_rs = 0.0f;
        int count = 0;

        for (int dy = -patch_radius; dy <= patch_radius; dy++) {
            for (int dx = -patch_radius; dx <= patch_radius; dx++) {
                // Reference value from shared memory
                const float ref_val = s_ref[(ty + patch_radius + dy) * NCC_TILE_DIM
                                            + (tx + patch_radius + dx)];

                // Warp patch pixel to source view using precomputed A, b
                const float wpx = fx + static_cast<float>(dx);
                const float wpy = fy + static_cast<float>(dy);

                // q = A * [wpx, wpy, 1]
                const float qx = sv.A[0] * wpx + sv.A[1] * wpy + sv.A[2];
                const float qy = sv.A[3] * wpx + sv.A[4] * wpy + sv.A[5];
                const float qz = sv.A[6] * wpx + sv.A[7] * wpy + sv.A[8];

                // Homography correction: factor = dot(nK, [wpx, wpy, 1]) / n_dot_P
                const float sval = nK0 * wpx + nK1 * wpy + nK2;
                const float factor = sval * inv_n_dot_P;

                // warped = q + b * factor
                const float wz = qz + sv.b[2] * factor;
                if (fabsf(wz) < 1e-8f) continue;
                const float inv_wz = 1.0f / wz;
                const float src_u = (qx + sv.b[0] * factor) * inv_wz;
                const float src_v = (qy + sv.b[1] * factor) * inv_wz;

                // Bounds check
                if (src_u < 0.0f || src_u >= fwidth - 1.0f ||
                    src_v < 0.0f || src_v >= fheight - 1.0f) continue;

                // Sample source texture (+0.5 for CUDA texel center offset)
                const float src_val = tex2D<float>(src_images[s], src_u + 0.5f, src_v + 0.5f);

                sum_ref += ref_val;
                sum_src += src_val;
                sum_ref2 += ref_val * ref_val;
                sum_src2 += src_val * src_val;
                sum_rs += ref_val * src_val;
                count++;
            }
        }

        float c;
        if (count < min_count) {
            c = 2.0f;
        } else {
            const float fc = static_cast<float>(count);
            const float num = fc * sum_rs - sum_ref * sum_src;
            const float den_r = fc * sum_ref2 - sum_ref * sum_ref;
            const float den_s = fc * sum_src2 - sum_src * sum_src;
            if (den_r < 1e-6f || den_s < 1e-6f) {
                c = 2.0f; // uniform patch (no contrast)
            } else {
                float ncc = num / (sqrtf(den_r) * sqrtf(den_s));
                ncc = fmaxf(-1.0f, fminf(1.0f, ncc));
                c = 1.0f - ncc; // [0, 2], lower = better
            }
        }

        // Insert into top-K sorted array
        if (c < top_costs[NCC_TOP_K - 1]) {
            top_costs[NCC_TOP_K - 1] = c;
            #pragma unroll
            for (int i = NCC_TOP_K - 2; i >= 0; i--) {
                if (top_costs[i + 1] < top_costs[i]) {
                    float tmp = top_costs[i];
                    top_costs[i] = top_costs[i + 1];
                    top_costs[i + 1] = tmp;
                } else break;
            }
        }
    }

    // Mean of top-K costs
    int k = min(NCC_TOP_K, num_sources);
    float sum = 0.0f;
    for (int i = 0; i < k; i++) sum += top_costs[i];
    cost[idx] = sum / static_cast<float>(k);
}

// ============================================================================
// CUDA Kernel 3: Propagation & Random Refinement (red-black checkerboard)
// ============================================================================

__global__ void propagate_and_refine_kernel(
    cudaTextureObject_t ref_image,
    const cudaTextureObject_t* __restrict__ src_images,
    float* __restrict__ depth,
    float* __restrict__ normal,
    float* __restrict__ cost,
    const float* __restrict__ ref_K_inv,
    const GPUSourceView* __restrict__ src_views,
    float depth_min, float depth_max,
    int patch_radius,
    int width, int height,
    int num_sources,
    int iteration,      // PatchMatch iteration index (controls perturbation range)
    int red_black,      // 0 = red pixels ((x+y)%2==0), 1 = black pixels
    unsigned long long seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Red-black checkerboard: only process matching parity
    if ((x + y) % 2 != red_black) return;

    const int idx = y * width + x;

    float best_depth = depth[idx];
    float best_nx = normal[idx * 3 + 0];
    float best_ny = normal[idx * 3 + 1];
    float best_nz = normal[idx * 3 + 2];
    float best_cost = cost[idx];

    // --- Spatial propagation: test 4-neighbor hypotheses ---
    const int dx4[4] = {-1, 1, 0, 0};
    const int dy4[4] = {0, 0, -1, 1};

    for (int n = 0; n < 4; n++) {
        int nx_coord = x + dx4[n];
        int ny_coord = y + dy4[n];
        if (nx_coord < 0 || nx_coord >= width ||
            ny_coord < 0 || ny_coord >= height) continue;

        int nidx = ny_coord * width + nx_coord;
        float nd = depth[nidx];
        float nnx = normal[nidx * 3 + 0];
        float nny = normal[nidx * 3 + 1];
        float nnz = normal[nidx * 3 + 2];

        if (nd < depth_min || nd > depth_max) continue;

        float c = computePixelCost(ref_image, src_images,
            nd, nnx, nny, nnz,
            ref_K_inv, src_views,
            x, y, patch_radius, width, height, num_sources);

        if (c < best_cost) {
            best_cost = c;
            best_depth = nd;
            best_nx = nnx;
            best_ny = nny;
            best_nz = nnz;
        }
    }

    // --- Multi-scale random refinement ---
    // Standard PatchMatch: try multiple perturbation levels per visit,
    // each with geometrically shrinking range for finer convergence.
    curandStatePhilox4_32_10_t rng;
    unsigned long long pixel_seed = seed + static_cast<unsigned long long>(idx) * 17ull;
    curand_init(pixel_seed, static_cast<unsigned long long>(iteration), 0, &rng);

    // Precompute view ray for normal flipping (done once, reused across refinement steps)
    const float ki0 = ref_K_inv[0], ki1 = ref_K_inv[1], ki2 = ref_K_inv[2];
    const float ki3 = ref_K_inv[3], ki4 = ref_K_inv[4], ki5 = ref_K_inv[5];
    const float ki6 = ref_K_inv[6], ki7 = ref_K_inv[7], ki8 = ref_K_inv[8];
    float rx = ki0 * x + ki1 * y + ki2;
    float ry = ki3 * x + ki4 * y + ki5;
    float rz = ki6 * x + ki7 * y + ki8;
    float ray_len = sqrtf(rx*rx + ry*ry + rz*rz);
    if (ray_len > 1e-12f) { rx /= ray_len; ry /= ray_len; rz /= ray_len; }

    // Start range from coarse, shrink by 0.5 per refinement step
    constexpr int NUM_REFINE_STEPS = 2;
    float depth_range = (depth_max - depth_min) * 0.5f;
    float normal_range = 1.0f;
    // Shrink initial range by iteration (outer PatchMatch loop)
    for (int i = 0; i < iteration; i++) {
        depth_range *= 0.6f;
        normal_range *= 0.6f;
    }

    for (int step = 0; step < NUM_REFINE_STEPS; step++) {
        // Perturb depth
        float rand_d = curand_uniform(&rng) * 2.0f - 1.0f;
        float cand_depth = best_depth + rand_d * depth_range;
        cand_depth = fmaxf(depth_min, fminf(depth_max, cand_depth));

        // Perturb normal
        float cand_nx = best_nx + (curand_uniform(&rng) * 2.0f - 1.0f) * normal_range;
        float cand_ny = best_ny + (curand_uniform(&rng) * 2.0f - 1.0f) * normal_range;
        float cand_nz = best_nz + (curand_uniform(&rng) * 2.0f - 1.0f) * normal_range;

        // Normalize
        float nlen = sqrtf(cand_nx*cand_nx + cand_ny*cand_ny + cand_nz*cand_nz);
        if (nlen > 1e-12f) {
            cand_nx /= nlen; cand_ny /= nlen; cand_nz /= nlen;
        }

        // Ensure normal faces camera
        if (cand_nx * rx + cand_ny * ry + cand_nz * rz > 0.0f) {
            cand_nx = -cand_nx; cand_ny = -cand_ny; cand_nz = -cand_nz;
        }

        float c = computePixelCost(ref_image, src_images,
            cand_depth, cand_nx, cand_ny, cand_nz,
            ref_K_inv, src_views,
            x, y, patch_radius, width, height, num_sources);

        if (c < best_cost) {
            best_cost = c;
            best_depth = cand_depth;
            best_nx = cand_nx;
            best_ny = cand_ny;
            best_nz = cand_nz;
        }

        // Shrink perturbation range for next refinement step
        depth_range *= 0.5f;
        normal_range *= 0.5f;
    }

    // Write back
    depth[idx] = best_depth;
    normal[idx * 3 + 0] = best_nx;
    normal[idx * 3 + 1] = best_ny;
    normal[idx * 3 + 2] = best_nz;
    cost[idx] = best_cost;
}

// ============================================================================
// CUDA Kernel 4: Geometric Consistency Filter
// ============================================================================

// Per-source-view camera params for the geometric filter
struct GeomFilterSrcView {
    float K[9];      // source intrinsics (row-major)
    float R[9];      // source rotation, world-to-cam (row-major)
    float t[3];      // source translation
};

__global__ void geometric_consistency_filter_kernel(
    float* __restrict__ ref_depth,         // H × W, modified in-place (0 = invalid)
    const float* __restrict__ ref_normal,  // H × W × 3, camera frame
    float* __restrict__ ref_confidence,    // H × W, output
    const float* __restrict__ ref_K_inv,   // 3×3 row-major
    const float* __restrict__ ref_R,       // 3×3 row-major (world-to-cam)
    const float* __restrict__ ref_t,       // 3-vec
    const float* __restrict__ src_depths_packed,   // [num_sources × H × W]
    const float* __restrict__ src_normals_packed,  // [num_sources × H × W × 3]
    const GeomFilterSrcView* __restrict__ src_views,
    float depth_tolerance,       // relative, e.g. 0.02 = 2%
    float normal_tolerance_cos,  // cos(angle_tolerance)
    int min_consistent,
    int num_sources,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float d = ref_depth[idx];
    if (d <= 0.0f) {
        ref_confidence[idx] = 0.0f;
        return;
    }

    // Reference K_inv
    const float ki0 = ref_K_inv[0], ki1 = ref_K_inv[1], ki2 = ref_K_inv[2];
    const float ki3 = ref_K_inv[3], ki4 = ref_K_inv[4], ki5 = ref_K_inv[5];
    const float ki6 = ref_K_inv[6], ki7 = ref_K_inv[7], ki8 = ref_K_inv[8];

    // Backproject to camera frame: P_cam = d * K_inv * [x, y, 1]
    const float fx = static_cast<float>(x);
    const float fy = static_cast<float>(y);
    float pcx = d * (ki0 * fx + ki1 * fy + ki2);
    float pcy = d * (ki3 * fx + ki4 * fy + ki5);
    float pcz = d * (ki6 * fx + ki7 * fy + ki8);

    // To world: P_world = R^T * (P_cam - t)
    float px_mt = pcx - ref_t[0];
    float py_mt = pcy - ref_t[1];
    float pz_mt = pcz - ref_t[2];
    // R^T multiply: columns of R become rows
    float pwx = ref_R[0]*px_mt + ref_R[3]*py_mt + ref_R[6]*pz_mt;
    float pwy = ref_R[1]*px_mt + ref_R[4]*py_mt + ref_R[7]*pz_mt;
    float pwz = ref_R[2]*px_mt + ref_R[5]*py_mt + ref_R[8]*pz_mt;

    // Reference normal: camera frame → world: n_world = R^T * n_cam
    float ncx = ref_normal[idx * 3 + 0];
    float ncy = ref_normal[idx * 3 + 1];
    float ncz = ref_normal[idx * 3 + 2];
    float nwx = ref_R[0]*ncx + ref_R[3]*ncy + ref_R[6]*ncz;
    float nwy = ref_R[1]*ncx + ref_R[4]*ncy + ref_R[7]*ncz;
    float nwz = ref_R[2]*ncx + ref_R[5]*ncy + ref_R[8]*ncz;

    int consistent_count = 0;
    const int pixels_per_view = width * height;

    for (int s = 0; s < num_sources; s++) {
        const GeomFilterSrcView& sv = src_views[s];

        // Project to source camera: P_src_cam = R_s * P_world + t_s
        float psx = sv.R[0]*pwx + sv.R[1]*pwy + sv.R[2]*pwz + sv.t[0];
        float psy = sv.R[3]*pwx + sv.R[4]*pwy + sv.R[5]*pwz + sv.t[1];
        float psz = sv.R[6]*pwx + sv.R[7]*pwy + sv.R[8]*pwz + sv.t[2];

        if (psz <= 0.0f) continue;

        // Project to pixel: p = K_s * P_src_cam
        float proj_x = sv.K[0]*psx + sv.K[1]*psy + sv.K[2]*psz;
        float proj_y = sv.K[3]*psx + sv.K[4]*psy + sv.K[5]*psz;
        float proj_z = sv.K[6]*psx + sv.K[7]*psy + sv.K[8]*psz;

        float u = proj_x / proj_z;
        float v = proj_y / proj_z;

        int ui = __float2int_rn(u);
        int vi = __float2int_rn(v);
        if (ui < 0 || ui >= width || vi < 0 || vi >= height) continue;

        // Read source depth
        int src_pix = vi * width + ui;
        float d_src = src_depths_packed[s * pixels_per_view + src_pix];
        if (d_src <= 0.0f) continue;

        // Depth consistency: |d_src - expected_z| / expected_z < tolerance
        float d_expected = psz;
        float depth_err = fabsf(d_src - d_expected) / d_expected;
        if (depth_err > depth_tolerance) continue;

        // Normal consistency: transform source normal to world and compare
        int src_n_base = (s * pixels_per_view + src_pix) * 3;
        float nsx = src_normals_packed[src_n_base + 0];
        float nsy = src_normals_packed[src_n_base + 1];
        float nsz = src_normals_packed[src_n_base + 2];

        // n_src_world = R_s^T * n_src_cam
        float nswx = sv.R[0]*nsx + sv.R[3]*nsy + sv.R[6]*nsz;
        float nswy = sv.R[1]*nsx + sv.R[4]*nsy + sv.R[7]*nsz;
        float nswz = sv.R[2]*nsx + sv.R[5]*nsy + sv.R[8]*nsz;

        float cos_angle = nwx*nswx + nwy*nswy + nwz*nswz;
        if (cos_angle < normal_tolerance_cos) continue;

        consistent_count++;
    }

    if (consistent_count < min_consistent) {
        ref_depth[idx] = 0.0f;
        ref_confidence[idx] = 0.0f;
    }
    // Retained pixels keep their NCC-based confidence (not overwritten)
}

// ============================================================================
// Texture object helpers
// ============================================================================

// Create a CUDA texture object from a grayscale float image (host data).
// Returns the texture object and the backing cudaArray (caller must free both).
static void createGrayscaleTexture(const float* host_data, int width, int height,
                                    cudaTextureObject_t& tex_obj, cudaArray_t& cuda_array) {
    // Allocate CUDA array
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    CUDA_CHECK(cudaMallocArray(&cuda_array, &desc, width, height));

    // Copy data to array
    CUDA_CHECK(cudaMemcpy2DToArray(cuda_array, 0, 0, host_data,
                                    width * sizeof(float),
                                    width * sizeof(float), height,
                                    cudaMemcpyHostToDevice));

    // Create texture object
    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = cuda_array;

    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0] = cudaAddressModeClamp;
    tex_desc.addressMode[1] = cudaAddressModeClamp;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = 0;

    CUDA_CHECK(cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr));
}

static void destroyTexture(cudaTextureObject_t tex_obj, cudaArray_t cuda_array) {
    if (tex_obj) cudaDestroyTextureObject(tex_obj);
    if (cuda_array) cudaFreeArray(cuda_array);
}

// ============================================================================
// Host-side helpers: camera conversion & source view precomputation
// ============================================================================

struct GPUCameraParams {
    float K[9];      // 3x3 row-major intrinsics
    float K_inv[9];  // 3x3 row-major inverse intrinsics
    float R[9];      // 3x3 row-major rotation (world-to-camera)
    float t[3];      // translation (world-to-camera)
};

static GPUCameraParams eigenToGPUCamera(const CameraPose& cam) {
    GPUCameraParams g;
    Eigen::Matrix3d K_inv = cam.K.inverse();
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            g.K[r * 3 + c]     = static_cast<float>(cam.K(r, c));
            g.K_inv[r * 3 + c] = static_cast<float>(K_inv(r, c));
            g.R[r * 3 + c]     = static_cast<float>(cam.R(r, c));
        }
        g.t[r] = static_cast<float>(cam.t(r));
    }
    return g;
}

// Precompute A = K_src * R_rel * K_ref_inv and b = K_src * t_rel for each source
static std::vector<GPUSourceView> precomputeSourceViews(
    const GPUCameraParams& ref_cam,
    const std::vector<GPUCameraParams>& src_cams)
{
    Matrix3x3 R_ref, K_ref_inv;
    for (int i = 0; i < 9; i++) {
        R_ref.m[i] = ref_cam.R[i];
        K_ref_inv.m[i] = ref_cam.K_inv[i];
    }
    Matrix3x3 R_ref_T = R_ref.transpose();
    float3 t_ref = make_float3(ref_cam.t[0], ref_cam.t[1], ref_cam.t[2]);

    std::vector<GPUSourceView> result(src_cams.size());
    for (size_t s = 0; s < src_cams.size(); s++) {
        Matrix3x3 K_s, R_s;
        for (int i = 0; i < 9; i++) {
            K_s.m[i] = src_cams[s].K[i];
            R_s.m[i] = src_cams[s].R[i];
        }
        float3 t_s = make_float3(src_cams[s].t[0], src_cams[s].t[1], src_cams[s].t[2]);

        // R_rel = R_s * R_ref^T
        Matrix3x3 R_rel = R_s * R_ref_T;
        // t_rel = t_s - R_rel * t_ref
        float3 t_rel = t_s - R_rel * t_ref;

        // A = K_s * R_rel * K_ref_inv
        Matrix3x3 A = K_s * (R_rel * K_ref_inv);
        // b = K_s * t_rel
        float3 b_vec = K_s * t_rel;

        for (int i = 0; i < 9; i++) result[s].A[i] = A.m[i];
        result[s].b[0] = b_vec.x;
        result[s].b[1] = b_vec.y;
        result[s].b[2] = b_vec.z;
    }
    return result;
}

// ============================================================================
// Kernel launch wrappers
// ============================================================================

static void launchInitKernel(float* d_depth, float* d_normal,
                              float depth_min, float depth_max,
                              const float* d_K_inv,
                              int width, int height,
                              unsigned long long seed) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    init_depth_normal_kernel<<<grid, block>>>(
        d_depth, d_normal, depth_min, depth_max,
        d_K_inv, width, height, seed);
    CUDA_CHECK_KERNEL();
}

static void launchNCCCostKernel(
    cudaTextureObject_t ref_tex,
    const std::vector<cudaTextureObject_t>& src_textures,
    float* d_depth, float* d_normal,
    const float* d_K_inv,
    const std::vector<GPUSourceView>& src_views_host,
    float* d_cost,
    int patch_radius,
    int width, int height)
{
    int num_sources = static_cast<int>(src_textures.size());

    // Upload source texture object handles and precomputed view params
    DeviceBuffer<cudaTextureObject_t> d_src_tex;
    d_src_tex.upload(src_textures.data(), num_sources);

    DeviceBuffer<GPUSourceView> d_src_views;
    d_src_views.upload(src_views_host.data(), num_sources);

    dim3 block(NCC_BLOCK_DIM, NCC_BLOCK_DIM);
    dim3 grid((width + NCC_BLOCK_DIM - 1) / NCC_BLOCK_DIM,
              (height + NCC_BLOCK_DIM - 1) / NCC_BLOCK_DIM);

    compute_ncc_cost_kernel<<<grid, block>>>(
        ref_tex, d_src_tex.ptr,
        d_depth, d_normal,
        d_K_inv, d_src_views.ptr,
        d_cost, patch_radius,
        width, height, num_sources);
    CUDA_CHECK_KERNEL();
}

static void launchPropagateRefineKernel(
    cudaTextureObject_t ref_tex,
    const cudaTextureObject_t* d_src_tex,
    float* d_depth, float* d_normal, float* d_cost,
    const float* d_K_inv,
    const GPUSourceView* d_src_views,
    float depth_min, float depth_max,
    int patch_radius,
    int width, int height,
    int num_sources,
    int iteration,
    int red_black,
    unsigned long long seed)
{
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    propagate_and_refine_kernel<<<grid, block>>>(
        ref_tex, d_src_tex,
        d_depth, d_normal, d_cost,
        d_K_inv, d_src_views,
        depth_min, depth_max,
        patch_radius, width, height, num_sources,
        iteration, red_black, seed);
    CUDA_CHECK_KERNEL();
}

static void launchGeometricFilterKernel(
    float* d_ref_depth, const float* d_ref_normal, float* d_ref_confidence,
    const float* d_ref_K_inv, const float* d_ref_R, const float* d_ref_t,
    const float* d_src_depths_packed, const float* d_src_normals_packed,
    const GeomFilterSrcView* d_src_views,
    float depth_tolerance, float normal_tolerance_cos,
    int min_consistent, int num_sources,
    int width, int height)
{
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    geometric_consistency_filter_kernel<<<grid, block>>>(
        d_ref_depth, d_ref_normal, d_ref_confidence,
        d_ref_K_inv, d_ref_R, d_ref_t,
        d_src_depths_packed, d_src_normals_packed,
        d_src_views,
        depth_tolerance, normal_tolerance_cos,
        min_consistent, num_sources,
        width, height);
    CUDA_CHECK_KERNEL();
}

// ============================================================================
// PFM save/load for checkpointing
// ============================================================================

bool saveDepthMapPFM(const DepthMap& dm, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    // PFM header: Pf = grayscale float
    f << "Pf\n" << dm.width << " " << dm.height << "\n";
    f << "-1.0\n"; // little-endian

    // PFM stores bottom-to-top
    for (int y = dm.height - 1; y >= 0; y--) {
        const float* row = dm.depth.ptr<float>(y);
        f.write(reinterpret_cast<const char*>(row), dm.width * sizeof(float));
    }
    return f.good();
}

bool loadDepthMapPFM(DepthMap& dm, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    std::string magic;
    f >> magic;
    if (magic != "Pf") return false;
    f >> dm.width >> dm.height;
    float scale;
    f >> scale;
    f.get();
    dm.depth = cv::Mat(dm.height, dm.width, CV_32F);
    for (int y = dm.height - 1; y >= 0; y--) {
        float* row = dm.depth.ptr<float>(y);
        f.read(reinterpret_cast<char*>(row), dm.width * sizeof(float));
    }
    return f.good();
}

// ============================================================================
// Top-level API: runPatchMatchMVS
// ============================================================================

std::vector<DepthMap> runPatchMatchMVS(
    const std::vector<ImageData>& images,
    const std::vector<CameraPose>& cameras,
    const SfMResult& sfm,
    const PipelineConfig& config)
{
    printf("[MVS] Starting PatchMatch MVS on %d cameras, %d iterations\n",
           sfm.num_registered, config.mvs_iterations);

    // 1. Select source views
    int num_sources = std::min(MAX_SRC_VIEWS, std::max(3, sfm.num_registered - 1));
    auto view_params = selectSourceViews(sfm, num_sources);
    printf("[MVS] Source view selection done for %zu reference views\n", view_params.size());

    // 2. Create texture objects for all images upfront
    int num_images = static_cast<int>(images.size());
    std::vector<cudaTextureObject_t> all_textures(num_images, 0);
    std::vector<cudaArray_t> all_arrays(num_images, nullptr);

    for (int i = 0; i < num_images; i++) {
        if (!cameras[i].is_registered) continue;
        const cv::Mat& gray = images[i].gray;
        if (gray.empty()) continue;
        createGrayscaleTexture(
            reinterpret_cast<const float*>(gray.data),
            gray.cols, gray.rows,
            all_textures[i], all_arrays[i]);
    }

    // 3. Precompute GPU camera params for all cameras
    std::vector<GPUCameraParams> gpu_cameras(num_images);
    for (int i = 0; i < num_images; i++) {
        if (cameras[i].is_registered) {
            gpu_cameras[i] = eigenToGPUCamera(cameras[i]);
        }
    }

    int patch_radius = std::max(1, (config.mvs_patch_size - 1) / 2);
    if (patch_radius > NCC_PATCH_RADIUS) {
        printf("[MVS] Warning: requested patch_radius %d exceeds max %d, clamping\n",
               patch_radius, NCC_PATCH_RADIUS);
        patch_radius = NCC_PATCH_RADIUS;
    }

    std::vector<DepthMap> depth_maps;
    depth_maps.reserve(view_params.size());

    // 4. Process each reference view
    for (size_t vi = 0; vi < view_params.size(); vi++) {
        const auto& vp = view_params[vi];
        int ref_id = vp.ref_image_id;
        const auto& ref_img = images[ref_id];
        int w = ref_img.width;
        int h = ref_img.height;

        if (config.verbose) {
            printf("[MVS] Processing view %d/%zu (image %d): %dx%d, depth=[%.4f, %.4f], %zu sources\n",
                   static_cast<int>(vi + 1), view_params.size(), ref_id, w, h,
                   vp.depth_min, vp.depth_max, vp.source_views.size());
        }

        if (vp.source_views.empty()) {
            printf("[MVS] Warning: no source views for image %d, skipping\n", ref_id);
            continue;
        }

        // Allocate GPU buffers
        size_t num_pixels = static_cast<size_t>(w) * h;
        DeviceBuffer<float> d_depth, d_normal, d_cost;
        d_depth.alloc(num_pixels);
        d_normal.alloc(num_pixels * 3);
        d_cost.alloc(num_pixels);

        // Upload reference camera K_inv
        DeviceBuffer<float> d_K_inv;
        d_K_inv.upload(gpu_cameras[ref_id].K_inv, 9);

        // Initialize depth and normal maps (random)
        unsigned long long seed = 42ull + static_cast<unsigned long long>(ref_id) * 1000000ull;
        launchInitKernel(d_depth.ptr, d_normal.ptr,
                         vp.depth_min, vp.depth_max,
                         d_K_inv.ptr, w, h, seed);

        // SfM-seeded initialization: project sparse points into this view
        // and overwrite random depths with known accurate depths + dilate
        {
            std::vector<float> h_depth(num_pixels);
            std::vector<float> h_normal(num_pixels * 3);
            CUDA_CHECK(cudaMemcpy(h_depth.data(), d_depth.ptr,
                                   num_pixels * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_normal.data(), d_normal.ptr,
                                   num_pixels * 3 * sizeof(float), cudaMemcpyDeviceToHost));

            const auto& cam = cameras[ref_id];
            Eigen::Matrix3d K = cam.K;
            int seeded = 0;
            constexpr int SEED_RADIUS = 50;  // large seed radius — flat-depth error is negligible for smooth surfaces

            for (const auto& pt : sfm.points) {
                // Check if this point is visible from this camera
                bool visible = false;
                for (int id : pt.image_ids) {
                    if (id == ref_id) { visible = true; break; }
                }
                if (!visible) continue;

                // Project to camera frame
                Eigen::Vector3d p_cam = cam.R * pt.position + cam.t;
                if (p_cam.z() <= 0) continue;

                // Project to pixel
                Eigen::Vector3d p_px = K * p_cam;
                int px = static_cast<int>(std::round(p_px.x() / p_px.z()));
                int py = static_cast<int>(std::round(p_px.y() / p_px.z()));

                float depth_val = static_cast<float>(p_cam.z());
                if (depth_val < vp.depth_min || depth_val > vp.depth_max) continue;

                // Compute normal toward camera at this point
                Eigen::Vector3d cam_center = cam.cam_center();
                Eigen::Vector3d to_cam = (cam_center - pt.position).normalized();
                // Transform normal to camera frame
                Eigen::Vector3d n_cam = cam.R * to_cam;
                float nx = static_cast<float>(n_cam.x());
                float ny = static_cast<float>(n_cam.y());
                float nz = static_cast<float>(n_cam.z());

                // Seed a small patch around the projected point
                for (int dy = -SEED_RADIUS; dy <= SEED_RADIUS; dy++) {
                    for (int dx = -SEED_RADIUS; dx <= SEED_RADIUS; dx++) {
                        int sx = px + dx;
                        int sy = py + dy;
                        if (sx < 0 || sx >= w || sy < 0 || sy >= h) continue;
                        int sidx = sy * w + sx;
                        h_depth[sidx] = depth_val;
                        h_normal[sidx * 3 + 0] = nx;
                        h_normal[sidx * 3 + 1] = ny;
                        h_normal[sidx * 3 + 2] = nz;
                        seeded++;
                    }
                }
            }

            if (seeded > 0) {
                CUDA_CHECK(cudaMemcpy(d_depth.ptr, h_depth.data(),
                                       num_pixels * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_normal.ptr, h_normal.data(),
                                       num_pixels * 3 * sizeof(float), cudaMemcpyHostToDevice));
            }
        }

        // Collect source textures and precompute homography params
        std::vector<cudaTextureObject_t> src_textures;
        std::vector<GPUCameraParams> src_gpu_cams;
        src_textures.reserve(vp.source_views.size());
        src_gpu_cams.reserve(vp.source_views.size());
        for (const auto& sv : vp.source_views) {
            src_textures.push_back(all_textures[sv.image_id]);
            src_gpu_cams.push_back(gpu_cameras[sv.image_id]);
        }

        auto src_views = precomputeSourceViews(gpu_cameras[ref_id], src_gpu_cams);

        // Compute initial NCC cost
        launchNCCCostKernel(
            all_textures[ref_id], src_textures,
            d_depth.ptr, d_normal.ptr,
            d_K_inv.ptr, src_views,
            d_cost.ptr, patch_radius, w, h);

        // PatchMatch iteration loop: propagate + refine (red-black checkerboard)
        if (config.mvs_iterations > 0) {
            int num_src = static_cast<int>(src_textures.size());

            // Upload source data once, reuse across all iterations
            DeviceBuffer<cudaTextureObject_t> d_src_tex;
            d_src_tex.upload(src_textures.data(), num_src);
            DeviceBuffer<GPUSourceView> d_src_views;
            d_src_views.upload(src_views.data(), num_src);

            for (int iter = 0; iter < config.mvs_iterations; iter++) {
                // Red pass: pixels where (x+y)%2 == 0
                launchPropagateRefineKernel(
                    all_textures[ref_id], d_src_tex.ptr,
                    d_depth.ptr, d_normal.ptr, d_cost.ptr,
                    d_K_inv.ptr, d_src_views.ptr,
                    vp.depth_min, vp.depth_max,
                    patch_radius, w, h, num_src,
                    iter, 0, seed + 100ull * iter);

                // Black pass: pixels where (x+y)%2 == 1
                launchPropagateRefineKernel(
                    all_textures[ref_id], d_src_tex.ptr,
                    d_depth.ptr, d_normal.ptr, d_cost.ptr,
                    d_K_inv.ptr, d_src_views.ptr,
                    vp.depth_min, vp.depth_max,
                    patch_radius, w, h, num_src,
                    iter, 1, seed + 100ull * iter + 50ull);

                if (config.verbose) {
                    printf("[MVS]   Iteration %d/%d done\n",
                           iter + 1, config.mvs_iterations);
                }
            }
        }

        // Download results
        DepthMap dm;
        dm.image_id = ref_id;
        dm.width = w;
        dm.height = h;
        dm.depth = cv::Mat(h, w, CV_32F);
        dm.normal = cv::Mat(h, w, CV_32FC3);
        dm.confidence = cv::Mat(h, w, CV_32F, cv::Scalar(0.0f));

        d_depth.download(reinterpret_cast<float*>(dm.depth.data), num_pixels);
        d_normal.download(reinterpret_cast<float*>(dm.normal.data), num_pixels * 3);

        // Download cost into confidence (inverted: confidence = 1 - cost/2)
        std::vector<float> cost_host(num_pixels);
        d_cost.download(cost_host.data(), num_pixels);
        for (size_t p = 0; p < num_pixels; p++) {
            dm.confidence.at<float>(static_cast<int>(p / w), static_cast<int>(p % w))
                = 1.0f - cost_host[p] * 0.5f;
        }

        if (config.save_intermediate && !config.output_path.empty()) {
            std::string dir = config.output_path.substr(0, config.output_path.find_last_of('/'));
            saveDepthMapPFM(dm, dir + "/depth_" + std::to_string(ref_id) + ".pfm");
        }

        depth_maps.push_back(std::move(dm));
    }

    // 5. Cleanup textures
    for (int i = 0; i < num_images; i++) {
        destroyTexture(all_textures[i], all_arrays[i]);
    }

    // 6. Geometric consistency filter across all depth maps
    //    Only run if iterations > 0 (init-only depth maps are random noise)
    if (config.mvs_iterations > 0 && depth_maps.size() >= 2) {
        printf("[MVS] Running geometric consistency filter on %zu depth maps...\n",
               depth_maps.size());

        static constexpr float GEOM_DEPTH_TOL = 0.12f;        // 12% relative
        static constexpr float GEOM_NORMAL_TOL_DEG = 180.0f;  // disabled (normals converge slowly)
        static constexpr int   GEOM_MIN_CONSISTENT = 2;
        float normal_tol_cos = std::cos(GEOM_NORMAL_TOL_DEG * static_cast<float>(M_PI) / 180.0f);

        // Build lookup: image_id → index in depth_maps
        std::vector<int> id_to_dm(num_images, -1);
        for (size_t i = 0; i < depth_maps.size(); i++) {
            id_to_dm[depth_maps[i].image_id] = static_cast<int>(i);
        }

        int total_retained = 0, total_pixels = 0;

        for (size_t vi = 0; vi < depth_maps.size(); vi++) {
            auto& dm = depth_maps[vi];
            int ref_id = dm.image_id;
            int w = dm.width;
            int h = dm.height;
            size_t num_pix = static_cast<size_t>(w) * h;

            // Find source views for this reference
            const MVSViewParams* vp_ptr = nullptr;
            for (const auto& vp : view_params) {
                if (vp.ref_image_id == ref_id) { vp_ptr = &vp; break; }
            }
            if (!vp_ptr || vp_ptr->source_views.empty()) continue;

            // Gather source views that have computed depth maps with matching dimensions
            std::vector<GeomFilterSrcView> filter_src;
            std::vector<float> packed_depths;
            std::vector<float> packed_normals;

            for (const auto& sv : vp_ptr->source_views) {
                if (sv.image_id < 0 || sv.image_id >= num_images) continue;
                int dm_idx = id_to_dm[sv.image_id];
                if (dm_idx < 0) continue;
                const auto& src_dm = depth_maps[dm_idx];
                if (src_dm.width != w || src_dm.height != h) continue;

                GeomFilterSrcView fsv;
                for (int r = 0; r < 9; r++) {
                    fsv.K[r] = gpu_cameras[sv.image_id].K[r];
                    fsv.R[r] = gpu_cameras[sv.image_id].R[r];
                }
                for (int r = 0; r < 3; r++) {
                    fsv.t[r] = gpu_cameras[sv.image_id].t[r];
                }
                filter_src.push_back(fsv);

                const float* d_data = reinterpret_cast<const float*>(src_dm.depth.data);
                packed_depths.insert(packed_depths.end(), d_data, d_data + num_pix);

                const float* n_data = reinterpret_cast<const float*>(src_dm.normal.data);
                packed_normals.insert(packed_normals.end(), n_data, n_data + num_pix * 3);
            }

            int num_filter_src = static_cast<int>(filter_src.size());
            if (num_filter_src < GEOM_MIN_CONSISTENT) continue;

            // Upload to GPU
            DeviceBuffer<float> d_depth_buf, d_normal_buf, d_conf_buf;
            d_depth_buf.upload(reinterpret_cast<const float*>(dm.depth.data), num_pix);
            d_normal_buf.upload(reinterpret_cast<const float*>(dm.normal.data), num_pix * 3);
            d_conf_buf.upload(reinterpret_cast<const float*>(dm.confidence.data), num_pix);

            DeviceBuffer<float> d_K_inv, d_R, d_t;
            d_K_inv.upload(gpu_cameras[ref_id].K_inv, 9);
            d_R.upload(gpu_cameras[ref_id].R, 9);
            d_t.upload(gpu_cameras[ref_id].t, 3);

            DeviceBuffer<float> d_src_depths, d_src_normals;
            d_src_depths.upload(packed_depths.data(), packed_depths.size());
            d_src_normals.upload(packed_normals.data(), packed_normals.size());

            DeviceBuffer<GeomFilterSrcView> d_filter_views;
            d_filter_views.upload(filter_src.data(), num_filter_src);

            launchGeometricFilterKernel(
                d_depth_buf.ptr, d_normal_buf.ptr, d_conf_buf.ptr,
                d_K_inv.ptr, d_R.ptr, d_t.ptr,
                d_src_depths.ptr, d_src_normals.ptr,
                d_filter_views.ptr,
                GEOM_DEPTH_TOL, normal_tol_cos,
                GEOM_MIN_CONSISTENT, num_filter_src,
                w, h);

            // Download filtered results
            d_depth_buf.download(reinterpret_cast<float*>(dm.depth.data), num_pix);
            d_conf_buf.download(reinterpret_cast<float*>(dm.confidence.data), num_pix);

            // Count retained pixels
            int retained = 0;
            for (size_t p = 0; p < num_pix; p++) {
                if (dm.depth.at<float>(static_cast<int>(p / w), static_cast<int>(p % w)) > 0.0f)
                    retained++;
            }
            total_retained += retained;
            total_pixels += static_cast<int>(num_pix);

            if (config.verbose) {
                printf("[MVS] Filter view %d: %d/%zu pixels retained (%.1f%%)\n",
                       ref_id, retained, num_pix,
                       100.0f * retained / static_cast<float>(num_pix));
            }
        }

        if (total_pixels > 0) {
            printf("[MVS] Geometric filter: %d/%d total pixels retained (%.1f%%)\n",
                   total_retained, total_pixels,
                   100.0f * total_retained / static_cast<float>(total_pixels));
        }
    }

    printf("[MVS] PatchMatch MVS complete: %zu depth maps generated\n", depth_maps.size());
    return depth_maps;
}

// ============================================================================
// Legacy wrappers
// ============================================================================

DepthMap patchMatchMVS(const ImageData& /*ref_image*/,
                        const std::vector<ImageData>& /*src_images*/,
                        const CameraPose& /*ref_pose*/,
                        const std::vector<CameraPose>& /*src_poses*/,
                        int /*iterations*/, int /*patch_size*/) {
    DepthMap dm;
    dm.image_id = -1;
    dm.width = 0;
    dm.height = 0;
    return dm;
}

std::vector<DepthMap> runDenseReconstruction(const std::vector<ImageData>& /*images*/,
                                              const std::vector<CameraPose>& /*poses*/,
                                              const PipelineConfig& /*config*/) {
    return {};
}
