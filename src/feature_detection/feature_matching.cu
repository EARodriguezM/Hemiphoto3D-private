#include "feature_detection/feature_matching.h"
#include "utils/cuda_utils.cuh"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <queue>
#include <vector>
#include <set>

// ============================================================================
// Constants
// ============================================================================
static constexpr int TILE_SIZE = 32;       // descriptors per shared-memory tile
static constexpr int DESC_DIM  = 128;      // SIFT descriptor length
static constexpr int BLOCK_SIZE = 128;     // threads per block for KNN kernel

// ============================================================================
// Device helpers
// ============================================================================

// KNN match result: best two distances and their indices
struct KNNResult {
    float dist1, dist2;
    int   idx1,  idx2;
};

// ============================================================================
// CUDA Kernel: Brute-force KNN (K=2) with tiled shared memory
// ============================================================================
// Grid:  (num_queries,) — one thread per query descriptor
// Block: (BLOCK_SIZE,)
// Each thread finds the 2 nearest reference descriptors for its query.
__global__ void bf_knn_match_kernel(
    const float* __restrict__ d_query,   // [num_query x 128]
    const float* __restrict__ d_ref,     // [num_ref x 128]
    int num_query,
    int num_ref,
    float* __restrict__ d_dist1,   // [num_query]  best distance
    float* __restrict__ d_dist2,   // [num_query]  second-best distance
    int*   __restrict__ d_idx1,    // [num_query]  best index
    int*   __restrict__ d_idx2)    // [num_query]  second-best index
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;
    if (qid >= num_query) return;

    float best1 = 1e30f, best2 = 1e30f;
    int   bidx1 = -1,    bidx2 = -1;

    // Load query descriptor into registers
    float q[DESC_DIM];
    for (int d = 0; d < DESC_DIM; d++) {
        q[d] = d_query[qid * DESC_DIM + d];
    }

    // Tile over reference descriptors using shared memory
    __shared__ float s_ref[TILE_SIZE][DESC_DIM];

    int num_tiles = (num_ref + TILE_SIZE - 1) / TILE_SIZE;
    for (int tile = 0; tile < num_tiles; tile++) {
        // Cooperatively load tile of reference descriptors
        int tile_start = tile * TILE_SIZE;
        // Each thread loads multiple elements to fill shared memory
        for (int row = threadIdx.x; row < TILE_SIZE; row += blockDim.x) {
            int rid = tile_start + row;
            if (rid < num_ref) {
                for (int d = 0; d < DESC_DIM; d++) {
                    s_ref[row][d] = d_ref[rid * DESC_DIM + d];
                }
            }
        }
        __syncthreads();

        // Compute distances to all refs in this tile
        int tile_count = min(TILE_SIZE, num_ref - tile_start);
        for (int r = 0; r < tile_count; r++) {
            float dist = 0.0f;
            for (int d = 0; d < DESC_DIM; d++) {
                float diff = q[d] - s_ref[r][d];
                dist += diff * diff;
            }
            // dist is squared L2 — no sqrt needed for comparison

            int rid = tile_start + r;
            if (dist < best1) {
                best2 = best1; bidx2 = bidx1;
                best1 = dist;  bidx1 = rid;
            } else if (dist < best2) {
                best2 = dist;  bidx2 = rid;
            }
        }
        __syncthreads();
    }

    d_dist1[qid] = best1;
    d_dist2[qid] = best2;
    d_idx1[qid]  = bidx1;
    d_idx2[qid]  = bidx2;
}

// ============================================================================
// CUDA Kernel: Ratio test + cross-check combined
// ============================================================================
// For each query q, keep match if dist1/dist2 < ratio^2 (squared distances)
// and if the best reference's best match back is q (mutual nearest neighbor).
__global__ void ratio_crosscheck_kernel(
    const float* __restrict__ d_dist1_fwd,
    const float* __restrict__ d_dist2_fwd,
    const int*   __restrict__ d_idx1_fwd,
    const float* __restrict__ d_dist1_rev,
    const int*   __restrict__ d_idx1_rev,
    int num_query,
    float ratio_sq,
    int*   __restrict__ d_match_qi,    // output: query indices of surviving matches
    int*   __restrict__ d_match_ri,    // output: ref indices of surviving matches
    float* __restrict__ d_match_dist,  // output: distances
    int*   __restrict__ d_num_matches) // output: atomic counter
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;
    if (qid >= num_query) return;

    float d1 = d_dist1_fwd[qid];
    float d2 = d_dist2_fwd[qid];
    int   rid = d_idx1_fwd[qid];

    if (rid < 0) return;

    // Ratio test (on squared distances)
    if (d2 > 0.0f && d1 / d2 >= ratio_sq) return;

    // Cross-check: ref's best match must be this query
    if (d_idx1_rev[rid] != qid) return;

    // Survived both tests — add to output
    int slot = atomicAdd(d_num_matches, 1);
    d_match_qi[slot]   = qid;
    d_match_ri[slot]   = rid;
    d_match_dist[slot] = sqrtf(d1);  // store actual L2 distance
}

// ============================================================================
// Host: GPU KNN matching for one pair
// ============================================================================
static std::vector<FeatureMatch> gpuMatchPair(
    const cv::Mat& desc_i, const cv::Mat& desc_j,
    float ratio_thresh)
{
    int n_i = desc_i.rows;
    int n_j = desc_j.rows;

    if (n_i == 0 || n_j == 0) return {};

    // Upload descriptors
    DeviceBuffer<float> d_desc_i, d_desc_j;
    d_desc_i.upload(desc_i.ptr<float>(), n_i * DESC_DIM);
    d_desc_j.upload(desc_j.ptr<float>(), n_j * DESC_DIM);

    // Forward KNN: i -> j
    DeviceBuffer<float> d_dist1_fwd, d_dist2_fwd;
    DeviceBuffer<int>   d_idx1_fwd,  d_idx2_fwd;
    d_dist1_fwd.alloc(n_i); d_dist2_fwd.alloc(n_i);
    d_idx1_fwd.alloc(n_i);  d_idx2_fwd.alloc(n_i);

    int grid_fwd = (n_i + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bf_knn_match_kernel<<<grid_fwd, BLOCK_SIZE>>>(
        d_desc_i.ptr, d_desc_j.ptr, n_i, n_j,
        d_dist1_fwd.ptr, d_dist2_fwd.ptr,
        d_idx1_fwd.ptr, d_idx2_fwd.ptr);
    CUDA_CHECK_KERNEL();

    // Reverse KNN: j -> i
    DeviceBuffer<float> d_dist1_rev, d_dist2_rev;
    DeviceBuffer<int>   d_idx1_rev,  d_idx2_rev;
    d_dist1_rev.alloc(n_j); d_dist2_rev.alloc(n_j);
    d_idx1_rev.alloc(n_j);  d_idx2_rev.alloc(n_j);

    int grid_rev = (n_j + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bf_knn_match_kernel<<<grid_rev, BLOCK_SIZE>>>(
        d_desc_j.ptr, d_desc_i.ptr, n_j, n_i,
        d_dist1_rev.ptr, d_dist2_rev.ptr,
        d_idx1_rev.ptr, d_idx2_rev.ptr);
    CUDA_CHECK_KERNEL();

    // Ratio test + cross-check (fused kernel)
    int max_matches = n_i;  // at most one match per query
    DeviceBuffer<int>   d_match_qi, d_match_ri;
    DeviceBuffer<float> d_match_dist;
    DeviceBuffer<int>   d_num_matches;
    d_match_qi.alloc(max_matches);
    d_match_ri.alloc(max_matches);
    d_match_dist.alloc(max_matches);
    d_num_matches.alloc(1);
    CUDA_CHECK(cudaMemset(d_num_matches.ptr, 0, sizeof(int)));

    float ratio_sq = ratio_thresh * ratio_thresh;
    ratio_crosscheck_kernel<<<grid_fwd, BLOCK_SIZE>>>(
        d_dist1_fwd.ptr, d_dist2_fwd.ptr, d_idx1_fwd.ptr,
        d_dist1_rev.ptr, d_idx1_rev.ptr,
        n_i, ratio_sq,
        d_match_qi.ptr, d_match_ri.ptr, d_match_dist.ptr,
        d_num_matches.ptr);
    CUDA_CHECK_KERNEL();

    // Download results
    int num_matches = 0;
    d_num_matches.download(&num_matches, 1);

    if (num_matches == 0) return {};

    std::vector<int>   h_qi(num_matches), h_ri(num_matches);
    std::vector<float> h_dist(num_matches);
    d_match_qi.download(h_qi.data(), num_matches);
    d_match_ri.download(h_ri.data(), num_matches);
    d_match_dist.download(h_dist.data(), num_matches);

    std::vector<FeatureMatch> matches(num_matches);
    for (int i = 0; i < num_matches; i++) {
        matches[i].idx_i    = h_qi[i];
        matches[i].idx_j    = h_ri[i];
        matches[i].distance = h_dist[i];
    }
    return matches;
}

// ============================================================================
// Host: Geometric verification (RANSAC fundamental matrix)
// ============================================================================
static ImagePairMatches geometricVerification(
    const SIFTFeatures& feat_i, const SIFTFeatures& feat_j,
    const cv::Mat& K_i, const cv::Mat& K_j,
    const std::vector<FeatureMatch>& raw_matches,
    int min_inliers)
{
    ImagePairMatches result;
    result.image_i = feat_i.image_id;
    result.image_j = feat_j.image_id;
    result.num_inliers = 0;
    result.median_triangulation_angle = 0.0;

    if (static_cast<int>(raw_matches.size()) < min_inliers) return result;

    // Build point arrays
    std::vector<cv::Point2f> pts_i, pts_j;
    pts_i.reserve(raw_matches.size());
    pts_j.reserve(raw_matches.size());
    for (const auto& m : raw_matches) {
        pts_i.push_back(feat_i.keypoints[m.idx_i].pt);
        pts_j.push_back(feat_j.keypoints[m.idx_j].pt);
    }

    // Need at least 8 points for fundamental matrix estimation
    if (pts_i.size() < 8) return result;

    // RANSAC fundamental matrix (catch degenerate configurations)
    cv::Mat inlier_mask;
    cv::Mat F;
    try {
        F = cv::findFundamentalMat(pts_i, pts_j, cv::FM_RANSAC, 3.0, 0.99, inlier_mask);
    } catch (const cv::Exception&) {
        // Degenerate configuration (e.g., self-match with identical points)
        return result;
    }
    if (F.empty() || F.rows < 3 || F.cols < 3) return result;

    // Count inliers and collect surviving matches
    int num_inliers = 0;
    std::vector<FeatureMatch> inlier_matches;
    for (int i = 0; i < static_cast<int>(raw_matches.size()); i++) {
        if (inlier_mask.at<uchar>(i)) {
            num_inliers++;
            inlier_matches.push_back(raw_matches[i]);
        }
    }

    if (num_inliers < min_inliers) return result;

    // Essential matrix: E = K_j^T * F * K_i
    cv::Mat E = K_j.t() * F * K_i;

    // Compute median triangulation angle (approximate via epipolar geometry)
    // Use the angle between bearing vectors from camera centers
    // For now, estimate from the essential matrix decomposition
    cv::Mat R1, R2, tvec;
    cv::decomposeEssentialMat(E, R1, R2, tvec);

    // Approximate median triangulation angle from baseline direction
    // The actual angle depends on point positions, but we can estimate
    // using the translation vector magnitude relative to scene depth
    double baseline_angle = std::asin(std::min(1.0, cv::norm(tvec))) * 180.0 / CV_PI;

    result.F = F;
    result.E = E;
    result.matches = inlier_matches;
    result.num_inliers = num_inliers;
    result.median_triangulation_angle = baseline_angle;

    return result;
}

// ============================================================================
// Public: matchFeatures — match one pair
// ============================================================================
ImagePairMatches matchFeatures(const SIFTFeatures& feat_i, const SIFTFeatures& feat_j,
                                const cv::Mat& K_i, const cv::Mat& K_j,
                                float ratio_thresh, int min_inliers)
{
    // GPU brute-force KNN + ratio test + cross-check
    auto raw_matches = gpuMatchPair(feat_i.descriptors, feat_j.descriptors, ratio_thresh);

    // Geometric verification (CPU RANSAC)
    return geometricVerification(feat_i, feat_j, K_i, K_j, raw_matches, min_inliers);
}

// ============================================================================
// Public: matchAllPairs — match all image pairs
// ============================================================================
std::vector<ImagePairMatches> matchAllPairs(
    const std::vector<SIFTFeatures>& features,
    const std::vector<ImageData>& images,
    float ratio_thresh,
    int min_inliers,
    bool verbose)
{
    int N = static_cast<int>(features.size());
    if (N < 2) return {};

    // Build list of pairs to evaluate
    std::vector<std::pair<int,int>> pairs;
    if (N <= 50) {
        // Exhaustive: all N*(N-1)/2 pairs
        for (int i = 0; i < N; i++)
            for (int j = i + 1; j < N; j++)
                pairs.emplace_back(i, j);
    } else {
        // Sequential: each image vs K=10 nearest neighbors by index
        std::set<std::pair<int,int>> pair_set;
        int K = 10;
        for (int i = 0; i < N; i++) {
            for (int k = 1; k <= K; k++) {
                int j = (i + k) % N;  // wrap around for circular capture
                int a = std::min(i, j), b = std::max(i, j);
                pair_set.insert({a, b});
            }
        }
        pairs.assign(pair_set.begin(), pair_set.end());
    }

    int total_pairs = static_cast<int>(pairs.size());
    if (verbose) {
        printf("[Matching] %d images, %d pairs to evaluate\n", N, total_pairs);
    }

    std::vector<ImagePairMatches> verified;
    verified.reserve(total_pairs);

    for (int p = 0; p < total_pairs; p++) {
        int i = pairs[p].first;
        int j = pairs[p].second;

        auto result = matchFeatures(features[i], features[j],
                                     images[i].K, images[j].K,
                                     ratio_thresh, min_inliers);

        if (result.num_inliers >= min_inliers) {
            verified.push_back(std::move(result));
        }

        if (verbose && ((p + 1) % 50 == 0 || p == total_pairs - 1)) {
            printf("[Matching] %d/%d pairs processed, %d verified\n",
                   p + 1, total_pairs, static_cast<int>(verified.size()));
        }
    }

    // Check match graph connectivity
    int components = checkMatchGraph(verified, N);
    if (verbose) {
        printf("[Matching] Match graph: %d verified pairs, %d connected components\n",
               static_cast<int>(verified.size()), components);
    }
    if (components > 1) {
        fprintf(stderr, "[WARNING] Match graph is disconnected (%d components). "
                        "Reconstruction will be partial.\n", components);
    }

    // Report isolated images
    if (verbose) {
        std::vector<bool> connected(N, false);
        for (const auto& m : verified) {
            connected[m.image_i] = true;
            connected[m.image_j] = true;
        }
        for (int i = 0; i < N; i++) {
            if (!connected[i]) {
                fprintf(stderr, "[WARNING] Image %d is isolated (no verified matches).\n", i);
            }
        }
    }

    return verified;
}

// ============================================================================
// Public: checkMatchGraph — BFS connectivity check
// ============================================================================
int checkMatchGraph(const std::vector<ImagePairMatches>& matches, int num_images)
{
    // Build adjacency list
    std::vector<std::vector<int>> adj(num_images);
    for (const auto& m : matches) {
        adj[m.image_i].push_back(m.image_j);
        adj[m.image_j].push_back(m.image_i);
    }

    // BFS to count connected components
    std::vector<bool> visited(num_images, false);
    int components = 0;
    for (int i = 0; i < num_images; i++) {
        if (visited[i]) continue;
        // Check if this node has any edges (skip truly isolated nodes that
        // don't participate in any pair — they form their own component)
        components++;
        std::queue<int> q;
        q.push(i);
        visited[i] = true;
        while (!q.empty()) {
            int node = q.front(); q.pop();
            for (int nb : adj[node]) {
                if (!visited[nb]) {
                    visited[nb] = true;
                    q.push(nb);
                }
            }
        }
    }
    return components;
}
