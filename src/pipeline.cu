#include "pipeline.h"
#include "utils/cuda_utils.cuh"
#include <cstdio>
#include <unordered_map>
#include <algorithm>

Pipeline::Pipeline(const PipelineConfig& config) : config_(config) {
}

Pipeline::~Pipeline() {
}

bool Pipeline::run() {
    // TODO: implement full pipeline orchestration
    printf("[Pipeline] Not yet implemented.\n");
    return false;
}

bool Pipeline::loadImages() { return false; }
bool Pipeline::detectFeatures() { return false; }
bool Pipeline::matchFeatures() { return false; }
bool Pipeline::reconstructSparse() { return false; }
bool Pipeline::bundleAdjust() { return false; }
bool Pipeline::denseReconstruction() { return false; }
bool Pipeline::fusePointClouds() { return false; }
bool Pipeline::generateMesh() { return false; }
bool Pipeline::exportMesh() { return false; }

// PipelineConfig::applyPreset — set derived parameters from quality string
void PipelineConfig::applyPreset() {
    if (quality == "low") {
        max_image_size = 1600;
        mvs_resolution = 0.5f;
        mvs_iterations = 4;
        poisson_depth = 8;
    } else if (quality == "medium") {
        max_image_size = 2400;
        mvs_resolution = 0.75f;
        mvs_iterations = 6;
        poisson_depth = 9;
    } else if (quality == "high") {
        // defaults are already "high"
    } else if (quality == "ultra") {
        max_image_size = 4800;
        mvs_resolution = 1.0f;
        mvs_iterations = 12;
        poisson_depth = 12;
    }
}

// DensePointCloud::freeGPU — release device memory
void DensePointCloud::freeGPU() {
    if (d_positions) { cudaFree(d_positions); d_positions = nullptr; }
    if (d_normals)   { cudaFree(d_normals);   d_normals   = nullptr; }
    if (d_colors)    { cudaFree(d_colors);    d_colors    = nullptr; }
    on_gpu = false;
}

// Mesh::is_watertight — check if all edges are shared by exactly 2 faces
bool Mesh::is_watertight() const {
    if (faces.empty()) return false;

    // Count each edge occurrence. An edge is identified by its sorted vertex pair.
    std::unordered_map<uint64_t, int> edge_count;
    auto edgeId = [](int a, int b) -> uint64_t {
        int lo = std::min(a, b), hi = std::max(a, b);
        return ((uint64_t)lo << 32) | (uint64_t)hi;
    };

    for (const auto& f : faces) {
        int v[3] = {f.x(), f.y(), f.z()};
        for (int e = 0; e < 3; e++) {
            edge_count[edgeId(v[e], v[(e + 1) % 3])]++;
        }
    }

    // Every edge must appear exactly 2 times (shared by 2 faces)
    for (const auto& [eid, count] : edge_count) {
        if (count != 2) return false;
    }
    return true;
}
