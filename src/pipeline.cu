#include "pipeline.h"
#include "utils/cuda_utils.cuh"
#include <cstdio>

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
    // TODO: implement edge-pair counting
    return false;
}
