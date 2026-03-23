#pragma once

#include "types.h"
#include <vector>
#include <string>

// ============================================================================
// Dense Point Cloud Generation — CUDA implementation
//
// Fuses per-view depth maps into a single dense point cloud via:
//   1. Depth-to-points backprojection (per-pixel GPU kernel)
//   2. Fusion: voxel grid downsampling (Option A) or TSDF (Option B)
//   3. Statistical outlier removal (uniform grid neighbor count)
//   4. Normal refinement via PCA with camera-oriented flipping
// ============================================================================

// Fuse depth maps into a dense point cloud.
// Main API for Step 8.
DensePointCloud fuseDepthMaps(
    const std::vector<DepthMap>& depth_maps,
    const std::vector<CameraPose>& cameras,
    const std::vector<ImageData>& images,
    const PipelineConfig& config
);

// Legacy overload (no config — uses defaults)
DensePointCloud fuseDepthMaps(
    const std::vector<DepthMap>& depth_maps,
    const std::vector<CameraPose>& poses,
    const std::vector<ImageData>& images
);

// Save dense point cloud as binary PLY
bool saveDensePointCloudPLY(const DensePointCloud& cloud, const std::string& path);

// Load dense point cloud from PLY
bool loadDensePointCloudPLY(DensePointCloud& cloud, const std::string& path);
