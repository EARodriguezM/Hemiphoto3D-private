#pragma once

#include "types.h"
#include <vector>

// Run bundle adjustment on cameras and points using Ceres Solver.
// Optimizes camera poses (angle-axis + translation) and 3D point positions.
// fix_first_camera: hold the first registered camera constant (gauge freedom)
// optimize_intrinsics: if true, also optimize fx/fy/cx/cy (only if enough images)
void bundleAdjust(
    std::vector<CameraPose>& cameras,
    std::vector<SparsePoint>& points,
    const std::vector<SIFTFeatures>& features,
    const PipelineConfig& config,
    bool fix_first_camera = true,
    bool optimize_intrinsics = false,
    bool fix_all_cameras = false
);

// Overload for backward compat (used by Step 5 periodic calls)
bool bundleAdjust(
    std::vector<CameraPose>& cameras,
    std::vector<SparsePoint>& points,
    const std::vector<SIFTFeatures>& features,
    double max_reproj_error = 4.0
);

// Post-BA: remove points with high reprojection error or short tracks
void filterPointCloud(
    std::vector<CameraPose>& cameras,
    std::vector<SparsePoint>& points,
    const std::vector<SIFTFeatures>& features,
    double max_reproj_error,
    int min_track_length
);
