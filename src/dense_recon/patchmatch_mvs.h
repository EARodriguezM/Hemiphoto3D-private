#pragma once

#include "types.h"
#include "sfm/camera_pose.h"
#include <vector>

// ============================================================================
// PatchMatch Multi-View Stereo — CUDA implementation
//
// For each reference image, estimates per-pixel depth and surface normal
// by finding photometric consistency with neighbor source views.
// Uses PatchMatch random search: init → propagate → evaluate → refine → repeat.
// ============================================================================

// Source view info for a reference image
struct SourceViewInfo {
    int image_id;       // index into the images/cameras arrays
    float score;        // selection score (higher = better)
    float baseline_deg; // baseline angle in degrees
};

// Per-reference-view MVS parameters computed from SfM
struct MVSViewParams {
    int ref_image_id;
    std::vector<SourceViewInfo> source_views; // sorted by score, top N_src
    float depth_min;
    float depth_max;
};

// Select source views for each reference image based on shared SfM points
// and baseline angle. Returns one MVSViewParams per registered camera.
std::vector<MVSViewParams> selectSourceViews(
    const SfMResult& sfm,
    int num_sources = 5  // N_src
);

// Run PatchMatch MVS for all images.
// Top-level API matching PLAN.md Step 7.
std::vector<DepthMap> runPatchMatchMVS(
    const std::vector<ImageData>& images,
    const std::vector<CameraPose>& cameras,
    const SfMResult& sfm,
    const PipelineConfig& config
);

// Run PatchMatch MVS for a single reference image (kept for compatibility).
DepthMap patchMatchMVS(const ImageData& ref_image,
                        const std::vector<ImageData>& src_images,
                        const CameraPose& ref_pose,
                        const std::vector<CameraPose>& src_poses,
                        int iterations = 8, int patch_size = 11);

// Run dense reconstruction (legacy wrapper).
std::vector<DepthMap> runDenseReconstruction(const std::vector<ImageData>& images,
                                              const std::vector<CameraPose>& poses,
                                              const PipelineConfig& config);

// Save depth map as Portable FloatMap (.pfm)
bool saveDepthMapPFM(const DepthMap& dm, const std::string& path);

// Load depth map from .pfm file
bool loadDepthMapPFM(DepthMap& dm, const std::string& path);
