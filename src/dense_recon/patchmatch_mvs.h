#pragma once

#include "types.h"
#include <vector>

// Run PatchMatch MVS for a single reference image
DepthMap patchMatchMVS(const ImageData& ref_image,
                        const std::vector<ImageData>& src_images,
                        const CameraPose& ref_pose,
                        const std::vector<CameraPose>& src_poses,
                        int iterations = 8, int patch_size = 11);

// Run PatchMatch MVS for all images
std::vector<DepthMap> runDenseReconstruction(const std::vector<ImageData>& images,
                                              const std::vector<CameraPose>& poses,
                                              const PipelineConfig& config);
