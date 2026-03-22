#include "dense_recon/patchmatch_mvs.h"

// Stub — implemented in Step 7
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
