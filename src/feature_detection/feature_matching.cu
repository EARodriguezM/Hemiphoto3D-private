#include "feature_detection/feature_matching.h"

// Stub — implemented in Step 4
ImagePairMatches matchFeatures(const SIFTFeatures& /*feat_i*/, const SIFTFeatures& /*feat_j*/,
                                float /*ratio_thresh*/) {
    ImagePairMatches result;
    result.image_i = -1;
    result.image_j = -1;
    result.num_inliers = 0;
    result.median_triangulation_angle = 0.0;
    return result;
}

std::vector<ImagePairMatches> matchAllPairs(const std::vector<SIFTFeatures>& /*features*/,
                                             float /*ratio_thresh*/,
                                             int /*min_matches*/) {
    return {};
}
