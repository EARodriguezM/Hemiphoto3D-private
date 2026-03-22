#pragma once

#include "types.h"
#include <vector>

// Match features between two images
ImagePairMatches matchFeatures(const SIFTFeatures& feat_i, const SIFTFeatures& feat_j,
                                float ratio_thresh = 0.75f);

// Match all image pairs
std::vector<ImagePairMatches> matchAllPairs(const std::vector<SIFTFeatures>& features,
                                             float ratio_thresh = 0.75f,
                                             int min_matches = 30);
