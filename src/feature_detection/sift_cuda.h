#pragma once

#include "types.h"
#include <vector>

// Detect SIFT features on a single image.
// Uses OpenCV's cv::SIFT (CPU). Descriptors stored in SIFTFeatures.descriptors (Nx128 float).
// max_features: maximum number of features to detect (0 = unlimited).
SIFTFeatures detectSIFT(const ImageData& image, int max_features = 8000);

// Detect features for all images (sequential).
std::vector<SIFTFeatures> detectAllFeatures(
    const std::vector<ImageData>& images,
    int max_features = 8000,
    bool verbose = false);
