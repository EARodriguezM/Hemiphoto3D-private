#pragma once

#include "types.h"
#include <vector>

// Detect SIFT features on a single image (custom CUDA kernels — GPU-accelerated).
// Descriptors stored in SIFTFeatures.descriptors (Nx128 float32).
// max_features: maximum number of features to retain (0 = unlimited).
SIFTFeatures detectSIFT(const ImageData& image, int max_features = 8000);

// Detect features for all images (sequential, each image processed on GPU).
std::vector<SIFTFeatures> detectAllFeatures(
    const std::vector<ImageData>& images,
    int max_features = 8000,
    bool verbose = false);

// Explicit path selection (for testing / comparison):
SIFTFeatures detectSIFT_CUDA(const ImageData& image, int max_features = 8000);
SIFTFeatures detectSIFT_OpenCV(const ImageData& image, int max_features = 8000);
