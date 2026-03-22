#pragma once

#include "types.h"
#include <vector>

// Detect SIFT features on a single image.
// Default path: CudaSift (fast GPU) if available, else custom CUDA kernels, else OpenCV CPU.
// Descriptors: Nx128 float32.
SIFTFeatures detectSIFT(const ImageData& image, int max_features = 8000);

// Detect features for all images.
std::vector<SIFTFeatures> detectAllFeatures(
    const std::vector<ImageData>& images,
    int max_features = 8000,
    bool verbose = false);

// Explicit path selection (for testing / comparison):
SIFTFeatures detectSIFT_CudaSift(const ImageData& image, int max_features = 8000);
SIFTFeatures detectSIFT_CUDA(const ImageData& image, int max_features = 8000);
SIFTFeatures detectSIFT_OpenCV(const ImageData& image, int max_features = 8000);
