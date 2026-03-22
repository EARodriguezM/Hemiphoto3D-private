#pragma once

#include "types.h"
#include <vector>

// Match features between two images using GPU brute-force KNN,
// Lowe's ratio test, cross-check, and geometric verification (RANSAC).
ImagePairMatches matchFeatures(const SIFTFeatures& feat_i, const SIFTFeatures& feat_j,
                                const cv::Mat& K_i, const cv::Mat& K_j,
                                float ratio_thresh = 0.75f,
                                int min_inliers = 30);

// Match all image pairs with pair selection strategy:
//   N <= 50: exhaustive
//   N > 50: sequential (each vs K=10 nearest neighbors)
// Returns only verified pairs with >= min_inliers.
// Also builds match graph and warns about disconnected components.
std::vector<ImagePairMatches> matchAllPairs(const std::vector<SIFTFeatures>& features,
                                             const std::vector<ImageData>& images,
                                             float ratio_thresh = 0.75f,
                                             int min_inliers = 30,
                                             bool verbose = false);

// Check if match graph is fully connected (single component).
// Returns number of connected components.
int checkMatchGraph(const std::vector<ImagePairMatches>& matches, int num_images);
