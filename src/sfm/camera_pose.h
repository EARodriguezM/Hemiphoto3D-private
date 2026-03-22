#pragma once

#include "types.h"
#include <vector>

// Estimate relative pose from essential matrix
bool estimateRelativePose(const ImagePairMatches& matches,
                           const cv::Mat& K_i, const cv::Mat& K_j,
                           const std::vector<cv::KeyPoint>& kp_i,
                           const std::vector<cv::KeyPoint>& kp_j,
                           CameraPose& pose_i, CameraPose& pose_j);

// Triangulate 3D points from two views
std::vector<SparsePoint> triangulatePoints(const CameraPose& pose_i, const CameraPose& pose_j,
                                            const ImagePairMatches& matches,
                                            const std::vector<cv::KeyPoint>& kp_i,
                                            const std::vector<cv::KeyPoint>& kp_j);

// Incremental SfM: register a new image to existing reconstruction
bool registerImage(const ImageData& image,
                   const SIFTFeatures& features,
                   const std::vector<SparsePoint>& points,
                   CameraPose& pose);
