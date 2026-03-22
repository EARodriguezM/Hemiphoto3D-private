#include "sfm/camera_pose.h"

// Stub — implemented in Step 5
bool estimateRelativePose(const ImagePairMatches& /*matches*/,
                           const cv::Mat& /*K_i*/, const cv::Mat& /*K_j*/,
                           const std::vector<cv::KeyPoint>& /*kp_i*/,
                           const std::vector<cv::KeyPoint>& /*kp_j*/,
                           CameraPose& /*pose_i*/, CameraPose& /*pose_j*/) {
    return false;
}

std::vector<SparsePoint> triangulatePoints(const CameraPose& /*pose_i*/, const CameraPose& /*pose_j*/,
                                            const ImagePairMatches& /*matches*/,
                                            const std::vector<cv::KeyPoint>& /*kp_i*/,
                                            const std::vector<cv::KeyPoint>& /*kp_j*/) {
    return {};
}

bool registerImage(const ImageData& /*image*/,
                   const SIFTFeatures& /*features*/,
                   const std::vector<SparsePoint>& /*points*/,
                   CameraPose& /*pose*/) {
    return false;
}
