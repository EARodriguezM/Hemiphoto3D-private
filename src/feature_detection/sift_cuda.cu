#include "feature_detection/sift_cuda.h"
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdio>

SIFTFeatures detectSIFT(const ImageData& image, int max_features) {
    SIFTFeatures features;
    features.image_id = image.id;
    features.d_descriptors = nullptr;
    features.count = 0;

    if (image.gray.empty()) {
        fprintf(stderr, "Warning: Image %d has no grayscale data.\n", image.id);
        return features;
    }

    // Create SIFT detector
    auto sift = cv::SIFT::create(
        max_features,   // nfeatures
        3,              // nOctaveLayers
        0.04,           // contrastThreshold
        10.0,           // edgeThreshold
        1.6             // sigma
    );

    // SIFT expects 8-bit grayscale input
    cv::Mat gray_u8;
    if (image.gray.type() == CV_32F) {
        image.gray.convertTo(gray_u8, CV_8U, 255.0);
    } else {
        gray_u8 = image.gray;
    }

    // Detect and compute
    sift->detectAndCompute(gray_u8, cv::noArray(), features.keypoints, features.descriptors);

    features.count = static_cast<int>(features.keypoints.size());

    return features;
}

std::vector<SIFTFeatures> detectAllFeatures(
    const std::vector<ImageData>& images,
    int max_features,
    bool verbose) {

    std::vector<SIFTFeatures> all_features;
    all_features.reserve(images.size());

    for (const auto& image : images) {
        SIFTFeatures feat = detectSIFT(image, max_features);

        if (verbose) {
            printf("  [%02d] %-30s  %d features\n",
                   image.id, image.filename.c_str(), feat.count);
        }

        all_features.push_back(std::move(feat));
    }

    if (verbose) {
        int total = 0;
        for (const auto& f : all_features) total += f.count;
        printf("Total features: %d across %zu images (avg %.0f/image)\n",
               total, all_features.size(),
               all_features.empty() ? 0.0 : double(total) / all_features.size());
    }

    return all_features;
}
