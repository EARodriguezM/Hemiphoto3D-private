#include <gtest/gtest.h>
#include "types.h"
#include "utils/image_loader.h"
#include "utils/synthetic_data.h"
#include "feature_detection/sift_cuda.h"
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;

static const std::string SYNTH_DIR = "data/synthetic";

class SIFTTest : public ::testing::Test {
protected:
    static std::vector<ImageData> images_;

    static void SetUpTestSuite() {
        if (!fs::exists(SYNTH_DIR + "/view_000.jpg")) {
            generateSyntheticData(SYNTH_DIR);
        }
        if (images_.empty()) {
            images_ = loadImages(SYNTH_DIR);
        }
    }
};

std::vector<ImageData> SIFTTest::images_;

TEST_F(SIFTTest, DetectOnSingleImage) {
    ASSERT_FALSE(images_.empty());
    SIFTFeatures feat = detectSIFT(images_[0], 8000);

    EXPECT_EQ(feat.image_id, 0);
    EXPECT_GE(feat.count, 100);      // synthetic sphere should have features
    EXPECT_LE(feat.count, 10000);
    EXPECT_EQ(feat.count, static_cast<int>(feat.keypoints.size()));
}

TEST_F(SIFTTest, DescriptorDimension128) {
    ASSERT_FALSE(images_.empty());
    SIFTFeatures feat = detectSIFT(images_[0], 8000);

    ASSERT_GT(feat.count, 0);
    EXPECT_EQ(feat.descriptors.cols, 128);
    EXPECT_EQ(feat.descriptors.rows, feat.count);
    // OpenCV SIFT produces float32 descriptors
    EXPECT_EQ(feat.descriptors.type(), CV_32F);
}

TEST_F(SIFTTest, KeypointsWithinBounds) {
    ASSERT_FALSE(images_.empty());
    SIFTFeatures feat = detectSIFT(images_[0], 8000);

    for (const auto& kp : feat.keypoints) {
        EXPECT_GE(kp.pt.x, 0.0f);
        EXPECT_GE(kp.pt.y, 0.0f);
        EXPECT_LT(kp.pt.x, static_cast<float>(images_[0].width));
        EXPECT_LT(kp.pt.y, static_cast<float>(images_[0].height));
    }
}

TEST_F(SIFTTest, DetectAllSyntheticImages) {
    ASSERT_EQ(images_.size(), 36u);
    auto all_features = detectAllFeatures(images_, 8000, false);

    ASSERT_EQ(all_features.size(), 36u);

    for (size_t i = 0; i < all_features.size(); i++) {
        EXPECT_EQ(all_features[i].image_id, static_cast<int>(i));
        EXPECT_GT(all_features[i].count, 0)
            << "Image " << i << " has no features";
        EXPECT_EQ(all_features[i].descriptors.cols, 128);
    }
}

TEST_F(SIFTTest, DescriptorsNonZero) {
    ASSERT_FALSE(images_.empty());
    SIFTFeatures feat = detectSIFT(images_[0], 8000);

    ASSERT_GT(feat.count, 0);
    // Check that descriptors are not all zeros
    double sum = cv::sum(cv::abs(feat.descriptors))[0];
    EXPECT_GT(sum, 0.0);
}

TEST_F(SIFTTest, RepeatabilityOnRotatedImage) {
    ASSERT_FALSE(images_.empty());

    // Detect on original
    SIFTFeatures feat_orig = detectSIFT(images_[0], 8000);
    ASSERT_GT(feat_orig.count, 50);

    // Create a slightly rotated version (small rotation to keep most features)
    ImageData rotated = images_[0];
    cv::Mat center_mat = (cv::Mat_<double>(2, 1) << images_[0].width / 2.0, images_[0].height / 2.0);
    cv::Mat rot_mat = cv::getRotationMatrix2D(
        cv::Point2f(images_[0].width / 2.0f, images_[0].height / 2.0f),
        15.0, 1.0);  // 15 degree rotation
    cv::warpAffine(images_[0].image, rotated.image, rot_mat,
                   images_[0].image.size());
    cv::Mat gray_u8;
    cv::cvtColor(rotated.image, gray_u8, cv::COLOR_BGR2GRAY);
    gray_u8.convertTo(rotated.gray, CV_32F, 1.0 / 255.0);

    SIFTFeatures feat_rot = detectSIFT(rotated, 8000);
    ASSERT_GT(feat_rot.count, 50);

    // Both should detect a reasonable number of features
    // (SIFT is rotation-invariant, but some features will be lost at borders)
    EXPECT_GT(feat_rot.count, feat_orig.count / 3)
        << "Rotated image lost too many features";
}
