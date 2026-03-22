#include <gtest/gtest.h>
#include "types.h"
#include "utils/image_loader.h"
#include "utils/synthetic_data.h"
#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;

static const std::string SYNTH_DIR = "data/synthetic";

class ImageLoaderTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // Generate synthetic data once for all tests
        if (!fs::exists(SYNTH_DIR + "/view_000.jpg")) {
            ASSERT_TRUE(generateSyntheticData(SYNTH_DIR));
        }
    }
};

TEST_F(ImageLoaderTest, LoadsSyntheticImages) {
    auto images = loadImages(SYNTH_DIR);
    ASSERT_EQ(images.size(), 36u);
}

TEST_F(ImageLoaderTest, ImageDimensions) {
    auto images = loadImages(SYNTH_DIR);
    ASSERT_FALSE(images.empty());
    // Synthetic images are 1280x960, well under the 3200 default max
    EXPECT_EQ(images[0].width, 1280);
    EXPECT_EQ(images[0].height, 960);
}

TEST_F(ImageLoaderTest, SequentialIds) {
    auto images = loadImages(SYNTH_DIR);
    for (size_t i = 0; i < images.size(); i++) {
        EXPECT_EQ(images[i].id, static_cast<int>(i));
    }
}

TEST_F(ImageLoaderTest, GrayscaleFloat01) {
    auto images = loadImages(SYNTH_DIR);
    ASSERT_FALSE(images.empty());

    const cv::Mat& gray = images[0].gray;
    EXPECT_EQ(gray.type(), CV_32F);
    EXPECT_EQ(gray.cols, images[0].width);
    EXPECT_EQ(gray.rows, images[0].height);

    // Check values are in [0, 1]
    double min_val, max_val;
    cv::minMaxLoc(gray, &min_val, &max_val);
    EXPECT_GE(min_val, 0.0);
    EXPECT_LE(max_val, 1.0);
}

TEST_F(ImageLoaderTest, IntrinsicsMatrix) {
    auto images = loadImages(SYNTH_DIR);
    ASSERT_FALSE(images.empty());

    const cv::Mat& K = images[0].K;
    EXPECT_EQ(K.rows, 3);
    EXPECT_EQ(K.cols, 3);
    EXPECT_EQ(K.type(), CV_64F);

    // fx = fy (square pixels)
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    EXPECT_DOUBLE_EQ(fx, fy);

    // Principal point should be at image center
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);
    EXPECT_DOUBLE_EQ(cx, images[0].width / 2.0);
    EXPECT_DOUBLE_EQ(cy, images[0].height / 2.0);

    // Focal length should be positive and reasonable
    EXPECT_GT(fx, 100.0);

    // Skew = 0, bottom row = [0 0 1]
    EXPECT_DOUBLE_EQ(K.at<double>(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(K.at<double>(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(K.at<double>(2, 0), 0.0);
    EXPECT_DOUBLE_EQ(K.at<double>(2, 1), 0.0);
    EXPECT_DOUBLE_EQ(K.at<double>(2, 2), 1.0);
}

TEST_F(ImageLoaderTest, DistCoeffsZero) {
    auto images = loadImages(SYNTH_DIR);
    ASSERT_FALSE(images.empty());

    const cv::Mat& dc = images[0].dist_coeffs;
    EXPECT_EQ(dc.rows, 5);
    EXPECT_EQ(dc.cols, 1);
    for (int i = 0; i < 5; i++) {
        EXPECT_DOUBLE_EQ(dc.at<double>(i), 0.0);
    }
}

TEST_F(ImageLoaderTest, ResizeRespected) {
    // Load with small max_size to force resize
    PipelineConfig config;
    config.max_image_size = 640;
    auto images = loadImages(SYNTH_DIR, config);
    ASSERT_FALSE(images.empty());

    // Largest dimension should be ≤ 640
    int max_dim = std::max(images[0].width, images[0].height);
    EXPECT_LE(max_dim, 640);
    // Aspect ratio preserved (1280:960 = 4:3)
    double ratio = double(images[0].width) / images[0].height;
    EXPECT_NEAR(ratio, 1280.0 / 960.0, 0.01);
}

TEST_F(ImageLoaderTest, FocalLengthOverride) {
    PipelineConfig config;
    config.focal_length_mm = 4.0;
    config.sensor_width_mm = 6.0;
    auto images = loadImages(SYNTH_DIR, config);
    ASSERT_FALSE(images.empty());

    // fx = 4.0 * 1280 / 6.0 ≈ 853.33
    double expected_fx = 4.0 * images[0].width / 6.0;
    EXPECT_NEAR(images[0].focal_length_px, expected_fx, 0.01);
    EXPECT_NEAR(images[0].K.at<double>(0, 0), expected_fx, 0.01);
}

TEST_F(ImageLoaderTest, EmptyDirectoryReturnsEmpty) {
    auto images = loadImages("/tmp/nonexistent_dir_xyz");
    EXPECT_TRUE(images.empty());
}

TEST_F(ImageLoaderTest, BGRImageLoaded) {
    auto images = loadImages(SYNTH_DIR);
    ASSERT_FALSE(images.empty());
    EXPECT_EQ(images[0].image.channels(), 3);
    EXPECT_EQ(images[0].image.type(), CV_8UC3);
}

// Test EXIF orientation logic (unit test of the function itself)
TEST(EXIFOrientationTest, Orientation6Rotates90CW) {
    // Create a 4x2 image (cols=4, rows=2) with known pixel pattern
    cv::Mat img(2, 4, CV_8UC3, cv::Scalar(0, 0, 0));
    img.at<cv::Vec3b>(0, 0) = cv::Vec3b(255, 0, 0);  // top-left = blue

    applyEXIFOrientation(img, 6);  // 90° clockwise

    // After 90° CW rotation: (rows=2,cols=4) → (rows=4,cols=2)
    EXPECT_EQ(img.cols, 2);
    EXPECT_EQ(img.rows, 4);
    // 90° CW: (r,c) → (c, rows_orig-1-r)
    // (0,0) → (0, 2-1-0) = (0, 1) in the new image
    EXPECT_EQ(img.at<cv::Vec3b>(0, 1), cv::Vec3b(255, 0, 0));
}

TEST(EXIFOrientationTest, Orientation3Rotates180) {
    cv::Mat img(2, 4, CV_8UC3, cv::Scalar(0, 0, 0));
    img.at<cv::Vec3b>(0, 0) = cv::Vec3b(255, 0, 0);

    applyEXIFOrientation(img, 3);  // 180°

    EXPECT_EQ(img.cols, 4);
    EXPECT_EQ(img.rows, 2);
    EXPECT_EQ(img.at<cv::Vec3b>(1, 3), cv::Vec3b(255, 0, 0));
}

TEST(SensorWidthTest, KnownDevices) {
    EXPECT_GT(estimateSensorWidth("Apple", "iPhone 13"), 5.0);
    EXPECT_GT(estimateSensorWidth("Google", "Pixel 8"), 5.0);
    EXPECT_GT(estimateSensorWidth("Samsung", "Galaxy S24"), 5.0);
}

TEST(SensorWidthTest, UnknownFallback) {
    EXPECT_DOUBLE_EQ(estimateSensorWidth("Unknown", "Phone"), 6.0);
}

TEST(SyntheticDataTest, FilesGenerated) {
    ASSERT_TRUE(fs::exists(SYNTH_DIR + "/view_000.jpg"));
    ASSERT_TRUE(fs::exists(SYNTH_DIR + "/view_035.jpg"));
    ASSERT_TRUE(fs::exists(SYNTH_DIR + "/cameras_gt.json"));
    ASSERT_TRUE(fs::exists(SYNTH_DIR + "/ground_truth.obj"));
}
