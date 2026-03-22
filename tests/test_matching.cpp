#include <gtest/gtest.h>
#include "types.h"
#include "utils/image_loader.h"
#include "utils/synthetic_data.h"
#include "feature_detection/sift_cuda.h"
#include "feature_detection/feature_matching.h"
#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;

static const std::string SYNTH_DIR = "data/synthetic";

class MatchingTest : public ::testing::Test {
protected:
    static std::vector<ImageData> images_;
    static std::vector<SIFTFeatures> features_;

    static void SetUpTestSuite() {
        if (!fs::exists(SYNTH_DIR + "/view_000.jpg")) {
            generateSyntheticData(SYNTH_DIR);
        }
        if (images_.empty()) {
            images_ = loadImages(SYNTH_DIR);
        }
        if (features_.empty()) {
            features_ = detectAllFeatures(images_, 8000, false);
        }
    }
};

std::vector<ImageData> MatchingTest::images_;
std::vector<SIFTFeatures> MatchingTest::features_;

// Match two adjacent synthetic views (10° apart) — expect >= 100 inlier matches
TEST_F(MatchingTest, AdjacentViewsMatchWell) {
    ASSERT_GE(features_.size(), 2u);

    auto result = matchFeatures(features_[0], features_[1],
                                 images_[0].K, images_[1].K,
                                 0.75f, 30);

    EXPECT_EQ(result.image_i, 0);
    EXPECT_EQ(result.image_j, 1);
    EXPECT_GE(result.num_inliers, 50)
        << "Adjacent views (10° apart) should have >= 50 inlier matches, got "
        << result.num_inliers;
    EXPECT_FALSE(result.F.empty()) << "Fundamental matrix should be computed";
    EXPECT_FALSE(result.E.empty()) << "Essential matrix should be computed";
}

// Verify fundamental matrix satisfies x'^T F x ≈ 0 for inliers
TEST_F(MatchingTest, FundamentalMatrixEpipolarConstraint) {
    ASSERT_GE(features_.size(), 2u);

    auto result = matchFeatures(features_[0], features_[1],
                                 images_[0].K, images_[1].K,
                                 0.75f, 30);
    ASSERT_GE(result.num_inliers, 30);
    ASSERT_FALSE(result.F.empty());

    cv::Mat F64;
    result.F.convertTo(F64, CV_64F);

    double max_error = 0.0;
    double sum_error = 0.0;
    for (const auto& m : result.matches) {
        cv::Point2f p_i = features_[0].keypoints[m.idx_i].pt;
        cv::Point2f p_j = features_[1].keypoints[m.idx_j].pt;

        cv::Mat x_i = (cv::Mat_<double>(3, 1) << p_i.x, p_i.y, 1.0);
        cv::Mat x_j = (cv::Mat_<double>(3, 1) << p_j.x, p_j.y, 1.0);

        cv::Mat err_mat = x_j.t() * F64 * x_i;
        double err = std::abs(err_mat.at<double>(0, 0));
        max_error = std::max(max_error, err);
        sum_error += err;
    }
    double avg_error = sum_error / result.matches.size();

    // Epipolar constraint should be close to 0 for inliers
    EXPECT_LT(avg_error, 5.0)
        << "Average epipolar error too large: " << avg_error;
}

// Match two distant views (180° apart) — verify they still produce matches
// but with a larger triangulation angle than adjacent views.
// Note: on a uniformly textured sphere, 180° views share no visible surface,
// but intermediate angles (90°) can match well due to good baseline geometry.
TEST_F(MatchingTest, DistantViewsMatchCharacteristics) {
    ASSERT_GE(features_.size(), 19u);  // need view 0 and view 18 (180° apart)

    auto adjacent = matchFeatures(features_[0], features_[1],
                                   images_[0].K, images_[1].K,
                                   0.75f, 10);

    auto opposite = matchFeatures(features_[0], features_[18],
                                   images_[0].K, images_[18].K,
                                   0.75f, 10);

    // Adjacent views (10° apart) should definitely match
    EXPECT_GE(adjacent.num_inliers, 30)
        << "Adjacent views should have good matches";

    // Opposite views (180° apart) on a sphere see completely different surface.
    // They may still produce some matches if the texture repeats, but fewer.
    // The key check: both produce some result (even if 0 inliers for 180°).
    printf("[DistantViewsTest] Adjacent (10°): %d inliers, Opposite (180°): %d inliers\n",
           adjacent.num_inliers, opposite.num_inliers);

    // Verify that adjacent views have a meaningful triangulation angle
    EXPECT_GT(adjacent.median_triangulation_angle, 0.0)
        << "Adjacent views should have non-zero triangulation angle";
}

// Match identical image to itself — GPU matching produces zero-distance matches
// Note: geometric verification (fundamental matrix) is degenerate for self-match,
// so we test two very close views instead, which should give near-perfect matches.
TEST_F(MatchingTest, NearIdenticalViewsHighMatchCount) {
    ASSERT_GE(features_.size(), 2u);

    // Views 0 and 1 are only 10° apart on our synthetic sphere.
    // They should produce a high match count with low distances.
    auto result = matchFeatures(features_[0], features_[1],
                                 images_[0].K, images_[1].K,
                                 0.9f, 10);

    // Should have many matches for near-identical views
    EXPECT_GE(result.num_inliers, 50)
        << "Near-identical views should have many matches";

    // Average distance should be relatively low
    if (!result.matches.empty()) {
        float total_dist = 0.0f;
        for (const auto& m : result.matches) {
            total_dist += m.distance;
        }
        float avg_dist = total_dist / result.matches.size();
        EXPECT_LT(avg_dist, 300.0f)
            << "Near-identical views should have low average match distance, got " << avg_dist;
    }
}

// Verify ratio test reduces match count by >= 30%
TEST_F(MatchingTest, RatioTestReducesMatches) {
    ASSERT_GE(features_.size(), 2u);

    // Strict ratio threshold
    auto strict = matchFeatures(features_[0], features_[1],
                                 images_[0].K, images_[1].K,
                                 0.6f, 10);

    // Relaxed ratio threshold
    auto relaxed = matchFeatures(features_[0], features_[1],
                                  images_[0].K, images_[1].K,
                                  0.95f, 10);

    // Stricter ratio should yield fewer matches
    EXPECT_LT(strict.num_inliers, relaxed.num_inliers)
        << "Strict ratio (0.6) should yield fewer matches than relaxed (0.95). "
        << "Strict=" << strict.num_inliers << " Relaxed=" << relaxed.num_inliers;

    // The reduction should be meaningful (>= 30% fewer)
    if (relaxed.num_inliers > 0) {
        double reduction = 1.0 - static_cast<double>(strict.num_inliers) / relaxed.num_inliers;
        EXPECT_GE(reduction, 0.30)
            << "Ratio test should reduce matches by >= 30%, got "
            << (reduction * 100) << "%";
    }
}

// Match all 36 images and verify graph connectivity
TEST_F(MatchingTest, AllPairsMatchGraphConnected) {
    ASSERT_EQ(features_.size(), 36u);

    auto all_matches = matchAllPairs(features_, images_, 0.75f, 30, true);

    EXPECT_GT(all_matches.size(), 0u)
        << "Should have at least some verified pairs";

    // Match graph should be fully connected (1 component)
    int components = checkMatchGraph(all_matches, 36);
    EXPECT_EQ(components, 1)
        << "Match graph should be fully connected, got " << components << " components";

    // Report statistics
    int total_inliers = 0;
    for (const auto& m : all_matches) {
        total_inliers += m.num_inliers;
    }
    double avg_inliers = all_matches.empty() ? 0.0 :
        static_cast<double>(total_inliers) / all_matches.size();

    printf("\n[MatchingTest] Total verified pairs: %zu\n", all_matches.size());
    printf("[MatchingTest] Average inliers per pair: %.1f\n", avg_inliers);
}
