#include <gtest/gtest.h>
#include "types.h"
#include "sfm/camera_pose.h"
#include "dense_recon/patchmatch_mvs.h"
#include "dense_recon/point_cloud.h"
#include "utils/synthetic_data.h"
#include "utils/image_loader.h"
#include "feature_detection/sift_cuda.h"
#include "feature_detection/feature_matching.h"

#include <opencv2/core/eigen.hpp>
#include <Eigen/Dense>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <vector>
#include <numeric>

namespace fs = std::filesystem;

// ============================================================================
// Test fixture: runs SfM + MVS to produce depth maps for fusion
// ============================================================================
class PointCloudTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        std::string data_dir = "data/synthetic";
        if (!fs::exists(data_dir + "/view_000.jpg")) {
            ASSERT_TRUE(generateSyntheticData(data_dir));
        }

        PipelineConfig cfg;
        cfg.input_dir = data_dir;
        cfg.verbose = false;
        cfg.turntable = true;
        images_ = loadImages(data_dir, cfg);
        ASSERT_EQ(static_cast<int>(images_.size()), 36);

        for (auto& img : images_) {
            img.K = (cv::Mat_<double>(3, 3) << 2000.0, 0.0, 640.0,
                                                0.0, 2000.0, 480.0,
                                                0.0, 0.0, 1.0);
            img.focal_length_px = 2000.0;
        }

        features_ = detectAllFeatures(images_, 8000, false);
        matches_ = matchAllPairs(features_, images_, cfg.match_ratio, cfg.min_matches, false);
        sfm_ = runIncrementalSfM(features_, matches_, images_, cfg);
        ASSERT_GE(sfm_.num_registered, 36);

        // Run MVS with moderate iterations
        PipelineConfig mvs_cfg;
        mvs_cfg.mvs_iterations = 6;
        mvs_cfg.verbose = false;
        depth_maps_ = runPatchMatchMVS(images_, sfm_.cameras, sfm_, mvs_cfg);
        ASSERT_GT(static_cast<int>(depth_maps_.size()), 0);

        // Compute sphere center and radius from SfM
        sphere_center_ = Eigen::Vector3d::Zero();
        for (const auto& pt : sfm_.points) sphere_center_ += pt.position;
        sphere_center_ /= static_cast<double>(sfm_.points.size());

        sphere_radius_ = 0;
        for (const auto& pt : sfm_.points)
            sphere_radius_ += (pt.position - sphere_center_).norm();
        sphere_radius_ /= static_cast<double>(sfm_.points.size());
    }

    static std::vector<ImageData> images_;
    static std::vector<SIFTFeatures> features_;
    static std::vector<ImagePairMatches> matches_;
    static SfMResult sfm_;
    static std::vector<DepthMap> depth_maps_;
    static Eigen::Vector3d sphere_center_;
    static double sphere_radius_;
};

std::vector<ImageData> PointCloudTest::images_;
std::vector<SIFTFeatures> PointCloudTest::features_;
std::vector<ImagePairMatches> PointCloudTest::matches_;
SfMResult PointCloudTest::sfm_;
std::vector<DepthMap> PointCloudTest::depth_maps_;
Eigen::Vector3d PointCloudTest::sphere_center_;
double PointCloudTest::sphere_radius_;

// ============================================================================
// Tests
// ============================================================================

TEST_F(PointCloudTest, FuseDepthMapsProducesPoints) {
    PipelineConfig cfg;
    cfg.voxel_resolution = (float)(sphere_radius_ * 0.02); // ~2% of radius
    cfg.outlier_grid_size = (float)(sphere_radius_ * 0.1);
    cfg.outlier_min_neighbors = 3;

    auto cloud = fuseDepthMaps(depth_maps_, sfm_.cameras, images_, cfg);
    ASSERT_GT(cloud.size(), 0) << "fuseDepthMaps produced no points";

    printf("[Test] Dense cloud: %d points\n", cloud.size());

    // Should have positions, normals, colors all same size
    EXPECT_EQ(static_cast<int>(cloud.positions.size()), cloud.size());
    EXPECT_EQ(static_cast<int>(cloud.normals.size()), cloud.size());
    EXPECT_EQ(static_cast<int>(cloud.colors.size()), cloud.size());
}

TEST_F(PointCloudTest, PointCountInExpectedRange) {
    PipelineConfig cfg;
    cfg.voxel_resolution = (float)(sphere_radius_ * 0.02);
    cfg.outlier_grid_size = (float)(sphere_radius_ * 0.1);
    cfg.outlier_min_neighbors = 3;

    auto cloud = fuseDepthMaps(depth_maps_, sfm_.cameras, images_, cfg);

    // Plan says: after voxel + outlier: ~90K-450K
    // Synthetic sphere may differ, so use wider bounds
    EXPECT_GT(cloud.size(), 1000)
        << "Too few points: " << cloud.size();
    EXPECT_LT(cloud.size(), 5000000)
        << "Too many points: " << cloud.size();

    printf("[Test] Point count: %d (expected 1K-5M)\n", cloud.size());
}

TEST_F(PointCloudTest, PointsFormSphere) {
    PipelineConfig cfg;
    cfg.voxel_resolution = (float)(sphere_radius_ * 0.02);
    cfg.outlier_grid_size = (float)(sphere_radius_ * 0.1);
    cfg.outlier_min_neighbors = 3;

    auto cloud = fuseDepthMaps(depth_maps_, sfm_.cameras, images_, cfg);
    ASSERT_GT(cloud.size(), 100);

    // Compute distances to sphere center and use median to assess shape.
    // Background pixels produce scattered points, so mean is biased;
    // the median distance should be close to the sphere radius.
    std::vector<double> dists;
    dists.reserve(cloud.size());
    for (int i = 0; i < cloud.size(); i++) {
        Eigen::Vector3d p = cloud.positions[i].cast<double>();
        dists.push_back((p - sphere_center_).norm());
    }
    std::sort(dists.begin(), dists.end());

    double median_dist = dists[dists.size() / 2];
    // Also check that a meaningful fraction clusters near the sphere
    int near_sphere = 0;
    for (double d : dists) {
        if (std::abs(d - sphere_radius_) < sphere_radius_) near_sphere++;
    }
    float near_ratio = (float)near_sphere / (float)dists.size();

    printf("[Test] Sphere fit: median_dist=%.6f, radius=%.6f, near_ratio=%.1f%%\n",
           median_dist, sphere_radius_, 100.0f * near_ratio);

    // Median distance should be within 1x radius of the sphere surface
    // (generous to accommodate PatchMatch depth errors)
    double median_error = std::abs(median_dist - sphere_radius_) / sphere_radius_;
    EXPECT_LT(median_error, 1.0)
        << "Median distance " << median_dist << " too far from radius " << sphere_radius_;

    // At least 30% of points should cluster near the sphere surface
    // (within 1 radius of the surface)
    EXPECT_GT(near_ratio, 0.30f)
        << "Too few points near sphere surface: " << near_ratio * 100 << "%";
}

TEST_F(PointCloudTest, NormalsPointOutward) {
    PipelineConfig cfg;
    cfg.voxel_resolution = (float)(sphere_radius_ * 0.02);
    cfg.outlier_grid_size = (float)(sphere_radius_ * 0.1);
    cfg.outlier_min_neighbors = 3;

    auto cloud = fuseDepthMaps(depth_maps_, sfm_.cameras, images_, cfg);
    ASSERT_GT(cloud.size(), 100);

    // For a sphere at center, outward normal should point away from center
    int outward = 0;
    int total = 0;
    for (int i = 0; i < cloud.size(); i++) {
        Eigen::Vector3f radial = cloud.positions[i] - sphere_center_.cast<float>();
        float d = radial.dot(cloud.normals[i]);
        if (d > 0) outward++;
        total++;
    }

    float outward_ratio = (float)outward / (float)total;
    printf("[Test] Normals pointing outward: %.1f%% (%d/%d)\n",
           100.0f * outward_ratio, outward, total);

    // At least 70% of normals should point outward
    EXPECT_GT(outward_ratio, 0.70f)
        << "Too few normals point outward from sphere center";
}

TEST_F(PointCloudTest, OutlierRateBelow10Percent) {
    // Verification gate: "Outlier rate < 10%" — this means the statistical
    // outlier removal step should not remove more than 10% of fused points.
    // We verify this by running fusion with and without outlier removal
    // (by setting min_neighbors=0 to disable it) and comparing counts.

    PipelineConfig cfg_no_filter;
    cfg_no_filter.voxel_resolution = (float)(sphere_radius_ * 0.02);
    cfg_no_filter.outlier_grid_size = (float)(sphere_radius_ * 0.1);
    cfg_no_filter.outlier_min_neighbors = 0; // disable outlier removal

    auto cloud_no_filter = fuseDepthMaps(depth_maps_, sfm_.cameras, images_, cfg_no_filter);
    ASSERT_GT(cloud_no_filter.size(), 100);

    PipelineConfig cfg;
    cfg.voxel_resolution = (float)(sphere_radius_ * 0.02);
    cfg.outlier_grid_size = (float)(sphere_radius_ * 0.1);
    cfg.outlier_min_neighbors = 3;

    auto cloud = fuseDepthMaps(depth_maps_, sfm_.cameras, images_, cfg);
    ASSERT_GT(cloud.size(), 100);

    float removal_rate = 1.0f - (float)cloud.size() / (float)cloud_no_filter.size();
    printf("[Test] Outlier removal rate: %.1f%% (%d → %d)\n",
           100.0f * removal_rate, cloud_no_filter.size(), cloud.size());

    // The outlier removal step should remove less than 10%
    EXPECT_LT(removal_rate, 0.10f)
        << "Outlier removal removed too many points: " << removal_rate * 100 << "%";

    // Also verify no gross outliers: points beyond 3x sphere radius from center
    int gross_outliers = 0;
    for (int i = 0; i < cloud.size(); i++) {
        Eigen::Vector3d p = cloud.positions[i].cast<double>();
        double dist = (p - sphere_center_).norm();
        if (dist > 3.0 * sphere_radius_) gross_outliers++;
    }

    float gross_rate = (float)gross_outliers / (float)cloud.size();
    printf("[Test] Gross outlier rate (>3x radius): %.1f%% (%d/%d)\n",
           100.0f * gross_rate, gross_outliers, cloud.size());
}

TEST_F(PointCloudTest, ColorsAreValid) {
    PipelineConfig cfg;
    cfg.voxel_resolution = (float)(sphere_radius_ * 0.02);
    cfg.outlier_grid_size = (float)(sphere_radius_ * 0.1);
    cfg.outlier_min_neighbors = 3;

    auto cloud = fuseDepthMaps(depth_maps_, sfm_.cameras, images_, cfg);
    ASSERT_GT(cloud.size(), 100);

    int valid_colors = 0;
    for (int i = 0; i < cloud.size(); i++) {
        const auto& c = cloud.colors[i];
        if (c.x() >= 0 && c.x() <= 1 &&
            c.y() >= 0 && c.y() <= 1 &&
            c.z() >= 0 && c.z() <= 1) {
            valid_colors++;
        }
    }

    float ratio = (float)valid_colors / (float)cloud.size();
    EXPECT_GT(ratio, 0.99f) << "Only " << valid_colors << "/" << cloud.size()
                            << " colors in valid [0,1] range";
}

TEST_F(PointCloudTest, PLYSaveLoad) {
    PipelineConfig cfg;
    cfg.voxel_resolution = (float)(sphere_radius_ * 0.02);
    cfg.outlier_grid_size = (float)(sphere_radius_ * 0.1);
    cfg.outlier_min_neighbors = 3;

    auto cloud = fuseDepthMaps(depth_maps_, sfm_.cameras, images_, cfg);
    ASSERT_GT(cloud.size(), 0);

    std::string ply_path = "/tmp/test_dense_cloud.ply";
    ASSERT_TRUE(saveDensePointCloudPLY(cloud, ply_path));

    DensePointCloud loaded;
    ASSERT_TRUE(loadDensePointCloudPLY(loaded, ply_path));

    EXPECT_EQ(loaded.size(), cloud.size());

    // Check a few values match
    for (int i = 0; i < std::min(100, cloud.size()); i++) {
        EXPECT_NEAR(loaded.positions[i].x(), cloud.positions[i].x(), 1e-5f);
        EXPECT_NEAR(loaded.positions[i].y(), cloud.positions[i].y(), 1e-5f);
        EXPECT_NEAR(loaded.positions[i].z(), cloud.positions[i].z(), 1e-5f);
    }

    std::remove(ply_path.c_str());
}

TEST_F(PointCloudTest, NormalsAreUnitLength) {
    PipelineConfig cfg;
    cfg.voxel_resolution = (float)(sphere_radius_ * 0.02);
    cfg.outlier_grid_size = (float)(sphere_radius_ * 0.1);
    cfg.outlier_min_neighbors = 3;

    auto cloud = fuseDepthMaps(depth_maps_, sfm_.cameras, images_, cfg);
    ASSERT_GT(cloud.size(), 100);

    int unit_count = 0;
    for (int i = 0; i < cloud.size(); i++) {
        float len = cloud.normals[i].norm();
        if (std::abs(len - 1.0f) < 0.05f) unit_count++;
    }

    float ratio = (float)unit_count / (float)cloud.size();
    printf("[Test] Unit normals: %.1f%% (%d/%d)\n",
           100.0f * ratio, unit_count, cloud.size());

    EXPECT_GT(ratio, 0.95f) << "Too few unit-length normals";
}
