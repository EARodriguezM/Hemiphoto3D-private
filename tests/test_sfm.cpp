#include <gtest/gtest.h>
#include "types.h"
#include "sfm/camera_pose.h"
#include "utils/synthetic_data.h"
#include "utils/image_loader.h"
#include "feature_detection/sift_cuda.h"
#include "feature_detection/feature_matching.h"

#include <opencv2/core/eigen.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ============================================================================
// Test fixture: generate synthetic data, detect features, match
// ============================================================================
class SfMTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // Generate synthetic data
        std::string data_dir = "data/synthetic";
        if (!fs::exists(data_dir + "/view_000.jpg")) {
            ASSERT_TRUE(generateSyntheticData(data_dir));
        }

        // Load images
        PipelineConfig cfg;
        cfg.input_dir = data_dir;
        cfg.verbose = false;
        images_ = loadImages(data_dir, cfg);
        ASSERT_EQ(static_cast<int>(images_.size()), 36);

        // Override intrinsics with known ground truth (synthetic data: fx=fy=2000, cx=640, cy=480)
        for (auto& img : images_) {
            img.K = (cv::Mat_<double>(3, 3) << 2000.0, 0.0, 640.0,
                                                0.0, 2000.0, 480.0,
                                                0.0, 0.0, 1.0);
            img.focal_length_px = 2000.0;
        }

        // Detect features
        features_ = detectAllFeatures(images_, 8000, false);
        ASSERT_EQ(features_.size(), images_.size());

        // Match all pairs
        matches_ = matchAllPairs(features_, images_, 0.75f, 30, false);
        ASSERT_GT(static_cast<int>(matches_.size()), 0);

        // Load ground truth from JSON
        loadGroundTruth(data_dir + "/cameras_gt.json");
    }

    static void loadGroundTruth(const std::string& path) {
        // Simple parsing of ground truth camera centers from JSON
        // We just need camera centers for comparison
        std::ifstream f(path);
        ASSERT_TRUE(f.is_open());

        gt_centers_.resize(36);
        gt_rotations_.resize(36);

        std::string line;
        int cam_idx = -1;
        while (std::getline(f, line)) {
            // Parse "center": [x, y, z]
            auto pos = line.find("\"center\"");
            if (pos != std::string::npos) {
                auto bracket = line.find('[', pos);
                if (bracket != std::string::npos) {
                    double x, y, z;
                    sscanf(line.c_str() + bracket, "[%lf,%lf,%lf]", &x, &y, &z);
                    gt_centers_[cam_idx] = Eigen::Vector3d(x, y, z);
                }
            }

            // Parse "R": [[...],[...],[...]]
            pos = line.find("\"R\"");
            if (pos != std::string::npos) {
                auto bracket = line.find("[[", pos);
                if (bracket != std::string::npos) {
                    Eigen::Matrix3d R;
                    sscanf(line.c_str() + bracket, "[[%lf,%lf,%lf],[%lf,%lf,%lf],[%lf,%lf,%lf]]",
                           &R(0,0), &R(0,1), &R(0,2),
                           &R(1,0), &R(1,1), &R(1,2),
                           &R(2,0), &R(2,1), &R(2,2));
                    gt_rotations_[cam_idx] = R;
                }
            }

            // Track camera index from "angle_deg"
            pos = line.find("\"angle_deg\"");
            if (pos != std::string::npos) {
                double angle;
                sscanf(line.c_str() + pos, "\"angle_deg\": %lf", &angle);
                cam_idx = static_cast<int>(std::round(angle / 10.0));
            }
        }
    }

    // Align two sets of camera centers using Umeyama similarity transform
    // Returns (scale, R, t) such that: target ≈ s*R*source + t
    static void umeyamaAlign(
        const std::vector<Eigen::Vector3d>& source,
        const std::vector<Eigen::Vector3d>& target,
        double& scale, Eigen::Matrix3d& R, Eigen::Vector3d& t)
    {
        int n = static_cast<int>(source.size());
        ASSERT_GT(n, 0);

        // Compute centroids
        Eigen::Vector3d src_mean = Eigen::Vector3d::Zero();
        Eigen::Vector3d tgt_mean = Eigen::Vector3d::Zero();
        for (int i = 0; i < n; i++) {
            src_mean += source[i];
            tgt_mean += target[i];
        }
        src_mean /= n;
        tgt_mean /= n;

        // Center the points
        Eigen::MatrixXd src_c(3, n), tgt_c(3, n);
        for (int i = 0; i < n; i++) {
            src_c.col(i) = source[i] - src_mean;
            tgt_c.col(i) = target[i] - tgt_mean;
        }

        // Compute variances
        double src_var = 0;
        for (int i = 0; i < n; i++) src_var += src_c.col(i).squaredNorm();
        src_var /= n;

        // Cross-covariance
        Eigen::Matrix3d S = tgt_c * src_c.transpose() / n;

        // SVD
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();
        Eigen::Vector3d d = svd.singularValues();

        // Handle reflection
        Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
        if ((U * V.transpose()).determinant() < 0) {
            D(2, 2) = -1;
        }

        R = U * D * V.transpose();
        scale = (d.asDiagonal() * D).trace() / src_var;
        t = tgt_mean - scale * R * src_mean;
    }

    static std::vector<ImageData> images_;
    static std::vector<SIFTFeatures> features_;
    static std::vector<ImagePairMatches> matches_;
    static std::vector<Eigen::Vector3d> gt_centers_;
    static std::vector<Eigen::Matrix3d> gt_rotations_;
};

std::vector<ImageData> SfMTest::images_;
std::vector<SIFTFeatures> SfMTest::features_;
std::vector<ImagePairMatches> SfMTest::matches_;
std::vector<Eigen::Vector3d> SfMTest::gt_centers_;
std::vector<Eigen::Matrix3d> SfMTest::gt_rotations_;

// ============================================================================
// Test 1: All 36 cameras registered
// ============================================================================
TEST_F(SfMTest, AllCamerasRegistered) {
    PipelineConfig cfg;
    cfg.verbose = true;
    cfg.max_reprojection_error = 4.0;
    cfg.min_triangulation_angle = 2.0;

    SfMResult result = runIncrementalSfM(features_, matches_, images_, cfg);

    EXPECT_EQ(result.num_registered, 36)
        << "Expected all 36 cameras registered, got " << result.num_registered;
}

// ============================================================================
// Test 2: Mean reprojection error < 1.0 pixel
// ============================================================================
TEST_F(SfMTest, ReprojectionError) {
    PipelineConfig cfg;
    cfg.verbose = false;
    cfg.max_reprojection_error = 4.0;
    cfg.min_triangulation_angle = 2.0;

    SfMResult result = runIncrementalSfM(features_, matches_, images_, cfg);

    printf("Mean reprojection error: %.4f px\n", result.mean_reprojection_error);
    EXPECT_LT(result.mean_reprojection_error, 1.0)
        << "Mean reprojection error " << result.mean_reprojection_error << " >= 1.0 px";
}

// ============================================================================
// Test 3: Pose accuracy — rotation error < 2° per camera, translation error < 5%
// ============================================================================
TEST_F(SfMTest, PoseAccuracy) {
    PipelineConfig cfg;
    cfg.verbose = false;
    cfg.max_reprojection_error = 4.0;
    cfg.min_triangulation_angle = 2.0;

    SfMResult result = runIncrementalSfM(features_, matches_, images_, cfg);
    ASSERT_EQ(result.num_registered, 36);

    // Collect recovered camera centers
    std::vector<Eigen::Vector3d> recovered_centers(36);
    for (const auto& cam : result.cameras) {
        if (cam.is_registered) {
            recovered_centers[cam.image_id] = cam.cam_center();
        }
    }

    // Debug: print a few camera centers
    printf("DEBUG: First 5 recovered centers:\n");
    for (int i = 0; i < 5; i++) {
        printf("  cam %d: (%.6f, %.6f, %.6f)\n", i,
               recovered_centers[i].x(), recovered_centers[i].y(), recovered_centers[i].z());
    }
    printf("DEBUG: First 5 GT centers:\n");
    for (int i = 0; i < 5; i++) {
        printf("  cam %d: (%.6f, %.6f, %.6f)\n", i,
               gt_centers_[i].x(), gt_centers_[i].y(), gt_centers_[i].z());
    }

    // Align recovered to ground truth using Umeyama
    double scale;
    Eigen::Matrix3d R_align;
    Eigen::Vector3d t_align;
    umeyamaAlign(recovered_centers, gt_centers_, scale, R_align, t_align);

    printf("Umeyama alignment: scale=%.4f\n", scale);

    // Compute scene diameter for relative error
    double max_dist = 0;
    for (int i = 0; i < 36; i++)
        for (int j = i + 1; j < 36; j++)
            max_dist = std::max(max_dist, (gt_centers_[i] - gt_centers_[j]).norm());

    double max_rot_err = 0;
    double max_trans_err = 0;
    double mean_rot_err = 0;
    double mean_trans_err = 0;

    for (int i = 0; i < 36; i++) {
        // Translation error (after alignment)
        Eigen::Vector3d aligned_center = scale * R_align * recovered_centers[i] + t_align;
        double trans_err = (aligned_center - gt_centers_[i]).norm();
        double trans_err_pct = trans_err / max_dist * 100.0;

        // Rotation error
        // Align recovered R: R_gt ≈ R_align * R_recovered (but we need to account for the similarity)
        // The aligned rotation for camera i: R_aligned_i = R_recovered_i * R_align^T
        // Then compare R_aligned_i with R_gt_i
        Eigen::Matrix3d R_aligned_i = result.cameras[i].R * R_align.transpose();
        Eigen::Matrix3d R_err = R_aligned_i * gt_rotations_[i].transpose();
        double trace = R_err.trace();
        trace = std::clamp(trace, -1.0, 3.0);
        double rot_err_deg = std::acos((trace - 1.0) / 2.0) * 180.0 / M_PI;

        max_rot_err = std::max(max_rot_err, rot_err_deg);
        max_trans_err = std::max(max_trans_err, trans_err_pct);
        mean_rot_err += rot_err_deg;
        mean_trans_err += trans_err_pct;
    }

    mean_rot_err /= 36;
    mean_trans_err /= 36;

    printf("Rotation error: mean=%.2f°, max=%.2f°\n", mean_rot_err, max_rot_err);
    printf("Translation error: mean=%.2f%%, max=%.2f%% of scene diameter\n", mean_trans_err, max_trans_err);

    EXPECT_LT(max_rot_err, 2.0) << "Max rotation error " << max_rot_err << "° >= 2°";
    EXPECT_LT(max_trans_err, 5.0) << "Max translation error " << max_trans_err << "% >= 5%";
}

// ============================================================================
// Test 4: Sparse point cloud has > 1000 points
// ============================================================================
TEST_F(SfMTest, SparseCloudSize) {
    PipelineConfig cfg;
    cfg.verbose = false;
    cfg.max_reprojection_error = 4.0;
    cfg.min_triangulation_angle = 2.0;

    SfMResult result = runIncrementalSfM(features_, matches_, images_, cfg);

    printf("Sparse point cloud: %zu points\n", result.points.size());
    EXPECT_GT(static_cast<int>(result.points.size()), 1000)
        << "Sparse cloud has only " << result.points.size() << " points (need > 1000)";
}
