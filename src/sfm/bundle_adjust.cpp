#include "sfm/bundle_adjust.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>

#include <cstdio>
#include <cmath>
#include <thread>
#include <vector>

// ============================================================================
// Reprojection error functor for Ceres auto-differentiation
// ============================================================================
struct ReprojectionCost {
    double obs_x, obs_y;
    double fx, fy, cx, cy;

    ReprojectionCost(double ox, double oy, double fx_, double fy_, double cx_, double cy_)
        : obs_x(ox), obs_y(oy), fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}

    template <typename T>
    bool operator()(const T* const cam_angle_axis,   // 3 params
                    const T* const cam_translation,   // 3 params
                    const T* const point_3d,           // 3 params
                    T* residuals) const {
        // 1. Rotate point: P_cam = AngleAxis(aa) * P_world
        T p[3];
        ceres::AngleAxisRotatePoint(cam_angle_axis, point_3d, p);

        // 2. Translate: P_cam += t
        p[0] += cam_translation[0];
        p[1] += cam_translation[1];
        p[2] += cam_translation[2];

        // 3. Project: pixel = K * [P_cam.x/P_cam.z, P_cam.y/P_cam.z]
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];
        T predicted_x = T(fx) * xp + T(cx);
        T predicted_y = T(fy) * yp + T(cy);

        // 4. Residual
        residuals[0] = predicted_x - T(obs_x);
        residuals[1] = predicted_y - T(obs_y);
        return true;
    }

    static ceres::CostFunction* Create(double ox, double oy,
                                        double fx, double fy,
                                        double cx, double cy) {
        return new ceres::AutoDiffCostFunction<ReprojectionCost, 2, 3, 3, 3>(
            new ReprojectionCost(ox, oy, fx, fy, cx, cy));
    }
};

// ============================================================================
// Bundle adjustment implementation
// ============================================================================
static void bundleAdjustImpl(
    std::vector<CameraPose>& cameras,
    std::vector<SparsePoint>& points,
    const std::vector<SIFTFeatures>& features,
    bool fix_first_camera,
    bool optimize_intrinsics,
    bool verbose,
    bool fix_all_cameras = false)
{
    if (points.empty()) return;

    int num_cameras = static_cast<int>(cameras.size());

    // Convert camera rotations to angle-axis
    std::vector<double> cam_aa(num_cameras * 3, 0.0);
    std::vector<double> cam_t(num_cameras * 3, 0.0);

    for (int i = 0; i < num_cameras; i++) {
        if (!cameras[i].is_registered) continue;

        // Eigen rotation matrix to angle-axis via Ceres
        double R[9];
        // Ceres expects column-major rotation matrix
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                R[c * 3 + r] = cameras[i].R(r, c);

        ceres::RotationMatrixToAngleAxis(R, &cam_aa[i * 3]);
        cam_t[i * 3 + 0] = cameras[i].t.x();
        cam_t[i * 3 + 1] = cameras[i].t.y();
        cam_t[i * 3 + 2] = cameras[i].t.z();
    }

    // Convert points to parameter array
    int num_points = static_cast<int>(points.size());
    std::vector<double> pt_data(num_points * 3);
    for (int i = 0; i < num_points; i++) {
        pt_data[i * 3 + 0] = points[i].position.x();
        pt_data[i * 3 + 1] = points[i].position.y();
        pt_data[i * 3 + 2] = points[i].position.z();
    }

    // Build Ceres problem
    ceres::Problem problem;
    int num_residuals = 0;

    for (int pi = 0; pi < num_points; pi++) {
        const auto& pt = points[pi];

        for (size_t obs = 0; obs < pt.image_ids.size(); obs++) {
            int img_id = pt.image_ids[obs];
            int kp_id = pt.keypoint_ids[obs];

            if (img_id < 0 || img_id >= num_cameras) continue;
            if (!cameras[img_id].is_registered) continue;
            if (kp_id < 0 || kp_id >= static_cast<int>(features[img_id].keypoints.size())) continue;

            const cv::KeyPoint& kp = features[img_id].keypoints[kp_id];
            const Eigen::Matrix3d& K = cameras[img_id].K;

            ceres::CostFunction* cost = ReprojectionCost::Create(
                kp.pt.x, kp.pt.y,
                K(0, 0), K(1, 1), K(0, 2), K(1, 2));

            ceres::LossFunction* loss = new ceres::HuberLoss(1.0);

            problem.AddResidualBlock(cost, loss,
                &cam_aa[img_id * 3],
                &cam_t[img_id * 3],
                &pt_data[pi * 3]);

            num_residuals++;
        }
    }

    if (num_residuals == 0) return;

    // Fix cameras as requested
    if (fix_all_cameras) {
        for (int i = 0; i < num_cameras; i++) {
            if (cameras[i].is_registered) {
                problem.SetParameterBlockConstant(&cam_aa[i * 3]);
                problem.SetParameterBlockConstant(&cam_t[i * 3]);
            }
        }
    } else if (fix_first_camera) {
        for (int i = 0; i < num_cameras; i++) {
            if (cameras[i].is_registered) {
                problem.SetParameterBlockConstant(&cam_aa[i * 3]);
                problem.SetParameterBlockConstant(&cam_t[i * 3]);
                break;
            }
        }
    }

    // Solver options
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.preconditioner_type = ceres::SCHUR_JACOBI;
    options.max_num_iterations = 100;
    options.function_tolerance = 1e-6;
    options.gradient_tolerance = 1e-10;
    options.parameter_tolerance = 1e-8;
    options.num_threads = static_cast<int>(std::thread::hardware_concurrency());
    options.minimizer_progress_to_stdout = verbose;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (verbose) {
        printf("BA: %d residuals, %d iterations, initial_cost=%.4f, final_cost=%.4f, %s\n",
               num_residuals, static_cast<int>(summary.iterations.size()),
               summary.initial_cost, summary.final_cost,
               summary.termination_type == ceres::CONVERGENCE ? "CONVERGED" : "NOT_CONVERGED");
    }

    // Convert back: angle-axis to rotation matrix, update cameras
    for (int i = 0; i < num_cameras; i++) {
        if (!cameras[i].is_registered) continue;

        double R[9];
        ceres::AngleAxisToRotationMatrix(&cam_aa[i * 3], R);
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                cameras[i].R(r, c) = R[c * 3 + r];

        cameras[i].t.x() = cam_t[i * 3 + 0];
        cameras[i].t.y() = cam_t[i * 3 + 1];
        cameras[i].t.z() = cam_t[i * 3 + 2];
    }

    // Update points
    for (int i = 0; i < num_points; i++) {
        points[i].position.x() = pt_data[i * 3 + 0];
        points[i].position.y() = pt_data[i * 3 + 1];
        points[i].position.z() = pt_data[i * 3 + 2];
    }
}

// ============================================================================
// Public API: full config version
// ============================================================================
void bundleAdjust(
    std::vector<CameraPose>& cameras,
    std::vector<SparsePoint>& points,
    const std::vector<SIFTFeatures>& features,
    const PipelineConfig& config,
    bool fix_first_camera,
    bool optimize_intrinsics,
    bool fix_all_cameras)
{
    bundleAdjustImpl(cameras, points, features, fix_first_camera, optimize_intrinsics, config.verbose, fix_all_cameras);
}

// ============================================================================
// Public API: simple version (backward compat)
// ============================================================================
bool bundleAdjust(
    std::vector<CameraPose>& cameras,
    std::vector<SparsePoint>& points,
    const std::vector<SIFTFeatures>& features,
    double max_reproj_error)
{
    (void)max_reproj_error;
    bundleAdjustImpl(cameras, points, features, true, false, false);
    return true;
}

// ============================================================================
// Post-BA point cloud filter
// ============================================================================
void filterPointCloud(
    std::vector<CameraPose>& cameras,
    std::vector<SparsePoint>& points,
    const std::vector<SIFTFeatures>& features,
    double max_reproj_error,
    int min_track_length)
{
    // Recompute per-point reprojection errors
    for (auto& pt : points) {
        double total = 0;
        int count = 0;
        for (size_t obs = 0; obs < pt.image_ids.size(); obs++) {
            int img_id = pt.image_ids[obs];
            int kp_id = pt.keypoint_ids[obs];
            if (img_id < 0 || img_id >= static_cast<int>(cameras.size())) continue;
            const auto& cam = cameras[img_id];
            if (!cam.is_registered) continue;
            if (kp_id < 0 || kp_id >= static_cast<int>(features[img_id].keypoints.size())) continue;

            const cv::KeyPoint& kp = features[img_id].keypoints[kp_id];
            Eigen::Vector3d pt_cam = cam.R * pt.position + cam.t;
            if (pt_cam.z() <= 0) { total += 999; count++; continue; }
            Eigen::Vector3d proj = cam.K * pt_cam;
            double u = proj.x() / proj.z();
            double v = proj.y() / proj.z();
            double err = std::sqrt((u - kp.pt.x) * (u - kp.pt.x) + (v - kp.pt.y) * (v - kp.pt.y));
            total += err;
            count++;
        }
        pt.mean_reprojection_error = count > 0 ? static_cast<float>(total / count) : 999.0f;
        pt.track_length = static_cast<int>(pt.image_ids.size());
    }

    size_t before = points.size();
    points.erase(
        std::remove_if(points.begin(), points.end(), [&](const SparsePoint& sp) {
            return sp.mean_reprojection_error > max_reproj_error || sp.track_length < min_track_length;
        }),
        points.end());

    if (points.size() < before) {
        printf("filterPointCloud: removed %zu points (remaining: %zu)\n",
               before - points.size(), points.size());
    }
}
