#include "sfm/camera_pose.h"
#include "sfm/bundle_adjust.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <queue>
#include <cstdio>

namespace fs = std::filesystem;

// ============================================================================
// Helper: convert Eigen intrinsics to cv::Mat
// ============================================================================
static cv::Mat eigenKToCV(const Eigen::Matrix3d& K) {
    cv::Mat Kcv(3, 3, CV_64F);
    cv::eigen2cv(K, Kcv);
    return Kcv;
}

// ============================================================================
// Helper: triangulate a single pair of observations
// Returns the 3D point in world coordinates. Sets 'valid' flag.
// ============================================================================
static Eigen::Vector3d triangulateOne(
    const CameraPose& cam_i, const CameraPose& cam_j,
    const cv::Point2f& pt_i, const cv::Point2f& pt_j,
    double max_reproj_error, double min_tri_angle_deg,
    bool& valid)
{
    valid = false;

    // Build 3x4 projection matrices
    Eigen::Matrix<double, 3, 4> P_i = cam_i.projection();
    Eigen::Matrix<double, 3, 4> P_j = cam_j.projection();

    cv::Mat P1(3, 4, CV_64F), P2(3, 4, CV_64F);
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 4; c++) {
            P1.at<double>(r, c) = P_i(r, c);
            P2.at<double>(r, c) = P_j(r, c);
        }

    // Triangulate
    cv::Mat pts4d;
    std::vector<cv::Point2f> p1 = {pt_i};
    std::vector<cv::Point2f> p2 = {pt_j};
    cv::triangulatePoints(P1, P2, p1, p2, pts4d);

    // triangulatePoints returns CV_32F or CV_64F depending on input types
    cv::Mat pts4d_64;
    pts4d.convertTo(pts4d_64, CV_64F);

    double w = pts4d_64.at<double>(3, 0);
    if (std::abs(w) < 1e-10) return Eigen::Vector3d::Zero();

    Eigen::Vector3d pt_world(
        pts4d_64.at<double>(0, 0) / w,
        pts4d_64.at<double>(1, 0) / w,
        pts4d_64.at<double>(2, 0) / w
    );

    // Check depth in both cameras (must be positive)
    Eigen::Vector3d pt_cam_i = cam_i.R * pt_world + cam_i.t;
    Eigen::Vector3d pt_cam_j = cam_j.R * pt_world + cam_j.t;
    if (pt_cam_i.z() <= 0 || pt_cam_j.z() <= 0) return Eigen::Vector3d::Zero();

    // Check reprojection error in both views
    Eigen::Vector3d proj_i = cam_i.K * pt_cam_i;
    Eigen::Vector3d proj_j = cam_j.K * pt_cam_j;
    double u_i = proj_i.x() / proj_i.z();
    double v_i = proj_i.y() / proj_i.z();
    double u_j = proj_j.x() / proj_j.z();
    double v_j = proj_j.y() / proj_j.z();

    double err_i = std::sqrt((u_i - pt_i.x) * (u_i - pt_i.x) + (v_i - pt_i.y) * (v_i - pt_i.y));
    double err_j = std::sqrt((u_j - pt_j.x) * (u_j - pt_j.x) + (v_j - pt_j.y) * (v_j - pt_j.y));

    if (err_i > max_reproj_error || err_j > max_reproj_error) return Eigen::Vector3d::Zero();

    // Check triangulation angle
    Eigen::Vector3d center_i = cam_i.cam_center();
    Eigen::Vector3d center_j = cam_j.cam_center();
    Eigen::Vector3d ray_i = (pt_world - center_i).normalized();
    Eigen::Vector3d ray_j = (pt_world - center_j).normalized();
    double cos_angle = ray_i.dot(ray_j);
    cos_angle = std::clamp(cos_angle, -1.0, 1.0);
    double angle_deg = std::acos(cos_angle) * 180.0 / M_PI;

    if (angle_deg < min_tri_angle_deg) return Eigen::Vector3d::Zero();

    valid = true;
    return pt_world;
}

// ============================================================================
// Helper: compute mean reprojection error for all points
// ============================================================================
static double computeMeanReprojError(
    const std::vector<CameraPose>& cameras,
    const std::vector<SparsePoint>& points,
    const std::vector<SIFTFeatures>& features)
{
    double total_error = 0;
    int count = 0;

    for (const auto& pt : points) {
        for (size_t obs = 0; obs < pt.image_ids.size(); obs++) {
            int img_id = pt.image_ids[obs];
            int kp_id = pt.keypoint_ids[obs];

            // Find camera
            const CameraPose* cam = nullptr;
            for (const auto& c : cameras) {
                if (c.image_id == img_id && c.is_registered) {
                    cam = &c;
                    break;
                }
            }
            if (!cam) continue;

            const cv::KeyPoint& kp = features[img_id].keypoints[kp_id];
            Eigen::Vector3d pt_cam = cam->R * pt.position + cam->t;
            if (pt_cam.z() <= 0) continue;
            Eigen::Vector3d proj = cam->K * pt_cam;
            double u = proj.x() / proj.z();
            double v = proj.y() / proj.z();
            double err = std::sqrt((u - kp.pt.x) * (u - kp.pt.x) + (v - kp.pt.y) * (v - kp.pt.y));
            total_error += err;
            count++;
        }
    }

    return count > 0 ? total_error / count : 0;
}

// ============================================================================
// Helper: update per-point reprojection errors
// ============================================================================
static void updatePointErrors(
    std::vector<SparsePoint>& points,
    const std::vector<CameraPose>& cameras,
    const std::vector<SIFTFeatures>& features)
{
    for (auto& pt : points) {
        double total = 0;
        int count = 0;
        for (size_t obs = 0; obs < pt.image_ids.size(); obs++) {
            int img_id = pt.image_ids[obs];
            int kp_id = pt.keypoint_ids[obs];
            const CameraPose* cam = nullptr;
            for (const auto& c : cameras) {
                if (c.image_id == img_id && c.is_registered) {
                    cam = &c;
                    break;
                }
            }
            if (!cam) continue;
            const cv::KeyPoint& kp = features[img_id].keypoints[kp_id];
            Eigen::Vector3d pt_cam = cam->R * pt.position + cam->t;
            if (pt_cam.z() <= 0) continue;
            Eigen::Vector3d proj = cam->K * pt_cam;
            double u = proj.x() / proj.z();
            double v = proj.y() / proj.z();
            double err = std::sqrt((u - kp.pt.x) * (u - kp.pt.x) + (v - kp.pt.y) * (v - kp.pt.y));
            total += err;
            count++;
        }
        pt.mean_reprojection_error = count > 0 ? static_cast<float>(total / count) : 0;
        pt.track_length = static_cast<int>(pt.image_ids.size());
    }
}

// ============================================================================
// Helper: build lookup from (image_id, keypoint_id) -> point index
// ============================================================================
using ObsKey = std::pair<int, int>;
struct ObsKeyHash {
    size_t operator()(const ObsKey& k) const {
        return std::hash<int>()(k.first) ^ (std::hash<int>()(k.second) << 16);
    }
};
using ObsMap = std::unordered_map<ObsKey, int, ObsKeyHash>;

static ObsMap buildObservationMap(const std::vector<SparsePoint>& points) {
    ObsMap map;
    for (int i = 0; i < static_cast<int>(points.size()); i++) {
        for (size_t obs = 0; obs < points[i].image_ids.size(); obs++) {
            map[{points[i].image_ids[obs], points[i].keypoint_ids[obs]}] = i;
        }
    }
    return map;
}

// ============================================================================
// Helper: find matches involving a specific image
// ============================================================================
static std::vector<const ImagePairMatches*> findMatchesForImage(
    int image_id,
    const std::vector<ImagePairMatches>& all_matches)
{
    std::vector<const ImagePairMatches*> result;
    for (const auto& m : all_matches) {
        if (m.image_i == image_id || m.image_j == image_id)
            result.push_back(&m);
    }
    return result;
}

// ============================================================================
// Phase 1: Find best initial pair and initialize
// ============================================================================

// Try a pair: compute E, recoverPose, triangulate.
// Returns count of valid 3D points and median parallax angle.
static int tryPairFull(
    int pair_idx,
    const std::vector<SIFTFeatures>& features,
    const std::vector<ImagePairMatches>& matches,
    const std::vector<ImageData>& images,
    const PipelineConfig& config,
    cv::Mat& R_out, cv::Mat& t_out, cv::Mat& mask_out,
    double& median_angle_out)
{
    median_angle_out = 180.0;
    const auto& m = matches[pair_idx];
    int id_i = m.image_i;
    int id_j = m.image_j;

    std::vector<cv::Point2f> pts_i, pts_j;
    for (const auto& fm : m.matches) {
        pts_i.push_back(features[id_i].keypoints[fm.idx_i].pt);
        pts_j.push_back(features[id_j].keypoints[fm.idx_j].pt);
    }

    double focal = images[id_i].K.at<double>(0, 0);
    cv::Point2d pp(images[id_i].K.at<double>(0, 2), images[id_i].K.at<double>(1, 2));

    cv::Mat E, mask;
    E = cv::findEssentialMat(pts_i, pts_j, focal, pp, cv::RANSAC, 0.999, 1.0, mask);
    if (E.empty()) return 0;

    cv::Mat K = (cv::Mat_<double>(3, 3) << focal, 0, pp.x, 0, focal, pp.y, 0, 0, 1);
    cv::Mat R, t;
    int recover_good = cv::recoverPose(E, pts_i, pts_j, K, R, t, mask);
    if (recover_good < 10) return 0;

    // Build temporary cameras for triangulation test
    CameraPose cam_a, cam_b;
    cam_a.R = Eigen::Matrix3d::Identity();
    cam_a.t = Eigen::Vector3d::Zero();
    cv::cv2eigen(images[id_i].K, cam_a.K);

    Eigen::Matrix3d R_e;
    Eigen::Vector3d t_e;
    cv::cv2eigen(R, R_e);
    cv::cv2eigen(t, t_e);
    cam_b.R = R_e;
    cam_b.t = t_e;
    cv::cv2eigen(images[id_j].K, cam_b.K);

    Eigen::Vector3d center_a = cam_a.cam_center();
    Eigen::Vector3d center_b = cam_b.cam_center();

    // Count successfully triangulated points and collect parallax angles
    int valid_count = 0;
    std::vector<double> angles;
    for (size_t k = 0; k < m.matches.size(); k++) {
        if (mask.at<uchar>(static_cast<int>(k)) == 0) continue;

        bool valid = false;
        Eigen::Vector3d pt3d = triangulateOne(cam_a, cam_b,
                       features[id_i].keypoints[m.matches[k].idx_i].pt,
                       features[id_j].keypoints[m.matches[k].idx_j].pt,
                       config.max_reprojection_error, config.min_triangulation_angle,
                       valid);
        if (valid) {
            valid_count++;
            Eigen::Vector3d ray_a = (pt3d - center_a).normalized();
            Eigen::Vector3d ray_b = (pt3d - center_b).normalized();
            double cos_a = std::clamp(ray_a.dot(ray_b), -1.0, 1.0);
            angles.push_back(std::acos(cos_a) * 180.0 / M_PI);
        }
    }

    if (valid_count > 0) {
        R_out = R;
        t_out = t;
        mask_out = mask;
        std::sort(angles.begin(), angles.end());
        median_angle_out = angles[angles.size() / 2];
    }
    return valid_count;
}

static bool initializePair(
    const std::vector<SIFTFeatures>& features,
    const std::vector<ImagePairMatches>& matches,
    const std::vector<ImageData>& images,
    const PipelineConfig& config,
    std::vector<CameraPose>& cameras,
    std::vector<SparsePoint>& points,
    int& init_id_i, int& init_id_j,
    bool verbose)
{
    // Try all viable pairs with tryPairFull, scoring by triangulated count
    // weighted by parallax angle quality (prefer moderate angles 3-45°)
    struct PairCandidate {
        int idx;
        int num_inliers;
    };
    std::vector<PairCandidate> candidates;
    for (int i = 0; i < static_cast<int>(matches.size()); i++) {
        const auto& m = matches[i];
        if (m.num_inliers < 30) continue;
        candidates.push_back({i, m.num_inliers});
    }

    // Sort by inlier count descending
    std::sort(candidates.begin(), candidates.end(),
              [](const PairCandidate& a, const PairCandidate& b) { return a.num_inliers > b.num_inliers; });

    // Try candidates and score by triangulated count * parallax quality
    int chosen_idx = -1;
    cv::Mat R_rel, t_rel, mask_E;
    double best_score = 0;
    int best_tri = 0;

    int max_tries = std::min(static_cast<int>(candidates.size()), 300);
    for (int trial = 0; trial < max_tries; trial++) {
        int pi = candidates[trial].idx;
        cv::Mat R_try, t_try, mask_try;
        double median_angle = 180.0;
        int tri = tryPairFull(pi, features, matches, images, config, R_try, t_try, mask_try, median_angle);

        if (tri < 20) continue;

        // Score: prefer moderate parallax (5-45°), penalize very wide parallax
        double angle_weight = 1.0;
        if (median_angle >= 3.0 && median_angle <= 45.0) angle_weight = 2.0;
        else if (median_angle <= 60.0) angle_weight = 1.0;
        else angle_weight = 0.3;  // penalize wide-baseline pairs

        // Also prefer small image index difference (likely adjacent in turntable)
        int idx_diff = std::abs(matches[pi].image_i - matches[pi].image_j);
        int n_imgs = static_cast<int>(images.size());
        idx_diff = std::min(idx_diff, n_imgs - idx_diff);  // circular
        double prox_weight = (idx_diff <= 5) ? 2.0 : 1.0;

        double score = tri * angle_weight * prox_weight;

        if (verbose && (trial < 10 || score > best_score)) {
            printf("SfM: Pair (%d, %d): inliers=%d, triangulated=%d, median_angle=%.1f°, score=%.0f\n",
                   matches[pi].image_i, matches[pi].image_j, candidates[trial].num_inliers,
                   tri, median_angle, score);
        }

        if (score > best_score) {
            best_score = score;
            best_tri = tri;
            chosen_idx = pi;
            R_rel = R_try;
            t_rel = t_try;
            mask_E = mask_try;
        }
    }

    if (chosen_idx < 0 || best_tri < 10) {
        fprintf(stderr, "SfM: No pair produced enough triangulated points (best: %d)\n", best_tri);
        return false;
    }

    const auto& pair = matches[chosen_idx];
    int id_i = pair.image_i;
    int id_j = pair.image_j;
    init_id_i = id_i;
    init_id_j = id_j;

    if (verbose) {
        printf("SfM Phase 1: Chosen pair (%d, %d) — %d match inliers, %d triangulated points\n",
               id_i, id_j, pair.num_inliers, best_tri);
    }

    // Camera 0 = identity (world frame)
    CameraPose& cam_i = cameras[id_i];
    cam_i.R = Eigen::Matrix3d::Identity();
    cam_i.t = Eigen::Vector3d::Zero();
    cv::cv2eigen(images[id_i].K, cam_i.K);
    cam_i.image_id = id_i;
    cam_i.is_registered = true;

    // Camera 1 = relative pose from recoverPose
    CameraPose& cam_j = cameras[id_j];
    Eigen::Matrix3d R_eigen;
    Eigen::Vector3d t_eigen;
    cv::cv2eigen(R_rel, R_eigen);
    cv::cv2eigen(t_rel, t_eigen);
    cam_j.R = R_eigen;
    cam_j.t = t_eigen;
    cv::cv2eigen(images[id_j].K, cam_j.K);
    cam_j.image_id = id_j;
    cam_j.is_registered = true;

    // Triangulate initial points using the chosen pair's matches
    for (size_t m = 0; m < pair.matches.size(); m++) {
        if (mask_E.at<uchar>(static_cast<int>(m)) == 0) continue;

        const auto& fm = pair.matches[m];
        bool valid = false;
        Eigen::Vector3d pt3d = triangulateOne(
            cam_i, cam_j,
            features[id_i].keypoints[fm.idx_i].pt,
            features[id_j].keypoints[fm.idx_j].pt,
            config.max_reprojection_error, config.min_triangulation_angle,
            valid);

        if (valid) {
            SparsePoint sp;
            sp.position = pt3d;
            sp.color = Eigen::Vector3f(0.8f, 0.8f, 0.8f);
            sp.image_ids = {id_i, id_j};
            sp.keypoint_ids = {fm.idx_i, fm.idx_j};
            sp.track_length = 2;
            sp.mean_reprojection_error = 0;
            points.push_back(sp);
        }
    }

    if (verbose) {
        printf("SfM Phase 1: Triangulated %zu initial points\n", points.size());
    }

    return true;
}

// ============================================================================
// Phase 2: Incremental registration (plain PnP — mirror disambiguation is
// handled post-hoc by disambiguateTurntable)
// ============================================================================

static void incrementalRegistration(
    const std::vector<SIFTFeatures>& features,
    const std::vector<ImagePairMatches>& all_matches,
    const std::vector<ImageData>& images,
    const PipelineConfig& config,
    std::vector<CameraPose>& cameras,
    std::vector<SparsePoint>& points,
    int init_id_i, int init_id_j,
    bool verbose)
{
    int num_images = static_cast<int>(images.size());
    int newly_registered = 0;

    auto circDist = [&](int a, int b) -> int {
        int d = std::abs(a - b);
        return std::min(d, num_images - d);
    };

    // Build sequential order: sort by distance to nearest init camera.
    std::vector<int> seq_order;
    for (int i = 0; i < num_images; i++) {
        if (i == init_id_i || i == init_id_j) continue;
        seq_order.push_back(i);
    }
    std::sort(seq_order.begin(), seq_order.end(), [&](int a, int b) {
        int da = std::min(circDist(a, init_id_i), circDist(a, init_id_j));
        int db = std::min(circDist(b, init_id_i), circDist(b, init_id_j));
        return da < db;
    });

    if (verbose) {
        printf("SfM: Init pair (%d, %d), first 10 in seq_order: ", init_id_i, init_id_j);
        for (int i = 0; i < std::min(10, static_cast<int>(seq_order.size())); i++)
            printf("%d ", seq_order[i]);
        printf("\n");
    }

    // Helper: register a single camera via PnP.
    auto tryRegisterPnP = [&](int img) -> bool {
        ObsMap obs_map = buildObservationMap(points);

        std::vector<cv::Point3f> obj_pts;
        std::vector<cv::Point2f> img_pts;
        std::vector<int> pt_indices, kp_indices;
        std::unordered_set<int> used_kps;

        auto img_matches = findMatchesForImage(img, all_matches);
        for (const auto* mp : img_matches) {
            int other = (mp->image_i == img) ? mp->image_j : mp->image_i;
            if (!cameras[other].is_registered) continue;
            if (circDist(img, other) > 5) continue;

            for (const auto& fm : mp->matches) {
                int kp_this  = (mp->image_i == img) ? fm.idx_i : fm.idx_j;
                int kp_other = (mp->image_i == img) ? fm.idx_j : fm.idx_i;
                if (used_kps.count(kp_this)) continue;

                auto it = obs_map.find({other, kp_other});
                if (it == obs_map.end()) continue;

                const auto& sp = points[it->second];
                obj_pts.push_back(cv::Point3f(
                    static_cast<float>(sp.position.x()),
                    static_cast<float>(sp.position.y()),
                    static_cast<float>(sp.position.z())));
                img_pts.push_back(features[img].keypoints[kp_this].pt);
                pt_indices.push_back(it->second);
                kp_indices.push_back(kp_this);
                used_kps.insert(kp_this);
            }
        }

        if (static_cast<int>(obj_pts.size()) < 6) return false;

        cv::Mat K_cv = images[img].K.clone();
        cv::Mat dist = images[img].dist_coeffs;
        cv::Mat rvec, tvec, inlier_indices;

        bool ok = cv::solvePnPRansac(
            obj_pts, img_pts, K_cv, dist,
            rvec, tvec, false, 10000,
            static_cast<float>(config.max_reprojection_error),
            0.99, inlier_indices, cv::SOLVEPNP_EPNP);

        if (!ok || inlier_indices.empty() || inlier_indices.rows < 6) return false;

        // Extract inliers
        std::vector<cv::Point3f> inlier_obj;
        std::vector<cv::Point2f> inlier_img;
        std::vector<int> inlier_pt_idx, inlier_kp_idx;
        for (int idx = 0; idx < inlier_indices.rows; idx++) {
            int k = inlier_indices.at<int>(idx);
            inlier_obj.push_back(obj_pts[k]);
            inlier_img.push_back(img_pts[k]);
            inlier_pt_idx.push_back(pt_indices[k]);
            inlier_kp_idx.push_back(kp_indices[k]);
        }
        int num_inliers = static_cast<int>(inlier_obj.size());

        // Refine with iterative PnP
        cv::solvePnP(inlier_obj, inlier_img, K_cv, dist,
                     rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);

        cv::Mat R_cv;
        cv::Rodrigues(rvec, R_cv);
        Eigen::Matrix3d R_pnp;
        Eigen::Vector3d t_pnp;
        cv::cv2eigen(R_cv, R_pnp);
        cv::cv2eigen(tvec, t_pnp);

        Eigen::Vector3d center = -R_pnp.transpose() * t_pnp;
        if (!center.allFinite()) return false;

        // Register camera
        cameras[img].R = R_pnp;
        cameras[img].t = t_pnp;
        cv::cv2eigen(K_cv, cameras[img].K);
        cameras[img].image_id = img;
        cameras[img].is_registered = true;
        newly_registered++;

        if (verbose) {
            printf("SfM: Image %d registered (%d inliers, center=(%.4f,%.4f,%.4f))\n",
                   img, num_inliers, center.x(), center.y(), center.z());
        }

        // Add observations for inliers
        for (int idx = 0; idx < num_inliers; idx++) {
            int pt_idx = inlier_pt_idx[idx];
            int kp_idx = inlier_kp_idx[idx];
            bool already = false;
            for (size_t obs = 0; obs < points[pt_idx].image_ids.size(); obs++) {
                if (points[pt_idx].image_ids[obs] == img) { already = true; break; }
            }
            if (!already) {
                points[pt_idx].image_ids.push_back(img);
                points[pt_idx].keypoint_ids.push_back(kp_idx);
                points[pt_idx].track_length =
                    static_cast<int>(points[pt_idx].image_ids.size());
            }
        }

        // Triangulate new points with nearby registered cameras
        {
            auto img_matches2 = findMatchesForImage(img, all_matches);
            ObsMap obs_map_new = buildObservationMap(points);

            for (const auto* mp : img_matches2) {
                int other = (mp->image_i == img) ? mp->image_j : mp->image_i;
                if (!cameras[other].is_registered) continue;
                if (circDist(img, other) > 5) continue;

                for (const auto& fm : mp->matches) {
                    int kp_this  = (mp->image_i == img) ? fm.idx_i : fm.idx_j;
                    int kp_other = (mp->image_i == img) ? fm.idx_j : fm.idx_i;

                    if (obs_map_new.count({img, kp_this}) ||
                        obs_map_new.count({other, kp_other}))
                        continue;

                    bool valid = false;
                    Eigen::Vector3d pt3d = triangulateOne(
                        cameras[img], cameras[other],
                        features[img].keypoints[kp_this].pt,
                        features[other].keypoints[kp_other].pt,
                        config.max_reprojection_error,
                        config.min_triangulation_angle,
                        valid);

                    if (valid) {
                        SparsePoint sp;
                        sp.position = pt3d;
                        sp.color = Eigen::Vector3f(0.8f, 0.8f, 0.8f);
                        sp.image_ids = {img, other};
                        sp.keypoint_ids = {kp_this, kp_other};
                        sp.track_length = 2;
                        sp.mean_reprojection_error = 0;
                        points.push_back(sp);

                        int new_idx = static_cast<int>(points.size()) - 1;
                        obs_map_new[{img, kp_this}] = new_idx;
                        obs_map_new[{other, kp_other}] = new_idx;
                    }
                }
            }
        }

        if (verbose) {
            printf("SfM: Total points after image %d: %zu\n", img, points.size());
        }

        // Periodic BA every 5 cameras
        if (newly_registered % 5 == 0) {
            if (verbose)
                printf("SfM: Running periodic BA (%d new images)...\n", newly_registered);
            bundleAdjust(cameras, points, features, config.max_reprojection_error);
        }

        return true;
    };

    // Pass 1: sequential registration
    for (int img : seq_order) {
        if (cameras[img].is_registered) continue;
        tryRegisterPnP(img);
    }

    // Pass 2: retry any that failed (more neighbors now available)
    for (int img : seq_order) {
        if (cameras[img].is_registered) continue;
        if (verbose) printf("SfM: Retry pass for image %d\n", img);
        tryRegisterPnP(img);
    }
}

// ============================================================================
// Post-SfM: Force turntable geometry to resolve 180° texture ambiguity.
//
// When texture repeats every 180°, SfM compresses the angular scale — cameras
// 180° apart get nearly identical poses. This function:
// 1. Extracts the turntable rotation axis from consecutive camera rotations
// 2. Computes each camera's compressed rotation angle about that axis
// 3. Rescales to the correct equidistant spacing (360°/N per camera)
// 4. Recomputes all camera poses from the turntable model
// 5. Deletes all 3D points and re-triangulates with corrected poses
// ============================================================================
static void disambiguateTurntable(
    std::vector<CameraPose>& cameras,
    std::vector<SparsePoint>& points,
    const std::vector<SIFTFeatures>& features,
    const std::vector<ImagePairMatches>& all_matches,
    const PipelineConfig& config,
    int init_cam,
    bool verbose)
{
    int N = static_cast<int>(cameras.size());
    if (N < 6) return;

    auto circDist = [&](int a, int b) -> int {
        int d = std::abs(a - b);
        return std::min(d, N - d);
    };

    // Count registered cameras
    int n_reg = 0;
    for (const auto& cam : cameras)
        if (cam.is_registered) n_reg++;
    if (n_reg < 6) return;

    // Step 1: Extract turntable axis from camera center plane normal.
    // Camera centers lie approximately in a plane; the plane normal is the turntable axis.
    std::vector<Eigen::Vector3d> centers;
    for (int i = 0; i < N; i++) {
        if (!cameras[i].is_registered) continue;
        centers.push_back(cameras[i].cam_center());
    }
    Eigen::Vector3d cam_centroid = Eigen::Vector3d::Zero();
    for (const auto& c : centers) cam_centroid += c;
    cam_centroid /= static_cast<double>(centers.size());

    Eigen::MatrixXd A(static_cast<int>(centers.size()), 3);
    for (int i = 0; i < static_cast<int>(centers.size()); i++)
        A.row(i) = (centers[i] - cam_centroid).transpose();

    Eigen::JacobiSVD<Eigen::MatrixXd> svd_plane(A, Eigen::ComputeFullV);
    Eigen::Vector3d plane_axis = svd_plane.matrixV().col(2); // smallest singular value = plane normal

    // Use the init camera's relative rotation to determine turntable axis.
    // The init camera (ref) has R=I. The second init camera's R encodes the turntable rotation.
    // Find the camera nearest to init_cam+5 (the likely second init camera) for the most
    // reliable rotation axis. Fall back to plane normal if unavailable.
    Eigen::Vector3d turntable_axis = plane_axis;

    // Collect rotation axes from multiple camera pairs for a robust estimate
    Eigen::Vector3d rot_axis_sum = Eigen::Vector3d::Zero();
    double weight_sum = 0;
    for (int i = 0; i < N; i++) {
        if (!cameras[i].is_registered || i == init_cam) continue;
        Eigen::Matrix3d R_rel = cameras[i].R * cameras[init_cam].R.transpose();
        Eigen::AngleAxisd aa(R_rel);
        double angle_deg = aa.angle() * 180.0 / M_PI;
        // Only use cameras with small-to-moderate relative rotation (not near 0° or 180°)
        if (angle_deg < 0.3 || angle_deg > 100.0) continue;
        Eigen::Vector3d ax = aa.axis();
        if (ax.dot(plane_axis) < 0) ax = -ax;
        double weight = aa.angle(); // weight by angle magnitude
        rot_axis_sum += ax * weight;
        weight_sum += weight;
    }

    if (weight_sum > 0) {
        turntable_axis = rot_axis_sum.normalized();
    }

    // Step 2: For each camera, compute its rotation angle about the turntable axis
    // relative to the init camera (which has the most reliable pose from E-matrix).

    // Use init camera as reference — it was set to R=I, t=0 in initializePair
    int ref_cam = init_cam;
    if (ref_cam < 0 || ref_cam >= N || !cameras[ref_cam].is_registered) {
        // Fallback to first registered
        for (int i = 0; i < N; i++) {
            if (cameras[i].is_registered) { ref_cam = i; break; }
        }
    }
    if (ref_cam < 0) return;

    std::vector<double> theta_sfm(N, 0);
    for (int i = 0; i < N; i++) {
        if (!cameras[i].is_registered) continue;
        Eigen::Matrix3d R_rel = cameras[i].R * cameras[ref_cam].R.transpose();
        Eigen::AngleAxisd aa(R_rel);
        // Sign: project rotation axis onto turntable axis
        double sign = (aa.axis().dot(turntable_axis) >= 0) ? 1.0 : -1.0;
        theta_sfm[i] = sign * aa.angle();
    }

    // Step 3: Determine the turntable direction.
    // Check if increasing camera index corresponds to increasing θ_sfm.
    double dir_sum = 0;
    for (int i = 0; i < N; i++) {
        int j = (i + 1) % N;
        if (!cameras[i].is_registered || !cameras[j].is_registered) continue;
        double dtheta = theta_sfm[j] - theta_sfm[i];
        // Signed circular step in terms of camera index
        dir_sum += dtheta;
    }
    int dir = (dir_sum >= 0) ? +1 : -1;

    double target_step = 2.0 * M_PI / N;  // 10° for 36 cameras

    // Step 4: Detect if turntable correction is needed.
    // With texture periodicity (90° or 180°), cameras at different real positions
    // get placed at the same SfM position. Check for duplicate positions.
    int num_duplicates = 0;
    for (int i = 0; i < N; i++) {
        if (!cameras[i].is_registered) continue;
        // Check if any camera with distant index has nearly the same rotation
        for (int j = i + N/6; j < N; j++) {
            if (!cameras[j].is_registered) continue;
            double angle_diff = std::abs(theta_sfm[j] - theta_sfm[i]);
            if (angle_diff < 5.0 * M_PI / 180.0) { // < 5° apart in SfM
                num_duplicates++;
            }
        }
    }

    if (verbose) {
        printf("SfM disambiguate: turntable axis=(%.3f,%.3f,%.3f), dir=%+d\n",
               turntable_axis.x(), turntable_axis.y(), turntable_axis.z(), dir);
        printf("SfM disambiguate: Found %d duplicate-position camera pairs\n", num_duplicates);
    }

    if (num_duplicates == 0) {
        if (verbose) printf("SfM disambiguate: No duplicates found, skipping\n");
        return;
    }

    // Step 5: Estimate turntable center O from viewing ray intersection.
    // Each camera looks roughly at the object center. O minimizes distance to all viewing rays.
    // Viewing ray: C_i + t * d_i, where d_i = R_i^T * [0,0,1] (camera z-axis in world)
    // O = (sum (I - d*d^T))^{-1} * sum (I - d*d^T) * C_i
    Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();
    for (int i = 0; i < N; i++) {
        if (!cameras[i].is_registered) continue;
        Eigen::Vector3d d_i = cameras[i].R.transpose() * Eigen::Vector3d(0, 0, 1);
        d_i.normalize();
        Eigen::Matrix3d P = Eigen::Matrix3d::Identity() - d_i * d_i.transpose();
        M += P;
        b += P * cameras[i].cam_center();
    }
    Eigen::Vector3d O = M.ldlt().solve(b);

    // Camera radius: average distance from cameras to O (in turntable plane)
    double radius = 0;
    int r_count = 0;
    for (int i = 0; i < N; i++) {
        if (!cameras[i].is_registered) continue;
        Eigen::Vector3d d = cameras[i].cam_center() - O;
        d -= d.dot(turntable_axis) * turntable_axis; // project onto turntable plane
        radius += d.norm();
        r_count++;
    }
    radius /= r_count;

    if (verbose) {
        printf("SfM disambiguate: turntable center=(%.4f,%.4f,%.4f), radius=%.4f\n",
               O.x(), O.y(), O.z(), radius);
    }

    // Step 6: Compute corrected poses for all cameras.
    // The axis direction and orbit direction have a combined 4-way ambiguity.
    // We try all 4 and pick the one where triangulated points are in front of cameras.

    Eigen::Matrix3d R_ref = cameras[ref_cam].R;
    Eigen::Vector3d C_ref = cameras[ref_cam].cam_center();
    Eigen::Vector3d C_ref_from_O = C_ref - O;
    Eigen::Vector3d C_ref_plane = C_ref_from_O - C_ref_from_O.dot(turntable_axis) * turntable_axis;
    double C_ref_height = C_ref_from_O.dot(turntable_axis);

    // Step 6b: Verify axis direction — triangulate test points and check depth.
    // If most triangulated points are behind cameras, the axis is flipped.
    auto applyTurntableCorrection = [&](const Eigen::Vector3d& axis, int direction) {
        for (int i = 0; i < N; i++) {
            if (!cameras[i].is_registered) continue;
            int steps = i - ref_cam;
            double angle_orbit = steps * target_step * direction;
            cameras[i].R = R_ref * Eigen::AngleAxisd(-angle_orbit, axis).toRotationMatrix();
            Eigen::Vector3d C_i = O + Eigen::AngleAxisd(angle_orbit, axis).toRotationMatrix() * C_ref_plane
                                  + C_ref_height * axis;
            cameras[i].t = -cameras[i].R * C_i;
        }
    };

    // Count how many points can be successfully triangulated between consecutive cameras.
    // The correct axis/direction produces the most valid triangulations (cheirality check passes).
    auto countValidTriangulations = [&]() -> int {
        int valid_count = 0;
        int tested = 0;
        for (int i = 0; i < N && tested < 100; i++) {
            int j = (i + 1) % N;
            if (!cameras[i].is_registered || !cameras[j].is_registered) continue;
            auto img_matches = findMatchesForImage(i, all_matches);
            for (const auto* mp : img_matches) {
                int other = (mp->image_i == i) ? mp->image_j : mp->image_i;
                if (other != j || mp->matches.empty()) continue;
                int limit = std::min(10, static_cast<int>(mp->matches.size()));
                for (int m = 0; m < limit; m++) {
                    const auto& fm = mp->matches[m];
                    int kp_i = (mp->image_i == i) ? fm.idx_i : fm.idx_j;
                    int kp_j = (mp->image_i == i) ? fm.idx_j : fm.idx_i;
                    bool valid = false;
                    triangulateOne(
                        cameras[i], cameras[j],
                        features[i].keypoints[kp_i].pt, features[j].keypoints[kp_j].pt,
                        config.max_reprojection_error, config.min_triangulation_angle, valid);
                    if (valid) valid_count++;
                    tested++;
                }
                break;
            }
        }
        return valid_count;
    };

    // Try 4 configurations: {axis, -axis} × {dir, -dir}
    // The correct one has most triangulated points in front of cameras.
    struct Config { Eigen::Vector3d ax; int d; int score; };
    std::vector<Config> configs = {
        {turntable_axis, dir, 0},
        {-turntable_axis, dir, 0},
        {turntable_axis, -dir, 0},
        {-turntable_axis, -dir, 0}
    };

    for (auto& cfg : configs) {
        applyTurntableCorrection(cfg.ax, cfg.d);
        cfg.score = countValidTriangulations();
    }

    int best_cfg = 0;
    for (int i = 1; i < 4; i++) {
        if (configs[i].score > configs[best_cfg].score)
            best_cfg = i;
    }

    // Apply the best configuration
    turntable_axis = configs[best_cfg].ax;
    dir = configs[best_cfg].d;
    applyTurntableCorrection(turntable_axis, dir);

    if (verbose) {
        printf("SfM disambiguate: Best config: axis=(%.3f,%.3f,%.3f), dir=%+d (scores: %d,%d,%d,%d)\n",
               turntable_axis.x(), turntable_axis.y(), turntable_axis.z(), dir,
               configs[0].score, configs[1].score, configs[2].score, configs[3].score);
        for (int i = 0; i < std::min(N, 6); i++) {
            if (!cameras[i].is_registered) continue;
            Eigen::Vector3d c = cameras[i].cam_center();
            printf("  cam %d: (%.4f, %.4f, %.4f)\n", i, c.x(), c.y(), c.z());
        }
    }

    // Step 7: Delete all 3D points and re-triangulate from scratch
    points.clear();

    // Triangulate from all pairs of nearby cameras
    for (int i = 0; i < N; i++) {
        if (!cameras[i].is_registered) continue;
        ObsMap obs_map = buildObservationMap(points);
        auto img_matches = findMatchesForImage(i, all_matches);

        for (const auto* mp : img_matches) {
            int other = (mp->image_i == i) ? mp->image_j : mp->image_i;
            if (other <= i) continue; // avoid duplicates
            if (!cameras[other].is_registered) continue;
            if (circDist(i, other) > 5) continue; // nearby cameras

            for (const auto& fm : mp->matches) {
                int kp_i = (mp->image_i == i) ? fm.idx_i : fm.idx_j;
                int kp_other = (mp->image_i == i) ? fm.idx_j : fm.idx_i;

                if (obs_map.count({i, kp_i}) || obs_map.count({other, kp_other}))
                    continue;

                bool valid = false;
                Eigen::Vector3d pt3d = triangulateOne(
                    cameras[i], cameras[other],
                    features[i].keypoints[kp_i].pt,
                    features[other].keypoints[kp_other].pt,
                    config.max_reprojection_error, // triangulation reproj threshold
                    config.min_triangulation_angle,
                    valid);

                if (valid) {
                    SparsePoint sp;
                    sp.position = pt3d;
                    sp.color = Eigen::Vector3f(0.8f, 0.8f, 0.8f);
                    sp.image_ids = {i, other};
                    sp.keypoint_ids = {kp_i, kp_other};
                    sp.track_length = 2;
                    sp.mean_reprojection_error = 0;
                    points.push_back(sp);

                    int new_idx = static_cast<int>(points.size()) - 1;
                    obs_map[{i, kp_i}] = new_idx;
                    obs_map[{other, kp_other}] = new_idx;
                }
            }
        }
    }

    if (verbose) {
        printf("SfM disambiguate: Re-triangulated %zu points\n", points.size());
    }
}

// ============================================================================
// Post-registration: detect and fix mirrored cameras using BFS rotation
// consistency. On a turntable, consecutive cameras differ by ~10°. If the
// difference is ~170° instead, one camera is mirrored (180° texture ambiguity).
// ============================================================================
static void fixMirroredCameras(
    std::vector<CameraPose>& cameras,
    std::vector<SparsePoint>& points,
    const std::vector<SIFTFeatures>& features,
    const std::vector<ImagePairMatches>& all_matches,
    const std::vector<ImageData>& images,
    const PipelineConfig& config,
    int init_id_i, int init_id_j,
    bool verbose)
{
    int num_images = static_cast<int>(cameras.size());
    int num_registered = 0;
    for (const auto& cam : cameras)
        if (cam.is_registered) num_registered++;
    if (num_registered < 6) return;

    auto circDist = [&](int a, int b) -> int {
        int d = std::abs(a - b);
        return std::min(d, num_images - d);
    };

    auto rotAngle = [](const Eigen::Matrix3d& R1, const Eigen::Matrix3d& R2) -> double {
        Eigen::Matrix3d R_diff = R1 * R2.transpose();
        double trace = std::clamp(R_diff.trace(), -1.0, 3.0);
        return std::acos((trace - 1.0) / 2.0) * 180.0 / M_PI;
    };

    // =========================================================================
    // Step 1: BFS from init pair to label cameras as correct or mirrored.
    // Between consecutive cameras, rotation diff should be ~10° (correct-correct
    // or mirrored-mirrored) or ~170° (correct-mirrored boundary).
    // =========================================================================

    // Debug: print consecutive rotation diffs to understand mirror pattern
    if (verbose) {
        printf("SfM mirror fix: init_id_i=%d, init_id_j=%d\n", init_id_i, init_id_j);
        for (int k = 0; k < num_images; k++) {
            int next = (k + 1) % num_images;
            if (cameras[k].is_registered && cameras[next].is_registered) {
                double diff = rotAngle(cameras[k].R, cameras[next].R);
                printf("  rot_diff(%d,%d) = %.1f°%s\n", k, next, diff,
                       diff > 90 ? " *** BOUNDARY" : "");
            }
        }
    }

    std::vector<int> label(num_images, -1); // -1=unknown, 0=correct, 1=mirrored
    label[init_id_i] = 0; // init pair is always correct
    label[init_id_j] = 0;

    std::queue<int> bfs_queue;
    bfs_queue.push(init_id_i);
    bfs_queue.push(init_id_j);

    while (!bfs_queue.empty()) {
        int k = bfs_queue.front();
        bfs_queue.pop();

        // Check both circular neighbors
        for (int dir : {-1, 1}) {
            int nb = (k + dir + num_images) % num_images;
            if (label[nb] >= 0) continue; // already labeled
            if (!cameras[nb].is_registered) continue;

            double diff = rotAngle(cameras[k].R, cameras[nb].R);
            // Expected diff for adjacent cameras: ~10° (one step on turntable)
            // If diff > 90°, neighbor has opposite mirror status
            if (diff > 90.0) {
                label[nb] = 1 - label[k]; // opposite label
            } else {
                label[nb] = label[k]; // same label
            }
            bfs_queue.push(nb);
        }
    }

    // Count mirrored cameras
    std::vector<bool> is_mirrored(num_images, false);
    int num_mirrored = 0;
    for (int i = 0; i < num_images; i++) {
        if (label[i] == 1) {
            is_mirrored[i] = true;
            num_mirrored++;
        }
    }

    // Sanity: if more than half are "mirrored", flip the labels
    // (the init pair might be the minority)
    if (num_mirrored > num_registered / 2) {
        if (verbose)
            printf("SfM mirror fix: Majority labeled mirrored (%d/%d), flipping labels\n",
                   num_mirrored, num_registered);
        for (int i = 0; i < num_images; i++) {
            if (label[i] == 0) { label[i] = 1; is_mirrored[i] = true; }
            else if (label[i] == 1) { label[i] = 0; is_mirrored[i] = false; }
        }
        // But init pair must stay correct
        label[init_id_i] = 0; is_mirrored[init_id_i] = false;
        label[init_id_j] = 0; is_mirrored[init_id_j] = false;

        num_mirrored = 0;
        for (int i = 0; i < num_images; i++)
            if (is_mirrored[i]) num_mirrored++;
    }

    if (verbose) {
        printf("SfM mirror fix: Detected %d mirrored cameras: ", num_mirrored);
        for (int i = 0; i < num_images; i++)
            if (is_mirrored[i]) printf("%d ", i);
        printf("\n");
    }

    if (num_mirrored == 0) return;

    // =========================================================================
    // Step 2: Fix mirrored cameras using SLERP rotation and center interpolation
    //         from correct neighbors. No PnP (points are contaminated).
    // =========================================================================
    for (int i = 0; i < num_images; i++) {
        if (!is_mirrored[i]) continue;

        // Find nearest correct cameras on each side
        int left = -1, right = -1;
        for (int d = 1; d < num_images; d++) {
            int li = (i - d + num_images) % num_images;
            if (cameras[li].is_registered && !is_mirrored[li] && left < 0) left = li;
            int ri = (i + d) % num_images;
            if (cameras[ri].is_registered && !is_mirrored[ri] && right < 0) right = ri;
            if (left >= 0 && right >= 0) break;
        }
        if (left < 0 || right < 0) continue;

        int ld = circDist(i, left);
        int rd = circDist(i, right);
        double alpha = static_cast<double>(ld) / (ld + rd);

        // SLERP rotation
        Eigen::Quaterniond q_left(cameras[left].R);
        Eigen::Quaterniond q_right(cameras[right].R);
        // Ensure quaternions are in the same hemisphere for correct SLERP
        if (q_left.dot(q_right) < 0) q_right.coeffs() = -q_right.coeffs();
        cameras[i].R = q_left.slerp(alpha, q_right).toRotationMatrix();

        // Interpolate camera centers
        Eigen::Vector3d c_left = cameras[left].cam_center();
        Eigen::Vector3d c_right = cameras[right].cam_center();
        Eigen::Vector3d c_interp = (1.0 - alpha) * c_left + alpha * c_right;
        cameras[i].t = -cameras[i].R * c_interp;

        if (verbose) {
            printf("SfM mirror fix: Camera %d fixed from neighbors (%d, %d), alpha=%.2f, "
                   "center=(%.4f,%.4f,%.4f)\n",
                   i, left, right, alpha,
                   c_interp.x(), c_interp.y(), c_interp.z());
        }
    }

    // =========================================================================
    // Step 3: Remove points contaminated by mirrored cameras, re-triangulate
    // =========================================================================
    size_t before = points.size();
    points.erase(
        std::remove_if(points.begin(), points.end(), [&](const SparsePoint& sp) {
            for (int img_id : sp.image_ids) {
                if (is_mirrored[img_id]) return true;
            }
            return false;
        }),
        points.end());

    if (verbose && points.size() < before) {
        printf("SfM mirror fix: Removed %zu contaminated points (remaining: %zu)\n",
               before - points.size(), points.size());
    }

    // Re-triangulate for fixed cameras with nearby correct cameras
    for (int i = 0; i < num_images; i++) {
        if (!is_mirrored[i]) continue;
        ObsMap obs_map = buildObservationMap(points);
        auto img_matches = findMatchesForImage(i, all_matches);

        for (const auto* mp : img_matches) {
            int other = (mp->image_i == i) ? mp->image_j : mp->image_i;
            if (!cameras[other].is_registered) continue;
            if (circDist(i, other) > 5) continue;

            for (const auto& fm : mp->matches) {
                int kp_this = (mp->image_i == i) ? fm.idx_i : fm.idx_j;
                int kp_other = (mp->image_i == i) ? fm.idx_j : fm.idx_i;

                if (obs_map.count({i, kp_this}) || obs_map.count({other, kp_other}))
                    continue;

                bool valid = false;
                Eigen::Vector3d pt3d = triangulateOne(
                    cameras[i], cameras[other],
                    features[i].keypoints[kp_this].pt,
                    features[other].keypoints[kp_other].pt,
                    config.max_reprojection_error,
                    config.min_triangulation_angle,
                    valid);

                if (valid) {
                    SparsePoint sp;
                    sp.position = pt3d;
                    sp.color = Eigen::Vector3f(0.8f, 0.8f, 0.8f);
                    sp.image_ids = {i, other};
                    sp.keypoint_ids = {kp_this, kp_other};
                    sp.track_length = 2;
                    sp.mean_reprojection_error = 0;
                    points.push_back(sp);

                    int new_idx = static_cast<int>(points.size()) - 1;
                    obs_map[{i, kp_this}] = new_idx;
                    obs_map[{other, kp_other}] = new_idx;
                }
            }
        }
    }

    if (verbose) printf("SfM mirror fix: After re-triangulation: %zu points\n", points.size());
}

// ============================================================================
// Phase 3: Scale calibration
// ============================================================================
static void scaleCalibration(
    SfMResult& result,
    const PipelineConfig& config,
    bool verbose)
{
    if (config.scale_bar_length_mm <= 0) {
        result.scale_factor = 1.0;
        return;
    }

    // Scale calibration requires user-provided scale bar endpoints
    // For now, this is a placeholder — needs user input mechanism
    if (verbose) {
        printf("SfM Phase 3: Scale calibration requested (%.2f mm) but no scale bar endpoints provided\n",
               config.scale_bar_length_mm);
    }
    result.scale_factor = 1.0;
}

// ============================================================================
// Phase 4: Turntable constraint
// ============================================================================
static void turntableConstraint(
    std::vector<CameraPose>& cameras,
    bool verbose)
{
    // Collect registered camera centers
    std::vector<Eigen::Vector3d> centers;
    std::vector<int> indices;
    for (int i = 0; i < static_cast<int>(cameras.size()); i++) {
        if (!cameras[i].is_registered) continue;
        centers.push_back(cameras[i].cam_center());
        indices.push_back(i);
    }

    if (centers.size() < 4) return;

    // Fit a plane to camera centers using SVD
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (const auto& c : centers) centroid += c;
    centroid /= static_cast<double>(centers.size());

    Eigen::MatrixXd A(static_cast<int>(centers.size()), 3);
    for (int i = 0; i < static_cast<int>(centers.size()); i++) {
        A.row(i) = (centers[i] - centroid).transpose();
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::Vector3d normal = svd.matrixV().col(2); // smallest singular value = plane normal

    // Fit circle: project centers onto plane, fit circle in 2D
    // Build 2D coordinate system on the plane
    Eigen::Vector3d u_axis = svd.matrixV().col(0);
    Eigen::Vector3d v_axis = svd.matrixV().col(1);

    std::vector<Eigen::Vector2d> pts2d(centers.size());
    for (size_t i = 0; i < centers.size(); i++) {
        Eigen::Vector3d d = centers[i] - centroid;
        pts2d[i] = Eigen::Vector2d(d.dot(u_axis), d.dot(v_axis));
    }

    // Least-squares circle fit: (x-a)^2 + (y-b)^2 = r^2
    // Linearize: x^2 + y^2 = 2ax + 2by + (r^2 - a^2 - b^2)
    Eigen::MatrixXd M(static_cast<int>(pts2d.size()), 3);
    Eigen::VectorXd rhs(static_cast<int>(pts2d.size()));
    for (int i = 0; i < static_cast<int>(pts2d.size()); i++) {
        M(i, 0) = 2.0 * pts2d[i].x();
        M(i, 1) = 2.0 * pts2d[i].y();
        M(i, 2) = 1.0;
        rhs(i) = pts2d[i].x() * pts2d[i].x() + pts2d[i].y() * pts2d[i].y();
    }

    Eigen::Vector3d abc = M.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rhs);
    double cx = abc(0), cy = abc(1);
    double r = std::sqrt(abc(2) + cx * cx + cy * cy);

    Eigen::Vector3d circle_center_3d = centroid + cx * u_axis + cy * v_axis;

    if (verbose) {
        printf("SfM Phase 4: Turntable circle — center=(%.4f,%.4f,%.4f), radius=%.4f, normal=(%.3f,%.3f,%.3f)\n",
               circle_center_3d.x(), circle_center_3d.y(), circle_center_3d.z(),
               r, normal.x(), normal.y(), normal.z());
    }

    // Project camera centers onto the circle (soft constraint)
    for (size_t i = 0; i < centers.size(); i++) {
        Eigen::Vector3d d = centers[i] - circle_center_3d;
        // Project onto plane
        d -= d.dot(normal) * normal;
        // Project onto circle
        double dist = d.norm();
        if (dist > 1e-10) {
            Eigen::Vector3d new_center = circle_center_3d + d * (r / dist);
            // Update camera: C_new = new_center, t_new = -R * C_new
            int cam_idx = indices[i];
            cameras[cam_idx].t = -cameras[cam_idx].R * new_center;
        }
    }
}

// ============================================================================
// Checkpoint: save/load
// ============================================================================
void saveSfMCheckpoint(const SfMResult& result, const std::string& output_dir) {
    fs::create_directories(output_dir);

    // Save cameras JSON
    std::string cam_path = output_dir + "/sfm_cameras.json";
    std::ofstream cf(cam_path);
    if (!cf) {
        fprintf(stderr, "Warning: Cannot write %s\n", cam_path.c_str());
        return;
    }

    cf << "{\n  \"num_registered\": " << result.num_registered
       << ",\n  \"mean_reprojection_error\": " << result.mean_reprojection_error
       << ",\n  \"scale_factor\": " << result.scale_factor
       << ",\n  \"cameras\": [\n";

    bool first = true;
    for (const auto& cam : result.cameras) {
        if (!first) cf << ",\n";
        first = false;
        cf << "    {\"image_id\": " << cam.image_id
           << ", \"is_registered\": " << (cam.is_registered ? "true" : "false")
           << ", \"R\": [";
        for (int r = 0; r < 3; r++) {
            cf << "[";
            for (int c = 0; c < 3; c++) {
                cf << cam.R(r, c);
                if (c < 2) cf << ",";
            }
            cf << "]";
            if (r < 2) cf << ",";
        }
        cf << "], \"t\": [" << cam.t.x() << "," << cam.t.y() << "," << cam.t.z()
           << "], \"K\": [";
        for (int r = 0; r < 3; r++) {
            cf << "[";
            for (int c = 0; c < 3; c++) {
                cf << cam.K(r, c);
                if (c < 2) cf << ",";
            }
            cf << "]";
            if (r < 2) cf << ",";
        }
        cf << "]}";
    }
    cf << "\n  ]\n}\n";
    cf.close();

    // Save sparse point cloud as PLY
    std::string ply_path = output_dir + "/sparse.ply";
    std::ofstream pf(ply_path);
    if (!pf) return;

    pf << "ply\nformat ascii 1.0\n";
    pf << "element vertex " << result.points.size() << "\n";
    pf << "property float x\nproperty float y\nproperty float z\n";
    pf << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    pf << "end_header\n";

    for (const auto& pt : result.points) {
        pf << pt.position.x() << " " << pt.position.y() << " " << pt.position.z()
           << " " << static_cast<int>(pt.color.x() * 255) << " "
           << static_cast<int>(pt.color.y() * 255) << " "
           << static_cast<int>(pt.color.z() * 255) << "\n";
    }
    pf.close();

    printf("SfM checkpoint saved: %s, %s\n", cam_path.c_str(), ply_path.c_str());
}

bool loadSfMCheckpoint(SfMResult& /*result*/, const std::string& output_dir) {
    std::string cam_path = output_dir + "/sfm_cameras.json";
    if (!fs::exists(cam_path)) return false;
    // Full JSON parser would go here — for now return false to re-run SfM
    // TODO: implement JSON loading in a future step
    return false;
}

// ============================================================================
// Main entry point
// ============================================================================
SfMResult runIncrementalSfM(
    const std::vector<SIFTFeatures>& features,
    const std::vector<ImagePairMatches>& matches,
    const std::vector<ImageData>& images,
    const PipelineConfig& config)
{
    SfMResult result;
    result.scale_factor = 1.0;
    result.mean_reprojection_error = 0;
    result.num_registered = 0;

    bool verbose = config.verbose;
    int num_images = static_cast<int>(images.size());

    if (num_images == 0 || matches.empty()) {
        fprintf(stderr, "SfM: No images or matches provided\n");
        return result;
    }

    // Initialize cameras
    result.cameras.resize(num_images);
    for (int i = 0; i < num_images; i++) {
        result.cameras[i].image_id = i;
        result.cameras[i].is_registered = false;
        cv::cv2eigen(images[i].K, result.cameras[i].K);
    }

    // Set deterministic random seed for reproducible RANSAC results
    cv::setRNGSeed(42);

    if (verbose) printf("\n=== SfM: Incremental Structure from Motion ===\n");

    // Phase 1: Initialize from best pair
    int init_id_i = -1, init_id_j = -1;
    if (!initializePair(features, matches, images, config, result.cameras, result.points, init_id_i, init_id_j, verbose)) {
        fprintf(stderr, "SfM: Initialization failed\n");
        return result;
    }

    // Phase 2: Incremental registration
    incrementalRegistration(features, matches, images, config, result.cameras, result.points, init_id_i, init_id_j, verbose);

    // Post-registration BA
    if (verbose) printf("SfM: Running post-registration bundle adjustment...\n");
    bundleAdjust(result.cameras, result.points, features, config.max_reprojection_error);

    // Disambiguate 180° turntable ambiguity (texture repeats every 180°)
    if (verbose) printf("SfM: Disambiguating 180° turntable ambiguity...\n");
    disambiguateTurntable(result.cameras, result.points, features, matches, config, init_id_i, verbose);

    // After disambiguation, camera poses are forced to exact turntable geometry.
    // Run point-only BA to refine 3D point positions without touching cameras.
    if (verbose) printf("SfM: Running point-only BA (cameras fixed)...\n");
    bundleAdjust(result.cameras, result.points, features, config, true, false, /*fix_all_cameras=*/true);

    // Update point errors
    updatePointErrors(result.points, result.cameras, features);

    // Post-BA cleanup: aggressively filter to keep only well-triangulated points
    double reproj_threshold = std::min(config.max_reprojection_error, 2.0);
    size_t before = result.points.size();
    result.points.erase(
        std::remove_if(result.points.begin(), result.points.end(), [&](const SparsePoint& sp) {
            return sp.mean_reprojection_error > reproj_threshold
                   || sp.track_length < 2;
        }),
        result.points.end());
    if (verbose && result.points.size() < before) {
        printf("SfM: Final cleanup removed %zu points (remaining: %zu)\n",
               before - result.points.size(), result.points.size());
    }

    // Phase 3: Scale calibration
    scaleCalibration(result, config, verbose);

    // Phase 4: Turntable constraint
    if (config.turntable) {
        if (verbose) printf("SfM Phase 4: Applying turntable constraint...\n");
        turntableConstraint(result.cameras, verbose);
    }

    // Compute final stats
    result.num_registered = 0;
    for (const auto& cam : result.cameras) {
        if (cam.is_registered) result.num_registered++;
    }
    result.mean_reprojection_error = computeMeanReprojError(result.cameras, result.points, features);

    if (verbose) {
        printf("SfM complete: %d/%d cameras registered, %zu points, mean reproj error: %.3f px\n",
               result.num_registered, num_images, result.points.size(), result.mean_reprojection_error);
    }

    // Checkpoint
    if (config.save_intermediate && !config.output_path.empty()) {
        fs::path out_dir = fs::path(config.output_path).parent_path();
        if (out_dir.empty()) out_dir = ".";
        saveSfMCheckpoint(result, out_dir.string());
    }

    return result;
}
