#include "utils/synthetic_data.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

// ============================================================================
// Synthetic sphere renderer with checkerboard texture, pinhole camera + z-buffer
// ============================================================================

// Sphere parameters
static constexpr double SPHERE_RADIUS = 0.05;     // 5 cm
static constexpr double CAMERA_DISTANCE = 0.25;    // 25 cm
static constexpr int NUM_VIEWS = 36;
static constexpr double ANGLE_STEP = 10.0;         // degrees

// Image parameters (matching PLAN spec)
static constexpr int IMG_W = 1280;
static constexpr int IMG_H = 960;
static constexpr double FX = 2000.0;
static constexpr double FY = 2000.0;
static constexpr double CX = 640.0;
static constexpr double CY = 480.0;

// Simple hash for deterministic pseudo-random noise from coordinates
static double hashNoise(double x, double y) {
    // Sine-based hash
    double val = std::sin(x * 127.1 + y * 311.7) * 43758.5453;
    return val - std::floor(val);
}

// Rich texture on sphere surface — multi-frequency patterns + noise for SIFT
static cv::Vec3b sphereTexture(double theta, double phi) {
    // theta: azimuth [0, 2*pi], phi: elevation [-pi/2, pi/2]
    double u = theta / (2.0 * M_PI);     // [0, 1]
    double v = (phi + M_PI / 2.0) / M_PI; // [0, 1]

    // High-frequency patterns at multiple scales for SIFT detection
    double check1 = std::sin(theta * 8.0) * std::sin((phi + M_PI / 2.0) * 8.0);
    double check2 = std::sin(theta * 16.0 + 1.0) * std::sin((phi + M_PI / 2.0) * 16.0 + 0.5);
    double check3 = std::sin(theta * 32.0 + 2.0) * std::sin((phi + M_PI / 2.0) * 32.0 + 1.5);

    // Multiple layers of noise at different frequencies
    double n1 = hashNoise(u * 30.0, v * 30.0);
    double n2 = hashNoise(u * 60.0 + 7.3, v * 60.0 + 13.7);
    double n3 = hashNoise(u * 120.0 + 23.1, v * 120.0 + 41.9);
    double n4 = hashNoise(u * 200.0 + 53.7, v * 200.0 + 97.3);

    // Combine into high-contrast RGB channels
    double r = 0.5 + 0.25 * check1 + 0.15 * check2 + 0.10 * check3
             + 0.15 * (n1 - 0.5) + 0.10 * (n2 - 0.5) + 0.05 * (n3 - 0.5);
    double g = 0.4 + 0.20 * check1 - 0.15 * check2 + 0.12 * check3
             + 0.12 * (n2 - 0.5) + 0.08 * (n3 - 0.5) + 0.05 * (n4 - 0.5);
    double b = 0.6 - 0.15 * check1 + 0.20 * check2 - 0.10 * check3
             + 0.13 * (n3 - 0.5) + 0.10 * (n1 - 0.5) + 0.05 * (n2 - 0.5);

    r = std::clamp(r, 0.0, 1.0);
    g = std::clamp(g, 0.0, 1.0);
    b = std::clamp(b, 0.0, 1.0);

    return cv::Vec3b(
        static_cast<uint8_t>(b * 255),
        static_cast<uint8_t>(g * 255),
        static_cast<uint8_t>(r * 255)
    );
}

// Render the sphere from a given camera [R|t] (world-to-camera)
// R is 3x3, t is 3x1, both double
static cv::Mat renderView(const cv::Mat& R, const cv::Mat& t) {
    cv::Mat image(IMG_H, IMG_W, CV_8UC3, cv::Scalar(30, 30, 30));  // dark background

    // For each pixel, cast a ray and intersect with sphere at origin
    // Camera center in world: C = -R^T * t
    cv::Mat Rt = R.t();
    cv::Mat C_world = -Rt * t;  // 3x1

    double cx_w = C_world.at<double>(0);
    double cy_w = C_world.at<double>(1);
    double cz_w = C_world.at<double>(2);

    // K inverse for ray direction
    // ray_cam = K^{-1} * [u, v, 1]^T
    // ray_world = R^T * ray_cam

    for (int py = 0; py < IMG_H; py++) {
        for (int px = 0; px < IMG_W; px++) {
            // Ray direction in camera frame
            double ray_cam_x = (px - CX) / FX;
            double ray_cam_y = (py - CY) / FY;
            double ray_cam_z = 1.0;

            // Transform to world frame: d = R^T * ray_cam
            double dx = Rt.at<double>(0, 0) * ray_cam_x + Rt.at<double>(0, 1) * ray_cam_y + Rt.at<double>(0, 2) * ray_cam_z;
            double dy = Rt.at<double>(1, 0) * ray_cam_x + Rt.at<double>(1, 1) * ray_cam_y + Rt.at<double>(1, 2) * ray_cam_z;
            double dz = Rt.at<double>(2, 0) * ray_cam_x + Rt.at<double>(2, 1) * ray_cam_y + Rt.at<double>(2, 2) * ray_cam_z;

            // Normalize direction
            double dlen = std::sqrt(dx * dx + dy * dy + dz * dz);
            dx /= dlen; dy /= dlen; dz /= dlen;

            // Ray-sphere intersection: |C + t*d|^2 = r^2
            // a*t^2 + b*t + c = 0 where a=1, b=2*(C.d), c=|C|^2 - r^2
            double b = 2.0 * (cx_w * dx + cy_w * dy + cz_w * dz);
            double c = cx_w * cx_w + cy_w * cy_w + cz_w * cz_w - SPHERE_RADIUS * SPHERE_RADIUS;
            double disc = b * b - 4.0 * c;

            if (disc < 0) continue;

            double sqrt_disc = std::sqrt(disc);
            double t_hit = (-b - sqrt_disc) / 2.0;  // nearest intersection
            if (t_hit < 0) {
                t_hit = (-b + sqrt_disc) / 2.0;
                if (t_hit < 0) continue;
            }

            // Hit point in world coordinates
            double hx = cx_w + t_hit * dx;
            double hy = cy_w + t_hit * dy;
            double hz = cz_w + t_hit * dz;

            // Spherical coordinates for texture
            double theta = std::atan2(hy, hx);         // azimuth
            if (theta < 0) theta += 2.0 * M_PI;
            double phi = std::asin(std::clamp(hz / SPHERE_RADIUS, -1.0, 1.0));  // elevation

            // Simple diffuse shading (light from camera position)
            double nx = hx / SPHERE_RADIUS;
            double ny = hy / SPHERE_RADIUS;
            double nz = hz / SPHERE_RADIUS;
            // Light direction = normalize(camera_center - hit_point)
            double lx = cx_w - hx, ly = cy_w - hy, lz = cz_w - hz;
            double llen = std::sqrt(lx * lx + ly * ly + lz * lz);
            lx /= llen; ly /= llen; lz /= llen;
            double ndotl = std::max(0.0, nx * lx + ny * ly + nz * lz);
            double shade = 0.2 + 0.8 * ndotl;  // ambient + diffuse

            cv::Vec3b color = sphereTexture(theta, phi);
            image.at<cv::Vec3b>(py, px) = cv::Vec3b(
                cv::saturate_cast<uint8_t>(color[0] * shade),
                cv::saturate_cast<uint8_t>(color[1] * shade),
                cv::saturate_cast<uint8_t>(color[2] * shade)
            );
        }
    }

    // Add Gaussian noise to break up smoothness (helps SIFT detect more features)
    cv::Mat noise(image.size(), CV_32FC3);
    cv::randn(noise, cv::Scalar(0, 0, 0), cv::Scalar(8, 8, 8));
    cv::Mat image_f;
    image.convertTo(image_f, CV_32FC3);
    image_f += noise;
    image_f.convertTo(image, CV_8UC3);

    return image;
}

// Write a simple OBJ sphere mesh
static bool writeGroundTruthMesh(const std::string& path) {
    std::ofstream f(path);
    if (!f) return false;

    f << "# Ground truth sphere: radius=" << SPHERE_RADIUS << "m, center=(0,0,0)\n";

    int n_lat = 32, n_lon = 64;

    // Vertices
    // Top pole
    f << "v 0 0 " << SPHERE_RADIUS << "\n";
    for (int i = 1; i < n_lat; i++) {
        double phi = M_PI * i / n_lat - M_PI / 2.0;
        // Actually we want phi from pole: phi = pi * i / n_lat
        double phi_angle = M_PI * double(i) / double(n_lat);
        double sp = std::sin(phi_angle);
        double cp = std::cos(phi_angle);
        for (int j = 0; j < n_lon; j++) {
            double theta = 2.0 * M_PI * j / n_lon;
            f << "v " << SPHERE_RADIUS * sp * std::cos(theta) << " "
              << SPHERE_RADIUS * sp * std::sin(theta) << " "
              << SPHERE_RADIUS * cp << "\n";
        }
    }
    // Bottom pole
    f << "v 0 0 " << -SPHERE_RADIUS << "\n";

    int total_verts = 2 + (n_lat - 1) * n_lon;

    // Faces — top cap
    for (int j = 0; j < n_lon; j++) {
        int next = (j + 1) % n_lon;
        f << "f 1 " << (j + 2) << " " << (next + 2) << "\n";
    }

    // Middle strips
    for (int i = 0; i < n_lat - 2; i++) {
        for (int j = 0; j < n_lon; j++) {
            int next = (j + 1) % n_lon;
            int v00 = 2 + i * n_lon + j;
            int v01 = 2 + i * n_lon + next;
            int v10 = 2 + (i + 1) * n_lon + j;
            int v11 = 2 + (i + 1) * n_lon + next;
            f << "f " << v00 << " " << v10 << " " << v11 << "\n";
            f << "f " << v00 << " " << v11 << " " << v01 << "\n";
        }
    }

    // Bottom cap
    for (int j = 0; j < n_lon; j++) {
        int next = (j + 1) % n_lon;
        int ring = 2 + (n_lat - 2) * n_lon;
        f << "f " << total_verts << " " << (ring + next) << " " << (ring + j) << "\n";
    }

    return true;
}

bool generateSyntheticData(const std::string& output_dir) {
    fs::create_directories(output_dir);

    // Open ground-truth JSON
    std::string json_path = output_dir + "/cameras_gt.json";
    std::ofstream json(json_path);
    if (!json) {
        fprintf(stderr, "Error: Cannot create %s\n", json_path.c_str());
        return false;
    }

    json << "{\n";
    json << "  \"sphere_radius\": " << SPHERE_RADIUS << ",\n";
    json << "  \"camera_distance\": " << CAMERA_DISTANCE << ",\n";
    json << "  \"intrinsics\": { \"fx\": " << FX << ", \"fy\": " << FY
         << ", \"cx\": " << CX << ", \"cy\": " << CY
         << ", \"width\": " << IMG_W << ", \"height\": " << IMG_H << " },\n";
    json << "  \"cameras\": [\n";

    for (int i = 0; i < NUM_VIEWS; i++) {
        double angle_deg = i * ANGLE_STEP;
        double angle_rad = angle_deg * M_PI / 180.0;

        // Camera center in world: circle in XY plane at z=0
        double cam_x = CAMERA_DISTANCE * std::cos(angle_rad);
        double cam_y = CAMERA_DISTANCE * std::sin(angle_rad);
        double cam_z = 0.0;

        // Camera looks at origin. Build world-to-camera [R|t]:
        // Camera z-axis (forward) = normalize(origin - camera_center) = -normalize(cam_pos)
        double fwd_x = -cam_x, fwd_y = -cam_y, fwd_z = -cam_z;
        double fwd_len = std::sqrt(fwd_x * fwd_x + fwd_y * fwd_y + fwd_z * fwd_z);
        fwd_x /= fwd_len; fwd_y /= fwd_len; fwd_z /= fwd_len;

        // Camera up = world Z
        double up_x = 0, up_y = 0, up_z = 1;

        // Camera x-axis (right) = forward × up
        double right_x = fwd_y * up_z - fwd_z * up_y;
        double right_y = fwd_z * up_x - fwd_x * up_z;
        double right_z = fwd_x * up_y - fwd_y * up_x;
        double right_len = std::sqrt(right_x * right_x + right_y * right_y + right_z * right_z);
        right_x /= right_len; right_y /= right_len; right_z /= right_len;

        // Camera y-axis (down) = forward × right  (to get image y pointing down)
        // Actually: y = z × x for right-handed camera frame with y-down
        double down_x = fwd_y * right_z - fwd_z * right_y;
        double down_y = fwd_z * right_x - fwd_x * right_z;
        double down_z = fwd_x * right_y - fwd_y * right_x;

        // R matrix (world-to-camera): rows are right, down, forward
        cv::Mat R = (cv::Mat_<double>(3, 3) <<
                     right_x, right_y, right_z,
                     down_x,  down_y,  down_z,
                     fwd_x,   fwd_y,   fwd_z);

        // t = -R * C_world
        cv::Mat C = (cv::Mat_<double>(3, 1) << cam_x, cam_y, cam_z);
        cv::Mat t = -R * C;

        // Render
        cv::Mat image = renderView(R, t);

        // Save image
        char fname[64];
        snprintf(fname, sizeof(fname), "view_%03d.jpg", i);
        std::string img_path = output_dir + "/" + fname;
        cv::imwrite(img_path, image, {cv::IMWRITE_JPEG_QUALITY, 95});

        // Write JSON entry
        json << "    {\n";
        json << "      \"image\": \"" << fname << "\",\n";
        json << "      \"angle_deg\": " << angle_deg << ",\n";
        json << "      \"R\": [["
             << R.at<double>(0, 0) << "," << R.at<double>(0, 1) << "," << R.at<double>(0, 2) << "],["
             << R.at<double>(1, 0) << "," << R.at<double>(1, 1) << "," << R.at<double>(1, 2) << "],["
             << R.at<double>(2, 0) << "," << R.at<double>(2, 1) << "," << R.at<double>(2, 2) << "]],\n";
        json << "      \"t\": [" << t.at<double>(0) << "," << t.at<double>(1) << "," << t.at<double>(2) << "],\n";
        json << "      \"center\": [" << cam_x << "," << cam_y << "," << cam_z << "]\n";
        json << "    }" << (i < NUM_VIEWS - 1 ? "," : "") << "\n";
    }

    json << "  ]\n}\n";
    json.close();

    // Write ground-truth mesh
    writeGroundTruthMesh(output_dir + "/ground_truth.obj");

    printf("Synthetic data: %d views at %dx%d → %s\n", NUM_VIEWS, IMG_W, IMG_H, output_dir.c_str());
    return true;
}
