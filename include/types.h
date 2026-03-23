#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cuda_runtime.h>

// ============================================================================
// Shared data structures for the entire CUDA 3D reconstruction pipeline.
// ALL structs live here — do not redefine elsewhere.
// ============================================================================

struct ImageData {
    cv::Mat image;              // BGR uint8
    cv::Mat gray;               // Grayscale float32 [0,1]
    cv::Mat K;                  // 3x3 double intrinsic matrix
    cv::Mat dist_coeffs;        // distortion coefficients (k1,k2,p1,p2,k3)
    double focal_length_px;     // focal length in pixels
    std::string filename;
    int width, height;
    int id;                     // sequential index
};

struct Keypoint {
    float x, y;                 // pixel coordinates
    float scale;                // detection scale
    float orientation;          // radians
    int octave;
};

struct SIFTFeatures {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;        // Nx128 float32 (CPU)
    float* d_descriptors;       // GPU pointer (may be nullptr if transferred back)
    int count;
    int image_id;
};

struct FeatureMatch {
    int idx_i, idx_j;           // keypoint indices in respective images
    float distance;             // L2 descriptor distance
};

struct ImagePairMatches {
    int image_i, image_j;
    std::vector<FeatureMatch> matches;
    cv::Mat F;                  // 3x3 fundamental matrix
    cv::Mat E;                  // 3x3 essential matrix
    int num_inliers;
    double median_triangulation_angle;  // degrees
};

struct CameraPose {
    int image_id;
    Eigen::Matrix3d R;          // world-to-camera rotation
    Eigen::Vector3d t;          // world-to-camera translation
    Eigen::Matrix3d K;          // intrinsics
    bool is_registered = false;

    // P_cam = R * P_world + t
    Eigen::Matrix<double,3,4> projection() const {
        Eigen::Matrix<double,3,4> Rt;
        Rt.block<3,3>(0,0) = R;
        Rt.col(3) = t;
        return K * Rt;
    }

    // P_world = R^T * (P_cam - t)
    Eigen::Vector3d cam_center() const {
        return -R.transpose() * t;
    }
};

struct SparsePoint {
    Eigen::Vector3d position;
    Eigen::Vector3f color;
    std::vector<int> image_ids;
    std::vector<int> keypoint_ids;
    float mean_reprojection_error;
    int track_length;           // number of images that see this point
};

struct DepthMap {
    cv::Mat depth;              // float32, world units (0 = invalid)
    cv::Mat normal;             // float32x3, world-frame surface normal
    cv::Mat confidence;         // float32, [0, 1]
    int image_id;
    int width, height;
};

struct DensePointCloud {
    std::vector<Eigen::Vector3f> positions;
    std::vector<Eigen::Vector3f> normals;
    std::vector<Eigen::Vector3f> colors;
    int size() const { return static_cast<int>(positions.size()); }

    // GPU mirrors (allocated on demand)
    float3* d_positions = nullptr;
    float3* d_normals   = nullptr;
    float3* d_colors    = nullptr;
    bool on_gpu = false;

    void freeGPU();             // release device memory
};

struct Mesh {
    std::vector<Eigen::Vector3f> vertices;
    std::vector<Eigen::Vector3f> normals;       // per-vertex
    std::vector<Eigen::Vector3f> colors;        // per-vertex
    std::vector<Eigen::Vector2f> texcoords;     // optional UV
    std::vector<Eigen::Vector3i> faces;         // triangle indices (0-indexed)

    int num_vertices() const { return static_cast<int>(vertices.size()); }
    int num_faces()    const { return static_cast<int>(faces.size()); }
    bool is_watertight() const;                 // check if all edges shared by exactly 2 faces
};

struct PipelineConfig {
    // Input / Output
    std::string input_dir;
    std::string output_path;            // .obj, .stl, or .ply
    std::string pointcloud_path;        // optional .ply export
    std::string log_path;
    bool save_intermediate = false;
    bool verbose = false;

    // Quality preset: "low", "medium", "high", "ultra"
    std::string quality = "high";

    // Image
    int max_image_size = 3200;

    // Feature detection
    float match_ratio = 0.75f;          // Lowe's ratio
    int min_matches = 30;               // per pair

    // SfM
    double min_triangulation_angle = 2.0; // degrees
    double max_reprojection_error = 4.0;  // pixels
    bool turntable = false;

    // Camera overrides
    double focal_length_mm = 0.0;       // 0 = use EXIF
    double sensor_width_mm = 0.0;       // 0 = estimate

    // Dense reconstruction
    float mvs_resolution = 1.0f;        // downscale factor for MVS
    int mvs_iterations = 12;
    int mvs_patch_size = 11;

    // Meshing
    int poisson_depth = 10;
    float poisson_screening = 4.0f;
    int smooth_iterations = 3;
    float smooth_lambda = 0.5f;
    int decimate_target = 0;            // 0 = no decimation

    // Scale calibration
    double scale_bar_length_mm = 0.0;   // 0 = uncalibrated

    // GPU
    int gpu_id = 0;
    size_t gpu_memory_limit = 0;        // 0 = auto

    // Derived from quality preset (call applyPreset() after parsing)
    void applyPreset();
};
