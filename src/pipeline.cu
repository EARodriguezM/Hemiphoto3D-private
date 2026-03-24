#include "pipeline.h"
#include "utils/cuda_utils.cuh"
#include "utils/image_loader.h"
#include "feature_detection/sift_cuda.h"
#include "feature_detection/feature_matching.h"
#include "sfm/camera_pose.h"
#include "sfm/bundle_adjust.h"
#include "dense_recon/patchmatch_mvs.h"
#include "dense_recon/point_cloud.h"
#include "meshing/poisson_recon.h"
#include "export/mesh_exporter.h"

#include <cstdio>
#include <cstdarg>
#include <chrono>
#include <unordered_map>
#include <algorithm>
#include <filesystem>
#include <numeric>

namespace fs = std::filesystem;

// ============================================================================
// Timer — wall-clock timing per stage
// ============================================================================

class Timer {
public:
    void start(const std::string& name) {
        starts_[name] = std::chrono::high_resolution_clock::now();
        bool found = false;
        for (auto& p : order_) {
            if (p.first == name) { found = true; break; }
        }
        if (!found) order_.push_back({name, 0.0});
    }

    void stop(const std::string& name) {
        auto it = starts_.find(name);
        if (it == starts_.end()) return;
        double secs = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - it->second).count();
        for (auto& p : order_) {
            if (p.first == name) { p.second = secs; break; }
        }
    }

    void printSummary() const {
        printf("\n=== Timing Summary ===\n");
        for (const auto& [name, secs] : order_) {
            if (secs < 60.0)
                printf("  %-20s %7.1f s\n", name.c_str(), secs);
            else
                printf("  %-20s %6.1f min\n", name.c_str(), secs / 60.0);
        }
        printf("======================\n");
    }

private:
    std::unordered_map<std::string,
        std::chrono::high_resolution_clock::time_point> starts_;
    std::vector<std::pair<std::string, double>> order_;
};

static Timer g_timer;

// ============================================================================
// Pipeline — constructor / destructor
// ============================================================================

Pipeline::Pipeline(const PipelineConfig& config) : config_(config) {
    // Derive intermediate directory from output path
    fs::path out_path(config_.output_path);
    fs::path out_dir = out_path.parent_path();
    if (out_dir.empty()) out_dir = ".";
    intermediate_dir_ = (out_dir / "intermediate").string();

    // Open log file if configured
    if (!config_.log_path.empty()) {
        log_file_ = fopen(config_.log_path.c_str(), "w");
        if (!log_file_) {
            fprintf(stderr, "[WARNING] Could not open log file '%s'\n",
                    config_.log_path.c_str());
        }
    }
}

Pipeline::~Pipeline() {
    if (log_file_) {
        fclose(log_file_);
        log_file_ = nullptr;
    }
}

// ============================================================================
// Logging helpers
// ============================================================================

void Pipeline::log(const char* fmt, ...) const {
    char buf[2048];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    printf("[recon3d] %s\n", buf);
    if (log_file_) {
        fprintf(log_file_, "[recon3d] %s\n", buf);
        fflush(log_file_);
    }
}

void Pipeline::vlog(const char* fmt, ...) const {
    if (!config_.verbose) return;
    char buf[2048];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    printf("[recon3d] %s\n", buf);
    if (log_file_) {
        fprintf(log_file_, "[recon3d] %s\n", buf);
        fflush(log_file_);
    }
}

void Pipeline::warn(const char* fmt, ...) const {
    char buf[2048];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    printf("[WARNING] %s\n", buf);
    if (log_file_) {
        fprintf(log_file_, "[WARNING] %s\n", buf);
        fflush(log_file_);
    }
}

void Pipeline::error(const char* fmt, ...) const {
    char buf[2048];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    fprintf(stderr, "[ERROR] %s\n", buf);
    if (log_file_) {
        fprintf(log_file_, "[ERROR] %s\n", buf);
        fflush(log_file_);
    }
}

std::string Pipeline::intermediatePath(const std::string& filename) const {
    return intermediate_dir_ + "/" + filename;
}

void Pipeline::ensureDir(const std::string& dir) const {
    fs::create_directories(dir);
}

// ============================================================================
// Main pipeline entry point
// ============================================================================

bool Pipeline::run() {
    g_timer = Timer();  // reset
    g_timer.start("total");

    // 0. GPU info
    log("GPU device %d:", config_.gpu_id);
    printGPUInfo(config_.gpu_id);
    CUDA_CHECK(cudaSetDevice(config_.gpu_id));

    // Create output directory
    fs::path out_dir = fs::path(config_.output_path).parent_path();
    if (!out_dir.empty()) ensureDir(out_dir.string());

    if (config_.save_intermediate || config_.resume) {
        ensureDir(intermediate_dir_);
    }

    vlog("Quality preset: %s", config_.quality.c_str());
    vlog("Max image size: %d px", config_.max_image_size);
    vlog("MVS resolution: %.2f, iterations: %d", config_.mvs_resolution, config_.mvs_iterations);
    vlog("Poisson depth: %d, smooth iterations: %d", config_.poisson_depth, config_.smooth_iterations);
    if (config_.resume) vlog("Resume mode: enabled");
    if (config_.save_intermediate) vlog("Save intermediate: enabled");

    // Pipeline stages
    if (!stageLoadImages())      return false;
    if (!stageDetectFeatures())  return false;
    if (!stageMatchFeatures())   return false;
    if (!stageSfM())             return false;
    if (!stageMVS())             return false;
    if (!stageFusion())          return false;
    if (!stageMeshing())         return false;
    if (!stageExport())          return false;

    g_timer.stop("total");
    g_timer.printSummary();

    log("Reconstruction complete: %s", config_.output_path.c_str());
    return true;
}

// ============================================================================
// Stage 1: Load images
// ============================================================================

bool Pipeline::stageLoadImages() {
    g_timer.start("load");
    log("Stage 1/8: Loading images from '%s' ...", config_.input_dir.c_str());

    images_ = loadImages(config_.input_dir, config_);

    g_timer.stop("load");

    if (images_.empty()) {
        error("No images loaded from '%s'.", config_.input_dir.c_str());
        return false;
    }

    log("Loaded %zu images (%dx%d)", images_.size(),
        images_[0].width, images_[0].height);

    if ((int)images_.size() < 10) {
        error("Need at least 10 images, got %zu.", images_.size());
        return false;
    }

    return true;
}

// ============================================================================
// Stage 2: Feature detection
// ============================================================================

bool Pipeline::stageDetectFeatures() {
    g_timer.start("features");
    log("Stage 2/8: Detecting features ...");

    features_ = detectAllFeatures(images_, 8000, config_.verbose);

    g_timer.stop("features");

    // Compute average and check for low-count images
    double avg = 0;
    int low_count = 0;
    for (const auto& f : features_) {
        avg += f.count;
        if (f.count < 100) low_count++;
    }
    avg /= (double)features_.size();

    log("Detected avg %.0f features per image", avg);

    if (low_count > 0) {
        warn("%d image(s) have fewer than 100 features — may hurt reconstruction.", low_count);
    }

    return true;
}

// ============================================================================
// Stage 3: Feature matching
// ============================================================================

bool Pipeline::stageMatchFeatures() {
    g_timer.start("matching");
    log("Stage 3/8: Matching features (ratio=%.2f, min_inliers=%d) ...",
        config_.match_ratio, config_.min_matches);

    matches_ = matchAllPairs(features_, images_, config_.match_ratio,
                              config_.min_matches, config_.verbose);

    g_timer.stop("matching");

    log("Matched %zu image pairs", matches_.size());

    if (matches_.empty()) {
        error("No image pairs matched.");
        error("Possible causes: images too different, wrong focal length, insufficient overlap.");
        error("Tips: ensure >= 60%% overlap between consecutive images, use consistent lighting.");
        return false;
    }

    // Check connectivity
    int components = checkMatchGraph(matches_, (int)images_.size());
    if (components > 1) {
        warn("Match graph has %d disconnected components — reconstruction may be incomplete.", components);
    }

    return true;
}

// ============================================================================
// Stage 4: Structure from Motion
// ============================================================================

bool Pipeline::stageSfM() {
    // Resume: check for cached SfM checkpoint
    std::string sfm_dir = intermediatePath("sfm");
    if (config_.resume && loadSfMCheckpoint(sfm_, sfm_dir)) {
        log("Stage 4/8: Loaded cached SfM (%d cameras, %zu points)",
            sfm_.num_registered, sfm_.points.size());
        return true;
    }

    g_timer.start("sfm");
    log("Stage 4/8: Running Structure from Motion ...");

    sfm_ = runIncrementalSfM(features_, matches_, images_, config_);

    g_timer.stop("sfm");

    log("Registered %d/%zu cameras, %zu sparse points, mean reproj %.2f px",
        sfm_.num_registered, images_.size(),
        sfm_.points.size(), sfm_.mean_reprojection_error);

    if (sfm_.num_registered < 3) {
        error("Only %d cameras registered — need at least 3.", sfm_.num_registered);
        return false;
    }

    if (sfm_.num_registered < (int)images_.size() / 2) {
        warn("Only %d/%zu cameras registered (< 50%%) — partial reconstruction.",
             sfm_.num_registered, images_.size());
    }

    // Final bundle adjustment + filtering
    vlog("Running final bundle adjustment ...");
    bundleAdjust(sfm_.cameras, sfm_.points, features_, config_);
    filterPointCloud(sfm_.cameras, sfm_.points, features_,
                     config_.max_reprojection_error, 2);
    vlog("After BA + filter: %zu sparse points", sfm_.points.size());

    // Save checkpoint
    if (config_.save_intermediate) {
        ensureDir(sfm_dir);
        saveSfMCheckpoint(sfm_, sfm_dir);
        vlog("Saved SfM checkpoint to %s", sfm_dir.c_str());
    }

    return true;
}

// ============================================================================
// Stage 5: Dense MVS
// ============================================================================

bool Pipeline::stageMVS() {
    // Resume: check for cached depth maps
    std::string dm_dir = intermediatePath("depth_maps");
    if (config_.resume && fs::exists(dm_dir)) {
        int count = 0;
        for (const auto& entry : fs::directory_iterator(dm_dir)) {
            if (entry.path().extension() == ".pfm") count++;
        }
        if (count > 0) {
            depth_maps_.clear();
            for (int i = 0; i < count; i++) {
                std::string path = dm_dir + "/depth_" + std::to_string(i) + ".pfm";
                DepthMap dm;
                if (loadDepthMapPFM(dm, path)) {
                    depth_maps_.push_back(std::move(dm));
                }
            }
            if (!depth_maps_.empty()) {
                log("Stage 5/8: Loaded %zu cached depth maps", depth_maps_.size());
                return true;
            }
        }
    }

    g_timer.start("mvs");
    log("Stage 5/8: Running PatchMatch MVS (res=%.2f, iter=%d) ...",
        config_.mvs_resolution, config_.mvs_iterations);

    depth_maps_ = runPatchMatchMVS(images_, sfm_.cameras, sfm_, config_);

    g_timer.stop("mvs");

    log("Generated %zu depth maps", depth_maps_.size());

    if (depth_maps_.empty()) {
        warn("MVS produced no depth maps — falling back to sparse point cloud.");
        // Build sparse cloud as fallback
        for (const auto& pt : sfm_.points) {
            cloud_.positions.push_back(pt.position.cast<float>());
            cloud_.normals.push_back(Eigen::Vector3f(0, 1, 0));
            cloud_.colors.push_back(pt.color);
        }
        if (!config_.pointcloud_path.empty()) {
            saveDensePointCloudPLY(cloud_, config_.pointcloud_path);
            log("Exported sparse cloud (%d points) to %s",
                cloud_.size(), config_.pointcloud_path.c_str());
        }
        // Continue — fusion will detect cloud_ is already populated
        return true;
    }

    // Save depth maps
    if (config_.save_intermediate) {
        ensureDir(dm_dir);
        for (size_t i = 0; i < depth_maps_.size(); i++) {
            std::string path = dm_dir + "/depth_" + std::to_string(i) + ".pfm";
            saveDepthMapPFM(depth_maps_[i], path);
        }
        vlog("Saved %zu depth maps to %s", depth_maps_.size(), dm_dir.c_str());
    }

    return true;
}

// ============================================================================
// Stage 6: Point cloud fusion
// ============================================================================

bool Pipeline::stageFusion() {
    // If cloud was already set by MVS fallback, skip fusion
    if (!cloud_.positions.empty()) {
        log("Stage 6/8: Using sparse cloud fallback (%d points)", cloud_.size());
        return true;
    }

    // Resume: check for cached point cloud
    std::string cloud_path = intermediatePath("dense_cloud.ply");
    if (config_.resume && fs::exists(cloud_path)) {
        if (loadDensePointCloudPLY(cloud_, cloud_path)) {
            log("Stage 6/8: Loaded cached dense cloud (%d points)", cloud_.size());
            // Still export point cloud if requested
            if (!config_.pointcloud_path.empty()) {
                exportPointCloud(cloud_, config_.pointcloud_path);
                vlog("Exported point cloud to %s", config_.pointcloud_path.c_str());
            }
            return true;
        }
    }

    g_timer.start("fusion");
    log("Stage 6/8: Fusing depth maps into point cloud ...");

    cloud_ = fuseDepthMaps(depth_maps_, sfm_.cameras, images_, config_);

    g_timer.stop("fusion");

    log("Dense cloud: %d points", cloud_.size());

    if (cloud_.size() < 100) {
        error("Point cloud too sparse (%d points).", cloud_.size());
        return false;
    }

    // Export point cloud if requested
    if (!config_.pointcloud_path.empty()) {
        exportPointCloud(cloud_, config_.pointcloud_path);
        log("Exported point cloud to %s", config_.pointcloud_path.c_str());
    }

    // Save intermediate
    if (config_.save_intermediate) {
        saveDensePointCloudPLY(cloud_, cloud_path);
        vlog("Saved dense cloud to %s", cloud_path.c_str());
    }

    return true;
}

// ============================================================================
// Stage 7: Surface reconstruction + post-processing
// ============================================================================

bool Pipeline::stageMeshing() {
    g_timer.start("meshing");
    log("Stage 7/8: Poisson surface reconstruction (depth=%d) ...",
        config_.poisson_depth);

    mesh_ = poissonReconstruct(cloud_, config_.poisson_depth,
                                config_.poisson_screening);

    if (mesh_.num_vertices() == 0 || mesh_.num_faces() == 0) {
        error("Poisson reconstruction produced empty mesh.");
        error("If PoissonRecon is not installed:");
        error("  git clone https://github.com/mkazhdan/PoissonRecon");
        error("  cd PoissonRecon && make && sudo cp Bin/Linux/PoissonRecon /usr/local/bin/");
        return false;
    }

    vlog("Raw mesh: %d vertices, %d faces",
         mesh_.num_vertices(), mesh_.num_faces());

    // Post-processing: smoothing, decimation, connected components, hole filling
    log("Post-processing mesh (smooth=%d, decimate=%d) ...",
        config_.smooth_iterations, config_.decimate_target);
    postProcessMesh(mesh_, config_);

    // Transfer colors from point cloud to mesh vertices
    transferColors(mesh_, cloud_);

    g_timer.stop("meshing");

    log("Mesh: %d vertices, %d faces, watertight: %s",
        mesh_.num_vertices(), mesh_.num_faces(),
        mesh_.is_watertight() ? "yes" : "no");

    if (mesh_.num_faces() == 0) {
        error("Mesh is empty after post-processing.");
        return false;
    }

    return true;
}

// ============================================================================
// Stage 8: Export
// ============================================================================

bool Pipeline::stageExport() {
    g_timer.start("export");
    log("Stage 8/8: Exporting mesh to '%s' ...", config_.output_path.c_str());

    bool ok = exportMesh(mesh_, config_.output_path);

    g_timer.stop("export");

    if (!ok) {
        error("Failed to export mesh to '%s'.", config_.output_path.c_str());
        return false;
    }

    // Also save intermediate mesh copy
    if (config_.save_intermediate) {
        std::string ply_path = intermediatePath("final_mesh.ply");
        exportPLY(mesh_, ply_path);
        vlog("Saved intermediate mesh to %s", ply_path.c_str());
    }

    return true;
}

// ============================================================================
// PipelineConfig::applyPreset — set derived parameters from quality string
// ============================================================================

void PipelineConfig::applyPreset() {
    if (quality == "low") {
        max_image_size = 1600;
        mvs_resolution = 0.5f;
        mvs_iterations = 4;
        poisson_depth = 8;
    } else if (quality == "medium") {
        max_image_size = 2400;
        mvs_resolution = 0.75f;
        mvs_iterations = 6;
        poisson_depth = 9;
    } else if (quality == "high") {
        // defaults are already "high"
    } else if (quality == "ultra") {
        max_image_size = 4800;
        mvs_resolution = 1.0f;
        mvs_iterations = 12;
        poisson_depth = 12;
    }
}

// ============================================================================
// DensePointCloud::freeGPU — release device memory
// ============================================================================

void DensePointCloud::freeGPU() {
    if (d_positions) { cudaFree(d_positions); d_positions = nullptr; }
    if (d_normals)   { cudaFree(d_normals);   d_normals   = nullptr; }
    if (d_colors)    { cudaFree(d_colors);    d_colors    = nullptr; }
    on_gpu = false;
}

// ============================================================================
// Mesh::is_watertight — check if all edges are shared by exactly 2 faces
// ============================================================================

bool Mesh::is_watertight() const {
    if (faces.empty()) return false;

    std::unordered_map<uint64_t, int> edge_count;
    auto edgeId = [](int a, int b) -> uint64_t {
        int lo = std::min(a, b), hi = std::max(a, b);
        return ((uint64_t)lo << 32) | (uint64_t)hi;
    };

    for (const auto& f : faces) {
        int v[3] = {f.x(), f.y(), f.z()};
        for (int e = 0; e < 3; e++) {
            edge_count[edgeId(v[e], v[(e + 1) % 3])]++;
        }
    }

    for (const auto& [eid, count] : edge_count) {
        if (count != 2) return false;
    }
    return true;
}
