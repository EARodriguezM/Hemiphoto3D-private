#pragma once

#include "types.h"
#include "sfm/camera_pose.h"  // SfMResult
#include <vector>
#include <string>
#include <cstdio>

class Pipeline {
public:
    explicit Pipeline(const PipelineConfig& config);
    ~Pipeline();

    // Run the full reconstruction pipeline. Returns true on success.
    bool run();

private:
    PipelineConfig config_;
    std::string intermediate_dir_;
    FILE* log_file_ = nullptr;

    // Intermediate data (populated by stages)
    std::vector<ImageData> images_;
    std::vector<SIFTFeatures> features_;
    std::vector<ImagePairMatches> matches_;
    SfMResult sfm_;
    std::vector<DepthMap> depth_maps_;
    DensePointCloud cloud_;
    Mesh mesh_;

    // Pipeline stages
    bool stageLoadImages();
    bool stageDetectFeatures();
    bool stageMatchFeatures();
    bool stageSfM();
    bool stageMVS();
    bool stageFusion();
    bool stageMeshing();
    bool stageExport();

    // Logging
    void log(const char* fmt, ...) const;
    void vlog(const char* fmt, ...) const;  // verbose only
    void warn(const char* fmt, ...) const;
    void error(const char* fmt, ...) const;

    // Helpers
    std::string intermediatePath(const std::string& filename) const;
    void ensureDir(const std::string& dir) const;
};
