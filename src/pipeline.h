#pragma once

#include "types.h"

class Pipeline {
public:
    explicit Pipeline(const PipelineConfig& config);
    ~Pipeline();

    // Run the full reconstruction pipeline
    bool run();

private:
    PipelineConfig config_;

    // Pipeline stages (implemented in later steps)
    bool loadImages();
    bool detectFeatures();
    bool matchFeatures();
    bool reconstructSparse();
    bool bundleAdjust();
    bool denseReconstruction();
    bool fusePointClouds();
    bool generateMesh();
    bool exportMesh();
};
