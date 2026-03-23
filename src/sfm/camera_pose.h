#pragma once

#include "types.h"
#include <vector>
#include <string>

// Result of incremental Structure from Motion
struct SfMResult {
    std::vector<CameraPose> cameras;
    std::vector<SparsePoint> points;
    int num_registered;
    double mean_reprojection_error;
    double scale_factor;  // world-units per reconstruction-unit (1.0 if uncalibrated)
};

// Run incremental SfM on matched images.
// Phase 1: Initialize from best pair (E decomposition + triangulation)
// Phase 2: Incremental PnP registration loop
// Phase 3: Scale calibration (if scale_bar_length_mm > 0)
// Phase 4: Turntable constraint (if --turntable)
SfMResult runIncrementalSfM(
    const std::vector<SIFTFeatures>& features,
    const std::vector<ImagePairMatches>& matches,
    const std::vector<ImageData>& images,
    const PipelineConfig& config
);

// Save SfM results to intermediate files (cameras JSON + sparse PLY)
void saveSfMCheckpoint(const SfMResult& result,
                       const std::string& output_dir);

// Load SfM results from checkpoint files. Returns false if not found.
bool loadSfMCheckpoint(SfMResult& result,
                       const std::string& output_dir);
