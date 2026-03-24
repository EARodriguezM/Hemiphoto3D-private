#pragma once

#include "types.h"
#include <string>

// Run Poisson surface reconstruction by invoking the PoissonRecon binary.
// Primary meshing approach — wraps the external binary with PLY I/O.
Mesh poissonReconstruct(const DensePointCloud& cloud,
                         int depth = 10,
                         float screening = 4.0f);

// Post-process a mesh: Taubin smoothing, decimation, connected components, hole filling.
void postProcessMesh(Mesh& mesh, const PipelineConfig& config);

// Transfer colors from point cloud to mesh vertices (nearest-neighbor lookup).
void transferColors(Mesh& mesh, const DensePointCloud& cloud);

// Remove faces whose vertices are far from the input point cloud (density trimming).
void trimLowDensityFaces(Mesh& mesh, const DensePointCloud& cloud,
                          float percentile = 0.10f);
