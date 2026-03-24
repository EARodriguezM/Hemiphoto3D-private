#pragma once

#include "types.h"

// GPU marching cubes on a volumetric TSDF grid.
// volume: nx*ny*nz float array (row-major, z-fastest) with signed distance values.
// iso_value: isosurface threshold (typically 0.0 for TSDF).
// voxel_size: physical size of each voxel (for correct vertex positions).
// origin: world-space position of grid corner (0,0,0).
Mesh marchingCubes(const float* volume, int nx, int ny, int nz,
                    float iso_value = 0.0f,
                    float voxel_size = 1.0f,
                    float origin_x = 0.0f, float origin_y = 0.0f, float origin_z = 0.0f);
