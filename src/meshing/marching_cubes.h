#pragma once

#include "types.h"

// GPU marching cubes on a volumetric grid
Mesh marchingCubes(const float* volume, int nx, int ny, int nz,
                    float iso_value = 0.0f);
