#pragma once

#include "types.h"

// Run Poisson surface reconstruction by invoking the PoissonRecon binary
Mesh poissonReconstruct(const DensePointCloud& cloud,
                         int depth = 10,
                         float screening = 4.0f);
