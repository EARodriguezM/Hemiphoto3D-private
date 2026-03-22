#pragma once

#include "types.h"
#include <vector>

// Run bundle adjustment on cameras and points
bool bundleAdjust(std::vector<CameraPose>& cameras,
                  std::vector<SparsePoint>& points,
                  const std::vector<SIFTFeatures>& features,
                  double max_reproj_error = 4.0);
