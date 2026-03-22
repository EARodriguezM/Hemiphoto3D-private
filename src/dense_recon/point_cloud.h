#pragma once

#include "types.h"
#include <vector>

// Fuse depth maps into a dense point cloud
DensePointCloud fuseDepthMaps(const std::vector<DepthMap>& depth_maps,
                               const std::vector<CameraPose>& poses,
                               const std::vector<ImageData>& images);
