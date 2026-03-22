#pragma once

#include "types.h"
#include <string>

// Export mesh to various formats
bool exportOBJ(const Mesh& mesh, const std::string& path);
bool exportSTL(const Mesh& mesh, const std::string& path);
bool exportPLY(const Mesh& mesh, const std::string& path);

// Export point cloud as PLY
bool exportPointCloudPLY(const DensePointCloud& cloud, const std::string& path);
