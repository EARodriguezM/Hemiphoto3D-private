#pragma once

#include "types.h"
#include <string>

enum class ExportFormat { OBJ, STL, PLY, AUTO };

// Unified interface — AUTO detects from file extension
bool exportMesh(const Mesh& mesh, const std::string& filepath,
                ExportFormat format = ExportFormat::AUTO);

// Always exports as PLY
bool exportPointCloud(const DensePointCloud& cloud, const std::string& filepath);

// Individual exporters
bool exportOBJ(const Mesh& mesh, const std::string& path);
bool exportSTL(const Mesh& mesh, const std::string& path);
bool exportPLY(const Mesh& mesh, const std::string& path);

// Export point cloud as PLY
bool exportPointCloudPLY(const DensePointCloud& cloud, const std::string& path);
