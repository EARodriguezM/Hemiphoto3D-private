#include "export/mesh_exporter.h"

#include <fstream>
#include <cstdio>
#include <algorithm>

// Stub — implemented in Step 10
bool exportPLY(const Mesh& /*mesh*/, const std::string& /*path*/) {
    return false;
}

bool exportPointCloudPLY(const DensePointCloud& cloud, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "[PLY] Cannot open %s for writing\n", path.c_str());
        return false;
    }

    int n = cloud.size();

    f << "ply\n";
    f << "format binary_little_endian 1.0\n";
    f << "element vertex " << n << "\n";
    f << "property float x\n";
    f << "property float y\n";
    f << "property float z\n";
    f << "property float nx\n";
    f << "property float ny\n";
    f << "property float nz\n";
    f << "property uchar red\n";
    f << "property uchar green\n";
    f << "property uchar blue\n";
    f << "end_header\n";

    for (int i = 0; i < n; i++) {
        float x = cloud.positions[i].x(), y = cloud.positions[i].y(), z = cloud.positions[i].z();
        float nx = cloud.normals[i].x(), ny = cloud.normals[i].y(), nz = cloud.normals[i].z();
        uint8_t r = (uint8_t)std::clamp(cloud.colors[i].x() * 255.0f, 0.0f, 255.0f);
        uint8_t g = (uint8_t)std::clamp(cloud.colors[i].y() * 255.0f, 0.0f, 255.0f);
        uint8_t b = (uint8_t)std::clamp(cloud.colors[i].z() * 255.0f, 0.0f, 255.0f);

        f.write(reinterpret_cast<const char*>(&x), 4);
        f.write(reinterpret_cast<const char*>(&y), 4);
        f.write(reinterpret_cast<const char*>(&z), 4);
        f.write(reinterpret_cast<const char*>(&nx), 4);
        f.write(reinterpret_cast<const char*>(&ny), 4);
        f.write(reinterpret_cast<const char*>(&nz), 4);
        f.write(reinterpret_cast<const char*>(&r), 1);
        f.write(reinterpret_cast<const char*>(&g), 1);
        f.write(reinterpret_cast<const char*>(&b), 1);
    }

    return true;
}
