#include "export/mesh_exporter.h"

#include <fstream>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

bool exportSTL(const Mesh& mesh, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        fprintf(stderr, "[STL] Cannot open %s for writing\n", path.c_str());
        return false;
    }

    int nf = mesh.num_faces();
    size_t total = 84 + (size_t)nf * 50;
    std::vector<char> buffer(total, 0);

    // Header (80 bytes) — no "solid" prefix
    const char* hdr = "cuda_3d_recon binary STL";
    std::memcpy(buffer.data(), hdr, std::strlen(hdr));

    // Triangle count
    uint32_t tri_count = (uint32_t)nf;
    std::memcpy(buffer.data() + 80, &tri_count, 4);

    // Per-triangle data
    char* ptr = buffer.data() + 84;
    for (int i = 0; i < nf; i++) {
        const auto& f = mesh.faces[i];
        Eigen::Vector3f v1 = mesh.vertices[f.x()];
        Eigen::Vector3f v2 = mesh.vertices[f.y()];
        Eigen::Vector3f v3 = mesh.vertices[f.z()];

        Eigen::Vector3f edge1 = v2 - v1;
        Eigen::Vector3f edge2 = v3 - v1;
        Eigen::Vector3f normal = edge1.cross(edge2);
        float len = normal.norm();
        if (len > 0) normal /= len;

        // Normal
        std::memcpy(ptr, normal.data(), 12); ptr += 12;
        // Vertices
        std::memcpy(ptr, v1.data(), 12); ptr += 12;
        std::memcpy(ptr, v2.data(), 12); ptr += 12;
        std::memcpy(ptr, v3.data(), 12); ptr += 12;
        // Attribute byte count
        uint16_t attr = 0;
        std::memcpy(ptr, &attr, 2); ptr += 2;
    }

    out.write(buffer.data(), (std::streamsize)total);
    printf("[STL] Exported mesh: %d faces → %s\n", nf, path.c_str());
    return true;
}
