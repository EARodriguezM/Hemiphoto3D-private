#include "export/mesh_exporter.h"

#include <fstream>
#include <cstdio>
#include <algorithm>

bool exportPLY(const Mesh& mesh, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "[PLY] Cannot open %s for writing\n", path.c_str());
        return false;
    }

    int nv = mesh.num_vertices();
    int nf = mesh.num_faces();
    bool has_normals = (int)mesh.normals.size() == nv;
    bool has_colors  = (int)mesh.colors.size() == nv;

    f << "ply\n";
    f << "format binary_little_endian 1.0\n";
    f << "element vertex " << nv << "\n";
    f << "property float x\n";
    f << "property float y\n";
    f << "property float z\n";
    if (has_normals) {
        f << "property float nx\n";
        f << "property float ny\n";
        f << "property float nz\n";
    }
    if (has_colors) {
        f << "property uchar red\n";
        f << "property uchar green\n";
        f << "property uchar blue\n";
    }
    if (nf > 0) {
        f << "element face " << nf << "\n";
        f << "property list uchar int vertex_indices\n";
    }
    f << "end_header\n";

    // Write vertices
    for (int i = 0; i < nv; i++) {
        float x = mesh.vertices[i].x(), y = mesh.vertices[i].y(), z = mesh.vertices[i].z();
        f.write(reinterpret_cast<const char*>(&x), 4);
        f.write(reinterpret_cast<const char*>(&y), 4);
        f.write(reinterpret_cast<const char*>(&z), 4);

        if (has_normals) {
            float nx = mesh.normals[i].x(), ny = mesh.normals[i].y(), nz = mesh.normals[i].z();
            f.write(reinterpret_cast<const char*>(&nx), 4);
            f.write(reinterpret_cast<const char*>(&ny), 4);
            f.write(reinterpret_cast<const char*>(&nz), 4);
        }

        if (has_colors) {
            uint8_t r = (uint8_t)std::clamp(mesh.colors[i].x() * 255.0f, 0.0f, 255.0f);
            uint8_t g = (uint8_t)std::clamp(mesh.colors[i].y() * 255.0f, 0.0f, 255.0f);
            uint8_t b = (uint8_t)std::clamp(mesh.colors[i].z() * 255.0f, 0.0f, 255.0f);
            f.write(reinterpret_cast<const char*>(&r), 1);
            f.write(reinterpret_cast<const char*>(&g), 1);
            f.write(reinterpret_cast<const char*>(&b), 1);
        }
    }

    // Write faces
    for (int i = 0; i < nf; i++) {
        uint8_t count = 3;
        int32_t idx[3] = {mesh.faces[i].x(), mesh.faces[i].y(), mesh.faces[i].z()};
        f.write(reinterpret_cast<const char*>(&count), 1);
        f.write(reinterpret_cast<const char*>(idx), 12);
    }

    printf("[PLY] Exported mesh: %d vertices, %d faces → %s\n", nv, nf, path.c_str());
    return true;
}

// Unified interface
bool exportMesh(const Mesh& mesh, const std::string& filepath, ExportFormat format) {
    if (format == ExportFormat::AUTO) {
        std::string ext = filepath.substr(filepath.find_last_of('.') + 1);
        if (ext == "obj" || ext == "OBJ") format = ExportFormat::OBJ;
        else if (ext == "stl" || ext == "STL") format = ExportFormat::STL;
        else format = ExportFormat::PLY;
    }
    switch (format) {
        case ExportFormat::OBJ: return exportOBJ(mesh, filepath);
        case ExportFormat::STL: return exportSTL(mesh, filepath);
        case ExportFormat::PLY: return exportPLY(mesh, filepath);
        default: return false;
    }
}

bool exportPointCloud(const DensePointCloud& cloud, const std::string& filepath) {
    return exportPointCloudPLY(cloud, filepath);
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
