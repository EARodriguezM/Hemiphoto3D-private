#include "meshing/poisson_recon.h"
#include "export/mesh_exporter.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

// ============================================================================
// PLY I/O helpers (binary little-endian)
// ============================================================================

// Write oriented point cloud (positions + normals) as binary PLY
static bool writeOrientedPLY(const DensePointCloud& cloud,
                              const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "[Poisson] Cannot write %s\n", path.c_str());
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
    f << "end_header\n";

    for (int i = 0; i < n; i++) {
        float vals[6] = {
            cloud.positions[i].x(), cloud.positions[i].y(), cloud.positions[i].z(),
            cloud.normals[i].x(),   cloud.normals[i].y(),   cloud.normals[i].z()
        };
        f.write(reinterpret_cast<const char*>(vals), sizeof(vals));
    }

    return true;
}

// Read a PLY mesh (vertices + faces), supports binary little-endian and ASCII
static bool readPLYMesh(const std::string& path, Mesh& mesh) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "[Poisson] Cannot read %s\n", path.c_str());
        return false;
    }

    std::string line;
    int num_vertices = 0, num_faces = 0;
    bool binary = false;
    int vertex_props = 0;     // number of float properties per vertex
    bool has_normals = false;
    bool has_colors = false;
    bool face_count_is_int = false; // true if face list uses int count (PoissonRecon), false for uchar
    bool in_vertex_element = false;

    // Parse header
    while (std::getline(f, line)) {
        // Strip trailing \r if present
        if (!line.empty() && line.back() == '\r') line.pop_back();

        if (line.find("format binary_little_endian") != std::string::npos)
            binary = true;
        if (line.find("element vertex") != std::string::npos) {
            sscanf(line.c_str(), "element vertex %d", &num_vertices);
            in_vertex_element = true;
        }
        if (line.find("element face") != std::string::npos) {
            sscanf(line.c_str(), "element face %d", &num_faces);
            in_vertex_element = false;
        }
        // Only count vertex properties (not face properties)
        if (in_vertex_element &&
            (line.find("property float") != std::string::npos ||
             line.find("property double") != std::string::npos))
            vertex_props++;
        if (line.find("property float nx") != std::string::npos)
            has_normals = true;
        if (line.find("property uchar red") != std::string::npos ||
            line.find("property float red") != std::string::npos)
            has_colors = true;
        // Detect face list count type: "property list int int" vs "property list uchar int"
        if (line.find("property list") != std::string::npos) {
            if (line.find("property list int") == 0 ||
                line.find("property list int") != std::string::npos)
                face_count_is_int = (line.find("list int ") != std::string::npos ||
                                     line.find("list int32") != std::string::npos);
        }
        if (line == "end_header")
            break;
    }

    if (num_vertices == 0) {
        fprintf(stderr, "[Poisson] PLY has 0 vertices\n");
        return false;
    }

    mesh.vertices.resize(num_vertices);
    if (has_normals) mesh.normals.resize(num_vertices);

    if (binary) {
        // Read vertices: each vertex has vertex_props float values
        std::vector<float> buf(vertex_props);
        for (int i = 0; i < num_vertices; i++) {
            f.read(reinterpret_cast<char*>(buf.data()), vertex_props * sizeof(float));
            mesh.vertices[i] = Eigen::Vector3f(buf[0], buf[1], buf[2]);
            if (has_normals && vertex_props >= 6)
                mesh.normals[i] = Eigen::Vector3f(buf[3], buf[4], buf[5]);
        }

        // Read faces
        mesh.faces.resize(num_faces);
        for (int i = 0; i < num_faces; i++) {
            int count;
            if (face_count_is_int) {
                int32_t cnt;
                f.read(reinterpret_cast<char*>(&cnt), sizeof(int32_t));
                count = cnt;
            } else {
                uint8_t cnt;
                f.read(reinterpret_cast<char*>(&cnt), 1);
                count = cnt;
            }
            if (count == 3) {
                int32_t idx[3];
                f.read(reinterpret_cast<char*>(idx), 3 * sizeof(int32_t));
                mesh.faces[i] = Eigen::Vector3i(idx[0], idx[1], idx[2]);
            } else if (count > 3) {
                // Triangulate polygon fan for quads/polygons
                std::vector<int32_t> idx(count);
                f.read(reinterpret_cast<char*>(idx.data()), count * sizeof(int32_t));
                mesh.faces[i] = Eigen::Vector3i(idx[0], idx[1], idx[2]);
                for (int j = 3; j < count; j++) {
                    mesh.faces.push_back(Eigen::Vector3i(idx[0], idx[j - 1], idx[j]));
                }
            }
        }
    } else {
        // ASCII
        for (int i = 0; i < num_vertices; i++) {
            std::getline(f, line);
            float x, y, z, nx, ny, nz;
            int parsed = sscanf(line.c_str(), "%f %f %f %f %f %f", &x, &y, &z, &nx, &ny, &nz);
            mesh.vertices[i] = Eigen::Vector3f(x, y, z);
            if (has_normals && parsed >= 6)
                mesh.normals[i] = Eigen::Vector3f(nx, ny, nz);
        }

        mesh.faces.resize(num_faces);
        for (int i = 0; i < num_faces; i++) {
            std::getline(f, line);
            int count, a, b, c;
            sscanf(line.c_str(), "%d %d %d %d", &count, &a, &b, &c);
            mesh.faces[i] = Eigen::Vector3i(a, b, c);
            if (count > 3) {
                // fan triangulation
                std::vector<int> idx(count);
                idx[0] = a; idx[1] = b; idx[2] = c;
                // re-parse for all indices
                std::istringstream iss(line);
                iss >> count;
                for (int j = 0; j < count; j++) iss >> idx[j];
                for (int j = 3; j < count; j++)
                    mesh.faces.push_back(Eigen::Vector3i(idx[0], idx[j - 1], idx[j]));
            }
        }
    }

    printf("[Poisson] Loaded PLY: %d vertices, %d faces\n",
           mesh.num_vertices(), mesh.num_faces());
    return true;
}

// ============================================================================
// Density trimming — remove faces far from input point cloud
// ============================================================================
void trimLowDensityFaces(Mesh& mesh, const DensePointCloud& cloud,
                          float percentile) {
    if (mesh.faces.empty() || cloud.size() == 0) return;

    int nv = mesh.num_vertices();

    // Compute bounding box of point cloud for grid acceleration
    Eigen::Vector3f bmin = cloud.positions[0], bmax = cloud.positions[0];
    for (int i = 1; i < cloud.size(); i++) {
        bmin = bmin.cwiseMin(cloud.positions[i]);
        bmax = bmax.cwiseMax(cloud.positions[i]);
    }

    Eigen::Vector3f extent = bmax - bmin;
    float cell_size = extent.maxCoeff() / 50.0f; // ~50 cells along longest axis
    if (cell_size < 1e-10f) return;

    // Build a uniform grid of point cloud points
    auto cellKey = [&](const Eigen::Vector3f& p) -> uint64_t {
        int cx = (int)((p.x() - bmin.x()) / cell_size);
        int cy = (int)((p.y() - bmin.y()) / cell_size);
        int cz = (int)((p.z() - bmin.z()) / cell_size);
        return ((uint64_t)cx * 73856093ULL) ^ ((uint64_t)cy * 19349663ULL) ^ ((uint64_t)cz * 83492791ULL);
    };

    std::unordered_map<uint64_t, std::vector<int>> grid;
    for (int i = 0; i < cloud.size(); i++) {
        grid[cellKey(cloud.positions[i])].push_back(i);
    }

    // For each mesh vertex, find distance to nearest point cloud point
    std::vector<float> vertex_dist(nv, std::numeric_limits<float>::max());
    for (int vi = 0; vi < nv; vi++) {
        const Eigen::Vector3f& v = mesh.vertices[vi];
        int cx = (int)((v.x() - bmin.x()) / cell_size);
        int cy = (int)((v.y() - bmin.y()) / cell_size);
        int cz = (int)((v.z() - bmin.z()) / cell_size);

        float best_d2 = std::numeric_limits<float>::max();

        // Search 3x3x3 neighborhood
        for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
        for (int dz = -1; dz <= 1; dz++) {
            int nx = cx + dx, ny = cy + dy, nz = cz + dz;
            uint64_t key = ((uint64_t)nx * 73856093ULL) ^ ((uint64_t)ny * 19349663ULL) ^ ((uint64_t)nz * 83492791ULL);
            auto it = grid.find(key);
            if (it == grid.end()) continue;
            for (int pi : it->second) {
                float d2 = (cloud.positions[pi] - v).squaredNorm();
                if (d2 < best_d2) best_d2 = d2;
            }
        }

        vertex_dist[vi] = std::sqrt(best_d2);
    }

    // Compute per-face max distance (all 3 vertices)
    std::vector<float> face_dist(mesh.num_faces());
    for (int fi = 0; fi < mesh.num_faces(); fi++) {
        const auto& f = mesh.faces[fi];
        face_dist[fi] = std::max({vertex_dist[f.x()], vertex_dist[f.y()], vertex_dist[f.z()]});
    }

    // Compute threshold at given percentile
    std::vector<float> sorted_dist = face_dist;
    std::sort(sorted_dist.begin(), sorted_dist.end());
    int cutoff_idx = std::min((int)(sorted_dist.size() * (1.0f - percentile)),
                              (int)sorted_dist.size() - 1);
    float threshold = sorted_dist[cutoff_idx];

    // Remove faces above threshold
    std::vector<Eigen::Vector3i> kept_faces;
    kept_faces.reserve(mesh.num_faces());
    for (int fi = 0; fi < mesh.num_faces(); fi++) {
        if (face_dist[fi] <= threshold) {
            kept_faces.push_back(mesh.faces[fi]);
        }
    }

    int removed = mesh.num_faces() - (int)kept_faces.size();
    mesh.faces = std::move(kept_faces);
    printf("[Poisson] Density trim: removed %d faces (threshold=%.6f), %d remaining\n",
           removed, threshold, mesh.num_faces());
}

// ============================================================================
// Color transfer — nearest-neighbor from point cloud to mesh vertices
// ============================================================================
void transferColors(Mesh& mesh, const DensePointCloud& cloud) {
    if (cloud.size() == 0 || mesh.num_vertices() == 0) return;

    // Build uniform grid for fast nearest-neighbor lookup
    Eigen::Vector3f bmin = cloud.positions[0], bmax = cloud.positions[0];
    for (int i = 1; i < cloud.size(); i++) {
        bmin = bmin.cwiseMin(cloud.positions[i]);
        bmax = bmax.cwiseMax(cloud.positions[i]);
    }

    Eigen::Vector3f extent = bmax - bmin;
    float cell_size = extent.maxCoeff() / 50.0f;
    if (cell_size < 1e-10f) cell_size = 1.0f;

    std::unordered_map<uint64_t, std::vector<int>> grid;
    auto cellKey = [&](int cx, int cy, int cz) -> uint64_t {
        return ((uint64_t)cx * 73856093ULL) ^ ((uint64_t)cy * 19349663ULL) ^ ((uint64_t)cz * 83492791ULL);
    };

    for (int i = 0; i < cloud.size(); i++) {
        int cx = (int)((cloud.positions[i].x() - bmin.x()) / cell_size);
        int cy = (int)((cloud.positions[i].y() - bmin.y()) / cell_size);
        int cz = (int)((cloud.positions[i].z() - bmin.z()) / cell_size);
        grid[cellKey(cx, cy, cz)].push_back(i);
    }

    mesh.colors.resize(mesh.num_vertices());

    for (int vi = 0; vi < mesh.num_vertices(); vi++) {
        const Eigen::Vector3f& v = mesh.vertices[vi];
        int cx = (int)((v.x() - bmin.x()) / cell_size);
        int cy = (int)((v.y() - bmin.y()) / cell_size);
        int cz = (int)((v.z() - bmin.z()) / cell_size);

        float best_d2 = std::numeric_limits<float>::max();
        int best_idx = 0;

        // Search expanding neighborhood until we find something
        for (int r = 1; r <= 5; r++) {
            for (int dx = -r; dx <= r; dx++)
            for (int dy = -r; dy <= r; dy++)
            for (int dz = -r; dz <= r; dz++) {
                // Only search the shell (skip inner cells already searched)
                if (r > 1 && std::abs(dx) < r && std::abs(dy) < r && std::abs(dz) < r)
                    continue;
                auto it = grid.find(cellKey(cx + dx, cy + dy, cz + dz));
                if (it == grid.end()) continue;
                for (int pi : it->second) {
                    float d2 = (cloud.positions[pi] - v).squaredNorm();
                    if (d2 < best_d2) {
                        best_d2 = d2;
                        best_idx = pi;
                    }
                }
            }
            if (best_d2 < std::numeric_limits<float>::max()) break;
        }

        if (!cloud.colors.empty()) {
            mesh.colors[vi] = cloud.colors[best_idx];
        } else {
            mesh.colors[vi] = Eigen::Vector3f(0.7f, 0.7f, 0.7f);
        }
    }

    printf("[Poisson] Color transfer: %d vertices colored\n", mesh.num_vertices());
}

// ============================================================================
// Connected components — keep only the largest
// ============================================================================
static void keepLargestComponent(Mesh& mesh) {
    if (mesh.faces.empty()) return;

    int nv = mesh.num_vertices();
    int nf = mesh.num_faces();

    // Build vertex-to-face adjacency
    std::vector<std::vector<int>> v2f(nv);
    for (int fi = 0; fi < nf; fi++) {
        const auto& f = mesh.faces[fi];
        v2f[f.x()].push_back(fi);
        v2f[f.y()].push_back(fi);
        v2f[f.z()].push_back(fi);
    }

    // BFS on face adjacency (faces share an edge → adjacent)
    std::vector<int> face_comp(nf, -1);
    int num_components = 0;
    std::vector<int> comp_sizes;

    for (int fi = 0; fi < nf; fi++) {
        if (face_comp[fi] >= 0) continue;
        int comp_id = num_components++;
        comp_sizes.push_back(0);

        std::queue<int> q;
        q.push(fi);
        face_comp[fi] = comp_id;

        while (!q.empty()) {
            int cur = q.front(); q.pop();
            comp_sizes[comp_id]++;

            const auto& face = mesh.faces[cur];
            // For each vertex in this face, find adjacent faces
            for (int vi = 0; vi < 3; vi++) {
                int v = face[vi];
                for (int adj_fi : v2f[v]) {
                    if (face_comp[adj_fi] < 0) {
                        face_comp[adj_fi] = comp_id;
                        q.push(adj_fi);
                    }
                }
            }
        }
    }

    if (num_components <= 1) {
        printf("[Mesh] Single connected component (%d faces)\n", nf);
        return;
    }

    // Find largest component
    int largest = (int)(std::max_element(comp_sizes.begin(), comp_sizes.end()) - comp_sizes.begin());

    // Keep only faces in the largest component
    std::vector<Eigen::Vector3i> kept_faces;
    kept_faces.reserve(comp_sizes[largest]);
    for (int fi = 0; fi < nf; fi++) {
        if (face_comp[fi] == largest)
            kept_faces.push_back(mesh.faces[fi]);
    }

    int removed_faces = nf - (int)kept_faces.size();
    mesh.faces = std::move(kept_faces);

    // Compact vertices: remove unreferenced ones
    std::vector<bool> used(nv, false);
    for (const auto& f : mesh.faces) {
        used[f.x()] = true;
        used[f.y()] = true;
        used[f.z()] = true;
    }

    std::vector<int> remap(nv, -1);
    int new_count = 0;
    for (int i = 0; i < nv; i++) {
        if (used[i]) remap[i] = new_count++;
    }

    std::vector<Eigen::Vector3f> new_verts(new_count);
    std::vector<Eigen::Vector3f> new_normals;
    std::vector<Eigen::Vector3f> new_colors;
    if (!mesh.normals.empty()) new_normals.resize(new_count);
    if (!mesh.colors.empty()) new_colors.resize(new_count);

    for (int i = 0; i < nv; i++) {
        if (remap[i] >= 0) {
            new_verts[remap[i]] = mesh.vertices[i];
            if (!mesh.normals.empty()) new_normals[remap[i]] = mesh.normals[i];
            if (!mesh.colors.empty()) new_colors[remap[i]] = mesh.colors[i];
        }
    }

    for (auto& f : mesh.faces) {
        f = Eigen::Vector3i(remap[f.x()], remap[f.y()], remap[f.z()]);
    }

    int removed_verts = nv - new_count;
    mesh.vertices = std::move(new_verts);
    mesh.normals = std::move(new_normals);
    mesh.colors = std::move(new_colors);

    printf("[Mesh] Kept largest component (%d/%d), removed %d faces, %d vertices\n",
           largest + 1, num_components, removed_faces, removed_verts);
}

// ============================================================================
// Taubin smoothing — alternating +lambda/-mu to prevent shrinkage
// ============================================================================
static void taubinSmooth(Mesh& mesh, int iterations, float lambda) {
    if (mesh.faces.empty() || mesh.vertices.empty()) return;

    int nv = mesh.num_vertices();
    float mu = -lambda / (1.0f - 0.1f * lambda); // Taubin's mu: slightly larger magnitude

    // Build adjacency
    std::vector<std::unordered_set<int>> adj(nv);
    for (const auto& f : mesh.faces) {
        adj[f.x()].insert(f.y()); adj[f.x()].insert(f.z());
        adj[f.y()].insert(f.x()); adj[f.y()].insert(f.z());
        adj[f.z()].insert(f.x()); adj[f.z()].insert(f.y());
    }

    std::vector<Eigen::Vector3f> new_verts(nv);

    for (int iter = 0; iter < iterations; iter++) {
        // Pass 1: shrink (+lambda)
        for (int i = 0; i < nv; i++) {
            if (adj[i].empty()) {
                new_verts[i] = mesh.vertices[i];
                continue;
            }
            Eigen::Vector3f mean = Eigen::Vector3f::Zero();
            for (int j : adj[i]) mean += mesh.vertices[j];
            mean /= (float)adj[i].size();
            new_verts[i] = mesh.vertices[i] + lambda * (mean - mesh.vertices[i]);
        }
        mesh.vertices = new_verts;

        // Pass 2: inflate (-mu, which acts as negative smoothing)
        for (int i = 0; i < nv; i++) {
            if (adj[i].empty()) {
                new_verts[i] = mesh.vertices[i];
                continue;
            }
            Eigen::Vector3f mean = Eigen::Vector3f::Zero();
            for (int j : adj[i]) mean += mesh.vertices[j];
            mean /= (float)adj[i].size();
            new_verts[i] = mesh.vertices[i] + mu * (mean - mesh.vertices[i]);
        }
        mesh.vertices = new_verts;
    }

    // Recompute normals after smoothing
    if (!mesh.normals.empty()) {
        std::fill(mesh.normals.begin(), mesh.normals.end(), Eigen::Vector3f::Zero());
        for (const auto& f : mesh.faces) {
            Eigen::Vector3f e1 = mesh.vertices[f.y()] - mesh.vertices[f.x()];
            Eigen::Vector3f e2 = mesh.vertices[f.z()] - mesh.vertices[f.x()];
            Eigen::Vector3f n = e1.cross(e2);
            mesh.normals[f.x()] += n;
            mesh.normals[f.y()] += n;
            mesh.normals[f.z()] += n;
        }
        for (auto& n : mesh.normals) {
            float len = n.norm();
            if (len > 1e-10f) n /= len;
        }
    }

    printf("[Mesh] Taubin smoothing: %d iterations, lambda=%.2f, mu=%.2f\n",
           iterations, lambda, mu);
}

// ============================================================================
// Edge-collapse mesh decimation (simplified Garland-Heckbert QEM)
// ============================================================================
static void decimateMesh(Mesh& mesh, int target_faces) {
    if (mesh.num_faces() <= target_faces || target_faces <= 0) return;

    int nv = mesh.num_vertices();
    int nf = mesh.num_faces();

    // Compute per-vertex quadric error matrices (4x4, stored as 10 unique elements)
    // Q = sum of plane equations p*p^T for each adjacent face
    struct Quadric {
        double q[10] = {}; // symmetric 4x4: q[0..9] = a2, ab, ac, ad, b2, bc, bd, c2, cd, d2

        void addPlane(double a, double b, double c, double d) {
            q[0] += a*a; q[1] += a*b; q[2] += a*c; q[3] += a*d;
            q[4] += b*b; q[5] += b*c; q[6] += b*d;
            q[7] += c*c; q[8] += c*d;
            q[9] += d*d;
        }

        Quadric operator+(const Quadric& o) const {
            Quadric r;
            for (int i = 0; i < 10; i++) r.q[i] = q[i] + o.q[i];
            return r;
        }

        double evaluate(const Eigen::Vector3f& v) const {
            double x = v.x(), y = v.y(), z = v.z();
            return q[0]*x*x + 2*q[1]*x*y + 2*q[2]*x*z + 2*q[3]*x
                 + q[4]*y*y + 2*q[5]*y*z + 2*q[6]*y
                 + q[7]*z*z + 2*q[8]*z
                 + q[9];
        }
    };

    std::vector<Quadric> vertex_Q(nv);

    for (int fi = 0; fi < nf; fi++) {
        const auto& f = mesh.faces[fi];
        Eigen::Vector3f e1 = mesh.vertices[f.y()] - mesh.vertices[f.x()];
        Eigen::Vector3f e2 = mesh.vertices[f.z()] - mesh.vertices[f.x()];
        Eigen::Vector3f n = e1.cross(e2).normalized();
        double a = n.x(), b = n.y(), c = n.z();
        double d = -n.dot(mesh.vertices[f.x()]);
        vertex_Q[f.x()].addPlane(a, b, c, d);
        vertex_Q[f.y()].addPlane(a, b, c, d);
        vertex_Q[f.z()].addPlane(a, b, c, d);
    }

    // Use edge collapse with priority queue
    // For simplicity, use iterative greedy collapse: pick the cheapest edge each round
    // This is O(n^2) in the worst case but works fine for moderate decimation

    // Track which vertices/faces are still alive
    std::vector<bool> face_alive(nf, true);
    std::vector<bool> vert_alive(nv, true);
    int current_faces = nf;

    // Build edge -> face adjacency
    struct EdgeKey {
        int v0, v1;
        bool operator==(const EdgeKey& o) const { return v0 == o.v0 && v1 == o.v1; }
    };
    struct EdgeHash {
        size_t operator()(const EdgeKey& e) const {
            return std::hash<int>()(e.v0) ^ (std::hash<int>()(e.v1) << 16);
        }
    };

    auto makeEdge = [](int a, int b) -> EdgeKey {
        return {std::min(a, b), std::max(a, b)};
    };

    // Priority queue: (cost, v0, v1)
    using EdgeCost = std::tuple<double, int, int>;
    std::priority_queue<EdgeCost, std::vector<EdgeCost>, std::greater<EdgeCost>> pq;

    // Initialize edge costs
    std::unordered_set<uint64_t> seen_edges;
    auto edgeId = [](int a, int b) -> uint64_t {
        int lo = std::min(a, b), hi = std::max(a, b);
        return ((uint64_t)lo << 32) | (uint64_t)hi;
    };

    for (int fi = 0; fi < nf; fi++) {
        const auto& f = mesh.faces[fi];
        int verts[3] = {f.x(), f.y(), f.z()};
        for (int e = 0; e < 3; e++) {
            int v0 = verts[e], v1 = verts[(e + 1) % 3];
            uint64_t eid = edgeId(v0, v1);
            if (seen_edges.count(eid)) continue;
            seen_edges.insert(eid);

            Quadric Q = vertex_Q[v0] + vertex_Q[v1];
            // Use midpoint as collapse target
            Eigen::Vector3f mid = 0.5f * (mesh.vertices[v0] + mesh.vertices[v1]);
            double cost = Q.evaluate(mid);
            pq.push({cost, std::min(v0, v1), std::max(v0, v1)});
        }
    }

    // Vertex merge tracking (union-find like)
    std::vector<int> remap(nv);
    std::iota(remap.begin(), remap.end(), 0);

    auto find = [&](int v) -> int {
        while (remap[v] != v) v = remap[v] = remap[remap[v]]; // path compression
        return v;
    };

    while (current_faces > target_faces && !pq.empty()) {
        auto [cost, v0_orig, v1_orig] = pq.top();
        pq.pop();

        int v0 = find(v0_orig);
        int v1 = find(v1_orig);
        if (v0 == v1 || !vert_alive[v0] || !vert_alive[v1]) continue;

        // Collapse v1 into v0
        mesh.vertices[v0] = 0.5f * (mesh.vertices[v0] + mesh.vertices[v1]);
        vertex_Q[v0] = vertex_Q[v0] + vertex_Q[v1];
        vert_alive[v1] = false;
        remap[v1] = v0;

        // Update all faces referencing v1 → v0
        for (int fi = 0; fi < nf; fi++) {
            if (!face_alive[fi]) continue;
            auto& f = mesh.faces[fi];
            bool ref_v1 = false;
            for (int k = 0; k < 3; k++) {
                if (find(f[k]) == v1 || f[k] == v1) {
                    f[k] = v0;
                    ref_v1 = true;
                }
                // Also resolve any stale remaps
                f[k] = find(f[k]);
            }

            // Check if face is degenerate (two or more same vertices)
            if (f.x() == f.y() || f.y() == f.z() || f.x() == f.z()) {
                face_alive[fi] = false;
                current_faces--;
            }
        }
    }

    // Compact the mesh
    std::vector<int> vert_remap(nv, -1);
    int new_nv = 0;
    for (int i = 0; i < nv; i++) {
        if (vert_alive[i] && find(i) == i) {
            vert_remap[i] = new_nv++;
        }
    }

    std::vector<Eigen::Vector3f> new_verts(new_nv);
    std::vector<Eigen::Vector3f> new_normals;
    std::vector<Eigen::Vector3f> new_colors;
    if (!mesh.normals.empty()) new_normals.resize(new_nv);
    if (!mesh.colors.empty()) new_colors.resize(new_nv);

    for (int i = 0; i < nv; i++) {
        if (vert_remap[i] >= 0) {
            new_verts[vert_remap[i]] = mesh.vertices[i];
            if (!mesh.normals.empty() && i < (int)mesh.normals.size())
                new_normals[vert_remap[i]] = mesh.normals[i];
            if (!mesh.colors.empty() && i < (int)mesh.colors.size())
                new_colors[vert_remap[i]] = mesh.colors[i];
        }
    }

    std::vector<Eigen::Vector3i> new_faces;
    new_faces.reserve(current_faces);
    for (int fi = 0; fi < nf; fi++) {
        if (!face_alive[fi]) continue;
        auto& f = mesh.faces[fi];
        int a = vert_remap[find(f.x())];
        int b = vert_remap[find(f.y())];
        int c = vert_remap[find(f.z())];
        if (a >= 0 && b >= 0 && c >= 0 && a != b && b != c && a != c) {
            new_faces.push_back(Eigen::Vector3i(a, b, c));
        }
    }

    mesh.vertices = std::move(new_verts);
    mesh.normals = std::move(new_normals);
    mesh.colors = std::move(new_colors);
    mesh.faces = std::move(new_faces);

    printf("[Mesh] Decimation: %d → %d faces, %d → %d vertices\n",
           nf, mesh.num_faces(), nv, mesh.num_vertices());
}

// ============================================================================
// Hole filling — fill small boundary loops
// ============================================================================
static void fillSmallHoles(Mesh& mesh, int max_boundary_edges) {
    if (mesh.faces.empty()) return;

    // Build edge → face count
    std::unordered_map<uint64_t, int> edge_count;
    auto edgeId = [](int a, int b) -> uint64_t {
        int lo = std::min(a, b), hi = std::max(a, b);
        return ((uint64_t)lo << 32) | (uint64_t)hi;
    };

    // Also build directed half-edge map for boundary traversal
    // key: (from, to), value: exists
    std::unordered_map<uint64_t, bool> half_edges;
    auto halfEdgeId = [](int from, int to) -> uint64_t {
        return ((uint64_t)from << 32) | (uint64_t)to;
    };

    for (const auto& f : mesh.faces) {
        int v[3] = {f.x(), f.y(), f.z()};
        for (int e = 0; e < 3; e++) {
            int a = v[e], b = v[(e + 1) % 3];
            edge_count[edgeId(a, b)]++;
            half_edges[halfEdgeId(a, b)] = true;
        }
    }

    // Find boundary half-edges (edges with count == 1)
    // A boundary half-edge (a,b) has no opposing half-edge (b,a)
    std::unordered_map<int, int> boundary_next; // vertex → next boundary vertex
    for (const auto& f : mesh.faces) {
        int v[3] = {f.x(), f.y(), f.z()};
        for (int e = 0; e < 3; e++) {
            int a = v[e], b = v[(e + 1) % 3];
            if (edge_count[edgeId(a, b)] == 1) {
                // This half-edge's opposite (b→a) is boundary
                boundary_next[b] = a;
            }
        }
    }

    if (boundary_next.empty()) return;

    // Trace boundary loops
    std::unordered_set<int> visited;
    int holes_filled = 0;

    for (auto& [start, _] : boundary_next) {
        if (visited.count(start)) continue;

        // Trace loop
        std::vector<int> loop;
        int cur = start;
        bool valid = true;
        while (true) {
            if (visited.count(cur)) {
                if (cur == start && !loop.empty()) break; // closed loop
                valid = false;
                break;
            }
            visited.insert(cur);
            loop.push_back(cur);
            auto it = boundary_next.find(cur);
            if (it == boundary_next.end()) { valid = false; break; }
            cur = it->second;
            if ((int)loop.size() > max_boundary_edges + 1) { valid = false; break; }
        }

        if (!valid || (int)loop.size() < 3 || (int)loop.size() > max_boundary_edges)
            continue;

        // Fan triangulate from first vertex
        for (int i = 1; i < (int)loop.size() - 1; i++) {
            mesh.faces.push_back(Eigen::Vector3i(loop[0], loop[i], loop[i + 1]));
        }
        holes_filled++;
    }

    printf("[Mesh] Hole filling: filled %d small holes (<= %d boundary edges)\n",
           holes_filled, max_boundary_edges);
}

// ============================================================================
// Post-processing pipeline
// ============================================================================
void postProcessMesh(Mesh& mesh, const PipelineConfig& config) {
    printf("[Mesh] Post-processing: %d vertices, %d faces\n",
           mesh.num_vertices(), mesh.num_faces());

    // 1. Taubin smoothing
    if (config.smooth_iterations > 0) {
        taubinSmooth(mesh, config.smooth_iterations, config.smooth_lambda);
    }

    // 2. Decimation
    if (config.decimate_target > 0) {
        decimateMesh(mesh, config.decimate_target);
    }

    // 3. Remove small connected components
    keepLargestComponent(mesh);

    // 4. Fill small holes
    fillSmallHoles(mesh, 20);

    // 5. Watertight check
    bool wt = mesh.is_watertight();
    printf("[Mesh] Watertight: %s\n", wt ? "yes" : "no");
    printf("[Mesh] Post-processing done: %d vertices, %d faces\n",
           mesh.num_vertices(), mesh.num_faces());
}

// ============================================================================
// Poisson Reconstruction — main entry point
// ============================================================================
Mesh poissonReconstruct(const DensePointCloud& cloud,
                         int depth,
                         float screening) {
    if (cloud.size() == 0) {
        fprintf(stderr, "[Poisson] Empty point cloud, nothing to reconstruct\n");
        return Mesh();
    }

    printf("[Poisson] Reconstructing mesh from %d points (depth=%d, screening=%.1f)\n",
           cloud.size(), depth, screening);

    // Create temporary directory for PLY I/O
    std::string tmp_dir = fs::temp_directory_path().string();
    std::string input_ply  = tmp_dir + "/poisson_input.ply";
    std::string output_ply = tmp_dir + "/poisson_output.ply";

    // 1. Export oriented point cloud
    if (!writeOrientedPLY(cloud, input_ply)) {
        return Mesh();
    }

    // 2. Call PoissonRecon binary
    char cmd[1024];

    snprintf(cmd, sizeof(cmd),
             "PoissonRecon --in \"%s\" --out \"%s\" --depth %d --pointWeight %.1f "
             "--samplesPerNode 1.5 --parallel 2 2>&1",
             input_ply.c_str(), output_ply.c_str(),
             depth, screening);

    printf("[Poisson] Running: %s\n", cmd);
    int ret = system(cmd);

    if (ret != 0) {
        printf("[Poisson] PoissonRecon failed (exit %d), retrying with depth %d\n",
               ret, depth - 1);
        snprintf(cmd, sizeof(cmd),
                 "PoissonRecon --in \"%s\" --out \"%s\" --depth %d --pointWeight %.1f "
                 "--samplesPerNode 1.5 --parallel 2 2>&1",
                 input_ply.c_str(), output_ply.c_str(),
                 depth - 1, screening);
        ret = system(cmd);
    }

    // Clean up input
    std::remove(input_ply.c_str());

    if (ret != 0) {
        fprintf(stderr, "[Poisson] PoissonRecon failed (exit %d)\n", ret);
        return Mesh();
    }

    // 3. Load output mesh
    Mesh mesh;
    if (!readPLYMesh(output_ply, mesh)) {
        fprintf(stderr, "[Poisson] Failed to read output mesh\n");
        std::remove(output_ply.c_str());
        return Mesh();
    }
    std::remove(output_ply.c_str());

    printf("[Poisson] Raw mesh: %d vertices, %d faces\n",
           mesh.num_vertices(), mesh.num_faces());

    // 4. Density trimming
    trimLowDensityFaces(mesh, cloud, 0.10f);

    // 5. Color transfer
    transferColors(mesh, cloud);

    return mesh;
}
