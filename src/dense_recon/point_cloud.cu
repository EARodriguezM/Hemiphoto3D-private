#include "dense_recon/point_cloud.h"
#include "utils/cuda_utils.cuh"
#include "utils/math_utils.h"

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Dense>

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

// ============================================================================
// GPU Hash Table for Voxel Downsampling (open-addressing, linear probing)
// ============================================================================

static constexpr int HASH_TABLE_MAX_PROBES = 128;

// Morton code for 3D coordinates (interleave bits of x, y, z)
__device__ __forceinline__ uint64_t expandBits(uint32_t v) {
    uint64_t x = v & 0x1FFFFF; // 21 bits
    x = (x | (x << 32)) & 0x1F00000000FFFF;
    x = (x | (x << 16)) & 0x1F0000FF0000FF;
    x = (x | (x << 8))  & 0x100F00F00F00F00F;
    x = (x | (x << 4))  & 0x10C30C30C30C30C3;
    x = (x | (x << 2))  & 0x1249249249249249;
    return x;
}

__device__ __forceinline__ uint64_t mortonCode3D(int x, int y, int z) {
    return expandBits((uint32_t)x) | (expandBits((uint32_t)y) << 1) | (expandBits((uint32_t)z) << 2);
}

// Hash function for the voxel hash table
__device__ __forceinline__ uint32_t hashMorton(uint64_t morton, uint32_t table_size) {
    // Murmur-like hash
    morton ^= morton >> 33;
    morton *= 0xFF51AFD7ED558CCD;
    morton ^= morton >> 33;
    morton *= 0xC4CEB9FE1A85EC53;
    morton ^= morton >> 33;
    return (uint32_t)(morton % table_size);
}

// Voxel hash table entry
struct VoxelEntry {
    uint64_t key;       // morton code (UINT64_MAX = empty)
    float pos_x, pos_y, pos_z;
    float nor_x, nor_y, nor_z;
    float col_r, col_g, col_b;
    int count;
};

// ============================================================================
// Kernel 1: depth_to_points_kernel
// ============================================================================

__global__ void depth_to_points_kernel(
    const float* __restrict__ depth,        // H x W
    const float* __restrict__ normals,      // H x W x 3 (interleaved)
    const float* __restrict__ confidence,   // H x W
    const unsigned char* __restrict__ image, // H x W x 3 (BGR)
    int width, int height,
    Matrix3x3 K_inv,
    Matrix3x3 R_T,   // R^T (transpose of world-to-camera R)
    float3 t_vec,     // translation vector t
    float conf_thresh,
    // Output
    float3* __restrict__ out_positions,
    float3* __restrict__ out_normals,
    float3* __restrict__ out_colors,
    int* __restrict__ out_count,
    int max_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;

    int x = idx % width;
    int y = idx / width;

    float d = depth[idx];
    float conf = confidence[idx];

    if (d <= 0.0f || conf < conf_thresh) return;

    // 1. Backproject to camera frame: P_cam = depth * K_inv * [x, y, 1]^T
    float3 pixel = make_float3((float)x, (float)y, 1.0f);
    float3 p_cam = K_inv * pixel;
    p_cam = p_cam * d;

    // 2. Transform to world frame: P_world = R^T * (P_cam - t)
    float3 diff = p_cam - t_vec;
    float3 p_world = R_T * diff;

    // 3. Transform normal to world frame: N_world = R^T * N_cam
    float3 n_cam = make_float3(
        normals[idx * 3 + 0],
        normals[idx * 3 + 1],
        normals[idx * 3 + 2]
    );
    float3 n_world = R_T * n_cam;
    n_world = normalize(n_world);

    // 4. Sample color from BGR image
    int pixel_idx = (y * width + x) * 3;
    float b = (float)image[pixel_idx + 0] / 255.0f;
    float g = (float)image[pixel_idx + 1] / 255.0f;
    float r = (float)image[pixel_idx + 2] / 255.0f;
    float3 color = make_float3(r, g, b);

    // 5. Write to output using atomicAdd on global counter
    int out_idx = atomicAdd(out_count, 1);
    if (out_idx < max_points) {
        out_positions[out_idx] = p_world;
        out_normals[out_idx] = n_world;
        out_colors[out_idx] = color;
    }
}

// ============================================================================
// Kernel 2: voxel_downsample_kernel (Option A)
// ============================================================================

__global__ void voxel_insert_kernel(
    const float3* __restrict__ positions,
    const float3* __restrict__ normals,
    const float3* __restrict__ colors,
    int num_points,
    float voxel_size,
    float3 origin,   // minimum corner of bounding box
    VoxelEntry* __restrict__ table,
    uint32_t table_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    float3 p = positions[idx];
    float3 n = normals[idx];
    float3 c = colors[idx];

    // Compute voxel coordinates
    int vx = (int)floorf((p.x - origin.x) / voxel_size);
    int vy = (int)floorf((p.y - origin.y) / voxel_size);
    int vz = (int)floorf((p.z - origin.z) / voxel_size);

    uint64_t morton = mortonCode3D(vx, vy, vz);
    uint32_t slot = hashMorton(morton, table_size);

    // Open-addressing linear probing insert
    for (int probe = 0; probe < HASH_TABLE_MAX_PROBES; probe++) {
        uint32_t s = (slot + probe) % table_size;

        // Try to claim this slot
        unsigned long long expected = 0xFFFFFFFFFFFFFFFF;
        unsigned long long desired = morton;
        unsigned long long old = atomicCAS(
            (unsigned long long*)&table[s].key,
            expected, desired
        );

        if (old == expected || old == desired) {
            // We own this slot or it's our key
            atomicAdd(&table[s].pos_x, p.x);
            atomicAdd(&table[s].pos_y, p.y);
            atomicAdd(&table[s].pos_z, p.z);
            atomicAdd(&table[s].nor_x, n.x);
            atomicAdd(&table[s].nor_y, n.y);
            atomicAdd(&table[s].nor_z, n.z);
            atomicAdd(&table[s].col_r, c.x);
            atomicAdd(&table[s].col_g, c.y);
            atomicAdd(&table[s].col_b, c.z);
            atomicAdd(&table[s].count, 1);
            return;
        }

        // Slot occupied by different key — probe next
        if ((uint64_t)old != morton) continue;
        // Should not get here, but just in case
        break;
    }
}

__global__ void voxel_extract_kernel(
    const VoxelEntry* __restrict__ table,
    uint32_t table_size,
    float3* __restrict__ out_positions,
    float3* __restrict__ out_normals,
    float3* __restrict__ out_colors,
    int* __restrict__ out_count,
    int max_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int)table_size) return;

    const VoxelEntry& entry = table[idx];
    if (entry.count <= 0) return;

    float inv = 1.0f / (float)entry.count;
    float3 pos = make_float3(entry.pos_x * inv, entry.pos_y * inv, entry.pos_z * inv);
    float3 nor = normalize(make_float3(entry.nor_x * inv, entry.nor_y * inv, entry.nor_z * inv));
    float3 col = make_float3(entry.col_r * inv, entry.col_g * inv, entry.col_b * inv);

    int out_idx = atomicAdd(out_count, 1);
    if (out_idx < max_points) {
        out_positions[out_idx] = pos;
        out_normals[out_idx] = nor;
        out_colors[out_idx] = col;
    }
}

// ============================================================================
// TSDF Volume (Option B) — Voxel Hashing with 8x8x8 blocks
// ============================================================================

static constexpr int TSDF_BLOCK_SIZE = 8;
static constexpr int TSDF_BLOCK_VOXELS = TSDF_BLOCK_SIZE * TSDF_BLOCK_SIZE * TSDF_BLOCK_SIZE;

struct TSDFBlock {
    int3 block_pos;     // block coordinate (in block units)
    float tsdf[TSDF_BLOCK_VOXELS];
    float weight[TSDF_BLOCK_VOXELS];
    float3 color[TSDF_BLOCK_VOXELS];
};

struct TSDFHashEntry {
    int3 block_pos;
    int block_idx;      // index into block array (-1 = empty)
};

__device__ __forceinline__ uint32_t hashBlockPos(int3 pos, uint32_t table_size) {
    uint32_t h = (uint32_t)(pos.x * 73856093 ^ pos.y * 19349663 ^ pos.z * 83492791);
    return h % table_size;
}

__global__ void tsdf_allocate_blocks_kernel(
    const float* __restrict__ depth,
    int width, int height,
    Matrix3x3 K_inv,
    Matrix3x3 R_T,
    float3 t_vec,
    float voxel_size,
    float truncation,
    float3 volume_origin,
    TSDFHashEntry* __restrict__ hash_table,
    uint32_t hash_table_size,
    int* __restrict__ block_alloc_counter,
    int max_blocks
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;

    int x = idx % width;
    int y = idx / width;
    float d = depth[idx];
    if (d <= 0.0f) return;

    // Backproject to world
    float3 pixel = make_float3((float)x, (float)y, 1.0f);
    float3 p_cam = K_inv * pixel;
    p_cam = p_cam * d;
    float3 p_world = R_T * (p_cam - t_vec);

    // Compute block coordinates for the point and its neighborhood (within truncation)
    float block_world_size = voxel_size * TSDF_BLOCK_SIZE;
    for (float dz = -truncation; dz <= truncation; dz += block_world_size) {
        float3 sample = make_float3(p_world.x, p_world.y, p_world.z + dz);
        int bx = (int)floorf((sample.x - volume_origin.x) / block_world_size);
        int by = (int)floorf((sample.y - volume_origin.y) / block_world_size);
        int bz = (int)floorf((sample.z - volume_origin.z) / block_world_size);

        int3 bp = make_int3(bx, by, bz);
        uint32_t slot = hashBlockPos(bp, hash_table_size);

        for (int probe = 0; probe < 64; probe++) {
            uint32_t s = (slot + probe) % hash_table_size;
            int old_idx = atomicCAS(&hash_table[s].block_idx, -1, -2); // claim slot

            if (old_idx == -1) {
                // New slot — allocate block
                int bidx = atomicAdd(block_alloc_counter, 1);
                if (bidx < max_blocks) {
                    hash_table[s].block_pos = bp;
                    hash_table[s].block_idx = bidx;
                }
                break;
            } else if (old_idx >= 0 &&
                       hash_table[s].block_pos.x == bp.x &&
                       hash_table[s].block_pos.y == bp.y &&
                       hash_table[s].block_pos.z == bp.z) {
                break; // already allocated
            }
            // collision — probe
        }
    }
}

__global__ void tsdf_integrate_kernel(
    const float* __restrict__ depth,
    const unsigned char* __restrict__ image,
    int width, int height,
    Matrix3x3 K,       // intrinsics
    Matrix3x3 R,       // world-to-camera rotation
    float3 t_vec,      // world-to-camera translation
    float voxel_size,
    float truncation,
    float3 volume_origin,
    TSDFHashEntry* __restrict__ hash_table,
    uint32_t hash_table_size,
    TSDFBlock* __restrict__ blocks,
    int num_blocks
) {
    // Each thread processes one voxel within one block
    int block_linear = blockIdx.x;
    if (block_linear >= num_blocks) return;

    TSDFBlock& blk = blocks[block_linear];
    int voxel_in_block = threadIdx.x;
    if (voxel_in_block >= TSDF_BLOCK_VOXELS) return;

    // Voxel local coords within block
    int lz = voxel_in_block / (TSDF_BLOCK_SIZE * TSDF_BLOCK_SIZE);
    int ly = (voxel_in_block / TSDF_BLOCK_SIZE) % TSDF_BLOCK_SIZE;
    int lx = voxel_in_block % TSDF_BLOCK_SIZE;

    // World position of this voxel center
    float block_world_size = voxel_size * TSDF_BLOCK_SIZE;
    float3 world_pos = make_float3(
        volume_origin.x + blk.block_pos.x * block_world_size + (lx + 0.5f) * voxel_size,
        volume_origin.y + blk.block_pos.y * block_world_size + (ly + 0.5f) * voxel_size,
        volume_origin.z + blk.block_pos.z * block_world_size + (lz + 0.5f) * voxel_size
    );

    // Project to camera: P_cam = R * P_world + t
    float3 p_cam = R * world_pos + t_vec;
    if (p_cam.z <= 0.0f) return;

    // Project to image
    float3 proj = K * p_cam;
    float u = proj.x / proj.z;
    float v = proj.y / proj.z;

    int px = (int)roundf(u);
    int py = (int)roundf(v);
    if (px < 0 || px >= width || py < 0 || py >= height) return;

    float depth_measured = depth[py * width + px];
    if (depth_measured <= 0.0f) return;

    float sdf = depth_measured - p_cam.z;

    if (sdf < -truncation) return;
    float tsdf_val = fminf(sdf / truncation, 1.0f);

    // Weighted running average
    float old_tsdf = blk.tsdf[voxel_in_block];
    float old_w = blk.weight[voxel_in_block];
    float new_w = old_w + 1.0f;
    blk.tsdf[voxel_in_block] = (old_tsdf * old_w + tsdf_val) / new_w;
    blk.weight[voxel_in_block] = fminf(new_w, 128.0f); // cap weight

    // Update color
    int pix_idx = (py * width + px) * 3;
    float3 c = make_float3(
        (float)image[pix_idx + 2] / 255.0f,  // R from BGR
        (float)image[pix_idx + 1] / 255.0f,  // G
        (float)image[pix_idx + 0] / 255.0f   // B
    );
    blk.color[voxel_in_block] = make_float3(
        (blk.color[voxel_in_block].x * old_w + c.x) / new_w,
        (blk.color[voxel_in_block].y * old_w + c.y) / new_w,
        (blk.color[voxel_in_block].z * old_w + c.z) / new_w
    );
}

__global__ void tsdf_extract_points_kernel(
    const TSDFBlock* __restrict__ blocks,
    int num_blocks,
    float voxel_size,
    float3 volume_origin,
    float3* __restrict__ out_positions,
    float3* __restrict__ out_normals,
    float3* __restrict__ out_colors,
    int* __restrict__ out_count,
    int max_points
) {
    int block_linear = blockIdx.x;
    if (block_linear >= num_blocks) return;

    const TSDFBlock& blk = blocks[block_linear];
    int voxel_in_block = threadIdx.x;
    if (voxel_in_block >= TSDF_BLOCK_VOXELS) return;

    float tsdf_val = blk.tsdf[voxel_in_block];
    float w = blk.weight[voxel_in_block];

    // Extract points near zero-crossing with sufficient weight
    if (w < 1.0f || fabsf(tsdf_val) > 0.5f) return;

    int lz = voxel_in_block / (TSDF_BLOCK_SIZE * TSDF_BLOCK_SIZE);
    int ly = (voxel_in_block / TSDF_BLOCK_SIZE) % TSDF_BLOCK_SIZE;
    int lx = voxel_in_block % TSDF_BLOCK_SIZE;

    float block_world_size = voxel_size * TSDF_BLOCK_SIZE;
    float3 pos = make_float3(
        volume_origin.x + blk.block_pos.x * block_world_size + (lx + 0.5f) * voxel_size,
        volume_origin.y + blk.block_pos.y * block_world_size + (ly + 0.5f) * voxel_size,
        volume_origin.z + blk.block_pos.z * block_world_size + (lz + 0.5f) * voxel_size
    );

    // Compute normal from TSDF gradient
    float3 normal = make_float3(0, 0, 1);  // fallback
    // Simple central differences within the block
    if (lx > 0 && lx < TSDF_BLOCK_SIZE - 1 &&
        ly > 0 && ly < TSDF_BLOCK_SIZE - 1 &&
        lz > 0 && lz < TSDF_BLOCK_SIZE - 1) {
        int idx_xm = lz * TSDF_BLOCK_SIZE * TSDF_BLOCK_SIZE + ly * TSDF_BLOCK_SIZE + (lx - 1);
        int idx_xp = lz * TSDF_BLOCK_SIZE * TSDF_BLOCK_SIZE + ly * TSDF_BLOCK_SIZE + (lx + 1);
        int idx_ym = lz * TSDF_BLOCK_SIZE * TSDF_BLOCK_SIZE + (ly - 1) * TSDF_BLOCK_SIZE + lx;
        int idx_yp = lz * TSDF_BLOCK_SIZE * TSDF_BLOCK_SIZE + (ly + 1) * TSDF_BLOCK_SIZE + lx;
        int idx_zm = (lz - 1) * TSDF_BLOCK_SIZE * TSDF_BLOCK_SIZE + ly * TSDF_BLOCK_SIZE + lx;
        int idx_zp = (lz + 1) * TSDF_BLOCK_SIZE * TSDF_BLOCK_SIZE + ly * TSDF_BLOCK_SIZE + lx;
        normal = make_float3(
            blk.tsdf[idx_xp] - blk.tsdf[idx_xm],
            blk.tsdf[idx_yp] - blk.tsdf[idx_ym],
            blk.tsdf[idx_zp] - blk.tsdf[idx_zm]
        );
        float len = length(normal);
        if (len > 1e-6f) normal = normal * (1.0f / len);
    }

    int out_idx = atomicAdd(out_count, 1);
    if (out_idx < max_points) {
        out_positions[out_idx] = pos;
        out_normals[out_idx] = normal;
        out_colors[out_idx] = blk.color[voxel_in_block];
    }
}

// ============================================================================
// Kernel 3: statistical_outlier_removal_kernel
// ============================================================================

__global__ void count_grid_neighbors_kernel(
    const float3* __restrict__ positions,
    int num_points,
    float cell_size,
    float3 origin,
    int grid_x, int grid_y, int grid_z,
    int* __restrict__ grid_counts  // grid_x * grid_y * grid_z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    float3 p = positions[idx];
    int cx = (int)floorf((p.x - origin.x) / cell_size);
    int cy = (int)floorf((p.y - origin.y) / cell_size);
    int cz = (int)floorf((p.z - origin.z) / cell_size);

    if (cx >= 0 && cx < grid_x && cy >= 0 && cy < grid_y && cz >= 0 && cz < grid_z) {
        atomicAdd(&grid_counts[cz * grid_y * grid_x + cy * grid_x + cx], 1);
    }
}

__global__ void outlier_filter_kernel(
    const float3* __restrict__ positions,
    const float3* __restrict__ normals,
    const float3* __restrict__ colors,
    int num_points,
    float cell_size,
    float3 origin,
    int grid_x, int grid_y, int grid_z,
    const int* __restrict__ grid_counts,
    int min_neighbors,
    float3* __restrict__ out_positions,
    float3* __restrict__ out_normals,
    float3* __restrict__ out_colors,
    int* __restrict__ out_count,
    int max_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    float3 p = positions[idx];
    int cx = (int)floorf((p.x - origin.x) / cell_size);
    int cy = (int)floorf((p.y - origin.y) / cell_size);
    int cz = (int)floorf((p.z - origin.z) / cell_size);

    // Count neighbors in 3x3x3 neighborhood
    int neighbor_count = 0;
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = cx + dx, ny = cy + dy, nz = cz + dz;
                if (nx >= 0 && nx < grid_x && ny >= 0 && ny < grid_y && nz >= 0 && nz < grid_z) {
                    neighbor_count += grid_counts[nz * grid_y * grid_x + ny * grid_x + nx];
                }
            }
        }
    }

    // Subtract self
    neighbor_count -= 1;

    if (neighbor_count >= min_neighbors) {
        int out_idx = atomicAdd(out_count, 1);
        if (out_idx < max_points) {
            out_positions[out_idx] = positions[idx];
            out_normals[out_idx] = normals[idx];
            out_colors[out_idx] = colors[idx];
        }
    }
}

// ============================================================================
// Kernel 4: refine_normals_kernel (PCA + camera orientation)
// ============================================================================

// Closed-form 3x3 symmetric eigendecomposition (Cardano's method)
// Returns the eigenvector corresponding to the smallest eigenvalue
__device__ float3 smallestEigenvector3x3(
    float a00, float a01, float a02,
    float a11, float a12, float a22
) {
    // Characteristic polynomial: lambda^3 + c2*lambda^2 + c1*lambda + c0 = 0
    // For symmetric matrix, trace = sum of eigenvalues, etc.
    float tr = a00 + a11 + a22;
    float q = tr / 3.0f;

    float s00 = a00 - q, s11 = a11 - q, s22 = a22 - q;
    float p = sqrtf(fmaxf(
        (s00 * s00 + s11 * s11 + s22 * s22 +
         2.0f * (a01 * a01 + a02 * a02 + a12 * a12)) / 6.0f,
        0.0f
    ));

    if (p < 1e-12f) {
        return make_float3(0.0f, 0.0f, 1.0f); // degenerate
    }

    float inv_p = 1.0f / p;
    float b00 = s00 * inv_p, b01 = a01 * inv_p, b02 = a02 * inv_p;
    float b11 = s11 * inv_p, b12 = a12 * inv_p;
    float b22 = s22 * inv_p;

    float det_b = b00 * (b11 * b22 - b12 * b12)
                - b01 * (b01 * b22 - b12 * b02)
                + b02 * (b01 * b12 - b11 * b02);
    float half_det = det_b * 0.5f;
    half_det = fminf(fmaxf(half_det, -1.0f), 1.0f);

    float phi = acosf(half_det) / 3.0f;

    // Smallest eigenvalue
    float e2 = q + 2.0f * p * cosf(phi + 2.0f * 3.14159265f / 3.0f);

    // Eigenvector for smallest eigenvalue e2:
    // (A - e2*I) * v = 0 — find null space via cross product of two rows
    float r00 = a00 - e2, r01 = a01,       r02 = a02;
    float r10 = a01,       r11 = a11 - e2, r12 = a12;
    float r22 = a22 - e2;

    float3 row0 = make_float3(r00, r01, r02);
    float3 row1 = make_float3(r10, r11, r12);
    float3 row2 = make_float3(a02, r12, r22);

    float3 c01 = cross(row0, row1);
    float3 c02 = cross(row0, row2);
    float3 c12 = cross(row1, row2);

    float len01 = dot(c01, c01);
    float len02 = dot(c02, c02);
    float len12 = dot(c12, c12);

    float3 result;
    if (len01 >= len02 && len01 >= len12) {
        result = c01;
    } else if (len02 >= len12) {
        result = c02;
    } else {
        result = c12;
    }

    float len = length(result);
    if (len < 1e-12f) return make_float3(0.0f, 0.0f, 1.0f);
    return result * (1.0f / len);
}

__global__ void refine_normals_kernel(
    const float3* __restrict__ positions,
    float3* __restrict__ normals,
    int num_points,
    float cell_size,
    float3 origin,
    int grid_x, int grid_y, int grid_z,
    const int* __restrict__ grid_counts,
    // Point-to-grid mapping: sorted point indices
    const int* __restrict__ grid_start,   // start index for each cell
    const int* __restrict__ sorted_indices,
    // Camera centers for orientation
    const float3* __restrict__ cam_centers,
    int num_cameras,
    int min_neighbors_pca
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    float3 p = positions[idx];
    int cx = (int)floorf((p.x - origin.x) / cell_size);
    int cy = (int)floorf((p.y - origin.y) / cell_size);
    int cz = (int)floorf((p.z - origin.z) / cell_size);

    // Gather neighbors from 3x3x3 cells
    float3 centroid = make_float3(0, 0, 0);
    int neighbor_count = 0;

    // First pass: compute centroid
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = cx + dx, ny = cy + dy, nz = cz + dz;
                if (nx < 0 || nx >= grid_x || ny < 0 || ny >= grid_y || nz < 0 || nz >= grid_z)
                    continue;
                int cell = nz * grid_y * grid_x + ny * grid_x + nx;
                int start = grid_start[cell];
                int count = grid_counts[cell];
                for (int k = 0; k < count && k < 64; k++) {
                    int pidx = sorted_indices[start + k];
                    centroid += positions[pidx];
                    neighbor_count++;
                }
            }
        }
    }

    if (neighbor_count < min_neighbors_pca) return; // keep existing normal

    float inv_n = 1.0f / (float)neighbor_count;
    centroid = centroid * inv_n;

    // Second pass: compute covariance matrix
    float c00 = 0, c01 = 0, c02 = 0, c11 = 0, c12 = 0, c22 = 0;
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = cx + dx, ny = cy + dy, nz = cz + dz;
                if (nx < 0 || nx >= grid_x || ny < 0 || ny >= grid_y || nz < 0 || nz >= grid_z)
                    continue;
                int cell = nz * grid_y * grid_x + ny * grid_x + nx;
                int start = grid_start[cell];
                int count = grid_counts[cell];
                for (int k = 0; k < count && k < 64; k++) {
                    int pidx = sorted_indices[start + k];
                    float3 d = positions[pidx] - centroid;
                    c00 += d.x * d.x;
                    c01 += d.x * d.y;
                    c02 += d.x * d.z;
                    c11 += d.y * d.y;
                    c12 += d.y * d.z;
                    c22 += d.z * d.z;
                }
            }
        }
    }

    c00 *= inv_n; c01 *= inv_n; c02 *= inv_n;
    c11 *= inv_n; c12 *= inv_n; c22 *= inv_n;

    float3 normal = smallestEigenvector3x3(c00, c01, c02, c11, c12, c22);

    // Orient toward nearest camera center
    float best_dist2 = 1e30f;
    float3 best_cam = make_float3(0, 0, 0);
    for (int i = 0; i < num_cameras; i++) {
        float3 diff = cam_centers[i] - p;
        float d2 = dot(diff, diff);
        if (d2 < best_dist2) {
            best_dist2 = d2;
            best_cam = cam_centers[i];
        }
    }

    float3 to_cam = best_cam - p;
    if (dot(normal, to_cam) < 0.0f) {
        normal = -normal;
    }

    normals[idx] = normal;
}

// ============================================================================
// Host helper: build uniform grid for point cloud
// ============================================================================

struct UniformGrid {
    DeviceBuffer<int> counts;
    DeviceBuffer<int> starts;
    DeviceBuffer<int> sorted_indices;
    float3 origin;
    int grid_x, grid_y, grid_z;
    float cell_size;
};

static void buildUniformGrid(
    const float3* d_positions,
    int num_points,
    float cell_size,
    float3 bbox_min, float3 bbox_max,
    UniformGrid& grid
) {
    grid.cell_size = cell_size;
    grid.origin = bbox_min;
    grid.grid_x = std::max(1, (int)ceilf((bbox_max.x - bbox_min.x) / cell_size) + 1);
    grid.grid_y = std::max(1, (int)ceilf((bbox_max.y - bbox_min.y) / cell_size) + 1);
    grid.grid_z = std::max(1, (int)ceilf((bbox_max.z - bbox_min.z) / cell_size) + 1);

    int total_cells = grid.grid_x * grid.grid_y * grid.grid_z;

    // Count points per cell
    grid.counts.alloc(total_cells);
    CUDA_CHECK(cudaMemset(grid.counts.ptr, 0, total_cells * sizeof(int)));

    int threads = 256;
    int blocks = (num_points + threads - 1) / threads;
    count_grid_neighbors_kernel<<<blocks, threads>>>(
        d_positions, num_points, cell_size, grid.origin,
        grid.grid_x, grid.grid_y, grid.grid_z, grid.counts.ptr
    );
    CUDA_CHECK_KERNEL();

    // Prefix sum for starts (CPU for simplicity)
    std::vector<int> h_counts(total_cells);
    grid.counts.download(h_counts.data(), total_cells);

    std::vector<int> h_starts(total_cells);
    int running = 0;
    for (int i = 0; i < total_cells; i++) {
        h_starts[i] = running;
        running += h_counts[i];
    }

    grid.starts.alloc(total_cells);
    grid.starts.upload(h_starts.data(), total_cells);

    // Sort points into grid cells (CPU)
    std::vector<float3> h_positions(num_points);
    CUDA_CHECK(cudaMemcpy(h_positions.data(), d_positions, num_points * sizeof(float3), cudaMemcpyDeviceToHost));

    std::vector<int> h_sorted(num_points);
    std::vector<int> h_offsets(total_cells, 0); // per-cell write offset

    for (int i = 0; i < num_points; i++) {
        float3 p = h_positions[i];
        int cx = (int)floorf((p.x - grid.origin.x) / cell_size);
        int cy = (int)floorf((p.y - grid.origin.y) / cell_size);
        int cz = (int)floorf((p.z - grid.origin.z) / cell_size);

        cx = std::max(0, std::min(cx, grid.grid_x - 1));
        cy = std::max(0, std::min(cy, grid.grid_y - 1));
        cz = std::max(0, std::min(cz, grid.grid_z - 1));

        int cell = cz * grid.grid_y * grid.grid_x + cy * grid.grid_x + cx;
        h_sorted[h_starts[cell] + h_offsets[cell]] = i;
        h_offsets[cell]++;
    }

    grid.sorted_indices.alloc(num_points);
    grid.sorted_indices.upload(h_sorted.data(), num_points);
}

// ============================================================================
// Host helper: compute bounding box on CPU
// ============================================================================

static void computeBBox(const std::vector<float3>& pts, float3& bbox_min, float3& bbox_max) {
    bbox_min = make_float3(1e30f, 1e30f, 1e30f);
    bbox_max = make_float3(-1e30f, -1e30f, -1e30f);
    for (const auto& p : pts) {
        bbox_min.x = std::min(bbox_min.x, p.x);
        bbox_min.y = std::min(bbox_min.y, p.y);
        bbox_min.z = std::min(bbox_min.z, p.z);
        bbox_max.x = std::max(bbox_max.x, p.x);
        bbox_max.y = std::max(bbox_max.y, p.y);
        bbox_max.z = std::max(bbox_max.z, p.z);
    }
}

// ============================================================================
// Helper: convert Eigen matrix to Matrix3x3
// ============================================================================

static Matrix3x3 eigenToMatrix3x3(const Eigen::Matrix3d& M) {
    Matrix3x3 result;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            result(r, c) = (float)M(r, c);
    return result;
}

// ============================================================================
// Main API: fuseDepthMaps
// ============================================================================

DensePointCloud fuseDepthMaps(
    const std::vector<DepthMap>& depth_maps,
    const std::vector<CameraPose>& cameras,
    const std::vector<ImageData>& images,
    const PipelineConfig& config
) {
    if (depth_maps.empty()) {
        printf("[PointCloud] No depth maps to fuse.\n");
        return DensePointCloud();
    }

    // Check for checkpoint
    if (config.save_intermediate && !config.output_path.empty()) {
        std::string dir = fs::path(config.output_path).parent_path().string();
        std::string ply_path = dir + "/dense.ply";
        if (fs::exists(ply_path)) {
            DensePointCloud cloud;
            if (loadDensePointCloudPLY(cloud, ply_path)) {
                printf("[PointCloud] Loaded checkpoint: %s (%d points)\n",
                       ply_path.c_str(), cloud.size());
                return cloud;
            }
        }
    }

    float conf_thresh = 0.5f;
    float voxel_size = config.voxel_resolution;

    // ========================================================================
    // Phase 1: Backproject all depth maps to raw 3D points
    // ========================================================================

    printf("[PointCloud] Phase 1: Backprojecting %d depth maps...\n",
           (int)depth_maps.size());

    // Estimate max points (all valid pixels across all depth maps)
    size_t max_raw_points = 0;
    for (const auto& dm : depth_maps) {
        max_raw_points += (size_t)dm.width * dm.height;
    }

    DeviceBuffer<float3> d_raw_positions, d_raw_normals, d_raw_colors;
    DeviceBuffer<int> d_raw_count;
    d_raw_positions.alloc(max_raw_points);
    d_raw_normals.alloc(max_raw_points);
    d_raw_colors.alloc(max_raw_points);
    d_raw_count.alloc(1);
    int zero = 0;
    d_raw_count.upload(&zero, 1);

    for (const auto& dm : depth_maps) {
        int img_id = dm.image_id;
        if (img_id < 0 || img_id >= (int)cameras.size()) continue;
        if (!cameras[img_id].is_registered) continue;

        const auto& cam = cameras[img_id];
        int W = dm.width, H = dm.height;
        int total_pixels = W * H;

        // Prepare K_inv and R^T
        Matrix3x3 K_inv = eigenToMatrix3x3(cam.K.inverse());
        Matrix3x3 R_T = eigenToMatrix3x3(cam.R.transpose());
        float3 t_vec = make_float3((float)cam.t.x(), (float)cam.t.y(), (float)cam.t.z());

        // Upload depth, normal, confidence, image
        DeviceBuffer<float> d_depth, d_confidence;
        DeviceBuffer<float> d_normals_map;
        DeviceBuffer<unsigned char> d_image;

        d_depth.alloc(total_pixels);
        d_confidence.alloc(total_pixels);
        d_normals_map.alloc(total_pixels * 3);

        // depth and confidence are float CV mats
        d_depth.upload((const float*)dm.depth.data, total_pixels);
        d_confidence.upload((const float*)dm.confidence.data, total_pixels);

        // Normal map: convert from Vec3f (interleaved) to flat float array
        // OpenCV stores Vec3f as [n0.x, n0.y, n0.z, n1.x, ...] which is what we need
        d_normals_map.upload((const float*)dm.normal.data, total_pixels * 3);

        // Image: find matching image
        const cv::Mat* img_ptr = nullptr;
        for (const auto& img : images) {
            if (img.id == img_id) {
                img_ptr = &img.image;
                break;
            }
        }
        if (!img_ptr || img_ptr->empty()) continue;

        // Make sure image is contiguous BGR uint8
        cv::Mat bgr;
        if (img_ptr->isContinuous() && img_ptr->type() == CV_8UC3 &&
            img_ptr->cols == W && img_ptr->rows == H) {
            bgr = *img_ptr;
        } else {
            cv::resize(*img_ptr, bgr, cv::Size(W, H));
        }

        d_image.alloc(W * H * 3);
        d_image.upload(bgr.data, W * H * 3);

        int threads = 256;
        int blocks = (total_pixels + threads - 1) / threads;
        depth_to_points_kernel<<<blocks, threads>>>(
            d_depth.ptr, d_normals_map.ptr, d_confidence.ptr, d_image.ptr,
            W, H, K_inv, R_T, t_vec, conf_thresh,
            d_raw_positions.ptr, d_raw_normals.ptr, d_raw_colors.ptr,
            d_raw_count.ptr, (int)max_raw_points
        );
        CUDA_CHECK_KERNEL();
    }

    int raw_count = 0;
    d_raw_count.download(&raw_count, 1);
    printf("[PointCloud] Raw points: %d\n", raw_count);

    if (raw_count == 0) {
        printf("[PointCloud] No raw points generated!\n");
        return DensePointCloud();
    }

    // Download raw points for bbox computation
    std::vector<float3> h_raw_pos(raw_count);
    CUDA_CHECK(cudaMemcpy(h_raw_pos.data(), d_raw_positions.ptr,
                           raw_count * sizeof(float3), cudaMemcpyDeviceToHost));

    float3 bbox_min, bbox_max;
    computeBBox(h_raw_pos, bbox_min, bbox_max);

    // ========================================================================
    // Phase 2: Fusion — Voxel Grid Downsampling (Option A) or TSDF (Option B)
    // ========================================================================

    DeviceBuffer<float3> d_fused_positions, d_fused_normals, d_fused_colors;
    DeviceBuffer<int> d_fused_count;
    int fused_count = 0;

    bool use_tsdf = config.use_tsdf;

    // Check VRAM for TSDF
    if (use_tsdf) {
        size_t avail = getAvailableGPUMemory(config.gpu_id);
        if (avail < 8ULL * 1024 * 1024 * 1024) {
            printf("[PointCloud] Insufficient VRAM for TSDF (%.1f GB available, 8 GB required). "
                   "Falling back to voxel downsampling.\n", avail / (1024.0 * 1024.0 * 1024.0));
            use_tsdf = false;
        }
    }

    if (use_tsdf) {
        // ==== Option B: TSDF Volume ====
        printf("[PointCloud] Phase 2: TSDF volume fusion...\n");

        float truncation = config.tsdf_truncation_mult * voxel_size;
        float3 margin = make_float3(
            (bbox_max.x - bbox_min.x) * 0.1f,
            (bbox_max.y - bbox_min.y) * 0.1f,
            (bbox_max.z - bbox_min.z) * 0.1f
        );
        float3 vol_origin = make_float3(
            bbox_min.x - margin.x,
            bbox_min.y - margin.y,
            bbox_min.z - margin.z
        );

        // Allocate hash table and blocks
        uint32_t hash_table_size = 1 << 20; // 1M entries
        int max_blocks = 500000;

        DeviceBuffer<TSDFHashEntry> d_hash_table;
        d_hash_table.alloc(hash_table_size);
        // Init entries to -1
        std::vector<TSDFHashEntry> h_hash_init(hash_table_size);
        for (auto& e : h_hash_init) { e.block_idx = -1; e.block_pos = make_int3(0,0,0); }
        d_hash_table.upload(h_hash_init.data(), hash_table_size);

        DeviceBuffer<int> d_block_counter;
        d_block_counter.alloc(1);
        d_block_counter.upload(&zero, 1);

        DeviceBuffer<TSDFBlock> d_blocks;
        d_blocks.alloc(max_blocks);
        CUDA_CHECK(cudaMemset(d_blocks.ptr, 0, max_blocks * sizeof(TSDFBlock)));

        // For each depth map: allocate blocks and integrate
        for (const auto& dm : depth_maps) {
            int img_id = dm.image_id;
            if (img_id < 0 || img_id >= (int)cameras.size()) continue;
            if (!cameras[img_id].is_registered) continue;

            const auto& cam = cameras[img_id];
            int W = dm.width, H = dm.height;
            int total_pixels = W * H;

            Matrix3x3 K_mat = eigenToMatrix3x3(cam.K);
            Matrix3x3 K_inv = eigenToMatrix3x3(cam.K.inverse());
            Matrix3x3 R_mat = eigenToMatrix3x3(cam.R);
            Matrix3x3 R_T = eigenToMatrix3x3(cam.R.transpose());
            float3 t_vec = make_float3((float)cam.t.x(), (float)cam.t.y(), (float)cam.t.z());

            DeviceBuffer<float> d_depth;
            d_depth.alloc(total_pixels);
            d_depth.upload((const float*)dm.depth.data, total_pixels);

            // Allocate blocks
            int threads = 256;
            int blocks_grid = (total_pixels + threads - 1) / threads;
            tsdf_allocate_blocks_kernel<<<blocks_grid, threads>>>(
                d_depth.ptr, W, H, K_inv, R_T, t_vec,
                voxel_size, truncation, vol_origin,
                d_hash_table.ptr, hash_table_size,
                d_block_counter.ptr, max_blocks
            );
            CUDA_CHECK_KERNEL();

            int num_blocks = 0;
            d_block_counter.download(&num_blocks, 1);
            num_blocks = std::min(num_blocks, max_blocks);

            if (num_blocks == 0) continue;

            // Copy block positions from hash table to block array
            // (blocks are allocated incrementally, positions are set in hash table)
            // We need to set block_pos in d_blocks from d_hash_table
            std::vector<TSDFHashEntry> h_hash(hash_table_size);
            d_hash_table.download(h_hash.data(), hash_table_size);
            std::vector<TSDFBlock> h_blocks(num_blocks);
            CUDA_CHECK(cudaMemcpy(h_blocks.data(), d_blocks.ptr,
                                   num_blocks * sizeof(TSDFBlock), cudaMemcpyDeviceToHost));
            for (const auto& entry : h_hash) {
                if (entry.block_idx >= 0 && entry.block_idx < num_blocks) {
                    h_blocks[entry.block_idx].block_pos = entry.block_pos;
                }
            }
            CUDA_CHECK(cudaMemcpy(d_blocks.ptr, h_blocks.data(),
                                   num_blocks * sizeof(TSDFBlock), cudaMemcpyHostToDevice));

            // Upload image
            const cv::Mat* img_ptr = nullptr;
            for (const auto& img : images) {
                if (img.id == img_id) { img_ptr = &img.image; break; }
            }
            if (!img_ptr) continue;

            cv::Mat bgr;
            if (img_ptr->cols == W && img_ptr->rows == H) bgr = *img_ptr;
            else cv::resize(*img_ptr, bgr, cv::Size(W, H));

            DeviceBuffer<unsigned char> d_image;
            d_image.alloc(W * H * 3);
            d_image.upload(bgr.data, W * H * 3);

            // Integrate
            tsdf_integrate_kernel<<<num_blocks, TSDF_BLOCK_VOXELS>>>(
                d_depth.ptr, d_image.ptr, W, H,
                K_mat, R_mat, t_vec,
                voxel_size, truncation, vol_origin,
                d_hash_table.ptr, hash_table_size,
                d_blocks.ptr, num_blocks
            );
            CUDA_CHECK_KERNEL();
        }

        // Extract points from TSDF
        int num_blocks = 0;
        d_block_counter.download(&num_blocks, 1);
        num_blocks = std::min(num_blocks, max_blocks);

        int max_tsdf_points = num_blocks * TSDF_BLOCK_VOXELS;
        d_fused_positions.alloc(max_tsdf_points);
        d_fused_normals.alloc(max_tsdf_points);
        d_fused_colors.alloc(max_tsdf_points);
        d_fused_count.alloc(1);
        d_fused_count.upload(&zero, 1);

        tsdf_extract_points_kernel<<<num_blocks, TSDF_BLOCK_VOXELS>>>(
            d_blocks.ptr, num_blocks, voxel_size, vol_origin,
            d_fused_positions.ptr, d_fused_normals.ptr, d_fused_colors.ptr,
            d_fused_count.ptr, max_tsdf_points
        );
        CUDA_CHECK_KERNEL();

        d_fused_count.download(&fused_count, 1);
        printf("[PointCloud] TSDF extracted %d points from %d blocks\n", fused_count, num_blocks);

    } else {
        // ==== Option A: Voxel Grid Downsampling ====
        printf("[PointCloud] Phase 2: Voxel grid downsampling (voxel_size=%.6f)...\n", voxel_size);

        // Hash table size: ~4x the expected number of voxels for low collision rate
        float3 extent = bbox_max - bbox_min;
        size_t est_voxels = (size_t)ceilf(extent.x / voxel_size) *
                            (size_t)ceilf(extent.y / voxel_size) *
                            (size_t)ceilf(extent.z / voxel_size);
        uint32_t table_size = (uint32_t)std::min((size_t)16 * 1024 * 1024,
                                                  std::max((size_t)raw_count * 2, est_voxels * 4));
        // Round up to power of 2 for better hash distribution
        uint32_t pot = 1;
        while (pot < table_size) pot <<= 1;
        table_size = pot;

        DeviceBuffer<VoxelEntry> d_voxel_table;
        d_voxel_table.alloc(table_size);
        // Init keys to empty (0xFFFFFFFFFFFFFFFF) and counts to 0
        // Use cudaMemset with 0xFF for keys (since empty = 0xFFFFFFFFFFFFFFFF)
        CUDA_CHECK(cudaMemset(d_voxel_table.ptr, 0xFF, table_size * sizeof(VoxelEntry)));
        // But we need counts = 0 and accumulators = 0. Re-initialize properly on CPU:
        // Actually, 0xFF for float is NaN and for int is -1. We need a proper init.
        {
            std::vector<VoxelEntry> h_table(table_size);
            for (auto& e : h_table) {
                e.key = 0xFFFFFFFFFFFFFFFF;
                e.pos_x = e.pos_y = e.pos_z = 0;
                e.nor_x = e.nor_y = e.nor_z = 0;
                e.col_r = e.col_g = e.col_b = 0;
                e.count = 0;
            }
            d_voxel_table.upload(h_table.data(), table_size);
        }

        int threads = 256;
        int blocks = (raw_count + threads - 1) / threads;
        voxel_insert_kernel<<<blocks, threads>>>(
            d_raw_positions.ptr, d_raw_normals.ptr, d_raw_colors.ptr,
            raw_count, voxel_size, bbox_min,
            d_voxel_table.ptr, table_size
        );
        CUDA_CHECK_KERNEL();

        // Extract voxels
        int max_fused = raw_count; // can't be more than raw
        d_fused_positions.alloc(max_fused);
        d_fused_normals.alloc(max_fused);
        d_fused_colors.alloc(max_fused);
        d_fused_count.alloc(1);
        d_fused_count.upload(&zero, 1);

        int extract_blocks = ((int)table_size + threads - 1) / threads;
        voxel_extract_kernel<<<extract_blocks, threads>>>(
            d_voxel_table.ptr, table_size,
            d_fused_positions.ptr, d_fused_normals.ptr, d_fused_colors.ptr,
            d_fused_count.ptr, max_fused
        );
        CUDA_CHECK_KERNEL();

        d_fused_count.download(&fused_count, 1);
        printf("[PointCloud] Voxel downsampled: %d → %d points\n", raw_count, fused_count);
    }

    // Free raw buffers
    d_raw_positions.free();
    d_raw_normals.free();
    d_raw_colors.free();
    d_raw_count.free();

    if (fused_count == 0) {
        printf("[PointCloud] No fused points!\n");
        return DensePointCloud();
    }

    // ========================================================================
    // Phase 3: Statistical Outlier Removal
    // ========================================================================

    printf("[PointCloud] Phase 3: Outlier removal (grid=%.4f, min_neighbors=%d)...\n",
           config.outlier_grid_size, config.outlier_min_neighbors);

    // Download fused points to compute bbox
    std::vector<float3> h_fused_pos(fused_count);
    CUDA_CHECK(cudaMemcpy(h_fused_pos.data(), d_fused_positions.ptr,
                           fused_count * sizeof(float3), cudaMemcpyDeviceToHost));

    computeBBox(h_fused_pos, bbox_min, bbox_max);

    float cell_size = config.outlier_grid_size;
    int grid_x = std::max(1, (int)ceilf((bbox_max.x - bbox_min.x) / cell_size) + 1);
    int grid_y = std::max(1, (int)ceilf((bbox_max.y - bbox_min.y) / cell_size) + 1);
    int grid_z = std::max(1, (int)ceilf((bbox_max.z - bbox_min.z) / cell_size) + 1);

    // Limit grid size to avoid excessive memory
    if ((size_t)grid_x * grid_y * grid_z > 100000000ULL) {
        // Increase cell size
        float scale = cbrtf((float)((size_t)grid_x * grid_y * grid_z) / 100000000.0f);
        cell_size *= scale;
        grid_x = std::max(1, (int)ceilf((bbox_max.x - bbox_min.x) / cell_size) + 1);
        grid_y = std::max(1, (int)ceilf((bbox_max.y - bbox_min.y) / cell_size) + 1);
        grid_z = std::max(1, (int)ceilf((bbox_max.z - bbox_min.z) / cell_size) + 1);
    }

    int total_cells = grid_x * grid_y * grid_z;

    DeviceBuffer<int> d_grid_counts;
    d_grid_counts.alloc(total_cells);
    CUDA_CHECK(cudaMemset(d_grid_counts.ptr, 0, total_cells * sizeof(int)));

    {
        int threads = 256;
        int blocks = (fused_count + threads - 1) / threads;
        count_grid_neighbors_kernel<<<blocks, threads>>>(
            d_fused_positions.ptr, fused_count, cell_size, bbox_min,
            grid_x, grid_y, grid_z, d_grid_counts.ptr
        );
        CUDA_CHECK_KERNEL();
    }

    DeviceBuffer<float3> d_clean_positions, d_clean_normals, d_clean_colors;
    DeviceBuffer<int> d_clean_count;
    d_clean_positions.alloc(fused_count);
    d_clean_normals.alloc(fused_count);
    d_clean_colors.alloc(fused_count);
    d_clean_count.alloc(1);
    d_clean_count.upload(&zero, 1);

    {
        int threads = 256;
        int blocks = (fused_count + threads - 1) / threads;
        outlier_filter_kernel<<<blocks, threads>>>(
            d_fused_positions.ptr, d_fused_normals.ptr, d_fused_colors.ptr,
            fused_count, cell_size, bbox_min, grid_x, grid_y, grid_z,
            d_grid_counts.ptr, config.outlier_min_neighbors,
            d_clean_positions.ptr, d_clean_normals.ptr, d_clean_colors.ptr,
            d_clean_count.ptr, fused_count
        );
        CUDA_CHECK_KERNEL();
    }

    int clean_count = 0;
    d_clean_count.download(&clean_count, 1);
    printf("[PointCloud] After outlier removal: %d → %d points (%.1f%% removed)\n",
           fused_count, clean_count,
           100.0f * (1.0f - (float)clean_count / (float)fused_count));

    // Free fused buffers
    d_fused_positions.free();
    d_fused_normals.free();
    d_fused_colors.free();
    d_fused_count.free();
    d_grid_counts.free();

    if (clean_count == 0) {
        printf("[PointCloud] No points after outlier removal!\n");
        return DensePointCloud();
    }

    // ========================================================================
    // Phase 4: Normal Refinement via PCA
    // ========================================================================

    printf("[PointCloud] Phase 4: Refining normals via PCA...\n");

    // Re-download cleaned points for bbox
    std::vector<float3> h_clean_pos(clean_count);
    CUDA_CHECK(cudaMemcpy(h_clean_pos.data(), d_clean_positions.ptr,
                           clean_count * sizeof(float3), cudaMemcpyDeviceToHost));
    computeBBox(h_clean_pos, bbox_min, bbox_max);

    // Build uniform grid for neighbor lookup (use same cell_size as outlier removal)
    UniformGrid ugrid;
    buildUniformGrid(d_clean_positions.ptr, clean_count, cell_size, bbox_min, bbox_max, ugrid);

    // Upload camera centers
    std::vector<float3> h_cam_centers;
    for (const auto& cam : cameras) {
        if (cam.is_registered) {
            Eigen::Vector3d c = cam.cam_center();
            h_cam_centers.push_back(make_float3((float)c.x(), (float)c.y(), (float)c.z()));
        }
    }

    DeviceBuffer<float3> d_cam_centers;
    d_cam_centers.alloc(h_cam_centers.size());
    d_cam_centers.upload(h_cam_centers.data(), h_cam_centers.size());

    {
        int threads = 256;
        int blocks = (clean_count + threads - 1) / threads;
        refine_normals_kernel<<<blocks, threads>>>(
            d_clean_positions.ptr, d_clean_normals.ptr, clean_count,
            ugrid.cell_size, ugrid.origin, ugrid.grid_x, ugrid.grid_y, ugrid.grid_z,
            ugrid.counts.ptr, ugrid.starts.ptr, ugrid.sorted_indices.ptr,
            d_cam_centers.ptr, (int)h_cam_centers.size(),
            6  // min_neighbors for PCA
        );
        CUDA_CHECK_KERNEL();
    }

    // ========================================================================
    // Phase 5: Download to CPU and return
    // ========================================================================

    DensePointCloud result;
    result.positions.resize(clean_count);
    result.normals.resize(clean_count);
    result.colors.resize(clean_count);

    std::vector<float3> h_normals(clean_count), h_colors(clean_count);
    CUDA_CHECK(cudaMemcpy(h_clean_pos.data(), d_clean_positions.ptr,
                           clean_count * sizeof(float3), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_normals.data(), d_clean_normals.ptr,
                           clean_count * sizeof(float3), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_colors.data(), d_clean_colors.ptr,
                           clean_count * sizeof(float3), cudaMemcpyDeviceToHost));

    for (int i = 0; i < clean_count; i++) {
        result.positions[i] = Eigen::Vector3f(h_clean_pos[i].x, h_clean_pos[i].y, h_clean_pos[i].z);
        result.normals[i]   = Eigen::Vector3f(h_normals[i].x, h_normals[i].y, h_normals[i].z);
        result.colors[i]    = Eigen::Vector3f(h_colors[i].x, h_colors[i].y, h_colors[i].z);
    }

    printf("[PointCloud] Done: %d final points\n", clean_count);

    // Checkpoint
    if (config.save_intermediate && !config.output_path.empty()) {
        std::string dir = fs::path(config.output_path).parent_path().string();
        fs::create_directories(dir);
        std::string ply_path = dir + "/dense.ply";
        saveDensePointCloudPLY(result, ply_path);
        printf("[PointCloud] Saved checkpoint: %s\n", ply_path.c_str());
    }

    return result;
}

// Legacy overload
DensePointCloud fuseDepthMaps(
    const std::vector<DepthMap>& depth_maps,
    const std::vector<CameraPose>& poses,
    const std::vector<ImageData>& images
) {
    PipelineConfig config;
    return fuseDepthMaps(depth_maps, poses, images, config);
}

// ============================================================================
// PLY I/O for DensePointCloud
// ============================================================================

bool saveDensePointCloudPLY(const DensePointCloud& cloud, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "[PointCloud] Cannot open %s for writing\n", path.c_str());
        return false;
    }

    int n = cloud.size();

    // ASCII header
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

    // Binary data
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

bool loadDensePointCloudPLY(DensePointCloud& cloud, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    // Parse header
    std::string line;
    int num_vertices = 0;
    bool in_header = true;
    while (in_header && std::getline(f, line)) {
        if (line.find("element vertex") != std::string::npos) {
            sscanf(line.c_str(), "element vertex %d", &num_vertices);
        }
        if (line == "end_header") {
            in_header = false;
        }
    }

    if (num_vertices <= 0) return false;

    cloud.positions.resize(num_vertices);
    cloud.normals.resize(num_vertices);
    cloud.colors.resize(num_vertices);

    // Read binary data: x y z nx ny nz r g b per vertex
    for (int i = 0; i < num_vertices; i++) {
        float x, y, z, nx, ny, nz;
        uint8_t r, g, b;
        f.read(reinterpret_cast<char*>(&x), 4);
        f.read(reinterpret_cast<char*>(&y), 4);
        f.read(reinterpret_cast<char*>(&z), 4);
        f.read(reinterpret_cast<char*>(&nx), 4);
        f.read(reinterpret_cast<char*>(&ny), 4);
        f.read(reinterpret_cast<char*>(&nz), 4);
        f.read(reinterpret_cast<char*>(&r), 1);
        f.read(reinterpret_cast<char*>(&g), 1);
        f.read(reinterpret_cast<char*>(&b), 1);

        cloud.positions[i] = Eigen::Vector3f(x, y, z);
        cloud.normals[i]   = Eigen::Vector3f(nx, ny, nz);
        cloud.colors[i]    = Eigen::Vector3f(r / 255.0f, g / 255.0f, b / 255.0f);
    }

    return f.good();
}
