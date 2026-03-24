#include "meshing/marching_cubes.h"
#include "utils/cuda_utils.cuh"
#include "utils/math_utils.h"

#include <Eigen/Core>
#include <cstdio>
#include <vector>
#include <cub/cub.cuh>

// ============================================================================
// Marching Cubes lookup tables (Paul Bourke)
// edgeTable: 256 entries, each a 12-bit mask of which edges are intersected.
// triTable: 256×16, triangle vertex indices (3 per triangle, -1 terminated).
// Stored in __constant__ memory for fast broadcast reads.
// ============================================================================

// Edge table: for each of the 256 cube configurations, which edges have crossings
static __constant__ int d_edgeTable[256] = {
    0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000
};

// Triangle table: 256 x 16. Each row lists up to 5 triangles (15 vertex indices),
// terminated by -1. Each group of 3 indices specifies one triangle.
static __constant__ int d_triTable[256][16] = {
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  1,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  8,  3,  9,  8,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  3,  1,  2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 9,  2, 10,  0,  2,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 2,  8,  3,  2, 10,  8, 10,  9,  8, -1, -1, -1, -1, -1, -1, -1},
    { 3, 11,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0, 11,  2,  8, 11,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  9,  0,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1, 11,  2,  1,  9, 11,  9,  8, 11, -1, -1, -1, -1, -1, -1, -1},
    { 3, 10,  1, 11, 10,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0, 10,  1,  0,  8, 10,  8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    { 3,  9,  0,  3, 11,  9, 11, 10,  9, -1, -1, -1, -1, -1, -1, -1},
    { 9,  8, 10, 10,  8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  7,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  3,  0,  7,  3,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  1,  9,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  1,  9,  4,  7,  1,  7,  3,  1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 10,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 3,  4,  7,  3,  0,  4,  1,  2, 10, -1, -1, -1, -1, -1, -1, -1},
    { 9,  2, 10,  9,  0,  2,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1},
    { 2, 10,  9,  2,  9,  7,  2,  7,  3,  7,  9,  4, -1, -1, -1, -1},
    { 8,  4,  7,  3, 11,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11,  4,  7, 11,  2,  4,  2,  0,  4, -1, -1, -1, -1, -1, -1, -1},
    { 9,  0,  1,  8,  4,  7,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1},
    { 4,  7, 11,  9,  4, 11,  9, 11,  2,  9,  2,  1, -1, -1, -1, -1},
    { 3, 10,  1,  3, 11, 10,  7,  8,  4, -1, -1, -1, -1, -1, -1, -1},
    { 1, 11, 10,  1,  4, 11,  1,  0,  4,  7, 11,  4, -1, -1, -1, -1},
    { 4,  7,  8,  9,  0, 11,  9, 11, 10, 11,  0,  3, -1, -1, -1, -1},
    { 4,  7, 11,  4, 11,  9,  9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    { 9,  5,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 9,  5,  4,  0,  8,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  5,  4,  1,  5,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 8,  5,  4,  8,  3,  5,  3,  1,  5, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 10,  9,  5,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 3,  0,  8,  1,  2, 10,  4,  9,  5, -1, -1, -1, -1, -1, -1, -1},
    { 5,  2, 10,  5,  4,  2,  4,  0,  2, -1, -1, -1, -1, -1, -1, -1},
    { 2, 10,  5,  3,  2,  5,  3,  5,  4,  3,  4,  8, -1, -1, -1, -1},
    { 9,  5,  4,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0, 11,  2,  0,  8, 11,  4,  9,  5, -1, -1, -1, -1, -1, -1, -1},
    { 0,  5,  4,  0,  1,  5,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1},
    { 2,  1,  5,  2,  5,  8,  2,  8, 11,  4,  8,  5, -1, -1, -1, -1},
    {10,  3, 11, 10,  1,  3,  9,  5,  4, -1, -1, -1, -1, -1, -1, -1},
    { 4,  9,  5,  0,  8,  1,  8, 10,  1,  8, 11, 10, -1, -1, -1, -1},
    { 5,  4,  0,  5,  0, 11,  5, 11, 10, 11,  0,  3, -1, -1, -1, -1},
    { 5,  4,  8,  5,  8, 10, 10,  8, 11, -1, -1, -1, -1, -1, -1, -1},
    { 9,  7,  8,  5,  7,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 9,  3,  0,  9,  5,  3,  5,  7,  3, -1, -1, -1, -1, -1, -1, -1},
    { 0,  7,  8,  0,  1,  7,  1,  5,  7, -1, -1, -1, -1, -1, -1, -1},
    { 1,  5,  3,  3,  5,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 9,  7,  8,  9,  5,  7, 10,  1,  2, -1, -1, -1, -1, -1, -1, -1},
    {10,  1,  2,  9,  5,  0,  5,  3,  0,  5,  7,  3, -1, -1, -1, -1},
    { 8,  0,  2,  8,  2,  5,  8,  5,  7, 10,  5,  2, -1, -1, -1, -1},
    { 2, 10,  5,  2,  5,  3,  3,  5,  7, -1, -1, -1, -1, -1, -1, -1},
    { 7,  9,  5,  7,  8,  9,  3, 11,  2, -1, -1, -1, -1, -1, -1, -1},
    { 9,  5,  7,  9,  7,  2,  9,  2,  0,  2,  7, 11, -1, -1, -1, -1},
    { 2,  3, 11,  0,  1,  8,  1,  7,  8,  1,  5,  7, -1, -1, -1, -1},
    {11,  2,  1, 11,  1,  7,  7,  1,  5, -1, -1, -1, -1, -1, -1, -1},
    { 9,  5,  8,  8,  5,  7, 10,  1,  3, 10,  3, 11, -1, -1, -1, -1},
    { 5,  7,  0,  5,  0,  9,  7, 11,  0,  1,  0, 10, 11, 10,  0, -1},
    {11, 10,  0, 11,  0,  3,  10,  5,  0,  8,  0,  7,  5,  7,  0, -1},
    {11, 10,  5,  7, 11,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10,  6,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  3,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 9,  0,  1,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  8,  3,  1,  9,  8,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1},
    { 1,  6,  5,  2,  6,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  6,  5,  1,  2,  6,  3,  0,  8, -1, -1, -1, -1, -1, -1, -1},
    { 9,  6,  5,  9,  0,  6,  0,  2,  6, -1, -1, -1, -1, -1, -1, -1},
    { 5,  9,  8,  5,  8,  2,  5,  2,  6,  3,  2,  8, -1, -1, -1, -1},
    { 2,  3, 11, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11,  0,  8, 11,  2,  0, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1},
    { 0,  1,  9,  2,  3, 11,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1},
    { 5, 10,  6,  1,  9,  2,  9, 11,  2,  9,  8, 11, -1, -1, -1, -1},
    { 6,  3, 11,  6,  5,  3,  5,  1,  3, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8, 11,  0, 11,  5,  0,  5,  1,  5, 11,  6, -1, -1, -1, -1},
    { 3, 11,  6,  0,  3,  6,  0,  6,  5,  0,  5,  9, -1, -1, -1, -1},
    { 6,  5,  9,  6,  9, 11, 11,  9,  8, -1, -1, -1, -1, -1, -1, -1},
    { 5, 10,  6,  4,  7,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  3,  0,  4,  7,  3,  6,  5, 10, -1, -1, -1, -1, -1, -1, -1},
    { 1,  9,  0,  5, 10,  6,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1},
    {10,  6,  5,  1,  9,  7,  1,  7,  3,  7,  9,  4, -1, -1, -1, -1},
    { 6,  1,  2,  6,  5,  1,  4,  7,  8, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2,  5,  5,  2,  6,  3,  0,  4,  3,  4,  7, -1, -1, -1, -1},
    { 8,  4,  7,  9,  0,  5,  0,  6,  5,  0,  2,  6, -1, -1, -1, -1},
    { 7,  3,  9,  7,  9,  4,  3,  2,  9,  5,  9,  6,  2,  6,  9, -1},
    { 3, 11,  2,  7,  8,  4, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1},
    { 5, 10,  6,  4,  7,  2,  4,  2,  0,  2,  7, 11, -1, -1, -1, -1},
    { 0,  1,  9,  4,  7,  8,  2,  3, 11,  5, 10,  6, -1, -1, -1, -1},
    { 9,  2,  1,  9, 11,  2,  9,  4, 11,  7, 11,  4,  5, 10,  6, -1},
    { 8,  4,  7,  3, 11,  5,  3,  5,  1,  5, 11,  6, -1, -1, -1, -1},
    { 5,  1, 11,  5, 11,  6,  1,  0, 11,  7, 11,  4,  0,  4, 11, -1},
    { 0,  5,  9,  0,  6,  5,  0,  3,  6, 11,  6,  3,  8,  4,  7, -1},
    { 6,  5,  9,  6,  9, 11,  4,  7,  9,  7, 11,  9, -1, -1, -1, -1},
    {10,  4,  9,  6,  4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4, 10,  6,  4,  9, 10,  0,  8,  3, -1, -1, -1, -1, -1, -1, -1},
    {10,  0,  1, 10,  6,  0,  6,  4,  0, -1, -1, -1, -1, -1, -1, -1},
    { 8,  3,  1,  8,  1,  6,  8,  6,  4,  6,  1, 10, -1, -1, -1, -1},
    { 1,  4,  9,  1,  2,  4,  2,  6,  4, -1, -1, -1, -1, -1, -1, -1},
    { 3,  0,  8,  1,  2,  9,  2,  4,  9,  2,  6,  4, -1, -1, -1, -1},
    { 0,  2,  4,  4,  2,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 8,  3,  2,  8,  2,  4,  4,  2,  6, -1, -1, -1, -1, -1, -1, -1},
    {10,  4,  9, 10,  6,  4, 11,  2,  3, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  2,  2,  8, 11,  4,  9, 10,  4, 10,  6, -1, -1, -1, -1},
    { 3, 11,  2,  0,  1,  6,  0,  6,  4,  6,  1, 10, -1, -1, -1, -1},
    { 6,  4,  1,  6,  1, 10,  4,  8,  1,  2,  1, 11,  8, 11,  1, -1},
    { 9,  6,  4,  9,  3,  6,  9,  1,  3, 11,  6,  3, -1, -1, -1, -1},
    { 8, 11,  1,  8,  1,  0, 11,  6,  1,  9,  1,  4,  6,  4,  1, -1},
    { 3, 11,  6,  3,  6,  0,  0,  6,  4, -1, -1, -1, -1, -1, -1, -1},
    { 6,  4,  8, 11,  6,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 7, 10,  6,  7,  8, 10,  8,  9, 10, -1, -1, -1, -1, -1, -1, -1},
    { 0,  7,  3,  0, 10,  7,  0,  9, 10,  6,  7, 10, -1, -1, -1, -1},
    {10,  6,  7,  1, 10,  7,  1,  7,  8,  1,  8,  0, -1, -1, -1, -1},
    {10,  6,  7, 10,  7,  1,  1,  7,  3, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2,  6,  1,  6,  8,  1,  8,  9,  8,  6,  7, -1, -1, -1, -1},
    { 2,  6,  9,  2,  9,  1,  6,  7,  9,  0,  9,  3,  7,  3,  9, -1},
    { 7,  8,  0,  7,  0,  6,  6,  0,  2, -1, -1, -1, -1, -1, -1, -1},
    { 7,  3,  2,  6,  7,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 2,  3, 11, 10,  6,  8, 10,  8,  9,  8,  6,  7, -1, -1, -1, -1},
    { 2,  0,  7,  2,  7, 11,  0,  9,  7,  6,  7, 10,  9, 10,  7, -1},
    { 1,  8,  0,  1,  7,  8,  1, 10,  7,  6,  7, 10,  2,  3, 11, -1},
    {11,  2,  1, 11,  1,  7, 10,  6,  1,  6,  7,  1, -1, -1, -1, -1},
    { 8,  9,  6,  8,  6,  7,  9,  1,  6, 11,  6,  3,  1,  3,  6, -1},
    { 0,  9,  1, 11,  6,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 7,  8,  0,  7,  0,  6,  3, 11,  0, 11,  6,  0, -1, -1, -1, -1},
    { 7, 11,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 7,  6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 3,  0,  8, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  1,  9, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 8,  1,  9,  8,  3,  1, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1},
    {10,  1,  2,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 10,  3,  0,  8,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1},
    { 2,  9,  0,  2, 10,  9,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1},
    { 6, 11,  7,  2, 10,  3, 10,  8,  3, 10,  9,  8, -1, -1, -1, -1},
    { 7,  2,  3,  6,  2,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 7,  0,  8,  7,  6,  0,  6,  2,  0, -1, -1, -1, -1, -1, -1, -1},
    { 2,  7,  6,  2,  3,  7,  0,  1,  9, -1, -1, -1, -1, -1, -1, -1},
    { 1,  6,  2,  1,  8,  6,  1,  9,  8,  8,  7,  6, -1, -1, -1, -1},
    {10,  7,  6, 10,  1,  7,  1,  3,  7, -1, -1, -1, -1, -1, -1, -1},
    {10,  7,  6,  1,  7, 10,  1,  8,  7,  1,  0,  8, -1, -1, -1, -1},
    { 0,  3,  7,  0,  7, 10,  0, 10,  9,  6, 10,  7, -1, -1, -1, -1},
    { 7,  6, 10,  7, 10,  8,  8, 10,  9, -1, -1, -1, -1, -1, -1, -1},
    { 6,  8,  4, 11,  8,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 3,  6, 11,  3,  0,  6,  0,  4,  6, -1, -1, -1, -1, -1, -1, -1},
    { 8,  6, 11,  8,  4,  6,  9,  0,  1, -1, -1, -1, -1, -1, -1, -1},
    { 9,  4,  6,  9,  6,  3,  9,  3,  1, 11,  3,  6, -1, -1, -1, -1},
    { 6,  8,  4,  6, 11,  8,  2, 10,  1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 10,  3,  0, 11,  0,  6, 11,  0,  4,  6, -1, -1, -1, -1},
    { 4, 11,  8,  4,  6, 11,  0,  2,  9,  2, 10,  9, -1, -1, -1, -1},
    {10,  9,  3, 10,  3,  2,  9,  4,  3, 11,  3,  6,  4,  6,  3, -1},
    { 8,  2,  3,  8,  4,  2,  4,  6,  2, -1, -1, -1, -1, -1, -1, -1},
    { 0,  4,  2,  4,  6,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  9,  0,  2,  3,  4,  2,  4,  6,  4,  3,  8, -1, -1, -1, -1},
    { 1,  9,  4,  1,  4,  2,  2,  4,  6, -1, -1, -1, -1, -1, -1, -1},
    { 8,  1,  3,  8,  6,  1,  8,  4,  6,  6, 10,  1, -1, -1, -1, -1},
    {10,  1,  0, 10,  0,  6,  6,  0,  4, -1, -1, -1, -1, -1, -1, -1},
    { 4,  6,  3,  4,  3,  8,  6, 10,  3,  0,  3,  9, 10,  9,  3, -1},
    {10,  9,  4,  6, 10,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  9,  5,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  3,  4,  9,  5, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1},
    { 5,  0,  1,  5,  4,  0,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1},
    {11,  7,  6,  8,  3,  4,  3,  5,  4,  3,  1,  5, -1, -1, -1, -1},
    { 9,  5,  4, 10,  1,  2,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1},
    { 6, 11,  7,  1,  2, 10,  0,  8,  3,  4,  9,  5, -1, -1, -1, -1},
    { 7,  6, 11,  5,  4, 10,  4,  2, 10,  4,  0,  2, -1, -1, -1, -1},
    { 3,  4,  8,  3,  5,  4,  3,  2,  5, 10,  5,  2, 11,  7,  6, -1},
    { 7,  2,  3,  7,  6,  2,  5,  4,  9, -1, -1, -1, -1, -1, -1, -1},
    { 9,  5,  4,  0,  8,  6,  0,  6,  2,  6,  8,  7, -1, -1, -1, -1},
    { 3,  6,  2,  3,  7,  6,  1,  5,  0,  5,  4,  0, -1, -1, -1, -1},
    { 6,  2,  8,  6,  8,  7,  2,  1,  8,  4,  8,  5,  1,  5,  8, -1},
    { 9,  5,  4, 10,  1,  6,  1,  7,  6,  1,  3,  7, -1, -1, -1, -1},
    { 1,  6, 10,  1,  7,  6,  1,  0,  7,  8,  7,  0,  9,  5,  4, -1},
    { 4,  0, 10,  4, 10,  5,  0,  3, 10,  6, 10,  7,  3,  7, 10, -1},
    { 7,  6, 10,  7, 10,  8,  5,  4, 10,  4,  8, 10, -1, -1, -1, -1},
    { 6,  9,  5,  6, 11,  9, 11,  8,  9, -1, -1, -1, -1, -1, -1, -1},
    { 3,  6, 11,  0,  6,  3,  0,  5,  6,  0,  9,  5, -1, -1, -1, -1},
    { 0, 11,  8,  0,  5, 11,  0,  1,  5,  5,  6, 11, -1, -1, -1, -1},
    { 6, 11,  3,  6,  3,  5,  5,  3,  1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 10,  9,  5, 11,  9, 11,  8, 11,  5,  6, -1, -1, -1, -1},
    { 0, 11,  3,  0,  6, 11,  0,  9,  6,  5,  6,  9,  1,  2, 10, -1},
    {11,  8,  5, 11,  5,  6,  8,  0,  5, 10,  5,  2,  0,  2,  5, -1},
    { 6, 11,  3,  6,  3,  5,  2, 10,  3, 10,  5,  3, -1, -1, -1, -1},
    { 5,  8,  9,  5,  2,  8,  5,  6,  2,  3,  8,  2, -1, -1, -1, -1},
    { 9,  5,  6,  9,  6,  0,  0,  6,  2, -1, -1, -1, -1, -1, -1, -1},
    { 1,  5,  8,  1,  8,  0,  5,  6,  8,  3,  8,  2,  6,  2,  8, -1},
    { 1,  5,  6,  2,  1,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  3,  6,  1,  6, 10,  3,  8,  6,  5,  6,  9,  8,  9,  6, -1},
    {10,  1,  0, 10,  0,  6,  9,  5,  0,  5,  6,  0, -1, -1, -1, -1},
    { 0,  3,  8,  5,  6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10,  5,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11,  5, 10,  7,  5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11,  5, 10, 11,  7,  5,  8,  3,  0, -1, -1, -1, -1, -1, -1, -1},
    { 5, 11,  7,  5, 10, 11,  1,  9,  0, -1, -1, -1, -1, -1, -1, -1},
    {10, 7, 5, 10, 11,  7,  9,  8,  1,  8,  3,  1, -1, -1, -1, -1},
    {11,  1,  2, 11,  7,  1,  7,  5,  1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  3,  1,  2,  7,  1,  7,  5,  7,  2, 11, -1, -1, -1, -1},
    { 9,  7,  5,  9,  2,  7,  9,  0,  2,  2, 11,  7, -1, -1, -1, -1},
    { 7,  5,  2,  7,  2, 11,  5,  9,  2,  3,  2,  8,  9,  8,  2, -1},
    { 2,  5, 10,  2,  3,  5,  3,  7,  5, -1, -1, -1, -1, -1, -1, -1},
    { 8,  2,  0,  8,  5,  2,  8,  7,  5, 10,  2,  5, -1, -1, -1, -1},
    { 9,  0,  1,  5, 10,  3,  5,  3,  7,  3, 10,  2, -1, -1, -1, -1},
    { 9,  8,  2,  9,  2,  1,  8,  7,  2, 10,  2,  5,  7,  5,  2, -1},
    { 1,  3,  5,  3,  7,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  7,  0,  7,  1,  1,  7,  5, -1, -1, -1, -1, -1, -1, -1},
    { 9,  0,  3,  9,  3,  5,  5,  3,  7, -1, -1, -1, -1, -1, -1, -1},
    { 9,  8,  7,  5,  9,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 5,  8,  4,  5, 10,  8, 10, 11,  8, -1, -1, -1, -1, -1, -1, -1},
    { 5,  0,  4,  5, 11,  0,  5, 10, 11, 11,  3,  0, -1, -1, -1, -1},
    { 0,  1,  9,  8,  4, 10,  8, 10, 11, 10,  4,  5, -1, -1, -1, -1},
    {10, 11,  4, 10,  4,  5, 11,  3,  4,  9,  4,  1,  3,  1,  4, -1},
    { 2,  5,  1,  2,  8,  5,  2, 11,  8,  4,  5,  8, -1, -1, -1, -1},
    { 0,  4, 11,  0, 11,  3,  4,  5, 11,  2, 11,  1,  5,  1, 11, -1},
    { 0,  2,  5,  0,  5,  9,  2, 11,  5,  4,  5,  8, 11,  8,  5, -1},
    { 9,  4,  5,  2, 11,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 2,  5, 10,  3,  5,  2,  3,  4,  5,  3,  8,  4, -1, -1, -1, -1},
    { 5, 10,  2,  5,  2,  4,  4,  2,  0, -1, -1, -1, -1, -1, -1, -1},
    { 3, 10,  2,  3,  5, 10,  3,  8,  5,  4,  5,  8,  0,  1,  9, -1},
    { 5, 10,  2,  5,  2,  4,  1,  9,  2,  9,  4,  2, -1, -1, -1, -1},
    { 8,  4,  5,  8,  5,  3,  3,  5,  1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  4,  5,  1,  0,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 8,  4,  5,  8,  5,  3,  9,  0,  5,  0,  3,  5, -1, -1, -1, -1},
    { 9,  4,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4, 11,  7,  4,  9, 11,  9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  3,  4,  9,  7,  9, 11,  7,  9, 10, 11, -1, -1, -1, -1},
    { 1, 10, 11,  1, 11,  4,  1,  4,  0,  7,  4, 11, -1, -1, -1, -1},
    { 3,  1,  4,  3,  4,  8,  1, 10,  4,  7,  4, 11, 10, 11,  4, -1},
    { 4, 11,  7,  9, 11,  4,  9,  2, 11,  9,  1,  2, -1, -1, -1, -1},
    { 9,  7,  4,  9, 11,  7,  9,  1, 11,  2, 11,  1,  0,  8,  3, -1},
    {11,  7,  4, 11,  4,  2,  2,  4,  0, -1, -1, -1, -1, -1, -1, -1},
    {11,  7,  4, 11,  4,  2,  8,  3,  4,  3,  2,  4, -1, -1, -1, -1},
    { 2,  9, 10,  2,  7,  9,  2,  3,  7,  7,  4,  9, -1, -1, -1, -1},
    { 9, 10,  7,  9,  7,  4, 10,  2,  7,  8,  7,  0,  2,  0,  7, -1},
    { 3,  7, 10,  3, 10,  2,  7,  4, 10,  1, 10,  0,  4,  0, 10, -1},
    { 1, 10,  2,  8,  7,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  9,  1,  4,  1,  7,  7,  1,  3, -1, -1, -1, -1, -1, -1, -1},
    { 4,  9,  1,  4,  1,  7,  0,  8,  1,  8,  7,  1, -1, -1, -1, -1},
    { 4,  0,  3,  7,  4,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  8,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 9, 10,  8,  10, 11,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 3,  0,  9,  3,  9, 11, 11,  9, 10, -1, -1, -1, -1, -1, -1, -1},
    { 0,  1, 10,  0, 10,  8,  8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    { 3,  1, 10, 11,  3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 11,  1, 11,  9,  9, 11,  8, -1, -1, -1, -1, -1, -1, -1},
    { 3,  0,  9,  3,  9, 11,  1,  2,  9,  2, 11,  9, -1, -1, -1, -1},
    { 0,  2, 11,  8,  0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 3,  2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 2,  3,  8,  2,  8, 10, 10,  8,  9, -1, -1, -1, -1, -1, -1, -1},
    { 9, 10,  2,  0,  9,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 2,  3,  8,  2,  8, 10,  0,  1,  8,  1, 10,  8, -1, -1, -1, -1},
    { 1, 10,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  3,  8,  9,  1,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  9,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  3,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
};

// ============================================================================
// CUDA Kernels
// ============================================================================

// Kernel 1: Classify each voxel — compute cube index, count triangles
__global__ void classifyVoxelsKernel(
    const float* __restrict__ volume,
    int nx, int ny, int nz,
    float iso_value,
    int* __restrict__ voxel_tri_count,   // output: triangles per voxel
    uint8_t* __restrict__ cube_indices   // output: cube configuration index
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cubes = (nx - 1) * (ny - 1) * (nz - 1);
    if (idx >= total_cubes) return;

    // Decompose linear index to (ix, iy, iz) — cube corner at (ix, iy, iz)
    int nxm = nx - 1, nym = ny - 1;
    int iz = idx / (nxm * nym);
    int rem = idx - iz * (nxm * nym);
    int iy = rem / nxm;
    int ix = rem - iy * nxm;

    // Sample TSDF at 8 cube corners
    // Corner ordering (Paul Bourke convention):
    //   0: (ix,   iy,   iz  )    4: (ix,   iy,   iz+1)
    //   1: (ix+1, iy,   iz  )    5: (ix+1, iy,   iz+1)
    //   2: (ix+1, iy+1, iz  )    6: (ix+1, iy+1, iz+1)
    //   3: (ix,   iy+1, iz  )    7: (ix,   iy+1, iz+1)
    #define VOL(x, y, z) volume[(z) * nx * ny + (y) * nx + (x)]

    float val[8];
    val[0] = VOL(ix,     iy,     iz    );
    val[1] = VOL(ix + 1, iy,     iz    );
    val[2] = VOL(ix + 1, iy + 1, iz    );
    val[3] = VOL(ix,     iy + 1, iz    );
    val[4] = VOL(ix,     iy,     iz + 1);
    val[5] = VOL(ix + 1, iy,     iz + 1);
    val[6] = VOL(ix + 1, iy + 1, iz + 1);
    val[7] = VOL(ix,     iy + 1, iz + 1);

    #undef VOL

    // Compute 8-bit cube index
    int cube_idx = 0;
    for (int i = 0; i < 8; i++) {
        if (val[i] < iso_value) cube_idx |= (1 << i);
    }

    cube_indices[idx] = (uint8_t)cube_idx;

    // Count triangles from triTable
    int count = 0;
    if (d_edgeTable[cube_idx] != 0) {
        for (int i = 0; d_triTable[cube_idx][i] != -1; i += 3) {
            count++;
        }
    }
    voxel_tri_count[idx] = count;
}

// Helper: interpolate vertex position along an edge
__device__ float3 interpolateEdge(float iso, float v0, float v1,
                                   float3 p0, float3 p1) {
    if (fabsf(iso - v0) < 1e-10f) return p0;
    if (fabsf(iso - v1) < 1e-10f) return p1;
    if (fabsf(v0 - v1) < 1e-10f) return p0;
    float t = (iso - v0) / (v1 - v0);
    t = fminf(fmaxf(t, 0.0f), 1.0f);
    return make_float3(
        p0.x + t * (p1.x - p0.x),
        p0.y + t * (p1.y - p0.y),
        p0.z + t * (p1.z - p0.z)
    );
}

// Kernel 2: Generate triangle vertices
__global__ void generateTrianglesKernel(
    const float* __restrict__ volume,
    int nx, int ny, int nz,
    float iso_value,
    float voxel_size,
    float origin_x, float origin_y, float origin_z,
    const uint8_t* __restrict__ cube_indices,
    const int* __restrict__ tri_offsets,  // prefix sum of triangle counts
    float3* __restrict__ out_vertices,    // 3 vertices per triangle
    float3* __restrict__ out_normals      // 3 normals per triangle
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cubes = (nx - 1) * (ny - 1) * (nz - 1);
    if (idx >= total_cubes) return;

    int cube_idx = cube_indices[idx];
    if (d_edgeTable[cube_idx] == 0) return;

    // Decompose index
    int nxm = nx - 1, nym = ny - 1;
    int iz = idx / (nxm * nym);
    int rem = idx - iz * (nxm * nym);
    int iy = rem / nxm;
    int ix = rem - iy * nxm;

    // Corner positions in world space
    float3 corners[8];
    corners[0] = make_float3(origin_x + ix       * voxel_size, origin_y + iy       * voxel_size, origin_z + iz       * voxel_size);
    corners[1] = make_float3(origin_x + (ix + 1) * voxel_size, origin_y + iy       * voxel_size, origin_z + iz       * voxel_size);
    corners[2] = make_float3(origin_x + (ix + 1) * voxel_size, origin_y + (iy + 1) * voxel_size, origin_z + iz       * voxel_size);
    corners[3] = make_float3(origin_x + ix       * voxel_size, origin_y + (iy + 1) * voxel_size, origin_z + iz       * voxel_size);
    corners[4] = make_float3(origin_x + ix       * voxel_size, origin_y + iy       * voxel_size, origin_z + (iz + 1) * voxel_size);
    corners[5] = make_float3(origin_x + (ix + 1) * voxel_size, origin_y + iy       * voxel_size, origin_z + (iz + 1) * voxel_size);
    corners[6] = make_float3(origin_x + (ix + 1) * voxel_size, origin_y + (iy + 1) * voxel_size, origin_z + (iz + 1) * voxel_size);
    corners[7] = make_float3(origin_x + ix       * voxel_size, origin_y + (iy + 1) * voxel_size, origin_z + (iz + 1) * voxel_size);

    // Corner values
    #define VOL(x, y, z) volume[(z) * nx * ny + (y) * nx + (x)]
    float val[8];
    val[0] = VOL(ix,     iy,     iz    );
    val[1] = VOL(ix + 1, iy,     iz    );
    val[2] = VOL(ix + 1, iy + 1, iz    );
    val[3] = VOL(ix,     iy + 1, iz    );
    val[4] = VOL(ix,     iy,     iz + 1);
    val[5] = VOL(ix + 1, iy,     iz + 1);
    val[6] = VOL(ix + 1, iy + 1, iz + 1);
    val[7] = VOL(ix,     iy + 1, iz + 1);
    #undef VOL

    // Interpolate edge vertices
    // Edges: 0-1, 1-2, 2-3, 3-0, 4-5, 5-6, 6-7, 7-4, 0-4, 1-5, 2-6, 3-7
    float3 edge_verts[12];
    int edge_flags = d_edgeTable[cube_idx];

    if (edge_flags &    1) edge_verts[ 0] = interpolateEdge(iso_value, val[0], val[1], corners[0], corners[1]);
    if (edge_flags &    2) edge_verts[ 1] = interpolateEdge(iso_value, val[1], val[2], corners[1], corners[2]);
    if (edge_flags &    4) edge_verts[ 2] = interpolateEdge(iso_value, val[2], val[3], corners[2], corners[3]);
    if (edge_flags &    8) edge_verts[ 3] = interpolateEdge(iso_value, val[3], val[0], corners[3], corners[0]);
    if (edge_flags &   16) edge_verts[ 4] = interpolateEdge(iso_value, val[4], val[5], corners[4], corners[5]);
    if (edge_flags &   32) edge_verts[ 5] = interpolateEdge(iso_value, val[5], val[6], corners[5], corners[6]);
    if (edge_flags &   64) edge_verts[ 6] = interpolateEdge(iso_value, val[6], val[7], corners[6], corners[7]);
    if (edge_flags &  128) edge_verts[ 7] = interpolateEdge(iso_value, val[7], val[4], corners[7], corners[4]);
    if (edge_flags &  256) edge_verts[ 8] = interpolateEdge(iso_value, val[0], val[4], corners[0], corners[4]);
    if (edge_flags &  512) edge_verts[ 9] = interpolateEdge(iso_value, val[1], val[5], corners[1], corners[5]);
    if (edge_flags & 1024) edge_verts[10] = interpolateEdge(iso_value, val[2], val[6], corners[2], corners[6]);
    if (edge_flags & 2048) edge_verts[11] = interpolateEdge(iso_value, val[3], val[7], corners[3], corners[7]);

    // Compute gradient (central differences) for normals at each edge vertex
    // We'll use the gradient at the interpolated position approximated by the gradient
    // at the edge midpoint grid position
    // For simplicity, compute face normals instead (cross product of triangle edges)

    // Write triangles
    int write_offset = tri_offsets[idx] * 3; // 3 vertices per triangle

    for (int i = 0; d_triTable[cube_idx][i] != -1; i += 3) {
        float3 v0 = edge_verts[d_triTable[cube_idx][i    ]];
        float3 v1 = edge_verts[d_triTable[cube_idx][i + 1]];
        float3 v2 = edge_verts[d_triTable[cube_idx][i + 2]];

        // Face normal
        float3 e1 = v1 - v0;
        float3 e2 = v2 - v0;
        float3 n = normalize(cross(e1, e2));

        out_vertices[write_offset    ] = v0;
        out_vertices[write_offset + 1] = v1;
        out_vertices[write_offset + 2] = v2;
        out_normals[write_offset    ] = n;
        out_normals[write_offset + 1] = n;
        out_normals[write_offset + 2] = n;
        write_offset += 3;
    }
}

// ============================================================================
// Host function
// ============================================================================
Mesh marchingCubes(const float* volume, int nx, int ny, int nz,
                    float iso_value, float voxel_size,
                    float origin_x, float origin_y, float origin_z) {
    if (!volume || nx < 2 || ny < 2 || nz < 2) {
        fprintf(stderr, "[MC] Invalid volume dimensions: %dx%dx%d\n", nx, ny, nz);
        return Mesh();
    }

    int total_cubes = (nx - 1) * (ny - 1) * (nz - 1);
    printf("[MC] Marching cubes: %dx%dx%d grid (%d cubes), iso=%.3f\n",
           nx, ny, nz, total_cubes, iso_value);

    // Upload volume to GPU
    DeviceBuffer<float> d_volume;
    d_volume.alloc(nx * ny * nz);
    d_volume.upload(volume, nx * ny * nz);

    // Allocate classification buffers
    DeviceBuffer<int> d_tri_count;
    DeviceBuffer<uint8_t> d_cube_indices;
    d_tri_count.alloc(total_cubes);
    d_cube_indices.alloc(total_cubes);

    // Kernel 1: Classify voxels
    int block = 256;
    int grid = (total_cubes + block - 1) / block;
    classifyVoxelsKernel<<<grid, block>>>(
        d_volume.ptr, nx, ny, nz, iso_value,
        d_tri_count.ptr, d_cube_indices.ptr
    );
    CUDA_CHECK_KERNEL();

    // Prefix sum to get write offsets
    DeviceBuffer<int> d_tri_offsets;
    d_tri_offsets.alloc(total_cubes);

    // CUB exclusive prefix sum
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes,
                                   d_tri_count.ptr, d_tri_offsets.ptr, total_cubes);
    DeviceBuffer<uint8_t> d_temp_buf;
    d_temp_buf.alloc(temp_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_buf.ptr, temp_bytes,
                                   d_tri_count.ptr, d_tri_offsets.ptr, total_cubes);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Get total triangle count
    int last_count, last_offset;
    CUDA_CHECK(cudaMemcpy(&last_count, d_tri_count.ptr + total_cubes - 1, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_offset, d_tri_offsets.ptr + total_cubes - 1, sizeof(int), cudaMemcpyDeviceToHost));
    int total_triangles = last_offset + last_count;

    printf("[MC] Total triangles: %d\n", total_triangles);

    if (total_triangles == 0) {
        printf("[MC] No triangles generated\n");
        return Mesh();
    }

    // Allocate output buffers (3 vertices per triangle)
    int total_verts = total_triangles * 3;
    DeviceBuffer<float3> d_out_verts, d_out_normals;
    d_out_verts.alloc(total_verts);
    d_out_normals.alloc(total_verts);

    // Kernel 2: Generate triangles
    generateTrianglesKernel<<<grid, block>>>(
        d_volume.ptr, nx, ny, nz, iso_value,
        voxel_size, origin_x, origin_y, origin_z,
        d_cube_indices.ptr, d_tri_offsets.ptr,
        d_out_verts.ptr, d_out_normals.ptr
    );
    CUDA_CHECK_KERNEL();

    // Download results
    std::vector<float3> h_verts(total_verts);
    std::vector<float3> h_normals(total_verts);
    d_out_verts.download(h_verts.data(), total_verts);
    d_out_normals.download(h_normals.data(), total_verts);

    // Build mesh with shared vertices (weld identical positions)
    Mesh mesh;
    mesh.vertices.reserve(total_verts);
    mesh.normals.reserve(total_verts);
    mesh.faces.reserve(total_triangles);

    // Simple approach: unique vertices per triangle (no welding for now — keeps it correct)
    mesh.vertices.resize(total_verts);
    mesh.normals.resize(total_verts);
    for (int i = 0; i < total_verts; i++) {
        mesh.vertices[i] = Eigen::Vector3f(h_verts[i].x, h_verts[i].y, h_verts[i].z);
        mesh.normals[i] = Eigen::Vector3f(h_normals[i].x, h_normals[i].y, h_normals[i].z);
    }
    for (int t = 0; t < total_triangles; t++) {
        mesh.faces.push_back(Eigen::Vector3i(t * 3, t * 3 + 1, t * 3 + 2));
    }

    // Weld vertices: merge vertices closer than voxel_size * 0.01
    // This makes the mesh manifold / watertight
    float weld_dist = voxel_size * 0.01f;
    float weld_dist2 = weld_dist * weld_dist;

    // Grid-based vertex welding
    Eigen::Vector3f bmin = mesh.vertices[0], bmax = mesh.vertices[0];
    for (const auto& v : mesh.vertices) {
        bmin = bmin.cwiseMin(v);
        bmax = bmax.cwiseMax(v);
    }

    float cell = voxel_size * 0.5f;
    if (cell < 1e-10f) cell = 1.0f;

    auto cellKey = [&](const Eigen::Vector3f& p) -> uint64_t {
        int cx = (int)((p.x() - bmin.x()) / cell);
        int cy = (int)((p.y() - bmin.y()) / cell);
        int cz = (int)((p.z() - bmin.z()) / cell);
        return ((uint64_t)(unsigned)cx * 73856093ULL) ^ ((uint64_t)(unsigned)cy * 19349663ULL) ^ ((uint64_t)(unsigned)cz * 83492791ULL);
    };

    std::unordered_map<uint64_t, std::vector<int>> vert_grid;
    std::vector<int> remap(total_verts, -1);
    std::vector<Eigen::Vector3f> welded_verts;
    std::vector<Eigen::Vector3f> welded_normals;

    for (int i = 0; i < total_verts; i++) {
        const auto& v = mesh.vertices[i];
        uint64_t key = cellKey(v);

        int found = -1;
        // Search this cell and neighbors
        int cx = (int)((v.x() - bmin.x()) / cell);
        int cy = (int)((v.y() - bmin.y()) / cell);
        int cz = (int)((v.z() - bmin.z()) / cell);
        for (int dx = -1; dx <= 1 && found < 0; dx++)
        for (int dy = -1; dy <= 1 && found < 0; dy++)
        for (int dz = -1; dz <= 1 && found < 0; dz++) {
            uint64_t nk = ((uint64_t)(unsigned)(cx+dx) * 73856093ULL) ^ ((uint64_t)(unsigned)(cy+dy) * 19349663ULL) ^ ((uint64_t)(unsigned)(cz+dz) * 83492791ULL);
            auto it = vert_grid.find(nk);
            if (it == vert_grid.end()) continue;
            for (int j : it->second) {
                if ((welded_verts[j] - v).squaredNorm() < weld_dist2) {
                    found = j;
                    break;
                }
            }
        }

        if (found >= 0) {
            remap[i] = found;
        } else {
            remap[i] = (int)welded_verts.size();
            welded_verts.push_back(v);
            welded_normals.push_back(mesh.normals[i]);
            vert_grid[key].push_back(remap[i]);
        }
    }

    mesh.vertices = std::move(welded_verts);
    mesh.normals = std::move(welded_normals);
    for (auto& f : mesh.faces) {
        f = Eigen::Vector3i(remap[f.x()], remap[f.y()], remap[f.z()]);
    }

    // Remove degenerate faces
    std::vector<Eigen::Vector3i> clean_faces;
    clean_faces.reserve(mesh.faces.size());
    for (const auto& f : mesh.faces) {
        if (f.x() != f.y() && f.y() != f.z() && f.x() != f.z())
            clean_faces.push_back(f);
    }
    mesh.faces = std::move(clean_faces);

    printf("[MC] Output mesh: %d vertices, %d faces (after welding)\n",
           mesh.num_vertices(), mesh.num_faces());
    return mesh;
}
