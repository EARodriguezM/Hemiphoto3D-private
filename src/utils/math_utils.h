#pragma once

#include <cmath>
#include <cuda_runtime.h>

// ============================================================================
// Lightweight math utilities for CUDA device code.
// All functions are __host__ __device__ inline.
// Do NOT use Eigen in this file.
// ============================================================================

#ifdef __CUDACC__
#define MATH_FUNC __host__ __device__ __forceinline__
#else
#define MATH_FUNC inline
#endif

// --- float3 operators -------------------------------------------------------

MATH_FUNC float3 make_float3_(float x, float y, float z) {
    return make_float3(x, y, z);
}

MATH_FUNC float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

MATH_FUNC float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

MATH_FUNC float3 operator*(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

MATH_FUNC float3 operator*(float s, float3 a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

MATH_FUNC float3 operator*(float3 a, float3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

MATH_FUNC float3 operator-(float3 a) {
    return make_float3(-a.x, -a.y, -a.z);
}

MATH_FUNC float3& operator+=(float3& a, float3 b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
    return a;
}

MATH_FUNC float3& operator-=(float3& a, float3 b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
    return a;
}

MATH_FUNC float3& operator*=(float3& a, float s) {
    a.x *= s; a.y *= s; a.z *= s;
    return a;
}

// --- float3 functions -------------------------------------------------------

MATH_FUNC float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

MATH_FUNC float3 cross(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

MATH_FUNC float length(float3 v) {
    return sqrtf(dot(v, v));
}

MATH_FUNC float3 normalize(float3 v) {
    float inv_len = 1.0f / fmaxf(length(v), 1e-12f);
    return v * inv_len;
}

// --- 3x3 matrix (row-major) -------------------------------------------------

struct Matrix3x3 {
    float m[9]; // row-major: m[row*3+col]

    MATH_FUNC float& operator()(int r, int c)       { return m[r * 3 + c]; }
    MATH_FUNC float  operator()(int r, int c) const  { return m[r * 3 + c]; }

    MATH_FUNC static Matrix3x3 identity() {
        Matrix3x3 I;
        I.m[0]=1; I.m[1]=0; I.m[2]=0;
        I.m[3]=0; I.m[4]=1; I.m[5]=0;
        I.m[6]=0; I.m[7]=0; I.m[8]=1;
        return I;
    }

    MATH_FUNC static Matrix3x3 zeros() {
        Matrix3x3 Z;
        for (int i = 0; i < 9; i++) Z.m[i] = 0.0f;
        return Z;
    }

    MATH_FUNC float3 row(int r) const {
        return make_float3(m[r*3], m[r*3+1], m[r*3+2]);
    }

    MATH_FUNC float3 col(int c) const {
        return make_float3(m[c], m[3+c], m[6+c]);
    }

    // Matrix-vector multiply: result = M * v
    MATH_FUNC float3 operator*(float3 v) const {
        return make_float3(
            m[0]*v.x + m[1]*v.y + m[2]*v.z,
            m[3]*v.x + m[4]*v.y + m[5]*v.z,
            m[6]*v.x + m[7]*v.y + m[8]*v.z
        );
    }

    // Matrix-matrix multiply
    MATH_FUNC Matrix3x3 operator*(const Matrix3x3& B) const {
        Matrix3x3 C = zeros();
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                for (int k = 0; k < 3; k++)
                    C(r,c) += (*this)(r,k) * B(k,c);
        return C;
    }

    MATH_FUNC Matrix3x3 transpose() const {
        Matrix3x3 T;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                T(r,c) = (*this)(c,r);
        return T;
    }

    MATH_FUNC float determinant() const {
        return m[0]*(m[4]*m[8]-m[5]*m[7])
             - m[1]*(m[3]*m[8]-m[5]*m[6])
             + m[2]*(m[3]*m[7]-m[4]*m[6]);
    }

    MATH_FUNC Matrix3x3 inverse() const {
        float det = determinant();
        float inv_det = 1.0f / det;
        Matrix3x3 inv;
        inv.m[0] = (m[4]*m[8] - m[5]*m[7]) * inv_det;
        inv.m[1] = (m[2]*m[7] - m[1]*m[8]) * inv_det;
        inv.m[2] = (m[1]*m[5] - m[2]*m[4]) * inv_det;
        inv.m[3] = (m[5]*m[6] - m[3]*m[8]) * inv_det;
        inv.m[4] = (m[0]*m[8] - m[2]*m[6]) * inv_det;
        inv.m[5] = (m[2]*m[3] - m[0]*m[5]) * inv_det;
        inv.m[6] = (m[3]*m[7] - m[4]*m[6]) * inv_det;
        inv.m[7] = (m[1]*m[6] - m[0]*m[7]) * inv_det;
        inv.m[8] = (m[0]*m[4] - m[1]*m[3]) * inv_det;
        return inv;
    }
};

// --- Angle-axis <-> Rotation matrix -----------------------------------------

// Rodrigues: angle-axis (3-vector, magnitude = angle) -> rotation matrix
MATH_FUNC Matrix3x3 angleAxisToRotation(float3 aa) {
    float angle = length(aa);
    if (angle < 1e-8f) {
        return Matrix3x3::identity();
    }
    float3 axis = aa * (1.0f / angle);
    float c = cosf(angle);
    float s = sinf(angle);
    float t = 1.0f - c;

    Matrix3x3 R;
    R(0,0) = t*axis.x*axis.x + c;
    R(0,1) = t*axis.x*axis.y - s*axis.z;
    R(0,2) = t*axis.x*axis.z + s*axis.y;
    R(1,0) = t*axis.x*axis.y + s*axis.z;
    R(1,1) = t*axis.y*axis.y + c;
    R(1,2) = t*axis.y*axis.z - s*axis.x;
    R(2,0) = t*axis.x*axis.z - s*axis.y;
    R(2,1) = t*axis.y*axis.z + s*axis.x;
    R(2,2) = t*axis.z*axis.z + c;
    return R;
}

// Rotation matrix -> angle-axis (3-vector, magnitude = angle)
MATH_FUNC float3 rotationToAngleAxis(const Matrix3x3& R) {
    float cos_angle = (R(0,0) + R(1,1) + R(2,2) - 1.0f) * 0.5f;
    cos_angle = fminf(fmaxf(cos_angle, -1.0f), 1.0f);
    float angle = acosf(cos_angle);

    if (angle < 1e-8f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    float3 axis;
    if (angle < 3.14159f - 1e-6f) {
        axis = make_float3(
            R(2,1) - R(1,2),
            R(0,2) - R(2,0),
            R(1,0) - R(0,1)
        );
        axis = normalize(axis);
    } else {
        // Near pi: use eigenvector of R corresponding to eigenvalue 1
        int col = 0;
        if (R(1,1) > R(0,0)) col = 1;
        if (R(2,2) > R(col,col)) col = 2;
        float3 v = make_float3(R(0,col) + (col==0?1:0),
                                R(1,col) + (col==1?1:0),
                                R(2,col) + (col==2?1:0));
        axis = normalize(v);
    }
    return axis * angle;
}

// --- Bilinear interpolation -------------------------------------------------

// Sample a float image at fractional coordinates (x, y)
// image is row-major, width x height
MATH_FUNC float bilinearSample(const float* image, int width, int height,
                                float x, float y) {
    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // Clamp to image bounds
    x0 = max(0, min(x0, width - 1));
    x1 = max(0, min(x1, width - 1));
    y0 = max(0, min(y0, height - 1));
    y1 = max(0, min(y1, height - 1));

    float fx = x - floorf(x);
    float fy = y - floorf(y);

    float v00 = image[y0 * width + x0];
    float v10 = image[y0 * width + x1];
    float v01 = image[y1 * width + x0];
    float v11 = image[y1 * width + x1];

    return (1-fx)*(1-fy)*v00 + fx*(1-fy)*v10 + (1-fx)*fy*v01 + fx*fy*v11;
}
