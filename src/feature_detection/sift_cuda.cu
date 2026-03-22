// sift_cuda.cu — Custom CUDA SIFT + OpenCV fallback
//
// Step 3b: Full GPU SIFT pipeline
//   1. Gaussian scale-space (separable blur, shared memory)
//   2. Difference of Gaussians (DoG)
//   3. Extrema detection with sub-pixel refinement
//   4. Orientation assignment (36-bin gradient histogram)
//   5. Descriptor computation (4×4×8 = 128-D, trilinear interp)

#include "feature_detection/sift_cuda.h"
#include "utils/cuda_utils.cuh"
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <cstring>
#include <vector>

// ========================================================================
// Constants
// ========================================================================
namespace {
constexpr int    SIFT_S               = 3;          // scales per octave
constexpr int    SIFT_GAUSS_PER_OCT   = SIFT_S + 3; // 6 Gaussian images
constexpr int    SIFT_DOG_PER_OCT     = SIFT_S + 2;  // 5 DoG images
constexpr float  SIFT_SIGMA0          = 1.6f;
constexpr float  SIFT_CONTRAST_THR    = 0.03f;
constexpr float  SIFT_EDGE_THR        = 10.0f;
constexpr float  SIFT_INIT_BLUR       = 0.5f;
constexpr int    SIFT_MAX_INTERP      = 5;
constexpr int    SIFT_ORI_NBINS       = 36;
constexpr float  SIFT_ORI_PEAK_RATIO  = 0.8f;
constexpr float  SIFT_ORI_RADIUS_FCTR = 3.0f;
constexpr float  SIFT_ORI_SIG_FCTR    = 1.5f;
constexpr int    SIFT_DESCR_W         = 4;
constexpr int    SIFT_DESCR_BINS      = 8;
constexpr float  SIFT_DESCR_MAG_THR   = 0.2f;
constexpr float  SIFT_DESCR_SCL       = 3.0f;
constexpr int    SIFT_BORDER          = 5;
constexpr int    MAX_KRADIUS           = 20;
constexpr int    MAX_KLEN              = 2 * MAX_KRADIUS + 1;
constexpr int    MAX_RAW_KP            = 16384;
constexpr int    MAX_ORI_KP            = 24576;
constexpr float  MY_PI                 = 3.14159265358979323846f;
constexpr float  MY_2PI                = 2.0f * MY_PI;
} // anon

// ========================================================================
// GPU keypoint structs (internal)
// ========================================================================
struct RawKeypoint {
    float x, y;        // octave-image coordinates
    float sigma;       // absolute sigma = sigma0 * 2^(oct + s/S)
    float response;    // |DoG| interpolated value
    int   octave;
    int   layer;       // DoG layer [1..S]
};

struct OrientedKeypoint {
    float x, y;        // octave-image coordinates
    float sigma;       // absolute sigma
    float angle;       // radians [0, 2π)
    float response;
    int   octave;
    int   layer;
};

// ========================================================================
// Constant memory — Gaussian kernel weights
// ========================================================================
__constant__ float c_gauss[MAX_KLEN];

// ========================================================================
// Kernel: Separable Gaussian Blur — Horizontal
// ========================================================================
__global__ void gaussian_blur_h_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int radius, int width, int height)
{
    extern __shared__ float smem[];
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int smem_w = (int)blockDim.x + 2 * radius;
    const int row = threadIdx.y * smem_w;

    for (int i = threadIdx.x; i < smem_w; i += blockDim.x) {
        int gx = (int)(blockIdx.x * blockDim.x) - radius + i;
        gx = max(0, min(gx, width - 1));
        smem[row + i] = (y < height) ? input[y * width + gx] : 0.0f;
    }
    __syncthreads();

    if (x < width && y < height) {
        float sum = 0.0f;
        const int base = row + threadIdx.x + radius;
        for (int k = -radius; k <= radius; k++)
            sum += smem[base + k] * c_gauss[k + radius];
        output[y * width + x] = sum;
    }
}

// ========================================================================
// Kernel: Separable Gaussian Blur — Vertical
// ========================================================================
__global__ void gaussian_blur_v_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int radius, int width, int height)
{
    extern __shared__ float smem[];
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int smem_h = (int)blockDim.y + 2 * radius;
    const int stride = blockDim.x;

    for (int i = threadIdx.y; i < smem_h; i += blockDim.y) {
        int gy = (int)(blockIdx.y * blockDim.y) - radius + i;
        gy = max(0, min(gy, height - 1));
        smem[i * stride + threadIdx.x] = (x < width) ? input[gy * width + x] : 0.0f;
    }
    __syncthreads();

    if (x < width && y < height) {
        float sum = 0.0f;
        const int br = threadIdx.y + radius;
        for (int k = -radius; k <= radius; k++)
            sum += smem[(br + k) * stride + threadIdx.x] * c_gauss[k + radius];
        output[y * width + x] = sum;
    }
}

// ========================================================================
// Kernel: Difference of Gaussians
// ========================================================================
__global__ void dog_subtract_kernel(
    const float* __restrict__ upper,
    const float* __restrict__ lower,
    float* __restrict__ dog,
    int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        dog[idx] = upper[idx] - lower[idx];
    }
}

// ========================================================================
// Kernel: Extrema Detection with Sub-pixel Refinement
// ========================================================================
__global__ void detect_extrema_kernel(
    const float* __restrict__ dogs,
    RawKeypoint* __restrict__ keypoints,
    int* __restrict__ kp_count,
    int max_kp,
    float contrast_thr, float edge_thr,
    int octave, int width, int height, int ndog)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < SIFT_BORDER || x >= width - SIFT_BORDER ||
        y < SIFT_BORDER || y >= height - SIFT_BORDER)
        return;

    const int img_sz = width * height;
    const float prelim = 0.5f * contrast_thr / SIFT_S;
    const float edge_sq = (edge_thr + 1.0f) * (edge_thr + 1.0f) / edge_thr;

    for (int s = 1; s < ndog - 1; s++) {
        const float* prev = dogs + (s - 1) * img_sz;
        const float* curr = dogs + s * img_sz;
        const float* next = dogs + (s + 1) * img_sz;
        float val = curr[y * width + x];

        if (fabsf(val) < prelim) continue;

        // 3×3×3 extremum check
        bool is_max = true, is_min = true;
        for (int dy = -1; dy <= 1 && (is_max || is_min); dy++) {
            for (int dx = -1; dx <= 1 && (is_max || is_min); dx++) {
                int idx = (y + dy) * width + (x + dx);
                float pv = prev[idx], cv2 = curr[idx], nv = next[idx];
                if (dx != 0 || dy != 0) {
                    if (cv2 >= val) is_max = false;
                    if (cv2 <= val) is_min = false;
                }
                if (pv >= val) is_max = false;
                if (pv <= val) is_min = false;
                if (nv >= val) is_max = false;
                if (nv <= val) is_min = false;
            }
        }
        if (!is_max && !is_min) continue;

        // Sub-pixel refinement (Taylor expansion)
        int xi = x, yi = y, si = s;
        for (int iter = 0; iter < SIFT_MAX_INTERP; iter++) {
            if (xi < SIFT_BORDER || xi >= width - SIFT_BORDER ||
                yi < SIFT_BORDER || yi >= height - SIFT_BORDER ||
                si < 1 || si >= ndog - 1) break;

            const float* pm = dogs + (si - 1) * img_sz;
            const float* pc = dogs + si * img_sz;
            const float* pn = dogs + (si + 1) * img_sz;
            float v0 = pc[yi * width + xi];

            // First derivatives
            float dx_v = 0.5f * (pc[yi * width + xi + 1] - pc[yi * width + xi - 1]);
            float dy_v = 0.5f * (pc[(yi+1) * width + xi] - pc[(yi-1) * width + xi]);
            float ds_v = 0.5f * (pn[yi * width + xi]     - pm[yi * width + xi]);

            // Second derivatives
            float dxx = pc[yi*width+xi+1] + pc[yi*width+xi-1] - 2.0f*v0;
            float dyy = pc[(yi+1)*width+xi] + pc[(yi-1)*width+xi] - 2.0f*v0;
            float dss = pn[yi*width+xi] + pm[yi*width+xi] - 2.0f*v0;
            float dxy = 0.25f*(pc[(yi+1)*width+xi+1] - pc[(yi+1)*width+xi-1]
                             - pc[(yi-1)*width+xi+1] + pc[(yi-1)*width+xi-1]);
            float dxs = 0.25f*(pn[yi*width+xi+1] - pn[yi*width+xi-1]
                             - pm[yi*width+xi+1] + pm[yi*width+xi-1]);
            float dys = 0.25f*(pn[(yi+1)*width+xi] - pn[(yi-1)*width+xi]
                             - pm[(yi+1)*width+xi] + pm[(yi-1)*width+xi]);

            // 3x3 Hessian solve via Cramer's rule
            float det = dxx*(dyy*dss - dys*dys)
                      - dxy*(dxy*dss - dys*dxs)
                      + dxs*(dxy*dys - dyy*dxs);
            if (fabsf(det) < 1e-10f) break;
            float id = 1.0f / det;

            float ox = -((dyy*dss-dys*dys)*dx_v - (dxy*dss-dys*dxs)*dy_v + (dxy*dys-dyy*dxs)*ds_v)*id;
            float oy = -(-(dxy*dss-dxs*dys)*dx_v + (dxx*dss-dxs*dxs)*dy_v - (dxx*dys-dxy*dxs)*ds_v)*id;
            float os = -((dxy*dys-dxs*dyy)*dx_v - (dxx*dys-dxs*dxy)*dy_v + (dxx*dyy-dxy*dxy)*ds_v)*id;

            if (fabsf(ox) < 0.5f && fabsf(oy) < 0.5f && fabsf(os) < 0.5f) {
                float xf = (float)xi + ox;
                float yf = (float)yi + oy;
                float sf = (float)si + os;

                // Interpolated contrast
                float d_hat = v0 + 0.5f*(dx_v*ox + dy_v*oy + ds_v*os);
                if (fabsf(d_hat) < contrast_thr / SIFT_S) break;

                // Edge response (Harris-like)
                float tr = dxx + dyy;
                float det2 = dxx*dyy - dxy*dxy;
                if (det2 <= 0.0f || tr*tr/det2 > edge_sq) break;

                int idx2 = atomicAdd(kp_count, 1);
                if (idx2 < max_kp) {
                    keypoints[idx2].x        = xf;
                    keypoints[idx2].y        = yf;
                    keypoints[idx2].sigma    = SIFT_SIGMA0 * powf(2.0f, (float)octave + sf/(float)SIFT_S);
                    keypoints[idx2].response = fabsf(d_hat);
                    keypoints[idx2].octave   = octave;
                    keypoints[idx2].layer    = si;
                }
                break;
            }
            xi += (int)roundf(ox);
            yi += (int)roundf(oy);
            si += (int)roundf(os);
        }
    }
}

// ========================================================================
// Kernel: Orientation Assignment  (one block per keypoint, 128 threads)
// ========================================================================
__global__ void compute_orientation_kernel(
    const float* __restrict__ gauss,
    const RawKeypoint* __restrict__ raw_kps, int num_raw,
    OrientedKeypoint* __restrict__ ori_kps,
    int* __restrict__ ori_count, int max_ori,
    int width, int height, int octave)
{
    int kp_idx = blockIdx.x;
    if (kp_idx >= num_raw) return;

    __shared__ float hist[SIFT_ORI_NBINS];
    __shared__ float smooth[SIFT_ORI_NBINS];
    __shared__ RawKeypoint kp;
    __shared__ float max_val;

    if (threadIdx.x == 0) kp = raw_kps[kp_idx];
    if (threadIdx.x < SIFT_ORI_NBINS) hist[threadIdx.x] = 0.0f;
    __syncthreads();

    const float* img = gauss + kp.layer * width * height;
    int kx = (int)roundf(kp.x);
    int ky = (int)roundf(kp.y);
    float sigma_oct = kp.sigma / (float)(1 << octave);
    float sig_w = SIFT_ORI_SIG_FCTR * sigma_oct;
    int radius = (int)roundf(SIFT_ORI_RADIUS_FCTR * sig_w);
    if (radius < 1) radius = 1;
    float wt_scale = -0.5f / (sig_w * sig_w);
    float bin_scale = (float)SIFT_ORI_NBINS / MY_2PI;
    int patch_n = (2*radius+1)*(2*radius+1);

    // Each thread processes a subset of the patch
    for (int i = threadIdx.x; i < patch_n; i += blockDim.x) {
        int dy = i / (2*radius+1) - radius;
        int dx = i % (2*radius+1) - radius;
        int px = kx + dx, py = ky + dy;
        if (px < 1 || px >= width-1 || py < 1 || py >= height-1) continue;

        float gx = img[py*width+px+1] - img[py*width+px-1];
        float gy = img[(py+1)*width+px] - img[(py-1)*width+px];
        float mag = sqrtf(gx*gx + gy*gy);
        float ori = atan2f(gy, gx);
        if (ori < 0.0f) ori += MY_2PI;

        float w = expf(wt_scale * (float)(dx*dx + dy*dy));
        int bin = ((int)floorf(ori * bin_scale)) % SIFT_ORI_NBINS;
        if (bin < 0) bin += SIFT_ORI_NBINS;
        atomicAdd(&hist[bin], mag * w);
    }
    __syncthreads();

    // Smooth histogram (6 passes of [1,1,1]/3)
    for (int it = 0; it < 6; it++) {
        __syncthreads();
        if (threadIdx.x < SIFT_ORI_NBINS) {
            int p = (threadIdx.x + SIFT_ORI_NBINS - 1) % SIFT_ORI_NBINS;
            int n = (threadIdx.x + 1) % SIFT_ORI_NBINS;
            smooth[threadIdx.x] = (hist[p] + hist[threadIdx.x] + hist[n]) / 3.0f;
        }
        __syncthreads();
        if (threadIdx.x < SIFT_ORI_NBINS)
            hist[threadIdx.x] = smooth[threadIdx.x];
    }
    __syncthreads();

    // Find max
    if (threadIdx.x == 0) {
        max_val = 0.0f;
        for (int i = 0; i < SIFT_ORI_NBINS; i++)
            if (hist[i] > max_val) max_val = hist[i];
    }
    __syncthreads();

    // Detect peaks — each of first 36 threads checks one bin
    if (threadIdx.x < SIFT_ORI_NBINS && max_val > 0.0f) {
        int b = threadIdx.x;
        int p = (b + SIFT_ORI_NBINS - 1) % SIFT_ORI_NBINS;
        int n = (b + 1) % SIFT_ORI_NBINS;
        if (hist[b] > SIFT_ORI_PEAK_RATIO * max_val &&
            hist[b] > hist[p] && hist[b] >= hist[n]) {
            // Parabolic interpolation
            float lv = hist[p], cv = hist[b], rv = hist[n];
            float denom = lv - 2.0f*cv + rv;
            float offset = (fabsf(denom) > 1e-10f) ? 0.5f*(lv - rv)/denom : 0.0f;
            float angle = ((float)b + offset + 0.5f) * (MY_2PI / (float)SIFT_ORI_NBINS);
            if (angle < 0.0f)    angle += MY_2PI;
            if (angle >= MY_2PI) angle -= MY_2PI;

            int idx = atomicAdd(ori_count, 1);
            if (idx < max_ori) {
                ori_kps[idx].x        = kp.x;
                ori_kps[idx].y        = kp.y;
                ori_kps[idx].sigma    = kp.sigma;
                ori_kps[idx].angle    = angle;
                ori_kps[idx].response = kp.response;
                ori_kps[idx].octave   = kp.octave;
                ori_kps[idx].layer    = kp.layer;
            }
        }
    }
}

// ========================================================================
// Kernel: Descriptor Computation (128-D, one block per keypoint, 256 thr)
// ========================================================================
__global__ void compute_descriptor_kernel(
    const float* __restrict__ gauss,
    const OrientedKeypoint* __restrict__ kps, int num_kps,
    float* __restrict__ descriptors,
    int width, int height, int octave)
{
    int kp_idx = blockIdx.x;
    if (kp_idx >= num_kps) return;

    __shared__ float desc[128];
    __shared__ OrientedKeypoint kp;
    __shared__ float norm_sq;

    // Init
    for (int i = threadIdx.x; i < 128; i += blockDim.x)
        desc[i] = 0.0f;
    if (threadIdx.x == 0) kp = kps[kp_idx];
    __syncthreads();

    const float scale_f = (float)(1 << octave);
    const float oct_x = kp.x;
    const float oct_y = kp.y;
    const float sigma_oct = kp.sigma / scale_f;
    const float* img = gauss + kp.layer * width * height;
    const float cos_t = cosf(kp.angle);
    const float sin_t = sinf(kp.angle);
    const float bins_per_rad = (float)SIFT_DESCR_BINS / MY_2PI;
    const float hist_w = SIFT_DESCR_SCL * sigma_oct;
    int radius = (int)roundf(hist_w * 1.4142135f * ((float)SIFT_DESCR_W + 1) * 0.5f);
    radius = min(radius, min(width, height) / 2 - 1);
    if (radius < 1) radius = 1;
    const float inv_bw2 = -0.5f / ((float)SIFT_DESCR_W * 0.5f * (float)SIFT_DESCR_W * 0.5f);
    const int patch_n = (2*radius+1)*(2*radius+1);

    // Each thread processes a subset of the patch
    for (int i = threadIdx.x; i < patch_n; i += blockDim.x) {
        int dy = i / (2*radius+1) - radius;
        int dx = i % (2*radius+1) - radius;

        float rot_x = ( cos_t*dx + sin_t*dy) / hist_w;
        float rot_y = (-sin_t*dx + cos_t*dy) / hist_w;

        float xbin = rot_x + (float)SIFT_DESCR_W * 0.5f - 0.5f;
        float ybin = rot_y + (float)SIFT_DESCR_W * 0.5f - 0.5f;
        if (xbin <= -1.0f || xbin >= (float)SIFT_DESCR_W ||
            ybin <= -1.0f || ybin >= (float)SIFT_DESCR_W) continue;

        int px = (int)roundf(oct_x) + dx;
        int py = (int)roundf(oct_y) + dy;
        if (px < 1 || px >= width-1 || py < 1 || py >= height-1) continue;

        float gx = img[py*width+px+1] - img[py*width+px-1];
        float gy = img[(py+1)*width+px] - img[(py-1)*width+px];
        float mag = sqrtf(gx*gx + gy*gy);
        float ori = atan2f(gy, gx) - kp.angle;
        if (ori < 0.0f)    ori += MY_2PI;
        if (ori >= MY_2PI)  ori -= MY_2PI;
        float obin = ori * bins_per_rad;

        float w = expf(inv_bw2 * (rot_x*rot_x + rot_y*rot_y));
        float contrib = mag * w;

        // Trilinear interpolation
        int xi0 = (int)floorf(xbin), yi0 = (int)floorf(ybin), oi0 = (int)floorf(obin);
        float xf = xbin - xi0, yf2 = ybin - yi0, of = obin - oi0;

        for (int yy = 0; yy <= 1; yy++) {
            int ybi = yi0 + yy;
            if (ybi < 0 || ybi >= SIFT_DESCR_W) continue;
            float yw = yy ? yf2 : (1.0f - yf2);
            for (int xx = 0; xx <= 1; xx++) {
                int xbi = xi0 + xx;
                if (xbi < 0 || xbi >= SIFT_DESCR_W) continue;
                float xw = xx ? xf : (1.0f - xf);
                for (int oo = 0; oo <= 1; oo++) {
                    int obi = (oi0 + oo) % SIFT_DESCR_BINS;
                    if (obi < 0) obi += SIFT_DESCR_BINS;
                    float ow = oo ? of : (1.0f - of);
                    int didx = (ybi * SIFT_DESCR_W + xbi) * SIFT_DESCR_BINS + obi;
                    atomicAdd(&desc[didx], contrib * xw * yw * ow);
                }
            }
        }
    }
    __syncthreads();

    // ---- Normalize → clamp → re-normalize (all threads participate in syncs) ----
    if (threadIdx.x == 0) norm_sq = 0.0f;
    __syncthreads();
    if (threadIdx.x < 128) atomicAdd(&norm_sq, desc[threadIdx.x] * desc[threadIdx.x]);
    __syncthreads();

    if (threadIdx.x < 128) {
        float inv_n = 1.0f / (sqrtf(norm_sq) + 1e-7f);
        desc[threadIdx.x] *= inv_n;
        if (desc[threadIdx.x] > SIFT_DESCR_MAG_THR)
            desc[threadIdx.x] = SIFT_DESCR_MAG_THR;
    }
    __syncthreads();

    if (threadIdx.x == 0) norm_sq = 0.0f;
    __syncthreads();
    if (threadIdx.x < 128) atomicAdd(&norm_sq, desc[threadIdx.x] * desc[threadIdx.x]);
    __syncthreads();

    if (threadIdx.x < 128) {
        float inv_n = 1.0f / (sqrtf(norm_sq) + 1e-7f);
        float val = fminf(desc[threadIdx.x] * inv_n * 512.0f, 255.0f);
        descriptors[kp_idx * 128 + threadIdx.x] = val;
    }
}

// ========================================================================
// Kernel: Downsample 2× (take every other pixel)
// ========================================================================
__global__ void downsample_2x_kernel(
    const float* __restrict__ input, float* __restrict__ output,
    int in_w, int in_h, int out_w, int out_h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < out_w && y < out_h)
        output[y * out_w + x] = input[(2*y) * in_w + (2*x)];
}

// ========================================================================
// Host helpers
// ========================================================================
static void computeGaussKernel(float sigma, std::vector<float>& kernel, int& radius) {
    radius = std::min((int)ceilf(3.0f * sigma), MAX_KRADIUS);
    if (radius < 1) radius = 1;
    int len = 2 * radius + 1;
    kernel.resize(len);
    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        float x = (float)(i - radius);
        kernel[i] = expf(-0.5f * x * x / (sigma * sigma));
        sum += kernel[i];
    }
    for (int i = 0; i < len; i++) kernel[i] /= sum;
}

static void gaussianBlur(const float* d_in, float* d_out, float* d_tmp,
                          float sigma, int w, int h) {
    std::vector<float> kern;
    int radius;
    computeGaussKernel(sigma, kern, radius);
    CUDA_CHECK(cudaMemcpyToSymbol(c_gauss, kern.data(), kern.size() * sizeof(float)));

    dim3 blk(32, 8);
    dim3 grd((w + blk.x - 1) / blk.x, (h + blk.y - 1) / blk.y);

    size_t smem_h = blk.y * (blk.x + 2 * radius) * sizeof(float);
    gaussian_blur_h_kernel<<<grd, blk, smem_h>>>(d_in, d_tmp, radius, w, h);
    CUDA_CHECK(cudaGetLastError());

    size_t smem_v = (blk.y + 2 * radius) * blk.x * sizeof(float);
    gaussian_blur_v_kernel<<<grd, blk, smem_v>>>(d_tmp, d_out, radius, w, h);
    CUDA_CHECK(cudaGetLastError());
}

// ========================================================================
// detectSIFT_CUDA — full custom GPU pipeline
// ========================================================================
SIFTFeatures detectSIFT_CUDA(const ImageData& image, int max_features) {
    SIFTFeatures result;
    result.image_id = image.id;
    result.d_descriptors = nullptr;
    result.count = 0;

    if (image.gray.empty()) {
        fprintf(stderr, "Warning: Image %d has no grayscale data.\n", image.id);
        return result;
    }

    const int orig_w = image.width;
    const int orig_h = image.height;
    if (orig_w < 2*SIFT_BORDER+1 || orig_h < 2*SIFT_BORDER+1) return result;

    // Number of octaves
    int num_oct = std::max(1, (int)floorf(log2f((float)std::min(orig_w, orig_h))) - 3);
    num_oct = std::min(num_oct, 4);

    // Upload image
    cv::Mat gray_f;
    if (image.gray.type() != CV_32F) image.gray.convertTo(gray_f, CV_32F, 1.0/255.0);
    else gray_f = image.gray;
    if (!gray_f.isContinuous()) gray_f = gray_f.clone();

    DeviceBuffer<float> d_image;
    d_image.upload((const float*)gray_f.data, orig_w * orig_h);

    // Temp buffer (max size)
    DeviceBuffer<float> d_tmp;
    d_tmp.alloc(orig_w * orig_h);

    // Initial blur: assumed σ=0.5 → σ_0
    float sigma_init = sqrtf(std::max(SIFT_SIGMA0*SIFT_SIGMA0 - SIFT_INIT_BLUR*SIFT_INIT_BLUR, 0.01f));
    DeviceBuffer<float> d_base_bufs[2];
    d_base_bufs[0].alloc(orig_w * orig_h);
    gaussianBlur(d_image.ptr, d_base_bufs[0].ptr, d_tmp.ptr, sigma_init, orig_w, orig_h);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Pre-allocate scale space for largest octave (reuse across octaves)
    DeviceBuffer<float> d_gauss, d_dogs;
    d_gauss.alloc((size_t)SIFT_GAUSS_PER_OCT * orig_w * orig_h);
    d_dogs.alloc((size_t)SIFT_DOG_PER_OCT * orig_w * orig_h);

    // Keypoint device buffers
    DeviceBuffer<RawKeypoint> d_raw_kps;  d_raw_kps.alloc(MAX_RAW_KP);
    DeviceBuffer<int> d_raw_cnt;           d_raw_cnt.alloc(1);
    DeviceBuffer<OrientedKeypoint> d_ori_kps; d_ori_kps.alloc(MAX_ORI_KP);
    DeviceBuffer<int> d_ori_cnt;              d_ori_cnt.alloc(1);

    // Collect results across octaves
    std::vector<OrientedKeypoint> all_kps;
    std::vector<float> all_desc;  // flat Nx128

    int cur_base = 0;
    int oct_w = orig_w, oct_h = orig_h;

    for (int oct = 0; oct < num_oct; oct++) {
        int img_sz = oct_w * oct_h;
        float* base_ptr = d_base_bufs[cur_base].ptr;

        // Ensure tmp is large enough
        if ((int)d_tmp.count < img_sz) d_tmp.alloc(img_sz);

        // Scale 0 = base image
        CUDA_CHECK(cudaMemcpy(d_gauss.ptr, base_ptr, img_sz * sizeof(float), cudaMemcpyDeviceToDevice));

        // Incremental Gaussian blurs for scales 1..(SIFT_GAUSS_PER_OCT-1)
        float k = powf(2.0f, 1.0f / (float)SIFT_S);
        for (int s = 1; s < SIFT_GAUSS_PER_OCT; s++) {
            float sig_prev = SIFT_SIGMA0 * powf(k, (float)(s - 1));
            float sig_curr = SIFT_SIGMA0 * powf(k, (float)s);
            float sig_incr = sqrtf(sig_curr*sig_curr - sig_prev*sig_prev);
            gaussianBlur(
                d_gauss.ptr + (size_t)(s-1) * img_sz,
                d_gauss.ptr + (size_t)s * img_sz,
                d_tmp.ptr, sig_incr, oct_w, oct_h);
        }

        // DoG
        dim3 blk2(32, 8);
        dim3 grd2((oct_w+31)/32, (oct_h+7)/8);
        for (int s = 0; s < SIFT_DOG_PER_OCT; s++) {
            dog_subtract_kernel<<<grd2, blk2>>>(
                d_gauss.ptr + (size_t)(s+1)*img_sz,
                d_gauss.ptr + (size_t)s*img_sz,
                d_dogs.ptr  + (size_t)s*img_sz,
                oct_w, oct_h);
            CUDA_CHECK(cudaGetLastError());
        }

        // Extrema detection
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_raw_cnt.ptr, &zero, sizeof(int), cudaMemcpyHostToDevice));
        detect_extrema_kernel<<<grd2, blk2>>>(
            d_dogs.ptr, d_raw_kps.ptr, d_raw_cnt.ptr,
            MAX_RAW_KP, SIFT_CONTRAST_THR, SIFT_EDGE_THR,
            oct, oct_w, oct_h, SIFT_DOG_PER_OCT);
        CUDA_CHECK(cudaGetLastError());

        int raw_cnt = 0;
        CUDA_CHECK(cudaMemcpy(&raw_cnt, d_raw_cnt.ptr, sizeof(int), cudaMemcpyDeviceToHost));
        raw_cnt = std::min(raw_cnt, MAX_RAW_KP);

        if (raw_cnt > 0) {
            // Orientation
            CUDA_CHECK(cudaMemcpy(d_ori_cnt.ptr, &zero, sizeof(int), cudaMemcpyHostToDevice));
            compute_orientation_kernel<<<raw_cnt, 128>>>(
                d_gauss.ptr, d_raw_kps.ptr, raw_cnt,
                d_ori_kps.ptr, d_ori_cnt.ptr, MAX_ORI_KP,
                oct_w, oct_h, oct);
            CUDA_CHECK(cudaGetLastError());

            int ori_cnt = 0;
            CUDA_CHECK(cudaMemcpy(&ori_cnt, d_ori_cnt.ptr, sizeof(int), cudaMemcpyDeviceToHost));
            ori_cnt = std::min(ori_cnt, MAX_ORI_KP);

            if (ori_cnt > 0) {
                // Descriptors
                DeviceBuffer<float> d_desc;
                d_desc.alloc((size_t)ori_cnt * 128);
                compute_descriptor_kernel<<<ori_cnt, 256>>>(
                    d_gauss.ptr, d_ori_kps.ptr, ori_cnt,
                    d_desc.ptr, oct_w, oct_h, oct);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());

                // Download
                std::vector<OrientedKeypoint> oct_kps(ori_cnt);
                CUDA_CHECK(cudaMemcpy(oct_kps.data(), d_ori_kps.ptr,
                    ori_cnt * sizeof(OrientedKeypoint), cudaMemcpyDeviceToHost));

                size_t prev_desc_sz = all_desc.size();
                all_desc.resize(prev_desc_sz + (size_t)ori_cnt * 128);
                d_desc.download(&all_desc[prev_desc_sz], (size_t)ori_cnt * 128);

                all_kps.insert(all_kps.end(), oct_kps.begin(), oct_kps.end());
            }
        }

        // Prepare next octave: downsample image[S]
        if (oct < num_oct - 1) {
            int nw = oct_w / 2, nh = oct_h / 2;
            if (nw < 2*SIFT_BORDER+1 || nh < 2*SIFT_BORDER+1) break;
            int next = 1 - cur_base;
            d_base_bufs[next].alloc(nw * nh);
            dim3 ds_grd((nw+31)/32, (nh+7)/8);
            downsample_2x_kernel<<<ds_grd, blk2>>>(
                d_gauss.ptr + (size_t)SIFT_S * img_sz,
                d_base_bufs[next].ptr,
                oct_w, oct_h, nw, nh);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            cur_base = next;
            oct_w = nw;
            oct_h = nh;
        }
    }

    // ---- Convert to OpenCV format ----
    int total = (int)all_kps.size();
    if (total == 0) return result;

    // Limit to max_features by response
    if (max_features > 0 && total > max_features) {
        std::vector<int> idx(total);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            return all_kps[a].response > all_kps[b].response;
        });
        idx.resize(max_features);
        std::sort(idx.begin(), idx.end());

        std::vector<OrientedKeypoint> top_kps(max_features);
        std::vector<float> top_desc(max_features * 128);
        for (int i = 0; i < max_features; i++) {
            top_kps[i] = all_kps[idx[i]];
            memcpy(&top_desc[i*128], &all_desc[idx[i]*128], 128*sizeof(float));
        }
        all_kps = std::move(top_kps);
        all_desc = std::move(top_desc);
        total = max_features;
    }

    result.keypoints.resize(total);
    for (int i = 0; i < total; i++) {
        const auto& kp = all_kps[i];
        float sf = (float)(1 << kp.octave);
        cv::KeyPoint& cvkp = result.keypoints[i];
        cvkp.pt.x      = kp.x * sf;
        cvkp.pt.y      = kp.y * sf;
        cvkp.size       = 2.0f * kp.sigma;
        cvkp.angle      = kp.angle * 180.0f / MY_PI;
        cvkp.response   = kp.response;
        cvkp.octave     = kp.octave;
        cvkp.class_id   = -1;
    }

    result.descriptors = cv::Mat(total, 128, CV_32F);
    if (total > 0)
        memcpy(result.descriptors.data, all_desc.data(), (size_t)total * 128 * sizeof(float));

    result.count = total;
    return result;
}

// ========================================================================
// detectSIFT_OpenCV — CPU fallback (original Step 3a wrapper)
// ========================================================================
SIFTFeatures detectSIFT_OpenCV(const ImageData& image, int max_features) {
    SIFTFeatures features;
    features.image_id = image.id;
    features.d_descriptors = nullptr;
    features.count = 0;

    if (image.gray.empty()) {
        fprintf(stderr, "Warning: Image %d has no grayscale data.\n", image.id);
        return features;
    }

    auto sift = cv::SIFT::create(max_features, 3, 0.04, 10.0, 1.6);

    cv::Mat gray_u8;
    if (image.gray.type() == CV_32F)
        image.gray.convertTo(gray_u8, CV_8U, 255.0);
    else
        gray_u8 = image.gray;

    sift->detectAndCompute(gray_u8, cv::noArray(), features.keypoints, features.descriptors);
    features.count = static_cast<int>(features.keypoints.size());
    return features;
}

// ========================================================================
// Public API — default routes to CUDA
// ========================================================================
SIFTFeatures detectSIFT(const ImageData& image, int max_features) {
    return detectSIFT_CUDA(image, max_features);
}

std::vector<SIFTFeatures> detectAllFeatures(
    const std::vector<ImageData>& images,
    int max_features, bool verbose)
{
    std::vector<SIFTFeatures> all;
    all.reserve(images.size());
    for (const auto& img : images) {
        SIFTFeatures feat = detectSIFT(img, max_features);
        if (verbose)
            printf("  [%02d] %-30s  %d features (CUDA)\n",
                   img.id, img.filename.c_str(), feat.count);
        all.push_back(std::move(feat));
    }
    if (verbose) {
        int total = 0;
        for (const auto& f : all) total += f.count;
        printf("Total features: %d across %zu images (avg %.0f/image)\n",
               total, all.size(),
               all.empty() ? 0.0 : (double)total / all.size());
    }
    return all;
}
