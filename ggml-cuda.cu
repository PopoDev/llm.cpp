#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdint.h>
#include <stdio.h>
#include <atomic>
#include <assert.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include "ggml-cuda.h"
#include "ggml.h"

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

#define CUDA_CHECK(err)                                                                 \
    do {                                                                                \
        cudaError_t err_ = (err);                                                       \
        if (err_ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %d at %s:%d: %s\n", err_, __FILE__, __LINE__,   \
                cudaGetErrorString(err_));                                              \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)

#if CUDART_VERSION >= 12
#define CUBLAS_CHECK(err)                                                               \
    do {                                                                                \
        cublasStatus_t err_ = (err);                                                    \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                            \
            fprintf(stderr, "\ncuBLAS error %d at %s:%d: %s\n",                         \
                    err_, __FILE__, __LINE__, cublasGetStatusString(err_));             \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)
#else
#define CUBLAS_CHECK(err)                                                               \
    do {                                                                                \
        cublasStatus_t err_ = (err);                                                    \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                            \
            fprintf(stderr, "\ncuBLAS error %d at %s:%d\n", err_, __FILE__, __LINE__);  \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)
#endif // CUDART_VERSION >= 11

typedef void (*dequantize_kernel_t)(const void * vx, const int ib, const int iqs, float & v0, float & v1);
typedef void (*to_fp32_cuda_t)(const void * x, float * y, int k, cudaStream_t stream);
typedef void (*to_fp16_cuda_t)(const void * x, half * y, int k, cudaStream_t stream);
typedef void (*dot_kernel_k_t)(const void * vx, const int ib, const int iqs, const float * y, float & v);
typedef void (*cpy_kernel_t)(const char * cx, char * cdst);
typedef void (*ggml_cuda_func_t)(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
typedef void (*ggml_cuda_op_t)(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i, float * src0_ddf_i,
    float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main);

#define WARP_SIZE 32

#define CUDA_ADD_BLOCK_SIZE 256
#define CUDA_MUL_BLOCK_SIZE 256
#define CUDA_SILU_BLOCK_SIZE 256
#define CUDA_CPY_BLOCK_SIZE 32
#define CUDA_SCALE_BLOCK_SIZE 256
#define CUDA_ROPE_BLOCK_SIZE 256
#define CUDA_DIAG_MASK_INF_BLOCK_SIZE 32
#define CUDA_DEQUANTIZE_BLOCK_SIZE 256

static __global__ void add_f32(const float * x, const float * y, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = x[i] + y[i];
}

static __global__ void mul_f32(const float * x, const float * y, float * dst, const int kx, const int ky) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= kx) {
        return;
    }
    dst[i] = x[i] * y[i%ky];
}

static __global__ void silu_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = x[i] / (1.0f + expf(-x[i]));
}

static __global__ void rms_norm_f32(const float * x, float * dst, const int ncols) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    const float eps = 1e-6;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int i = 0; i < ncols; i += WARP_SIZE) {
        const int col = i + tid;
        const float xi = x[row*ncols + col];
        tmp += xi * xi;
    }

    // sum up partial sums
    __syncthreads();
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    const float mean = tmp / ncols;
    const float scale = 1.0f / sqrtf(mean + eps);

    for (int i = 0; i < ncols; i += WARP_SIZE) {
        const int col = i + tid;
        dst[row*ncols + col] = scale * x[row*ncols + col];
    }
}

static __global__ void mul_mat_p021_f16_f32(const void * vx, const float * y, float * dst, const int ncols_x, const int nrows_x, const int nchannels_x) {
    const half * x = (half *) vx;

    const int row_x = blockDim.y*blockIdx.y + threadIdx.y;
    const int channel = blockDim.z*blockIdx.z + threadIdx.z;

    const int nrows_y = ncols_x;
    const int nrows_dst = nrows_x;
    const int row_dst = row_x;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x; col_x0 += blockDim.x) {
        const int col_x = col_x0 + threadIdx.x;

        if (col_x >= ncols_x) {
            break;
        }

        // x is transposed and permuted
        const int ix = row_x*nchannels_x*ncols_x + channel*ncols_x + col_x;
        const float xi = __half2float(x[ix]);

        const int row_y = col_x;


        // y is not transposed but permuted
        const int iy = channel*nrows_y + row_y;

        tmp += xi * y[iy];
    }

    // dst is not transposed and not permuted
    const int idst = channel*nrows_dst + row_dst;

    // sum up partial sums and write back result
    __syncthreads();
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (threadIdx.x == 0) {
        dst[idst] = tmp;
    }
}

static __global__ void mul_mat_vec_nc_f16_f32( // nc == non-contiguous
    const void * vx, const float * y, float * dst, const int ncols_x, const int nrows_x,
    const int row_stride_x, const int nchannels_x, const int channel_stride_x) {

    const half * x = (half *) vx;

    const int row_x = blockDim.y*blockIdx.y + threadIdx.y;
    const int channel = blockDim.z*blockIdx.z + threadIdx.z;

    const int nrows_y = ncols_x;
    const int nrows_dst = nrows_x;
    const int row_dst = row_x;

    const int idst = channel*nrows_dst + row_dst;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x; col_x0 += blockDim.x) {
        const int col_x = col_x0 + threadIdx.x;

        if (col_x >= ncols_x) {
            break;
        }

        const int ix = channel*channel_stride_x + row_x*row_stride_x + col_x;
        const float xi = __half2float(x[ix]);

        const int row_y = col_x;

        const int iy = channel*nrows_y + row_y;

        tmp += xi * y[iy];
    }

    // sum up partial sums and write back result
    __syncthreads();
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (threadIdx.x == 0) {
        dst[idst] = tmp;
    }
}

static __device__ void cpy_1_f32_f32(const char * cxi, char * cdsti) {
    const float * xi = (float *) cxi;
    float * dsti = (float *) cdsti;

    *dsti = *xi;
}

static __device__ void cpy_1_f32_f16(const char * cxi, char * cdsti) {
    const float * xi = (float *) cxi;
    half * dsti = (half *) cdsti;

    *dsti = __float2half(*xi);
}

template <cpy_kernel_t cpy_1>
static __global__ void cpy_f32_f16(const char * cx, char * cdst, const int ne,
                                   const int ne00, const int ne01, const int nb00, const int nb01, const int nb02,
                                   const int ne10, const int ne11, const int nb10, const int nb11, const int nb12) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= ne) {
        return;
    }

    // determine indices i02/i12, i01/i11, i00/i10 as a function of index i of flattened tensor
    // then combine those indices with the corresponding byte offsets to get the total offsets
    const int i02 = i / (ne00*ne01);
    const int i01 = (i - i02*ne01*ne00) / ne00;
    const int i00 = i - i02*ne01*ne00 - i01*ne00;
    const int x_offset = i00*nb00 + i01*nb01 + i02*nb02;

    const int i12 = i / (ne10*ne11);
    const int i11 = (i - i12*ne10*ne11) / ne10;
    const int i10 = i - i12*ne10*ne11 - i11*ne10;
    const int dst_offset = i10*nb10 + i11*nb11 + i12*nb12;

    cpy_1(cx + x_offset, cdst + dst_offset);
}

// rope == RoPE == rotary positional embedding
static __global__ void rope_f32(const float * x, float * dst, const int ncols, const float p, const float theta_scale) {
    const int col = 2*(blockDim.x*blockIdx.x + threadIdx.x);

    if (col >= ncols) {
        return;
    }

    const int row = blockDim.y*blockIdx.y + threadIdx.y;
    const int i = row*ncols + col;

    const float theta = p*powf(theta_scale, col/2);
    const float sin_theta = sinf(theta);
    const float cos_theta = cosf(theta);

    const float x0 = x[i + 0];
    const float x1 = x[i + 1];

    dst[i + 0] = x0*cos_theta - x1*sin_theta;
    dst[i + 1] = x0*sin_theta + x1*cos_theta;
}

static __global__ void diag_mask_inf_f32(const float * x, float * dst, const int ncols, const int rows_per_channel, const int n_past) {
    const int col = blockDim.x*blockIdx.x + threadIdx.x;
    const int row = blockDim.y*blockIdx.y + threadIdx.y;

    if (col >= ncols) {
        return;
    }

    const int i = row*ncols + col;
    // dst[i] = col > n_past + row ? -INFINITY : x[i];
    dst[i] = x[i] - (col > n_past + row % rows_per_channel) * INT_MAX; // equivalent within rounding error but slightly faster on GPU
}

// the CUDA soft max implementation differs from the CPU implementation
// instead of doubles floats are used
// values are also not normalized to the maximum value by subtracting it in the exponential function
// theoretically these changes could cause problems with rounding error and arithmetic overflow but for LLaMa it seems to be fine
static __global__ void soft_max_f32(const float * x, float * dst, const int ncols) {
    const int row = blockDim.y*blockIdx.y + threadIdx.y;
    const int block_size = blockDim.x;
    const int tid = threadIdx.x;

    float tmp = 0.0;

    for (int block_start = 0; block_start < ncols; block_start += block_size) {
        const int col = block_start + tid;

        if (col >= ncols) {
            break;
        }

        const int i = row*ncols + col;
        const float val = expf(x[i]);
        tmp += val;
        dst[i] = val;
    }

    // sum up partial sums
    __syncthreads();
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    for (int block_start = 0; block_start < ncols; block_start += block_size) {
        const int col = block_start + tid;

        if (col >= ncols) {
            break;
        }

        const int i = row*ncols + col;
        dst[i] /= tmp;
    }
}

static __global__ void scale_f32(const float * x, float * dst, const float scale, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = scale * x[i];
}

static void add_f32_cuda(const float * x, const float * y, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_ADD_BLOCK_SIZE - 1) / CUDA_ADD_BLOCK_SIZE;
    add_f32<<<num_blocks, CUDA_ADD_BLOCK_SIZE, 0, stream>>>(x, y, dst, k);
}

static void mul_f32_cuda(const float * x, const float * y, float * dst, const int kx, const int ky, cudaStream_t stream) {
    const int num_blocks = (kx + CUDA_MUL_BLOCK_SIZE - 1) / CUDA_MUL_BLOCK_SIZE;
    mul_f32<<<num_blocks, CUDA_MUL_BLOCK_SIZE, 0, stream>>>(x, y, dst, kx, ky);
}

static void silu_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SILU_BLOCK_SIZE - 1) / CUDA_SILU_BLOCK_SIZE;
    silu_f32<<<num_blocks, CUDA_SILU_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void rms_norm_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % WARP_SIZE == 0);
    const dim3 block_dims(WARP_SIZE, 1, 1);
    rms_norm_f32<<<nrows, block_dims, 0, stream>>>(x, dst, ncols);
}

static void ggml_mul_mat_p021_f16_f32_cuda(const void * vx, const float * y, float * dst, const int ncols_x, const int nrows_x, const int nchannels_x, cudaStream_t stream) {
    const dim3 block_nums(1, nrows_x, nchannels_x);
    const dim3 block_dims(WARP_SIZE, 1, 1);
    mul_mat_p021_f16_f32<<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols_x, nrows_x, nchannels_x);
}

static void ggml_mul_mat_vec_nc_f16_f32_cuda(
    const void * vx, const float * y, float * dst, const int ncols_x, const int nrows_x, const int row_stride_x,
    const int nchannels_x, const int channel_stride_x, cudaStream_t stream) {

    const dim3 block_nums(1, nrows_x, nchannels_x);
    const dim3 block_dims(WARP_SIZE, 1, 1);
    mul_mat_vec_nc_f16_f32<<<block_nums, block_dims, 0, stream>>>
        (vx, y, dst, ncols_x, nrows_x, row_stride_x, nchannels_x, channel_stride_x);
}

static void ggml_cpy_f32_f32_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int nb00, const int nb01, const int nb02,
    const int ne10, const int ne11, const int nb10, const int nb11, const int nb12, cudaStream_t stream) {

    const int num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    cpy_f32_f16<cpy_1_f32_f32><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, nb00, nb01, nb02, ne10, ne11, nb10, nb11, nb12);
}

static void ggml_cpy_f32_f16_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int nb00, const int nb01, const int nb02,
    const int ne10, const int ne11, const int nb10, const int nb11, const int nb12, cudaStream_t stream) {

    const int num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    cpy_f32_f16<cpy_1_f32_f16><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, nb00, nb01, nb02, ne10, ne11, nb10, nb11, nb12);
}

static void scale_f32_cuda(const float * x, float * dst, const float scale, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    scale_f32<<<num_blocks, CUDA_SCALE_BLOCK_SIZE, 0, stream>>>(x, dst, scale, k);
}

static void rope_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, const float p, const float theta_scale, cudaStream_t stream) {
    GGML_ASSERT(nrows % 2 == 0);
    const dim3 block_dims(2*CUDA_ROPE_BLOCK_SIZE, 1, 1);
    const int num_blocks_x = (ncols + 2*CUDA_ROPE_BLOCK_SIZE - 1) / (2*CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(num_blocks_x, nrows, 1);
    rope_f32<<<block_nums, block_dims, 0, stream>>>(x, dst, ncols, p, theta_scale);
}

static void diag_mask_inf_f32_cuda(const float * x, float * dst, const int ncols_x, const int nrows_x, const int rows_per_channel, const int n_past, cudaStream_t stream) {
    const dim3 block_dims(CUDA_DIAG_MASK_INF_BLOCK_SIZE, 1, 1);
    const int block_num_x = (ncols_x + CUDA_DIAG_MASK_INF_BLOCK_SIZE - 1) / CUDA_DIAG_MASK_INF_BLOCK_SIZE;
    const dim3 block_nums(block_num_x, nrows_x, 1);
    diag_mask_inf_f32<<<block_nums, block_dims, 0, stream>>>(x, dst, ncols_x, rows_per_channel, n_past);
}

static void soft_max_f32_cuda(const float * x, float * dst, const int ncols_x, const int nrows_x, cudaStream_t stream) {
    const dim3 block_dims(WARP_SIZE, 1, 1);
    const dim3 block_nums(1, nrows_x, 1);
    soft_max_f32<<<block_nums, block_dims, 0, stream>>>(x, dst, ncols_x);
}

// buffer pool for cuda
#define MAX_CUDA_BUFFERS 256

struct scoped_spin_lock {
    std::atomic_flag& lock;
    scoped_spin_lock(std::atomic_flag& lock) : lock(lock) {
        while (lock.test_and_set(std::memory_order_acquire)) {
            ; // spin
        }
    }
    ~scoped_spin_lock() {
        lock.clear(std::memory_order_release);
    }
    scoped_spin_lock(const scoped_spin_lock&) = delete;
    scoped_spin_lock& operator=(const scoped_spin_lock&) = delete;
};

struct cuda_buffer {
    void * ptr = nullptr;
    size_t size = 0;
    int access_count = 0;
};

static cuda_buffer g_cuda_buffer_pool[GGML_CUDA_MAX_DEVICES][MAX_CUDA_BUFFERS];
static std::atomic_flag g_cuda_pool_lock = ATOMIC_FLAG_INIT;

static void * ggml_cuda_pool_malloc(size_t size, size_t * actual_size) {
    scoped_spin_lock lock(g_cuda_pool_lock);
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    size_t min_size_diff = SIZE_MAX;
    size_t min_size_diff_ok = size * 0.05; // wiggle room
    cuda_buffer* best_fit = nullptr; // candidate pointer
    for (int i = 0; i < MAX_CUDA_BUFFERS; ++i) {
        cuda_buffer& b = g_cuda_buffer_pool[id][i];
        if (b.size >= size && b.ptr != nullptr) {
            size_t size_diff = b.size - size;
            if (size_diff < min_size_diff) {
                best_fit = &b;
                min_size_diff = size_diff;
                if (size_diff < min_size_diff_ok) {
                    break;
                }
            }
        }
    }
    if (best_fit != nullptr) {
        *actual_size = best_fit->size;
        void * ptr = best_fit->ptr;
        best_fit->ptr = nullptr;
        // best_fit->size = 0;
        best_fit->access_count++;
        return ptr;
    }
    //printf("CUDA MALLOC: Allocated MB: %.2f\n", (float)size/1024/1024);
    void * ptr;
    CUDA_CHECK(cudaMalloc((void **) &ptr, size));
    *actual_size = size;    
    return ptr;
}

static void ggml_cuda_pool_free(void * ptr, size_t size) {
    scoped_spin_lock lock(g_cuda_pool_lock);
    int id;
    CUDA_CHECK(cudaGetDevice(&id));

    for (int i = 0; i < MAX_CUDA_BUFFERS; ++i) {
        cuda_buffer& b = g_cuda_buffer_pool[id][i];
        if (b.ptr == nullptr) {
            b.ptr = ptr;
            b.size = size; // the original size should still be correct
            b.access_count = 1;
            return;
        }
    }
    fprintf(stderr, "WARNING: cuda buffer pool full, increase MAX_CUDA_BUFFERS\n");
    CUDA_CHECK(cudaFree(ptr));
}

// unallocates any "free" buffers that have not been used (or less than n times since last free)
// for example call after evaluation
int ggml_cuda_pool_purge_buffers_with_access_count(int min_access_count, int device_id) {
    scoped_spin_lock lock(g_cuda_pool_lock);
    int id;
    CUDA_CHECK(cudaGetDevice(&id));

    int total_purged = 0;

    for (int i = 0; i < MAX_CUDA_BUFFERS; ++i) {
        cuda_buffer& b = g_cuda_buffer_pool[device_id][i];
        if (b.ptr != nullptr && b.access_count < min_access_count) {
            if (id != device_id) {
                CUDA_CHECK(cudaSetDevice(device_id));
            }
            CUDA_CHECK(cudaFree(b.ptr));
            //printf("\n-----> CUDA: access count - purged buffer %d of size %zu for device %d\n", i, b.size, device_id);
            b.ptr = nullptr;
            b.size = 0;
            b.access_count = 0;
            
            total_purged++;
        }
    }
    return total_purged;
}
// resets access_count for all free buffers (for example before evaluation)
void ggml_cuda_pool_reset_all_counters(int device_id) {
    scoped_spin_lock lock(g_cuda_pool_lock);

    for (int i = 0; i < MAX_CUDA_BUFFERS; ++i) {
        cuda_buffer& b = g_cuda_buffer_pool[device_id][i];
        if (b.ptr != nullptr) {
            b.access_count = 0;
            //printf("CUDA: reset buffer %d of size %zu access_count %d for device %d\n", i, b.size, b.access_count, device_id);
        }
    }
}


static void * g_scratch_buffer = nullptr;
static size_t g_scratch_size = 1024*1024*1024; // 1 GB by default
static size_t g_scratch_offset = 0;

#define GGML_CUDA_MAX_STREAMS 8 // Set this to 1 for reproducible matrix multiplication.
#define GGML_CUDA_MAX_EVENTS 64

// Note: tensor_split defines the breakpoints of tensors that can be split {0,0.5}
static float g_tensor_split[GGML_CUDA_MAX_DEVICES] = {0};
static GPUStatus g_system_gpu_status;

static cublasHandle_t g_cublas_handles[GGML_CUDA_MAX_DEVICES] = {nullptr};

static cudaStream_t g_cudaStreams_main[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = { nullptr };

static cudaStream_t g_cudaStreams_memcpy_src1[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = { nullptr };
static cudaEvent_t g_cudaEvents_memcpy_src1[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_EVENTS] = { nullptr };

// Todo verify: free and total memory reported by cudaMemGetInfo differs from gpu_z which also differs from hwinfo64.
// Update the system status about available GPUs and memory usage
void ggml_cuda_update_gpu_status(int device_id) {
    int currentDevice = 0;
    CUDA_CHECK(cudaGetDevice(&currentDevice));
    if (device_id == -1) {
        // Update all devices 
        if (g_system_gpu_status.num_devices == 0)
        {
            CUDA_CHECK(cudaGetDeviceCount(&g_system_gpu_status.num_devices));
            if (g_system_gpu_status.num_devices > GGML_CUDA_MAX_DEVICES) {
                g_system_gpu_status.num_devices = GGML_CUDA_MAX_DEVICES;
                fprintf(stderr, "WARNING: GGML_CUDA_MAX_DEVICES is smaller than the number of devices on the system. Using first %d devices.\n", GGML_CUDA_MAX_DEVICES);
            }
            if (g_system_gpu_status.num_devices > g_system_gpu_status.max_gpus)
                    g_system_gpu_status.num_devices = g_system_gpu_status.max_gpus;
                
            g_system_gpu_status.total_vram = 0;
            for (int id = 0; id < g_system_gpu_status.num_devices; ++id) {
                CUDA_CHECK(cudaGetDeviceProperties(&g_system_gpu_status.device_props[id], id));
            }
        }
        g_system_gpu_status.total_vram = 0;
        g_system_gpu_status.total_free_vram = 0;
        for (int id = 0; id < g_system_gpu_status.num_devices; ++id) {
            CUDA_CHECK(cudaSetDevice(id));
            CUDA_CHECK(cudaMemGetInfo(&g_system_gpu_status.device_vram_free[id], &g_system_gpu_status.device_vram_total[id]));
            g_system_gpu_status.total_vram += g_system_gpu_status.device_vram_total[id];
            g_system_gpu_status.total_free_vram += g_system_gpu_status.device_vram_free[id];
        }
        // restore current device
        if (currentDevice != g_system_gpu_status.num_devices-1) {
            CUDA_CHECK(cudaSetDevice(currentDevice));
        }
    } else {
        // Update only the specified device
        CUDA_CHECK(cudaGetDeviceProperties(&g_system_gpu_status.device_props[device_id], device_id));
        CUDA_CHECK(cudaSetDevice(device_id));
        CUDA_CHECK(cudaMemGetInfo(&g_system_gpu_status.device_vram_free[device_id], &g_system_gpu_status.device_vram_total[device_id]));
        // go through all devices and update total/free
        g_system_gpu_status.total_vram = 0;
        g_system_gpu_status.total_free_vram = 0;
        for (int id = 0; id < g_system_gpu_status.num_devices; ++id) {
            g_system_gpu_status.total_vram += g_system_gpu_status.device_vram_total[id];
            g_system_gpu_status.total_free_vram += g_system_gpu_status.device_vram_free[id];
        }
        // restore current device
        if (device_id != currentDevice) {
            CUDA_CHECK(cudaSetDevice(currentDevice));
        }
    }
    
#if 1
    // required for proper vram distribution but split tensors require memory on primary GPU which could be disabled
    // remove unused GPUs from available calculation
    bool all_zero = true;
    for (int i = 0; i < g_system_gpu_status.num_devices; ++i) {
        if (g_tensor_split[i] != 0.0f) {
            all_zero = false;
        }
    }
    if (!all_zero)
    for (int id = 0; id < g_system_gpu_status.num_devices; ++id) {
        if (g_tensor_split[id] >= 1.0 || (id > 0 && g_tensor_split[id] == g_tensor_split[id-1])) {
            g_system_gpu_status.total_vram -= g_system_gpu_status.device_vram_total[id];
            g_system_gpu_status.total_free_vram -= g_system_gpu_status.device_vram_free[id];
        }
        
    }
#endif
}
void ggml_cuda_print_gpu_status(const GPUStatus *status, bool print_summary) {
    if (status == NULL) {
        fprintf(stderr,"Error: Invalid GPU status pointer.\n");
        return;
    }

    const char *divider = "+----+------------------------------------+------------+-----------+-----------+-----------+-----------+";
    fprintf(stderr,"%s\n", divider);
    fprintf(stderr,"| ID | %-25s %2d found | %10s | %9s | %9s | %9s | %9s |\n", "Device", status->num_devices, "VRAM Total", "VRAM Free", "VRAM Used","Split at ", "Device");
    fprintf(stderr,"%s\n", divider);

    for (int i = 0; i < status->num_devices; ++i) {
        const struct cudaDeviceProp *prop = &status->device_props[i];
        size_t vram_used = status->device_vram_total[i] - status->device_vram_free[i];
        float split_at_percentage = g_tensor_split[i] * 100;
        fprintf(stderr,"| %2d | %-34s | %7zu MB | %6zu MB | %6zu MB | %8.1f%% | %9s |\n", 
                i,prop->name, status->device_vram_total[i] / (1024 * 1024), status->device_vram_free[i] / (1024 * 1024), vram_used / (1024 * 1024),split_at_percentage, (i == status->main_device_id) ? "Primary" : "Secondary");
        // printf("%s\n", divider);
    }
    if (print_summary && status->num_devices > 1)
    {
        fprintf(stderr,"%s\n", divider);
        fprintf(stderr,"|    | %-34s | %7zu MB | %6zu MB | %6zu MB | %9s | %9s |\n", 
            "Device summary", status->total_vram / (1024 * 1024), status->total_free_vram / (1024 * 1024), (status->total_vram - status->total_free_vram) / (1024 * 1024), "N/A", "All");
    }
    fprintf(stderr,"%s\n", divider);
    
}

const GPUStatus* ggml_cuda_get_system_gpu_status(void) {
    return &g_system_gpu_status;
}
void ggml_cuda_set_max_gpus(int max_gpus) {
    g_system_gpu_status.max_gpus = max_gpus;
}

// can be called multithreaded to prevent 1-2 seconds delay on handle creation
bool ggml_init_cublas(bool check_only) {
    static volatile bool initialized = false;
    if (check_only || initialized) return initialized;
        
    int currentDevice = 0;
    CUDA_CHECK(cudaGetDevice(&currentDevice));
    if (!initialized) {
        //g_system_gpu_status.num_devices = 0;
        if (g_system_gpu_status.num_devices == 0)
            ggml_cuda_update_gpu_status(-1);

        bool all_zero = true;
        for (int i = 0; i < g_system_gpu_status.num_devices; ++i) {
            if (g_tensor_split[i] != 0.0f) {
                all_zero = false;
            }
        }
        if (all_zero)
        {
            int64_t total_vram = 0;
            for (int id = 0; id < g_system_gpu_status.num_devices; ++id) {
                g_tensor_split[id] = total_vram;
                size_t vram_free;
                // vram_total = g_system_gpu_status.device_vram_total[id];
                vram_free = g_system_gpu_status.device_vram_free[id];
                total_vram += vram_free;
            }
            for (int id = 0; id < g_system_gpu_status.num_devices; ++id) {
                g_tensor_split[id] /= total_vram;
            }
        }
        //ggml_cuda_print_gpu_status(&g_system_gpu_status,true);
        // printf("Preparing CUDA for %d devices: ",g_system_gpu_status.num_devices);
        for (int id = 0; id < g_system_gpu_status.num_devices; ++id) {
            CUDA_CHECK(cudaSetDevice(id));

            // create streams
            for (int i = 0; i < GGML_CUDA_MAX_STREAMS; ++i) {
                CUDA_CHECK(cudaStreamCreateWithFlags(&g_cudaStreams_main[id][i], cudaStreamNonBlocking));
                CUDA_CHECK(cudaStreamCreateWithFlags(&g_cudaStreams_memcpy_src1[id][i], cudaStreamNonBlocking));
            }
            // create events
            for (int i = 0; i < GGML_CUDA_MAX_EVENTS; ++i) {
                CUDA_CHECK(cudaEventCreateWithFlags(&g_cudaEvents_memcpy_src1[id][i], cudaEventDisableTiming));
            }

            // create cublas handle
            CUBLAS_CHECK(cublasCreate(&g_cublas_handles[id]));
            CUBLAS_CHECK(cublasSetMathMode(g_cublas_handles[id], CUBLAS_TF32_TENSOR_OP_MATH));
        }
        CUDA_CHECK(cudaSetDevice(currentDevice));

        // configure logging to stdout
        // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

        initialized = true;
        
    }
    return initialized;
}

// prepare tensor split before we've initizalized cublas
void ggml_cuda_set_tensor_split_prepare(const float * tensor_split, int num_devices) {
    int cur_num_devices = g_system_gpu_status.num_devices;
    g_system_gpu_status.num_devices = num_devices;
    ggml_cuda_set_tensor_split(tensor_split);
    g_system_gpu_status.num_devices = cur_num_devices;
}
// expect array of float proportions where to split each device id, if all 0.0 then no change to default split
void ggml_cuda_set_tensor_split(const float * tensor_split) {
    bool all_zero = true;
    for (int i = 0; i < g_system_gpu_status.num_devices; ++i) {
        if (tensor_split[i] != 0.0f) {
            all_zero = false;
            break;
        }
    }
    if (all_zero) {
        return;
    }
    float split_sum = 0.0f;
    for (int i = 0; i < g_system_gpu_status.num_devices; ++i) {
        g_tensor_split[i] = split_sum;
        split_sum += tensor_split[i];
    }
    for (int i = 0; i < g_system_gpu_status.num_devices; ++i) {
        float device_prop = tensor_split[i] / split_sum;
        if (device_prop == 0.0f) {
            g_tensor_split[i] = 1.0f;
        }
        else {
            g_tensor_split[i] /= split_sum;
        }

    }
}

void * ggml_cuda_host_malloc(size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }
    if (g_system_gpu_status.num_devices == 0) {
        return nullptr;
    }

    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) {
        // The allocation error can be bypassed. A null ptr will assigned out of this function.
        // This can fixed the OOM error in WSL.
        cudaGetLastError();
        fprintf(stderr, "WARNING: failed to allocate %.2f MB of pinned (CUDA optimized) memory: %s\n",
            size/1024.0/1024.0, cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

void ggml_cuda_host_free(void * ptr) {
    CUDA_CHECK(cudaFreeHost(ptr));
}

static cudaError_t ggml_cuda_cpy_tensor_2d(
    void * dst, const struct ggml_tensor * src, int64_t i3, int64_t i2, int64_t i1_low, int64_t i1_high, cudaStream_t stream) {

    cudaMemcpyKind kind;
    char * src_ptr;
    if (src->backend == GGML_BACKEND_CPU) {
        kind = cudaMemcpyHostToDevice;
        src_ptr = (char *) src->data;
    } else if (src->backend == GGML_BACKEND_GPU) {
        kind = cudaMemcpyDeviceToDevice;
        struct ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src->extra;
        int id;
        CUDA_CHECK(cudaGetDevice(&id));
        src_ptr = (char *) extra->data_device[id];
    } else {
        GGML_ASSERT(false);
    }
    char * dst_ptr = (char *) dst;

    const int64_t ne0 = src->ne[0];
    const int64_t nb0 = src->nb[0];
    const int64_t nb1 = src->nb[1];
    const int64_t nb2 = src->nb[2];
    const int64_t nb3 = src->nb[3];
    const enum ggml_type type = src->type;
    const int64_t ts = ggml_type_size(type);
    const int64_t bs = ggml_blck_size(type);
    int64_t i1_diff = i1_high - i1_low;

    const char * x = src_ptr + i1_low*nb1 + i2*nb2 + i3*nb3;
    if (nb0 == ts && nb1 == ts*ne0/bs) {
        return cudaMemcpyAsync(dst_ptr, x, i1_diff*nb1, kind, stream);
    } else if (nb0 == ts) {
        return cudaMemcpy2DAsync(dst_ptr, ts*ne0/bs, x, nb1, ts*ne0/bs, i1_diff, kind, stream);
    } else {
        for (int64_t i1 = 0; i1 < i1_diff; i1++) {
            const void * rx = (const void *) ((const char *) x + i1*nb1);
            void * rd = (void *) (dst_ptr + i1*ts*ne0/bs);
            // pretend the row is a matrix with cols=1
            cudaError_t r = cudaMemcpy2DAsync(rd, ts/bs, rx, nb0, ts/bs, ne0, kind, stream);
            if (r != cudaSuccess) return r;
        }
        return cudaSuccess;
    }
}

inline void ggml_cuda_op_add(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    float * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main){

    GGML_ASSERT(src0_ddf_i != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_ddf_i != nullptr);

    const int64_t ne0 = src0->ne[0];
    const int64_t i01_diff = i01_high - i01_low;

    // compute
    add_f32_cuda(src0_ddf_i, src1_ddf_i, dst_ddf_i, ne0*i01_diff, cudaStream_main);

    (void) src1;
    (void) dst;
    (void) src0_ddq_i;
    (void) i02;
    (void) i1;
}

inline void ggml_cuda_op_mul(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    float * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main){

    GGML_ASSERT(src0_ddf_i != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_ddf_i != nullptr);

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    for (int64_t i01 = i01_low; i01 < i01_high; i01++) {
        const int64_t i11 = i1*ne11 + i01%ne11; // broadcast src1 across src0

        float * src0_ddf_i01 = src0_ddf_i + i01*ne00;
        float * src1_ddf_i01 = src1_ddf_i + i11*ne10;
        float * dst_ddf_i01 = dst_ddf_i + i01*ne00;

        // compute
        mul_f32_cuda(src0_ddf_i01, src1_ddf_i01, dst_ddf_i01, ne00, ne10, cudaStream_main);
    }

    (void) dst;
    (void) src0_ddq_i;
    (void) i02;
}

inline void ggml_cuda_op_silu(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    float * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main){

    GGML_ASSERT(src0_ddf_i != nullptr);
    GGML_ASSERT(dst_ddf_i != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t i01_diff = i01_high - i01_low;

    // compute
    silu_f32_cuda(src0_ddf_i, dst_ddf_i, ne00*i01_diff, cudaStream_main);

    (void) src1;
    (void) dst;
    (void) src0_ddq_i;
    (void) src1_ddf_i;
    (void) i02;
    (void) i1;
}

inline void ggml_cuda_op_rms_norm(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    float * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main){

    GGML_ASSERT(src0_ddf_i != nullptr);
    GGML_ASSERT(dst_ddf_i != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t i01_diff = i01_high - i01_low;

    // compute
    rms_norm_f32_cuda(src0_ddf_i, dst_ddf_i, ne00, i01_diff, cudaStream_main);

    (void) src1;
    (void) dst;
    (void) src0_ddq_i;
    (void) src1_ddf_i;
    (void) i02;
    (void) i1;
}

inline void ggml_cuda_op_mul_mat_cublas(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    float * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main){

    GGML_ASSERT(src0_ddf_i != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_ddf_i != nullptr);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    const int64_t ne0 = dst->ne[0];
    const int64_t i01_diff = i01_high - i01_low;

    dst->meta.cuda_perf_mal_mul_type=32;
    int id;
    CUDA_CHECK(cudaGetDevice(&id));

    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    int ldc = dst->backend == GGML_BACKEND_GPU && id == g_system_gpu_status.main_device_id ? ne0 : i01_diff;

    CUBLAS_CHECK(cublasSetStream(g_cublas_handles[id], cudaStream_main));
    CUBLAS_CHECK(
        cublasSgemm(g_cublas_handles[id], CUBLAS_OP_T, CUBLAS_OP_N,
                i01_diff, ne11, ne10,
                &alpha, src0_ddf_i, ne00,
                        src1_ddf_i, ne10,
                &beta,  dst_ddf_i,  ldc));

    (void) dst;
    (void) src0_ddq_i;
    (void) i02;
    (void) i1;
}
__global__ void float_to_half(const float* src, __half* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}
// takes 16,32 bit returns 32 bit - internally converts src1 to 16 bit
inline void ggml_cuda_op_mul_mat_cublas_f16_f32(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    __half * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main){

    GGML_ASSERT(src0_ddf_i != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_ddf_i != nullptr);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    const int64_t ne0 = dst->ne[0];
    const int64_t i01_diff = i01_high - i01_low;
    dst->meta.cuda_perf_mal_mul_type=16;
    int id;
    CUDA_CHECK(cudaGetDevice(&id));

    // we need to convert src1_ddf_i to half precision
    __half* src1_ddf_i_half;
    size_t src1_size = ne10 * ne11 * sizeof(__half);
    size_t actual_size = 0;
    src1_ddf_i_half = (half *)ggml_cuda_pool_malloc(src1_size,&actual_size);
    float_to_half<<<(ne10 * ne11 + 255) / 256, 256, 0, cudaStream_main>>>(src1_ddf_i, src1_ddf_i_half, ne10 * ne11);
    CUDA_CHECK(cudaStreamSynchronize(cudaStream_main));

    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    int ldc = dst->backend == GGML_BACKEND_GPU && id == g_system_gpu_status.main_device_id ? ne0 : i01_diff;

    CUBLAS_CHECK(cublasSetStream(g_cublas_handles[id], cudaStream_main));
    CUBLAS_CHECK(
        cublasGemmEx(g_cublas_handles[id], CUBLAS_OP_T, CUBLAS_OP_N,
                i01_diff, ne11, ne10,
                &alpha, src0_ddf_i, CUDA_R_16F, ne00,
                        src1_ddf_i_half, CUDA_R_16F, ne10,
                &beta,  dst_ddf_i,  CUDA_R_32F, ldc,
                CUBLAS_COMPUTE_32F_FAST_16F,
                CUBLAS_GEMM_DEFAULT));

    (void) dst;
    (void) src0_ddq_i;
    (void) i02;
    (void) i1;
    ggml_cuda_pool_free(src1_ddf_i_half,actual_size);
}
// src0 is actually fp16, everything else is same as normal
inline void ggml_cuda_op_mul_mat_cublas_f16_f32_wrapper(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i, float * src0_ddf_i,
    float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main) {

    // Cast float pointers to half-precision pointers
    __half * src0_ddf_i_half = reinterpret_cast<__half *>(src0_ddf_i);
    // __half * src1_ddf_i_half = reinterpret_cast<__half *>(src1_ddf_i);
    // __half * dst_ddf_i_half = reinterpret_cast<__half *>(dst_ddf_i);

    // Call the modified function with the casted pointers
    ggml_cuda_op_mul_mat_cublas_f16_f32(src0, src1, dst, src0_ddq_i, src0_ddf_i_half, src1_ddf_i, dst_ddf_i, i02, i01_low, i01_high, i1, cudaStream_main);
}

inline void ggml_cuda_op_rope(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    float * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main){

    GGML_ASSERT(src0_ddf_i != nullptr);
    GGML_ASSERT(dst_ddf_i != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t i01_diff = i01_high - i01_low;

    const int n_past = ((int32_t *) src1->data)[0];
    const int n_dims = ((int32_t *) src1->data)[1];
    const int mode   = ((int32_t *) src1->data)[2];
    GGML_ASSERT(mode == 0);

    const float theta_scale = powf((float)(dst->meta.i_custom[GGML_CUSTOM_I_ROPE_ANG_FREQ]?dst->meta.i_custom[GGML_CUSTOM_I_ROPE_ANG_FREQ]:10000), -2.0f/n_dims);
    float p = ((mode & 1) == 0 ? n_past + i02 : i02);
    // custom 2d rotation angle scale (needs a test - blind adapted)
    if (dst->meta.f_custom[GGML_CUSTOM_F_ROPE_ANG_SCALE] != 0.0f) {
        p *= dst->meta.f_custom[GGML_CUSTOM_F_ROPE_ANG_SCALE];
    }
    // compute
    rope_f32_cuda(src0_ddf_i, dst_ddf_i, ne00, i01_diff, p, theta_scale, cudaStream_main);

    (void) dst;
    (void) src0_ddq_i;
    (void) src1_ddf_i;
    (void) i1;
}

inline void ggml_cuda_op_diag_mask_inf(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    float * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main){

    GGML_ASSERT(src0_ddf_i != nullptr);
    GGML_ASSERT(dst_ddf_i != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t i01_diff = i01_high - i01_low;

    const int n_past = ((int32_t *) src1->data)[0];

    // compute
    diag_mask_inf_f32_cuda(src0_ddf_i, dst_ddf_i, ne00, i01_diff, ne01, n_past, cudaStream_main);

    (void) dst;
    (void) src0_ddq_i;
    (void) src1_ddf_i;
    (void) i02;
    (void) i1;
}

inline void ggml_cuda_op_soft_max(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    float * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main){

    GGML_ASSERT(src0_ddf_i != nullptr);
    GGML_ASSERT(dst_ddf_i != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t i01_diff = i01_high - i01_low;

    // compute
    soft_max_f32_cuda(src0_ddf_i, dst_ddf_i, ne00, i01_diff, cudaStream_main);

    (void) src1;
    (void) dst;
    (void) src0_ddq_i;
    (void) src1_ddf_i;
    (void) i02;
    (void) i1;
}

inline void ggml_cuda_op_scale(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * src0_ddq_i,
    float * src0_ddf_i, float * src1_ddf_i, float * dst_ddf_i, int64_t i02, int64_t i01_low, int64_t i01_high, int i1,
    cudaStream_t & cudaStream_main){

    GGML_ASSERT(src0_ddf_i != nullptr);
    GGML_ASSERT(dst_ddf_i != nullptr);

    const float scale = ((float *) src1->data)[0];

    const int64_t ne00 = src0->ne[0];
    const int64_t i01_diff = i01_high - i01_low;

    // compute
    scale_f32_cuda(src0_ddf_i, dst_ddf_i, scale, ne00*i01_diff, cudaStream_main);

    (void) src1;
    (void) dst;
    (void) src0_ddq_i;
    (void) src1_ddf_i;
    (void) i02;
    (void) i1;
}

static void ggml_cuda_op(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                         ggml_cuda_op_t op, bool src0_needs_f32, bool flatten_rows) {
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    const int64_t nrows0 = ggml_nrows(src0);

    const bool use_src1 = src1 != nullptr;
    const int64_t ne10 = use_src1 ? src1->ne[0] : 1;
    const int64_t ne11 = use_src1 ? src1->ne[1] : 1;
    const int64_t ne12 = use_src1 ? src1->ne[2] : 1;
    const int64_t ne13 = use_src1 ? src1->ne[3] : 1;

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    GGML_ASSERT(dst->backend != GGML_BACKEND_GPU_SPLIT);
    GGML_ASSERT(!use_src1 || src1->backend != GGML_BACKEND_GPU_SPLIT);

    // strides for iteration over dims 3 and 2
    const int64_t num_iters = flatten_rows ? 1 : ne02 * ne03;
    const int64_t stride_mod = flatten_rows ? ne02 * ne03 : 1;
    const int64_t src0_stride = ne00 * ne01 * stride_mod;
    const int64_t src1_stride = ne10 * ne11 * stride_mod;
    const int64_t dst_stride = ne0 * ne1 * stride_mod;

    const size_t src0_ts = ggml_type_size(src0->type);
    const size_t src0_bs = ggml_blck_size(src0->type);

    struct ggml_tensor_extra_gpu * src0_extra =            (ggml_tensor_extra_gpu *) src0->extra;
    struct ggml_tensor_extra_gpu * src1_extra = use_src1 ? (ggml_tensor_extra_gpu *) src1->extra : nullptr;
    struct ggml_tensor_extra_gpu * dst_extra  =            (ggml_tensor_extra_gpu *) dst->extra;

    const bool src0_on_device = src0->backend == GGML_BACKEND_GPU || src0->backend == GGML_BACKEND_GPU_SPLIT;
    const bool src0_is_contiguous = ggml_is_contiguous(src0);
    const bool src0_is_f32 = src0->type == GGML_TYPE_F32;

    const bool src1_is_contiguous = use_src1 && ggml_is_contiguous(src1);
    const bool src1_stays_on_host = use_src1 && (
        dst->op == GGML_OP_SCALE || dst->op == GGML_OP_DIAG_MASK_INF || dst->op == GGML_OP_ROPE);

    const bool split = src0->backend == GGML_BACKEND_GPU_SPLIT;

    // dd = data device
    char  * src0_ddq[GGML_CUDA_MAX_DEVICES] = {nullptr}; // quantized
    float * src0_ddf[GGML_CUDA_MAX_DEVICES] = {nullptr}; // float
    float * src1_ddf[GGML_CUDA_MAX_DEVICES] = {nullptr};
    float *  dst_ddf[GGML_CUDA_MAX_DEVICES] = {nullptr};

    // asq = actual size quantized, asf = actual size float
    size_t src0_asq[GGML_CUDA_MAX_DEVICES] = {0};
    size_t src0_asf[GGML_CUDA_MAX_DEVICES] = {0};
    size_t src1_asf[GGML_CUDA_MAX_DEVICES] = {0};
    size_t  dst_asf[GGML_CUDA_MAX_DEVICES] = {0};

    for (int id = 0; id < g_system_gpu_status.num_devices; ++id) {
        if (!split && id != g_system_gpu_status.main_device_id) {
            continue;
        }

        const bool src1_on_device = use_src1 && src1->backend == GGML_BACKEND_GPU && id == g_system_gpu_status.main_device_id;
        const bool dst_on_device = dst->backend == GGML_BACKEND_GPU && id == g_system_gpu_status.main_device_id;

        int64_t row_low, row_high;
        if (split) {
            row_low = id == 0 ? 0 : nrows0*g_tensor_split[id];
            row_high = id == g_system_gpu_status.num_devices - 1 ? nrows0 : nrows0*g_tensor_split[id + 1];
        } else {
            row_low = 0;
            row_high = nrows0;
        }
        if (row_low == row_high) {
            continue;
        }

        int64_t row_diff = row_high - row_low;

        cudaSetDevice(id);

        if (src0_on_device && src0_is_contiguous) {
            if (src0_is_f32) {
                src0_ddf[id] = (float *) src0_extra->data_device[id];
            } else {
                src0_ddq[id] = (char *) src0_extra->data_device[id];
            }
        } else {
            if (src0_is_f32) {
                src0_ddf[id] = (float *) ggml_cuda_pool_malloc(row_diff*ne00 * sizeof(float), &src0_asf[id]);
            } else {
                src0_ddq[id] = (char *) ggml_cuda_pool_malloc(row_diff*ne00 * src0_ts/src0_bs, &src0_asq[id]);
            }
        }

        if (use_src1 && !src1_stays_on_host) {
            if (src1_on_device && src1_is_contiguous) {
                src1_ddf[id] = (float *) src1_extra->data_device[id];
            } else {
                src1_ddf[id] = (float *) ggml_cuda_pool_malloc(num_iters*src1_stride * sizeof(float), &src1_asf[id]);
            }
        }
        if (dst_on_device) {
            dst_ddf[id] = (float *) dst_extra->data_device[id];
        } else {
            size_t size_dst_ddf = split ? row_diff*ne1 * sizeof(float) : num_iters*dst_stride * sizeof(float);
            dst_ddf[id] = (float *) ggml_cuda_pool_malloc(size_dst_ddf, &dst_asf[id]);
        }

        const int64_t i03_max = flatten_rows ? 1 : ne03;
        const int64_t i02_max = flatten_rows ? 1 : ne02;
        const int64_t rows_per_iter = flatten_rows ? nrows0 : ne01;

        for (int64_t i03 = 0; i03 < i03_max; i03++) {
            const int64_t i13 = i03 % ne13;
            for (int64_t i02 = 0; i02 < i02_max; i02++) {
                const int64_t i12 = i02 % ne12;

                const int64_t i0 = i03*ne02 + i02;

                // i0 values that contain the lower/upper rows for a split tensor when using multiple GPUs
                const int64_t i0_offset_low = row_low/rows_per_iter;
                const int64_t i0_offset_high = row_high/rows_per_iter;

                int64_t i01_low = 0;
                int64_t i01_high = rows_per_iter;
                if (split) {
                    if (i0 < i0_offset_low || i0 > i0_offset_high) {
                        continue;
                    }
                    if (i0 == i0_offset_low) {
                        i01_low = row_low % rows_per_iter;
                    }
                    if (i0 == i0_offset_high) {
                        i01_high = row_high % rows_per_iter;
                    }
                }

                // There is possibly a bug in the Windows nvcc compiler regarding instruction reordering or optimizing out local variables.
                // Removing the first assert or changing the order of the arguments causes the second assert to fail.
                // Removing both asserts results in i01_high becoming 0 which in turn results in garbage output.
                // The root cause seems to be a problem with i0_offset_high becoming 0 when it should always be >0 (for single GPU).
                GGML_ASSERT(i01_low == 0 || g_system_gpu_status.num_devices > 1);
                GGML_ASSERT(i01_high == rows_per_iter || g_system_gpu_status.num_devices > 1);

                const int64_t i01_diff = i01_high - i01_low;
                if (i01_diff == 0) {
                    continue;
                }
                const int64_t i11 = i13*ne12 + i12;

                cudaStream_t cudaStream_main        =        g_cudaStreams_main[id][i0 % GGML_CUDA_MAX_STREAMS];
                cudaStream_t cudaStream_memcpy_src1 = g_cudaStreams_memcpy_src1[id][i0 % GGML_CUDA_MAX_STREAMS];
                cudaEvent_t  cudaEvent_memcpy_src1  =  g_cudaEvents_memcpy_src1[id][i0 % GGML_CUDA_MAX_EVENTS];

                // for split tensors the data begins at i0 == i0_offset_low
                char  * src0_ddq_i = src0_ddq[id] + (i0 - i0_offset_low)*src0_stride*src0_ts/src0_bs;
                float * src0_ddf_i = src0_ddf[id] + (i0 - i0_offset_low)*src0_stride;
                float * src1_ddf_i = src1_ddf[id] + i11*src1_stride;
                float * dst_ddf_i  =  dst_ddf[id] + (i0 - i0_offset_low)*dst_stride;

                // for split tensors the data pointer needs to be rounded down
                // to the bin edge for i03, i02 bins beyond the first
                if (i0 - i0_offset_low > 0) {
                    GGML_ASSERT(!flatten_rows);
                    src0_ddq_i -= (row_low % ne01)*ne00 * src0_ts/src0_bs;
                    src0_ddf_i -= (row_low % ne01)*ne00;
                    dst_ddf_i  -= (row_low % ne0)*ne1;
                }

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                if (dst->backend == GGML_BACKEND_GPU && id == g_system_gpu_status.main_device_id) {
                    dst_ddf_i += i01_low; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                if (use_src1 && !src1_stays_on_host) {
                    if (src1->backend == GGML_BACKEND_CPU) {
                        GGML_ASSERT(!flatten_rows || nrows0 == ggml_nrows(src1));
                        int64_t nrows1 = flatten_rows ? nrows0 : ne11;
                        CUDA_CHECK(ggml_cuda_cpy_tensor_2d(src1_ddf_i, src1, i03, i02, 0, nrows1, cudaStream_memcpy_src1));
                    } else if (src1->backend == GGML_BACKEND_GPU && src1_is_contiguous) {
                        if (id != g_system_gpu_status.main_device_id) {
                            GGML_ASSERT(!flatten_rows);
                            float * src1_ddf_i_source = (float *) src1_extra->data_device[g_system_gpu_status.main_device_id];
                            src1_ddf_i_source += i11*src1_stride;
                            CUDA_CHECK(cudaMemcpyAsync(src1_ddf_i, src1_ddf_i_source, src1_stride*sizeof(float),
                                                    cudaMemcpyDeviceToDevice, cudaStream_memcpy_src1));
                        }
                    } else if (src1_on_device && !src1_is_contiguous) {
                        GGML_ASSERT(!split);
                        CUDA_CHECK(ggml_cuda_cpy_tensor_2d(src1_ddf_i, src1, i03, i02, 0, ne11, cudaStream_main));
                    } else {
                        GGML_ASSERT(false);
                    }
                }
                CUDA_CHECK(cudaEventRecord(cudaEvent_memcpy_src1, cudaStream_memcpy_src1));

                if (!src0_on_device || !src0_is_contiguous) {
                    if (src0_is_f32) {
                        CUDA_CHECK(ggml_cuda_cpy_tensor_2d(src0_ddf_i, src0, i03, i02, i01_low, i01_high, cudaStream_main));
                    } else {
                        CUDA_CHECK(ggml_cuda_cpy_tensor_2d(src0_ddq_i, src0, i03, i02, i01_low, i01_high, cudaStream_main));
                    }
                }

                // wait with main stream until src1 memcpy is done
                CUDA_CHECK(cudaStreamWaitEvent(cudaStream_main, cudaEvent_memcpy_src1, 0));

                // do the computation
                op(src0, src1, dst, src0_ddq_i, src0_ddf_i, src1_ddf_i, dst_ddf_i, i02, i01_low, i01_high, i11, cudaStream_main);
                CUDA_CHECK(cudaGetLastError());

                // copy dst to host or other device if necessary
                if (!dst_on_device) {
                    void * dst_off_device;
                    cudaMemcpyKind kind;
                    if (dst->backend == GGML_BACKEND_CPU) {
                        dst_off_device = dst->data;
                        kind = cudaMemcpyDeviceToHost;
                    } else if (dst->backend == GGML_BACKEND_GPU) {
                        dst_off_device = dst_extra->data_device[g_system_gpu_status.main_device_id];
                        kind = cudaMemcpyDeviceToDevice;
                    } else {
                        GGML_ASSERT(false);
                    }
                    if (split) {
                        // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                        // dst is NOT transposed.
                        // The outputs of cuBLAS matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                        // Instead they need to be copied to the correct slice in ne0 = dst row index.
                        // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                        for (int64_t j = 0; j < ne1; ++j) {
                            float * dhf_dst_i = (float *) ((char *) dst_off_device + (j*ne0 + i01_low)*sizeof(float) + i02*nb2 + i03*nb3);
                            CUDA_CHECK(cudaMemcpyAsync(dhf_dst_i, dst_ddf_i + j*i01_diff, i01_diff*sizeof(float), kind, cudaStream_main));
                        }
                    } else {
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                        CUDA_CHECK(cudaMemcpyAsync(dhf_dst_i, dst_ddf_i, dst_stride*sizeof(float), kind, cudaStream_main));
                    }
                }
            }
        }
    }

    // wait until each device is finished, then free their buffers
    for (int id = 0; id < g_system_gpu_status.num_devices; ++id) {
        if (src0_asq[id] == 0 && src0_asf[id] == 0 && src1_asf[id] == 0 && dst_asf[id] == 0) {
            continue;
        }

        CUDA_CHECK(cudaSetDevice(id));
        CUDA_CHECK(cudaDeviceSynchronize());

        if (src0_asq[id] > 0) {
            ggml_cuda_pool_free(src0_ddq[id], src0_asq[id]);
        }
        if (src0_asf[id] > 0) {
            ggml_cuda_pool_free(src0_ddf[id], src0_asf[id]);
        }
        if (src1_asf[id] > 0) {
            ggml_cuda_pool_free(src1_ddf[id], src1_asf[id]);
        }
        if (dst_asf[id] > 0) {
            ggml_cuda_pool_free(dst_ddf[id], dst_asf[id]);
        }
    }
}

void ggml_cuda_add(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    ggml_cuda_op(src0, src1, dst, ggml_cuda_op_add, true, true);
}

void ggml_cuda_mul(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    ggml_cuda_op(src0, src1, dst, ggml_cuda_op_mul, true, false); // TODO ggml_cuda_op needs modification for flatten
}

void ggml_cuda_silu(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    ggml_cuda_op(src0, src1, dst, ggml_cuda_op_silu, true, true);
}

void ggml_cuda_rms_norm(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    ggml_cuda_op(src0, src1, dst, ggml_cuda_op_rms_norm, true, true);
}

bool ggml_cuda_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {
    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    // if cuda is disabled we reject
    if (g_system_gpu_status.num_devices == 0) {
        return false;
    }
    if (dst->meta.cuda_op_directive != -1) return dst->meta.cuda_op_directive? true : false;
    if (src0->meta.cuda_op_directive != -1) return src0->meta.cuda_op_directive? true : false;
    if (src1->meta.cuda_op_directive != -1) return src1->meta.cuda_op_directive? true : false;
    
    // TODO: find the optimal values for these
    if ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) &&
        src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32 &&
        (ne0 >= 32 && ne1 >= 32 && ne10 >= 32)) {
            //todo: wouldn't it make sense to switch based on flops required  instead ?
            // printf("can_mul_mat: true for shape: %ld %ld %ld\n", ne0, ne1, ne10);
        return true;
    }

    return false;
}

void ggml_cuda_mul_mat_vec_p021(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst){
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(src0->backend != GGML_BACKEND_GPU_SPLIT);
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]); // 0213 permutation
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]); // 0213 permutation
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    CUDA_CHECK(cudaSetDevice(g_system_gpu_status.main_device_id));
    cudaStream_t cudaStream_main = g_cudaStreams_main[g_system_gpu_status.main_device_id][0];

    struct ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;
    void * src0_ddq = src0_extra->data_device[g_system_gpu_status.main_device_id];

    struct ggml_tensor_extra_gpu * src1_extra = (ggml_tensor_extra_gpu *) src1->extra;
    float * src1_ddf = (float *) src1_extra->data_device[g_system_gpu_status.main_device_id];

    struct ggml_tensor_extra_gpu * dst_extra = (ggml_tensor_extra_gpu *) dst->extra;
    float * dst_ddf = (float *) dst_extra->data_device[g_system_gpu_status.main_device_id];

    ggml_mul_mat_p021_f16_f32_cuda(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, ne02, cudaStream_main);

    CUDA_CHECK(cudaDeviceSynchronize());
}

void ggml_cuda_mul_mat_vec_nc(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst){
    GGML_ASSERT(!ggml_is_contiguous(src0) && ggml_is_contiguous(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(src0->backend != GGML_BACKEND_GPU_SPLIT);
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];

    CUDA_CHECK(cudaSetDevice(g_system_gpu_status.main_device_id));
    cudaStream_t cudaStream_main = g_cudaStreams_main[g_system_gpu_status.main_device_id][0];

    struct ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;
    void * src0_ddq = src0_extra->data_device[g_system_gpu_status.main_device_id];

    struct ggml_tensor_extra_gpu * src1_extra = (ggml_tensor_extra_gpu *) src1->extra;
    float * src1_ddf = (float *) src1_extra->data_device[g_system_gpu_status.main_device_id];

    struct ggml_tensor_extra_gpu * dst_extra = (ggml_tensor_extra_gpu *) dst->extra;
    float * dst_ddf = (float *) dst_extra->data_device[g_system_gpu_status.main_device_id];

    const int row_stride_x = nb01 / sizeof(half);
    const int channel_stride_x = nb02 / sizeof(half);

    ggml_mul_mat_vec_nc_f16_f32_cuda(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, row_stride_x, ne02, channel_stride_x, cudaStream_main);

    CUDA_CHECK(cudaDeviceSynchronize());
}

void ggml_cuda_mul_mat(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    bool all_on_device = (src0->backend == GGML_BACKEND_GPU || src0->backend == GGML_BACKEND_GPU_SPLIT) &&
        src1->backend == GGML_BACKEND_GPU && dst->backend == GGML_BACKEND_GPU;

    dst->meta.cuda_perf_mal_mul_type=1;
    if (all_on_device && ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
        ggml_cuda_mul_mat_vec_p021(src0, src1, dst);
    } else if (all_on_device && !ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && src1->ne[1] == 1) {
        ggml_cuda_mul_mat_vec_nc(src0, src1, dst);
    }else if (src0->type == GGML_TYPE_F32) {
        ggml_cuda_op(src0, src1, dst, ggml_cuda_op_mul_mat_cublas, true, false);
    } else {
        GGML_ASSERT(false);
    }
}

void ggml_cuda_scale(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    ggml_cuda_op(src0, src1, dst, ggml_cuda_op_scale, true, true);
}

void ggml_cuda_cpy(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne == ggml_nelements(src1));

    GGML_ASSERT(src0->backend == GGML_BACKEND_GPU);
    GGML_ASSERT(src1->backend == GGML_BACKEND_GPU);

    GGML_ASSERT(ggml_nbytes(src0) <= INT_MAX);
    GGML_ASSERT(ggml_nbytes(src1) <= INT_MAX);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    GGML_ASSERT(src0->ne[3] == 1);

    const int64_t nb00 = src0->nb[0];
    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    GGML_ASSERT(src1->ne[3] == 1);

    const int64_t nb10 = src1->nb[0];
    const int64_t nb11 = src1->nb[1];
    const int64_t nb12 = src1->nb[2];

    CUDA_CHECK(cudaSetDevice(g_system_gpu_status.main_device_id));
    cudaStream_t cudaStream_main = g_cudaStreams_main[g_system_gpu_status.main_device_id][0];

    const struct ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;
    const struct ggml_tensor_extra_gpu * src1_extra = (ggml_tensor_extra_gpu *) src1->extra;

    char * src0_ddc = (char *) src0_extra->data_device[g_system_gpu_status.main_device_id];
    char * src1_ddc = (char *) src1_extra->data_device[g_system_gpu_status.main_device_id];

    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f32_f32_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, nb00, nb01, nb02,
                              ne10, ne11, nb10, nb11, nb12, cudaStream_main);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f32_f16_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, nb00, nb01, nb02,
                              ne10, ne11, nb10, nb11, nb12, cudaStream_main);
    } else {
        GGML_ASSERT(false);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    (void) dst;
}

void ggml_cuda_diag_mask_inf(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    ggml_cuda_op(src0, src1, dst, ggml_cuda_op_diag_mask_inf, true, true);
}

void ggml_cuda_soft_max(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    ggml_cuda_op(src0, src1, dst, ggml_cuda_op_soft_max, true, true);
}

void ggml_cuda_rope(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    ggml_cuda_op(src0, src1, dst, ggml_cuda_op_rope, true, false); // FIXME flatten changes results
}

void ggml_cuda_nop(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    (void) src0;
    (void) src1;
    (void) dst;
}

// copy tensor (*data) to the correct device(s) and assign tensor->extra->data_device[] with cuda mem pointer
void ggml_cuda_transform_tensor(void * data, struct ggml_tensor * tensor) {
    int nrows = ggml_nrows(tensor);
    const size_t nb1 = tensor->nb[1];
    ggml_backend backend = tensor->backend;
    struct ggml_tensor_extra_gpu * extra = new struct ggml_tensor_extra_gpu;
    memset(extra, 0, sizeof(*extra));

    for (int id = 0; id < g_system_gpu_status.num_devices; ++id) {
        if (backend == GGML_BACKEND_GPU && id != g_system_gpu_status.main_device_id) {
            continue;
        }

        cudaSetDevice(id);

        int row_low, row_high;
        if (backend == GGML_BACKEND_GPU) {
            row_low = 0;
            row_high = nrows;
        } else if (backend == GGML_BACKEND_GPU_SPLIT) {
            row_low = id == 0 ? 0 : nrows*g_tensor_split[id];
            row_high = id == g_system_gpu_status.num_devices - 1 ? nrows : nrows*g_tensor_split[id + 1];
        } else {
            GGML_ASSERT(false);
        }
        if (row_low == row_high) {
            continue;
        }

        int64_t nrows_split = row_high - row_low;

        const size_t offset_split = row_low*nb1;
        const size_t size = ggml_nbytes_split(tensor, nrows_split);

        void * buf;
        CUDA_CHECK(cudaMalloc(&buf, size));
        void * buf_host = (char*)data + offset_split;

        cudaMemcpy(buf, buf_host, size, cudaMemcpyHostToDevice);

        extra->data_device[id] = buf;
    }

    tensor->extra = extra;
}

void ggml_cuda_free_data(struct ggml_tensor * tensor) {
    if (tensor->backend != GGML_BACKEND_GPU && tensor->backend != GGML_BACKEND_GPU_SPLIT) {
        return;
    }

    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) tensor->extra;

    for (int id = 0; id < g_system_gpu_status.num_devices; ++id) {
        if (extra->data_device[id] == nullptr) {
            continue;
        }

        CUDA_CHECK(cudaSetDevice(id));
        CUDA_CHECK(cudaFree(extra->data_device[id]));
    }

    delete extra;
}

void ggml_cuda_assign_buffers_impl(struct ggml_tensor * tensor, bool scratch) {
    if (scratch && g_scratch_size == 0) {
        return;
    }

    // recursively assign CUDA buffers until a compute tensor is found
    if (tensor->src0 != nullptr && tensor->src0->backend == GGML_BACKEND_CPU) {
        const ggml_op src0_op = tensor->src0->op;
        if (src0_op == GGML_OP_RESHAPE || src0_op == GGML_OP_TRANSPOSE || src0_op == GGML_OP_VIEW) {
            ggml_cuda_assign_buffers_impl(tensor->src0, scratch);
        }
    }
    if (tensor->op == GGML_OP_CPY && tensor->src1->backend == GGML_BACKEND_CPU) {
        ggml_cuda_assign_buffers_impl(tensor->src1, scratch);
    }

    tensor->backend = GGML_BACKEND_GPU;
    struct ggml_tensor_extra_gpu * extra = new ggml_tensor_extra_gpu;

    const bool inplace = (tensor->src0 != nullptr && tensor->src0->data == tensor->data) ||
        tensor->op == GGML_OP_VIEW;
    const size_t size = ggml_nbytes(tensor);

    CUDA_CHECK(cudaSetDevice(g_system_gpu_status.main_device_id));
    if (inplace && tensor->src0->backend == GGML_BACKEND_GPU) {
        struct ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu * ) tensor->src0->extra;
        char * src0_ddc = (char *) src0_extra->data_device[g_system_gpu_status.main_device_id];
        size_t offset = 0;
        if (tensor->op == GGML_OP_VIEW) {
            memcpy(&offset, tensor->opt[0]->data, sizeof(size_t));
        }
        extra->data_device[g_system_gpu_status.main_device_id] = src0_ddc + offset;
    } else if (tensor->op == GGML_OP_CPY) {
        struct ggml_tensor_extra_gpu * src1_extra = (ggml_tensor_extra_gpu * ) tensor->src1->extra;
        void * src1_ddv = src1_extra->data_device[g_system_gpu_status.main_device_id];
        extra->data_device[g_system_gpu_status.main_device_id] = src1_ddv;
    } else if (scratch) {
        GGML_ASSERT(size <= g_scratch_size);
        if (g_scratch_offset + size > g_scratch_size) {
            g_scratch_offset = 0;
        }

        char * data = (char *) g_scratch_buffer;
        if (data == nullptr) {
            CUDA_CHECK(cudaMalloc(&data, g_scratch_size));
            g_scratch_buffer = data;
        }
        extra->data_device[g_system_gpu_status.main_device_id] = data + g_scratch_offset;

        g_scratch_offset += size;

        GGML_ASSERT(g_scratch_offset <= g_scratch_size);
    } else { // allocate new buffers outside of scratch
        void * data;
        CUDA_CHECK(cudaMalloc(&data, size));
        CUDA_CHECK(cudaMemset(data, 0, size));
        extra->data_device[g_system_gpu_status.main_device_id] = data;
    }

    tensor->extra = extra;
}

void ggml_cuda_assign_buffers(struct ggml_tensor * tensor) {
    ggml_cuda_assign_buffers_impl(tensor, true);
}

void ggml_cuda_assign_buffers_no_scratch(struct ggml_tensor * tensor) {
    ggml_cuda_assign_buffers_impl(tensor, false);
}

void ggml_cuda_set_main_device(int main_device) {
    // if (main_device >= g_system_gpu_status.num_devices) {
    //     fprintf(stderr, "warning: cannot set main_device=%d because there are only %d devices. Using device %d instead.\n",
    //             main_device, g_system_gpu_status.num_devices, g_system_gpu_status.main_device_id);
    //     return;
    // }
    // we accept setting it before initialization
    g_system_gpu_status.main_device_id = main_device;
}
void ggml_cuda_set_vram_reserved(int64_t vram_reserved_bytes) {
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i)
    {
        g_system_gpu_status.device_vram_reserved[i] = vram_reserved_bytes;
    }
}

void ggml_cuda_set_scratch_size(size_t scratch_size) {
    g_scratch_size = scratch_size;
}

void ggml_cuda_free_scratch() {
    if (g_scratch_buffer == nullptr) {
        return;
    }

    CUDA_CHECK(cudaFree(g_scratch_buffer));
    g_scratch_buffer = nullptr;
}

bool ggml_cuda_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor){
    ggml_cuda_func_t func;

    if (tensor->op == GGML_OP_NONE)
        return true;
    // user has disabled cuda or no devices found
    if (g_system_gpu_status.num_devices == 0) 
         return false;
    // allow manual skip
    if (tensor->meta.cuda_op_directive == 0) 
        return false;


    const bool any_on_device = tensor->backend == GGML_BACKEND_GPU
        || tensor->src0->backend == GGML_BACKEND_GPU || tensor->src0->backend == GGML_BACKEND_GPU_SPLIT
        || (tensor->src1 != nullptr && tensor->src1->backend == GGML_BACKEND_GPU);

    switch (tensor->op) {
        case GGML_OP_ADD:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cuda_add;
            break;
        case GGML_OP_MUL:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cuda_mul;
            break;
        case GGML_OP_SILU:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cuda_silu;
            break;
        case GGML_OP_RMS_NORM:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cuda_rms_norm;
            break;
        case GGML_OP_MUL_MAT:
            if (!any_on_device && !ggml_cuda_can_mul_mat(tensor->src0, tensor->src1, tensor)) {
                return false;
            }
            func = ggml_cuda_mul_mat;
            break;
        case GGML_OP_SCALE:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cuda_scale;
            break;
        case GGML_OP_CPY:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cuda_cpy;
            break;
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cuda_nop;
            break;
        case GGML_OP_DIAG_MASK_INF:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cuda_diag_mask_inf;
            break;
        case GGML_OP_SOFT_MAX:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cuda_soft_max;
            break;
        case GGML_OP_ROPE:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cuda_rope;
            break;
        default:
            return false;
    }

    if (params->ith != 0) {
        return true;
    }
    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return true;
    }
    func(tensor->src0, tensor->src1, tensor);
    return true;
}