#include <cstddef>
#include <cstdint>
#include <stdint.h>
#include <stdio.h>
#include <atomic>

#include <cuda_runtime.h>

#include "ggml-cuda.h"
#include "ggml.h"


#define CUDA_ADD_BLOCK_SIZE 256
#define CUDA_MUL_BLOCK_SIZE 256
#define CUDA_SILU_BLOCK_SIZE 256
#define CUDA_CPY_BLOCK_SIZE 32
#define CUDA_SCALE_BLOCK_SIZE 256
#define CUDA_ROPE_BLOCK_SIZE 256
#define CUDA_DIAG_MASK_INF_BLOCK_SIZE 32

#define CUDA_CHECK(err)                                                                 \
    do {                                                                                \
        cudaError_t err_ = (err);                                                       \
        if (err_ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %d at %s:%d: %s\n", err_, __FILE__, __LINE__,   \
                cudaGetErrorString(err_));                                              \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)

#define CUBLAS_CHECK(err)                                                               \
    do {                                                                                \
        cublasStatus_t err_ = (err);                                                    \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                            \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", err_, __FILE__, __LINE__);    \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)

static void ggml_cuda_mul_mat_f32(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int x_ne = ne01 * ne00;
    const int y_ne = ne11 * ne10;
    const int d_ne = ne11 * ne01;
    const int n_mm = ne03 * ne02;

    size_t x_size, y_size, d_size;
    float * d_X = (float *) ggml_cuda_pool_malloc(n_mm * sizeof(float) * x_ne, &x_size);
    float * d_Y = (float *) ggml_cuda_pool_malloc(n_mm * sizeof(float) * y_ne, &y_size);
    float * d_D = (float *) ggml_cuda_pool_malloc(n_mm * sizeof(float) * d_ne, &d_size);

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            int i = i03*ne02 + i02;
            cudaStream_t cudaStream = g_cudaStreams[i % GGML_CUDA_MAX_STREAMS];

            float * c_X = d_X + i * x_ne;
            float * c_Y = d_Y + i * y_ne;
            float * c_D = d_D + i * d_ne;

            // copy data to device
            CUDA_CHECK(ggml_cuda_h2d_tensor_2d(c_X, src0, i03, i02, cudaStream));
            CUDA_CHECK(ggml_cuda_h2d_tensor_2d(c_Y, src1, i03, i02, cudaStream));

            // compute
            CUBLAS_CHECK(cublasSetStream(g_cublasH, cudaStream));
            CUBLAS_CHECK(
                cublasSgemm(g_cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                        ne01, ne11, ne10,
                        &alpha, c_X, ne00,
                                c_Y, ne10,
                        &beta,  c_D, ne01));

            // copy dst to host
            float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);
            CUDA_CHECK(cudaMemcpyAsync(d, c_D, sizeof(float) * d_ne, cudaMemcpyDeviceToHost, cudaStream));
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    ggml_cuda_pool_free(d_X, x_size);
    ggml_cuda_pool_free(d_Y, y_size);
    ggml_cuda_pool_free(d_D, d_size);
}

static void * ggml_cuda_pool_malloc(size_t size, size_t * actual_size) {
    scoped_spin_lock lock(g_cuda_pool_lock);

    for (int i = 0; i < MAX_CUDA_BUFFERS; ++i) {
        cuda_buffer& b = g_cuda_buffer_pool[i];
        if (b.size >= size && b.ptr != nullptr) {
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
    }
    void * ptr;
    CUDA_CHECK(cudaMalloc((void **) &ptr, size));
    *actual_size = size;
    return ptr;
}


static void ggml_cuda_pool_free(void * ptr, size_t size) {
    scoped_spin_lock lock(g_cuda_pool_lock);

    for (int i = 0; i < MAX_CUDA_BUFFERS; ++i) {
        cuda_buffer& b = g_cuda_buffer_pool[i];
        if (b.ptr == nullptr) {
            b.ptr = ptr;
            b.size = size;
            return;
        }
    }
    fprintf(stderr, "WARNING: cuda buffer pool full, increase MAX_CUDA_BUFFERS\n");
    CUDA_CHECK(cudaFree(ptr));
}