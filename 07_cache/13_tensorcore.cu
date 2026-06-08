#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <chrono>
using namespace std;
using namespace nvcuda;

#define TILE_M 128
#define TILE_N 128
#define TILE_K 32
#define PAD    8

__global__ __launch_bounds__(512, 2)
void kernel(int dim_m, int dim_n, int dim_k,
            const float *__restrict__ d_a,
            const float *__restrict__ d_b,
            float *__restrict__ d_c) {
  const int om      = TILE_M * blockIdx.x;
  const int on      = TILE_N * blockIdx.y;
  const int tid     = threadIdx.x;
  const int warp_id = tid / 32;

  __shared__ half sA[2][TILE_K][TILE_M + PAD];
  __shared__ half sB[2][TILE_N][TILE_K + PAD];

  // Sixteen warps split a 128x128 CTA tile into 16x64 warp tiles.
  const int wr = warp_id / 2;
  const int wc = (warp_id % 2) * 4;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
  for (int c = 0; c < 4; c++)
    wmma::fill_fragment(acc[c], 0.0f);

  // Load the first K tile. A is vectorized along M, B along K.
  int buf = 0;
  for (int v = tid; v < TILE_K * TILE_M / 4; v += blockDim.x) {
    int kk = v / (TILE_M / 4);
    int row = (v % (TILE_M / 4)) * 4;
    float4 x = *reinterpret_cast<const float4 *>(
        &d_a[kk * dim_m + om + row]);
    *reinterpret_cast<half2 *>(&sA[0][kk][row]) =
        __floats2half2_rn(x.x, x.y);
    *reinterpret_cast<half2 *>(&sA[0][kk][row + 2]) =
        __floats2half2_rn(x.z, x.w);
  }
  for (int v = tid; v < TILE_N * TILE_K / 4; v += blockDim.x) {
    int col = v / (TILE_K / 4);
    int kk = (v % (TILE_K / 4)) * 4;
    float4 x = *reinterpret_cast<const float4 *>(
        &d_b[(on + col) * dim_k + kk]);
    *reinterpret_cast<half2 *>(&sB[0][col][kk]) =
        __floats2half2_rn(x.x, x.y);
    *reinterpret_cast<half2 *>(&sB[0][col][kk + 2]) =
        __floats2half2_rn(x.z, x.w);
  }
  __syncthreads();

  for (int k = 0; k < dim_k; k += TILE_K) {
    int nb = 1 - buf;

    if (k + TILE_K < dim_k) {
      for (int v = tid; v < TILE_K * TILE_M / 4; v += blockDim.x) {
        int kk = v / (TILE_M / 4);
        int row = (v % (TILE_M / 4)) * 4;
        float4 x = *reinterpret_cast<const float4 *>(
            &d_a[(k + TILE_K + kk) * dim_m + om + row]);
        *reinterpret_cast<half2 *>(&sA[nb][kk][row]) =
            __floats2half2_rn(x.x, x.y);
        *reinterpret_cast<half2 *>(&sA[nb][kk][row + 2]) =
            __floats2half2_rn(x.z, x.w);
      }
      for (int v = tid; v < TILE_N * TILE_K / 4; v += blockDim.x) {
        int col = v / (TILE_K / 4);
        int kk = (v % (TILE_K / 4)) * 4;
        float4 x = *reinterpret_cast<const float4 *>(
            &d_b[(on + col) * dim_k + k + TILE_K + kk]);
        *reinterpret_cast<half2 *>(&sB[nb][col][kk]) =
            __floats2half2_rn(x.x, x.y);
        *reinterpret_cast<half2 *>(&sB[nb][col][kk + 2]) =
            __floats2half2_rn(x.z, x.w);
      }
    }

    for (int kk = 0; kk < TILE_K; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
                     wmma::col_major> a_frag;
      wmma::load_matrix_sync(
          a_frag, &sA[buf][kk][wr * 16], TILE_M + PAD);

      for (int c = 0; c < 4; c++) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half,
                       wmma::col_major> b_frag;
        wmma::load_matrix_sync(
            b_frag, &sB[buf][(wc + c) * 16][kk], TILE_K + PAD);
        wmma::mma_sync(acc[c], a_frag, b_frag, acc[c]);
      }
    }

    __syncthreads();
    buf = nb;
  }

  for (int c = 0; c < 4; c++) {
    int cm = om + wr * 16;
    int cn = on + (wc + c) * 16;
    if (cm < dim_m && cn < dim_n)
      wmma::store_matrix_sync(
          &d_c[cn * dim_m + cm], acc[c], dim_m,
          wmma::mem_col_major);
  }
}

int main(int argc, const char **argv) {
  int m = 10240;
  int k = 4096;
  int n = 8192;
  float alpha = 1.0;
  float beta = 0.0;
  int Nt = 10;
  float *A, *B, *C, *C2;
  cudaMallocManaged(&A, m * k * sizeof(float));
  cudaMallocManaged(&B, k * n * sizeof(float));
  cudaMallocManaged(&C, m * n * sizeof(float));
  cudaMallocManaged(&C2, m * n * sizeof(float));
  for (int i=0; i<m; i++)
    for (int j=0; j<k; j++)
      A[k*i+j] = drand48();
  for (int i=0; i<k; i++)
    for (int j=0; j<n; j++)
      B[n*i+j] = drand48();
  for (int i=0; i<n; i++)
    for (int j=0; j<m; j++)
      C[m*i+j] = C2[m*i+j] = 0;

  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);

  auto tic = chrono::steady_clock::now();
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    cublasGemmEx(cublas_handle,
                 CUBLAS_OP_N, CUBLAS_OP_N,
                 m, n, k, &alpha,
                 A, CUDA_R_32F, m,
                 B, CUDA_R_32F, k,
                 &beta,
                 C, CUDA_R_32F, m,
                 CUBLAS_COMPUTE_32F_FAST_16F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();
  }
  auto toc = chrono::steady_clock::now();

  int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
  double tcublas = chrono::duration<double>(toc - tic).count() / Nt;
  double cublas_flops = double(num_flops) / tcublas / 1.0e9;

  dim3 block(512);
  dim3 grid((m + TILE_M - 1) / TILE_M,
            (n + TILE_N - 1) / TILE_N);

  tic = chrono::steady_clock::now();
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    kernel<<<grid, block>>>(m, n, k, A, B, C2);
    cudaDeviceSynchronize();
  }
  toc = chrono::steady_clock::now();

  double twmma = chrono::duration<double>(toc - tic).count() / Nt;
  double wmma_flops = double(num_flops) / twmma / 1.0e9;
  printf("CUBLAS: %.2f Gflops, WMMA: %.2f Gflops\n", cublas_flops, wmma_flops);

  double err = 0;
  for (int i=0; i<n; i++)
    for (int j=0; j<m; j++)
      err += fabs(C[m*i+j] - C2[m*i+j]);
  printf("error: %lf\n", err/n/m);

  cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(C2);
  cublasDestroy(cublas_handle);
}
