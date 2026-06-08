#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <mma.h>
#include <chrono>
using namespace std;
using namespace nvcuda;

#define TILE 128
#define PAD  8   // avoid shared memory bank conflicts

__global__ void kernel(int dim_m, int dim_n, int dim_k,
                       float *d_a, float *d_b, float *d_c) {
  const int om      = TILE * blockIdx.x;
  const int on      = TILE * blockIdx.y;
  const int i       = threadIdx.x;       // 0..127
  const int warp_id = i / 32;            // 0..3

  // Double-buffered shared memory with padding
  __shared__ half sA[2][16][TILE + PAD];
  __shared__ half sB[2][16][TILE + PAD];

  // Each warp owns 32 rows x 128 cols of the 128x128 output tile
  // warp 0: rows 0-31, warp 1: rows 32-63, warp 2: rows 64-95, warp 3: rows 96-127
  const int wr = warp_id * 2;  // first row-tile index (each tile = 16 rows)

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][8];
  for (int r = 0; r < 2; r++)
    for (int c = 0; c < 8; c++)
      wmma::fill_fragment(acc[r][c], 0.0f);

  // Load first k-slab into buffer 0
  int buf = 0;
  for (int j = 0; j < 16; j++) {
    sA[0][j][i] = __float2half(d_a[j * dim_m + om + i]);
    sB[0][j][i] = __float2half(d_b[(on + i) * dim_k + j]);
  }
  __syncthreads();

  for (int k = 0; k < dim_k; k += 16) {
    int nb = 1 - buf;

    // Prefetch next k-slab while computing current (double buffering)
    if (k + 16 < dim_k) {
      for (int j = 0; j < 16; j++) {
        sA[nb][j][i] = __float2half(d_a[(k+16+j) * dim_m + om + i]);
        sB[nb][j][i] = __float2half(d_b[(on + i) * dim_k + k+16+j]);
      }
    }

    // Compute using current buffer: 2 row-tiles x 8 col-tiles per warp
    for (int r = 0; r < 2; r++) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
      wmma::load_matrix_sync(a_frag, &sA[buf][0][(wr+r)*16], TILE+PAD);
      for (int c = 0; c < 8; c++) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
        wmma::load_matrix_sync(b_frag, &sB[buf][0][c*16], TILE+PAD);
        wmma::mma_sync(acc[r][c], a_frag, b_frag, acc[r][c]);
      }
    }

    __syncthreads();
    buf = nb;
  }

  // Write accumulators to C
  for (int r = 0; r < 2; r++)
    for (int c = 0; c < 8; c++) {
      int cm = om + (wr+r)*16, cn = on + c*16;
      if (cm < dim_m && cn < dim_n)
        wmma::store_matrix_sync(&d_c[cn * dim_m + cm], acc[r][c], dim_m, wmma::mem_col_major);
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

  dim3 block(TILE);
  dim3 grid((m+TILE-1)/TILE, (n+TILE-1)/TILE);

  tic = chrono::steady_clock::now();
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    kernel<<<grid, block>>>(m, n, k, A, B, C2);
    cudaDeviceSynchronize();
  }
  toc = chrono::steady_clock::now();

  double tcutlass = chrono::duration<double>(toc - tic).count() / Nt;
  double cutlass_flops = double(num_flops) / tcutlass / 1.0e9;
  printf("CUBLAS: %.2f Gflops, CUTLASS: %.2f Gflops\n", cublas_flops, cutlass_flops);

  double err = 0;
  for (int i=0; i<n; i++)
    for (int j=0; j<m; j++)
      err += fabs(C[m*i+j] - C2[m*i+j]);
  printf("error: %lf\n", err/n/m);

  cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(C2);
  cublasDestroy(cublas_handle);
}
