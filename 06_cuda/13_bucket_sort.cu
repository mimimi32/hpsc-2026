#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(err__));                                 \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// ① バケットのカウント: 各 key を対応するバケットに atomicAdd で加算
__global__ void count_buckets(const int *key, int *bucket, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    atomicAdd(&bucket[key[i]], 1);
  }
}

// ② 排他的プレフィックス和 (Hillis-Steele scan)
// bucket[0..range-1] を入力として、offset[i] = bucket[0] + ... + bucket[i-1] を計算する
// 注意: 1 ブロック内で完結させるため、range <= 1024 を前提とする
__global__ void exclusive_scan(const int *bucket, int *offset, int range) {
  extern __shared__ int temp[];
  int i = threadIdx.x;
  if (i >= range) return;

  // 排他的スキャンにするため、1つ右にずらして読み込む
  // offset[0] = 0, offset[i] = sum of bucket[0..i-1]
  temp[i] = (i > 0) ? bucket[i - 1] : 0;
  __syncthreads();

  // Hillis-Steele scan: log2(range) ステップで完了
  for (int j = 1; j < range; j <<= 1) {
    int val = temp[i];
    __syncthreads();
    if (i >= j) {
      temp[i] = val + temp[i - j];
    }
    __syncthreads();
  }

  offset[i] = temp[i];
}

// ③ 書き戻し: 各スレッドが 1 つのバケットを担当し、対応する範囲を埋める
__global__ void expand_buckets(const int *bucket, const int *offset, int *key,
                               int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < range) {
    int begin = offset[i];
    for (int j = 0; j < bucket[i]; ++j) {
      key[begin + j] = i;
    }
  }
}

int main() {
  int n = 50;
  int range = 5;

  std::vector<int> key(n);
  for (int i = 0; i < n; i++) {
    key[i] = rand() % range;
    std::printf("%d ", key[i]);
  }
  std::printf("\n");

  int *d_key = nullptr;
  int *d_bucket = nullptr;
  int *d_offset = nullptr;
  int *d_sorted = nullptr;

  CUDA_CHECK(cudaMalloc(&d_key, n * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_bucket, range * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_offset, range * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_sorted, n * sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_key, key.data(), n * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_bucket, 0, range * sizeof(int)));

  // ① カウント (n スレッド使用)
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  count_buckets<<<blocks, threads>>>(d_key, d_bucket, n);
  CUDA_CHECK(cudaGetLastError());

  // ② プレフィックス和 (range スレッド使用、1 ブロック内で完結)
  // shared memory のサイズは range * sizeof(int)
  exclusive_scan<<<1, range, range * sizeof(int)>>>(d_bucket, d_offset, range);
  CUDA_CHECK(cudaGetLastError());

  // ③ 書き戻し (range スレッド使用)
  expand_buckets<<<1, range>>>(d_bucket, d_offset, d_sorted, range);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<int> sorted(n);
  CUDA_CHECK(cudaMemcpy(sorted.data(), d_sorted, n * sizeof(int),
                        cudaMemcpyDeviceToHost));

  for (int i = 0; i < n; i++) {
    std::printf("%d ", sorted[i]);
  }
  std::printf("\n");

  CUDA_CHECK(cudaFree(d_key));
  CUDA_CHECK(cudaFree(d_bucket));
  CUDA_CHECK(cudaFree(d_offset));
  CUDA_CHECK(cudaFree(d_sorted));

  return 0;
}