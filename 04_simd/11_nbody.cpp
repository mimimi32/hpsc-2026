#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  // j ループの全データを一括ロード
  __m512 xvec = _mm512_load_ps(x);
  __m512 yvec = _mm512_load_ps(y);
  __m512 mvec = _mm512_load_ps(m);
  // j のインデックスベクトル {0,1,2,...,15}
  __m512 jvec = _mm512_setr_ps(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);

  for(int i=0; i<N; i++) {
    // i 番目の値を全レーンにブロードキャスト
    __m512 xi = _mm512_set1_ps(x[i]);
    __m512 yi = _mm512_set1_ps(y[i]);
    __m512 ii = _mm512_set1_ps((float)i);

    // rx = x[i] - x[j],  ry = y[i] - y[j]
    __m512 rx = _mm512_sub_ps(xi, xvec);
    __m512 ry = _mm512_sub_ps(yi, yvec);

    // r2 = rx*rx + ry*ry
    __m512 r2 = _mm512_add_ps(_mm512_mul_ps(rx, rx),
                              _mm512_mul_ps(ry, ry));

    // 1/r を rsqrt で計算
    __m512 invr  = _mm512_rsqrt14_ps(r2);
    __m512 invr3 = _mm512_mul_ps(_mm512_mul_ps(invr, invr), invr);

    // i != j のマスクを作る
    __mmask16 mask = _mm512_cmp_ps_mask(ii, jvec, _MM_CMPINT_NE);

    // fxv = rx * m[j] * (1/r^3),  fyv = ry * m[j] * (1/r^3)
    __m512 fxv = _mm512_mul_ps(_mm512_mul_ps(rx, mvec), invr3);
    __m512 fyv = _mm512_mul_ps(_mm512_mul_ps(ry, mvec), invr3);

    // i==j のレーンはゼロにして、横方向に総和を取る
    fx[i] -= _mm512_mask_reduce_add_ps(mask, fxv);
    fy[i] -= _mm512_mask_reduce_add_ps(mask, fyv);

    printf("%d %g %g\n", i, fx[i], fy[i]);
  }
}
