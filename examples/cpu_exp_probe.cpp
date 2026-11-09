// Small AVX512 measurement adapter; link against an existing CPU PyTorch build.
// No PyTorch rebuild or kernel replacement. See docs/attention-investigation.md.
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <limits>

using V = at::vec::Vectorized<float>;
static_assert(V::size() == 16, "This measurement adapter requires AVX512 float vectors");
// CPUBlas::gemm itself is not exported by these wheels. This is the exported
// BLAS entry point called by its fp32/highest-precision path for these sizes.
extern "C" void sgemm_(char*, char*, int*, int*, int*, float*, const float*, int*,
                       const float*, int*, float*, float*, int*);

extern "C" uint32_t measured_mxcsr() { return _mm_getcsr(); }
extern "C" int measured_vector_width() { return V::size(); }

extern "C" __attribute__((noinline)) void measured_exp(
    const float* input, float* output, int64_t size, int kind) {
  int64_t i = 0;
  for (; i + V::size() <= size; i += V::size()) {
    V value = V::loadu(input + i);
    V result = kind == 0 ? value.exp_u20() : value.exp();
    result.store(output + i);
  }
  if (i < size) {
    V value = V::loadu(input + i, size - i);
    V result = kind == 0 ? value.exp_u20() : value.exp();
    result.store(output + i, size - i);
  }
}

// Match the float32, unpacked, <= 191-query CPU flash score GEMM, in 32-query
// blocks. Original tensor strides are passed explicitly. The caller supplies
// one batch/head pointer. Column-major GEMM writes a row-major [query,key] view.
extern "C" void measured_scores(
    const float* query, const float* key, float* output,
    int64_t queries, int64_t keys, int64_t width,
    int64_t query_stride, int64_t key_stride, float scale, int backward) {
  for (int64_t first = 0; first < queries; first += 32) {
    const int64_t count = std::min<int64_t>(32, queries - first);
    char trans_a = 'T', trans_b = 'N';
    int m = static_cast<int>(keys), n = static_cast<int>(count), k = static_cast<int>(width);
    int lda = static_cast<int>(key_stride), ldb = static_cast<int>(query_stride), ldc = m;
    float alpha = backward ? scale : 1.0f, beta = 0.0f;
    sgemm_(&trans_a, &trans_b, &m, &n, &k, &alpha,
           key, &lda, query + first * query_stride, &ldb,
           &beta, output + first * keys, &ldc);
  }
  if (!backward) {
    V factor(scale);
    int64_t i = 0, size = queries * keys;
    for (; i + V::size() <= size; i += V::size()) {
      (V::loadu(output + i) * factor).store(output + i);
    }
    for (; i < size; ++i) output[i] *= scale;
  }
}

// Match the forward vector accumulation and scalar tail for one key block.
// The caller supplies already-scaled scores with the causal/additive mask.
extern "C" void measured_forward_probabilities(
    const float* scores, float* probability, float* logsumexp,
    int64_t rows, int64_t keys) {
  for (int64_t row = 0; row < rows; ++row) {
    const float* input = scores + row * keys;
    float* output = probability + row * keys;
    float maximum = at::vec::reduce_all<float>(
        [](V& a, V& b) { return at::vec::maximum(a,b); }, input, keys);
    V sum(0.0f), max_vector(maximum);
    int64_t i = 0;
    for (; i + V::size() <= keys; i += V::size()) {
      V value = (V::loadu(input + i) - max_vector).exp_u20();
      sum = sum + value;
      value.store(output + i);
    }
    sum = sum + V(0.0f);
    float total = at::vec::vec_reduce_all<float>([](V& a, V& b) { return a+b; }, sum);
    for (; i < keys; ++i) {
      output[i] = std::exp(input[i] - maximum);
      total += output[i];
    }
    logsumexp[row] = maximum + std::log(total);
    const float inverse = 1.0f / total;
    for (i = 0; i < keys; ++i) output[i] *= inverse;
  }
}
