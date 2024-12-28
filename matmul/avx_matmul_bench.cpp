#include <chrono>
#include <cmath>
#include <functional>
#include <immintrin.h>
#include <iostream>
#include <string>
#include "../common/common01.h"

constexpr size_t RUNS = 1000;
constexpr size_t N = 1024;

int32_t reduce_avx2(const __m256i &vec)
{
    // Horizontal addition
    // Step 1: Add adjacent pairs
    __m256i v1 = _mm256_hadd_epi32(vec, vec); // 0, 2, 4, 6, 1, 3, 5, 7
    __m256i v2 = _mm256_hadd_epi32(v1, v1);   // 0, 4, 1, 5, 2, 6, 3, 7

    // Step 2: Extract the final result
    int result = _mm256_extract_epi32(v2, 0) + _mm256_extract_epi32(v2, 4);

    return result;
}

void vector_matmul_scalar(
    const int32_t *__restrict__ a,
    const int32_t *__restrict__ b,
    int32_t *__restrict__ c)
{
    for (int j = 0; j < N; ++j)
    {
        for (int i = 0; i < N; ++i)
        {
            c[j * N + i] = 0;
            for (int k = 0; k < N; ++k)
            {
                c[j * N + i] += a[j * N + k] * b[i * N + k]; // b is col major
            }
        }
    }
}

void vector_matmul_scalar(
    const int32_t *__restrict__ a,
    const int32_t *__restrict__ b,
    int32_t *__restrict__ c)
{
    for (int j = 0; j < N; ++j)
    {
        for (int i = 0; i < N; ++i)
        {
            c[j * N + i] = 0;
            for (int k = 0; k < N; ++k)
            {
                c[j * N + i] += a[j * N + k] * b[i * N + k]; // b is col major
            }
        }
    }
}

void vector_matmul_avx2(
    const int32_t *__restrict__ a,
    const int32_t *__restrict__ b,
    int32_t *__restrict__ c)
{
    for (int j = 0; j < N; ++j)
    {
        for (int i = 0; i < N; ++i)
        {
            __m256i vec_s = _mm256_setzero_si256();
            for (int k = 0; k < N; k += 8)
            {
                auto *ptr_a = a + j * N + k; // `a` is row major
                auto *ptr_b = b + i * N + k; // `b` is col major
                __m256i vec_a = _mm256_load_si256((__m256i *)ptr_a);
                __m256i vec_b = _mm256_load_si256((__m256i *)ptr_b);
                __m256i vec_mul = _mm256_mullo_epi32(vec_a, vec_b);
                vec_s = _mm256_add_epi32(vec_s, vec_mul);
            }
            const int32_t sum = reduce_avx2(vec_s);
            c[j * N + i] = sum;
        }
    }
}

void bench_scalar(const int32_t *__restrict__ a_ptr,
                  const int32_t *__restrict__ b_ptr,
                  int32_t *__restrict__ c_scalar_ptr)
{
    {
        TimerStats te("Scalar Matmul With Mul (vector_matmul_scalar)");
        for (volatile size_t i = 0; i < RUNS; i++)
        {
            TimerScope ts(te);
            vector_matmul_scalar(a_ptr, b_ptr, c_scalar_ptr);
        }
    }
}

void bench_avx2(const int32_t *__restrict__ a_ptr,
                const int32_t *__restrict__ b_ptr,
                int32_t *__restrict__ c_avx_mul_ptr)
{
    {
        TimerStats te("AVX Matmul With Mul (vector_matmul_avx)");
        for (volatile size_t i = 0; i < RUNS; i++)
        {
            TimerScope ts(te);
            vector_matmul_avx2(a_ptr, b_ptr, c_avx_mul_ptr);
        }
    }
}

// Function to verify the results of scalar and RVV methods
void verify_results(const int32_t *c1, const int32_t *c2)
{
    for (size_t j = 0; j < N; j++)
    {
        for (size_t i = 0; i < N; i++)
        {
            if (c1[j * N + i] != c2[j * N + i])
            {
                std::cerr << "Results mismatch at index " << i << std::endl;
                std::cerr << "c1[" << j << ", " << i << "] = " << c1[j * N + i] << std::endl;
                std::cerr << "c2[" << j << ", " << i << "] = " << c2[j * N + i] << std::endl;
                return;
            }
        }
    }
    std::cout << "Results match!" << std::endl;
}

void wipe(int32_t *p, size_t len)
{
    for (size_t i = 0; i < len; i++)
    {
        p[i] = 0;
    }
}

int main(int argc, char **argv)
{
    constexpr size_t ALIGNMENT = 32; // 32-byte alignment

    auto *a_ptr = static_cast<int32_t *>(aligned_alloc(ALIGNMENT, N * N * sizeof(int32_t)));
    auto *b_ptr = static_cast<int32_t *>(aligned_alloc(ALIGNMENT, N * N * sizeof(int32_t)));
    auto *c_scalar_ptr = static_cast<int32_t *>(aligned_alloc(ALIGNMENT, N * N * sizeof(int32_t)));
    auto *c_avx_mul_ptr = static_cast<int32_t *>(aligned_alloc(ALIGNMENT, N * N * sizeof(int32_t)));

    wipe(c_scalar_ptr, N * N);
    wipe(c_avx_mul_ptr, N * N);

    for (size_t j = 0; j < N; j++)
    {
        for (size_t i = 0; i < N; i++)
        {
            a_ptr[j * N + i] = static_cast<int32_t>(rand()); // `a` is row major
            b_ptr[i * N + j] = static_cast<int32_t>(rand()); // `b` is col major
        }
    }

    bench_scalar(a_ptr, b_ptr, c_scalar_ptr);

    bench_avx2(a_ptr, b_ptr, c_avx_mul_ptr);

    verify_results(c_scalar_ptr, c_avx_mul_ptr);

    return 0;
}

// data points

// Baseline: 209.938

// Autovec: 34.6567

// SIMD : 33.6646
