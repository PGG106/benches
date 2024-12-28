#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <riscv_vector.h>
#include <string>
#include "../common/common01.h"

constexpr size_t RUNS = 100;
constexpr size_t N = 512;

void matmul_scalar(
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

void vector_matmul(
    const int32_t *__restrict__ a,
    const int32_t *__restrict__ b,
    int32_t *__restrict__ c)
{
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    size_t vl = 0;
    for (int j = 0; j < N; ++j)
    {
        for (int i = 0; i < N; ++i)
        {
            vint32m1_t vec_s = __riscv_vmv_v_x_i32m1(0, vlmax);
            vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vlmax);
            for (int k = 0; k < N; k += __riscv_vsetvl_e32m1(N - k))
            {
                vl = __riscv_vsetvl_e32m1(N - k);
                auto *ptr_a = a + j * N + k;
                auto *ptr_b = b + i * N + k;
                vint32m1_t vec_a = __riscv_vle32_v_i32m1(ptr_a, vl);
                vint32m1_t vec_b = __riscv_vle32_v_i32m1(ptr_b, vl);
                vec_s = __riscv_vmacc_vv_i32m1(vec_s, vec_a, vec_b, vl);
            }
            int sum = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m1_i32m1(vec_s, vec_zero, vlmax));
            c[j * N + i] = sum;
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

void bench_scalar(const int32_t *__restrict__ a_ptr,
                  const int32_t *__restrict__ b_ptr,
                  int32_t *__restrict__ c_scalar_ptr)
{
    {
        TimerStats te("Scalar Matmul With Mul (vector_matmul_scalar)");
        for (volatile size_t i = 0; i < RUNS; i++)
        {
            TimerScope ts(te);
            matmul_scalar(a_ptr, b_ptr, c_scalar_ptr);
        }
    }
}

void bench_rvv(const int32_t *__restrict__ a_ptr,
               const int32_t *__restrict__ b_ptr,
               int32_t *__restrict__ c_avx_mul_ptr)
{
    {
        TimerStats te("AVX Matmul With Mul (vector_matmul_avx)");
        for (volatile size_t i = 0; i < RUNS; i++)
        {
            TimerScope ts(te);
            vector_matmul(a_ptr, b_ptr, c_avx_mul_ptr);
        }
    }
}

int main(int argc, char **argv)
{
    constexpr size_t ALIGNMENT = 32; // 32-byte alignment

    auto *a_ptr = static_cast<int32_t *>(aligned_alloc(ALIGNMENT, N * N * sizeof(int32_t)));
    auto *b_ptr = static_cast<int32_t *>(aligned_alloc(ALIGNMENT, N * N * sizeof(int32_t)));
    auto *c_scalar_ptr = static_cast<int32_t *>(aligned_alloc(ALIGNMENT, N * N * sizeof(int32_t)));
    auto *c_rvv_ptr = static_cast<int32_t *>(aligned_alloc(ALIGNMENT, N * N * sizeof(int32_t)));

    wipe(c_scalar_ptr, N * N);
    wipe(c_rvv_ptr, N * N);

    for (size_t j = 0; j < N; j++)
    {
        for (size_t i = 0; i < N; i++)
        {
            a_ptr[j * N + i] = static_cast<int32_t>(rand()); // `a` is row major
            b_ptr[i * N + j] = static_cast<int32_t>(rand());
        }
    }

    bench_scalar(a_ptr, b_ptr, c_scalar_ptr);

    bench_rvv(a_ptr, b_ptr, c_rvv_ptr);

    verify_results(c_scalar_ptr, c_rvv_ptr);

    return 0;
}

// data points

// Baseline: 209.938

// Autovec: 89.39

// SIMD : 19.63
