/// @file simd_scan.h
/// AVX2 in-register inclusive-scan primitives shared by the blocked
/// (van Herk) kernels in rolling_basic.cpp and rolling_corr.cpp.
#pragma once

#include "fastwindow.h"

#if FW_SIMD

namespace fastwindow {
namespace simd {

/// Forward inclusive additive scan of 4 lanes; carry accumulates across
/// calls (broadcast of the highest lane).
inline __m256d scan_fwd_add(__m256d v, __m256d& carry) {
    const __m256d z = _mm256_setzero_pd();
    __m256d t = _mm256_add_pd(v, _mm256_blend_pd(
        _mm256_permute4x64_pd(v, _MM_SHUFFLE(2, 1, 0, 0)), z, 0x1));
    t = _mm256_add_pd(t, _mm256_blend_pd(
        _mm256_permute4x64_pd(t, _MM_SHUFFLE(1, 0, 0, 0)), z, 0x3));
    t = _mm256_add_pd(t, carry);
    carry = _mm256_permute4x64_pd(t, 0xFF);
    return t;
}

/// Backward (suffix) inclusive additive scan of 4 lanes; carry is the
/// broadcast of the lowest lane.
inline __m256d scan_bwd_add(__m256d v, __m256d& carry) {
    const __m256d z = _mm256_setzero_pd();
    __m256d t = _mm256_add_pd(v, _mm256_blend_pd(
        _mm256_permute4x64_pd(v, _MM_SHUFFLE(3, 3, 2, 1)), z, 0x8));
    t = _mm256_add_pd(t, _mm256_blend_pd(
        _mm256_permute4x64_pd(t, _MM_SHUFFLE(3, 3, 3, 2)), z, 0xC));
    t = _mm256_add_pd(t, carry);
    carry = _mm256_permute4x64_pd(t, 0x00);
    return t;
}

/// Lane-mask of non-finite entries: |v| >= inf, with NaN caught by the
/// unordered comparison (NLT_UQ is true for NaN and ±inf).
inline __m256d bad_lanes(__m256d v) {
    const __m256d absmask = _mm256_castsi256_pd(
        _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL));
    const __m256d vinf = _mm256_set1_pd(
        std::numeric_limits<double>::infinity());
    return _mm256_cmp_pd(_mm256_and_pd(v, absmask), vinf, _CMP_NLT_UQ);
}

} //namespace simd
} //namespace fastwindow

#endif //FW_SIMD
