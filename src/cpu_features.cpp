/// @file cpu_features.cpp
/// Runtime CPU feature detection for the AVX2 kernel dispatch.
/// The AVX2 paths require the AVX2 and FMA instruction sets plus OS support
/// for saving the YMM register state (OSXSAVE + XCR0 bits 1 and 2) — the
/// last part matters on old hypervisors and minimal kernels, where the CPU
/// flag alone is not enough.
#include "fastwindow.h"

#if FW_SIMD

#if defined(_MSC_VER)
  #include <intrin.h>
#else
  #include <cpuid.h>
#endif

namespace fastwindow {

static bool detect_avx2() noexcept {
#if defined(_MSC_VER)
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7) return false;
    __cpuid(r, 1);
    const bool osxsave = (r[2] & (1 << 27)) != 0;
    const bool avx     = (r[2] & (1 << 28)) != 0;
    const bool fma     = (r[2] & (1 << 12)) != 0;
    if (!(osxsave && avx && fma)) return false;
    if ((_xgetbv(0) & 0x6) != 0x6) return false;   //OS saves YMM state
    __cpuidex(r, 7, 0);
    return (r[1] & (1 << 5)) != 0;                 //AVX2
#else
    unsigned a, b, c, d;
    if (!__get_cpuid(0, &a, &b, &c, &d) || a < 7) return false;
    if (!__get_cpuid(1, &a, &b, &c, &d)) return false;
    const bool osxsave = (c & (1u << 27)) != 0;
    const bool avx     = (c & (1u << 28)) != 0;
    const bool fma     = (c & (1u << 12)) != 0;
    if (!(osxsave && avx && fma)) return false;
    unsigned eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    if ((eax & 0x6) != 0x6) return false;          //OS saves YMM state
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return false;
    return (b & (1u << 5)) != 0;                   //AVX2
#endif
}

bool cpu_has_avx2() noexcept {
    static const bool ok = detect_avx2();
    return ok;
}

} //namespace fastwindow

#else //!FW_SIMD

namespace fastwindow {
bool cpu_has_avx2() noexcept { return false; }
} //namespace fastwindow

#endif
