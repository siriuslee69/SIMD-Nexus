#============================================#
# simd/generic_f64.nim                       #
# <- Generic 64-bit float SIMD helpers.      #
#============================================#

import
    nimsimd/sse2,
    nimsimd/avx2,
    nimsimd/neon,
    ./base_operations

when defined(neon) or defined(arm64) or defined(aarch64):
    ## nimsimd 1.3.2 exposes `float64x2`, but it does not import the basic
    ## load/store/arithmetic intrinsics we need for a shared SSE2/NEON `f64x2`
    ## path. Keep this bridge here so future portable SIMD code can stay 1:1.
    func nxVld1qF64(p: pointer): float64x2 {.importc: "vld1q_f64", header: "arm_neon.h".}
    func nxVst1qF64(p: pointer, v: float64x2) {.importc: "vst1q_f64", header: "arm_neon.h".}
    func nxVaddqF64(a, b: float64x2): float64x2 {.importc: "vaddq_f64", header: "arm_neon.h".}
    func nxVsubqF64(a, b: float64x2): float64x2 {.importc: "vsubq_f64", header: "arm_neon.h".}
    func nxVmulqF64(a, b: float64x2): float64x2 {.importc: "vmulq_f64", header: "arm_neon.h".}
    func nxVdivqF64(a, b: float64x2): float64x2 {.importc: "vdivq_f64", header: "arm_neon.h".}

    template `+`*(A, B: float64x2): float64x2 =
        nxVaddqF64(A, B)

    template `-`*(A, B: float64x2): float64x2 =
        nxVsubqF64(A, B)

    template `*`*(A, B: float64x2): float64x2 =
        nxVmulqF64(A, B)

    template `/`*(A, B: float64x2): float64x2 =
        nxVdivqF64(A, B)

when defined(neon) or defined(arm64) or defined(aarch64):
    type
        SimdF64x2* = float64x2
else:
    type
        SimdF64x2* = f64x2

type
    SimdF64x4* = f64x4
    SimdF64* = SimdF64x2 | SimdF64x4


template lanesF64*[T: SimdF64](): int =
    when T is SimdF64x4:
        4
    else:
        2


proc set1F64*[T: SimdF64](v: float64): T =
    ## v: scalar value to broadcast across lanes.
    when T is SimdF64x4:
        result = mm256_set1_pd(v)
    elif T is float64x2:
        result = vmovq_n_f64(v)
    else:
        result = mm_set1_pd(v)


proc loadF64x2*[T: SimdF64x2](A: array[2, float64]): T =
    ## A: input array with 2 elements.
    when T is float64x2:
        result = nxVld1qF64(cast[pointer](unsafeAddr A[0]))
    else:
        result = mm_loadu_pd(cast[pointer](unsafeAddr A[0]))


proc loadF64x2Ptr*[T: SimdF64x2](p: pointer): T =
    ## p: input pointer to at least 2 float64 values.
    when T is float64x2:
        result = nxVld1qF64(p)
    else:
        result = mm_loadu_pd(p)


proc loadF64x4*[T: SimdF64x4](A: array[4, float64]): T =
    ## A: input array with 4 elements.
    result = mm256_loadu_pd(cast[pointer](unsafeAddr A[0]))


proc storeF64x2*[T: SimdF64x2](A: T): array[2, float64] =
    ## A: input SIMD vector.
    var
        R: array[2, float64]
    when T is float64x2:
        nxVst1qF64(cast[pointer](unsafeAddr R[0]), A)
    else:
        mm_storeu_pd(cast[pointer](unsafeAddr R[0]), A)
    result = R


proc storeF64x2Ptr*[T: SimdF64x2](p: pointer, A: T) =
    ## p: destination pointer to at least 2 float64 values.
    ## A: input SIMD vector.
    when T is float64x2:
        nxVst1qF64(p, A)
    else:
        mm_storeu_pd(p, A)


proc storeF64x4*[T: SimdF64x4](A: T): array[4, float64] =
    ## A: input SIMD vector.
    var
        R: array[4, float64]
    mm256_storeu_pd(cast[pointer](unsafeAddr R[0]), A)
    result = R


proc add3*[T: SimdF64](A, B, C: T): T =
    ## A: left operand.
    ## B: middle operand.
    ## C: right operand.
    result = (A + B) + C


proc mul3*[T: SimdF64](A, B, C: T): T =
    ## A: left operand.
    ## B: middle operand.
    ## C: right operand.
    result = (A * B) * C
