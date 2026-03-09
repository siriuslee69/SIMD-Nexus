#============================================#
# simd/generic_f64.nim                       #
# <- Generic 64-bit float SIMD helpers.      #
#============================================#

import
    nimsimd/avx2,
    ./base_operations


type
    SimdF64x2* = f64x2
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
    else:
        result = mm_set1_pd(v)


proc loadF64x2*[T: SimdF64x2](A: array[2, float64]): T =
    ## A: input array with 2 elements.
    result = mm_loadu_pd(cast[pointer](unsafeAddr A[0]))


proc loadF64x4*[T: SimdF64x4](A: array[4, float64]): T =
    ## A: input array with 4 elements.
    result = mm256_loadu_pd(cast[pointer](unsafeAddr A[0]))


proc storeF64x2*[T: SimdF64x2](A: T): array[2, float64] =
    ## A: input SIMD vector.
    var
        R: array[2, float64]
    mm_storeu_pd(cast[pointer](unsafeAddr R[0]), A)
    result = R


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
