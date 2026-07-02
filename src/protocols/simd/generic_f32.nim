#============================================#
# simd/generic_f32.nim                       #
# <- Generic 32-bit float SIMD helpers.      #
#============================================#

import
    nimsimd/avx2,
    ./base_operations


type
    SimdF32x4* = f32x4
    SimdF32x8* = f32x8
    SimdF32* = SimdF32x4 | SimdF32x8


template lanesF32*[T: SimdF32](): int =
    when T is SimdF32x8:
        8
    else:
        4


proc set1F32*[T: SimdF32](v: float32): T =
    ## v: scalar value to broadcast across lanes.
    when T is SimdF32x8:
        result = mm256_set1_ps(v)
    else:
        result = mm_set1_ps(v)


proc loadF32x4*[T: SimdF32x4](A: array[4, float32]): T =
    ## A: input array with 4 elements.
    result = mm_loadu_ps(cast[pointer](unsafeAddr A[0]))


proc loadF32x8*[T: SimdF32x8](A: array[8, float32]): T =
    ## A: input array with 8 elements.
    result = mm256_loadu_ps(cast[pointer](unsafeAddr A[0]))


proc storeF32x4*[T: SimdF32x4](A: T): array[4, float32] =
    ## A: input SIMD vector.
    var
        R: array[4, float32]
    mm_storeu_ps(cast[pointer](unsafeAddr R[0]), A)
    result = R


proc storeF32x8*[T: SimdF32x8](A: T): array[8, float32] =
    ## A: input SIMD vector.
    var
        R: array[8, float32]
    mm256_storeu_ps(cast[pointer](unsafeAddr R[0]), A)
    result = R


proc add3*[T: SimdF32](A, B, C: T): T =
    ## A: left operand.
    ## B: middle operand.
    ## C: right operand.
    result = (A + B) + C


proc mul3*[T: SimdF32](A, B, C: T): T =
    ## A: left operand.
    ## B: middle operand.
    ## C: right operand.
    result = (A * B) * C
