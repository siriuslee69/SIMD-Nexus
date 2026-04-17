#============================================#
# simd/generic_u64.nim                       #
# <- Generic 64-bit SIMD traits and helpers. #
#============================================#

import
    nimsimd/avx2,
    ./base_operations


type
    SimdU64x2* = u64x2
    SimdU64x4* = u64x4
    SimdU64* = SimdU64x2 | SimdU64x4


template lanesU64*[T: SimdU64](): int =
    when T is SimdU64x4:
        4
    else:
        2


proc set1U64*[T: SimdU64](v: uint64): T =
    ## v: scalar value to broadcast across lanes.
    when T is SimdU64x4:
        result = u64x4(mm256_set1_epi64x(cast[int64](v)))
    else:
        result = u64x2(mm_set1_epi64x(cast[int64](v)))


proc loadU64x2*[T: SimdU64x2](A: array[2, uint64]|array[2, int64]): T =
    ## A: input array with 2 elements.
    result = u64x2(mm_loadu_si128(cast[pointer](unsafeAddr A[0])))


proc loadU64x4*[T: SimdU64x4](A: array[4, uint64]|array[4, int64]): T =
    ## A: input array with 4 elements.
    result = u64x4(mm256_loadu_si256(cast[pointer](unsafeAddr A[0])))


proc storeU64x2*[T: SimdU64x2](A: T): array[2, uint64] =
    ## A: input SIMD vector.
    var
        R: array[2, uint64]
    mm_storeu_si128(cast[pointer](unsafeAddr R[0]), M128i(A))
    result = R


proc storeU64x4*[T: SimdU64x4](A: T): array[4, uint64] =
    ## A: input SIMD vector.
    var
        R: array[4, uint64]
    mm256_storeu_si256(cast[pointer](unsafeAddr R[0]), M256i(A))
    result = R


proc rotl64*[T: SimdU64](A: T, k: int32): T =
    ## A: input vector.
    ## k: rotation count.
    result = (A shl k) or (A shr (64 - k))


proc rotr64*[T: SimdU64](A: T, k: int32): T =
    ## A: input vector.
    ## k: rotation count.
    result = (A shr k) or (A shl (64 - k))


proc xor3*[T: SimdU64](A, B, C: T): T =
    ## A: left operand.
    ## B: middle operand.
    ## C: right operand.
    result = (A xor B) xor C


proc add3*[T: SimdU64](A, B, C: T): T =
    ## A: left operand.
    ## B: middle operand.
    ## C: right operand.
    result = (A + B) + C


proc andNot*[T: SimdU64](A, B: T): T =
    ## A: left operand.
    ## B: mask operand to invert.
    result = A and (not B)
