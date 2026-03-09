#============================================#
# simd/generic_i16.nim                       #
# <- Generic 16-bit SIMD traits and helpers. #
#============================================#

import
    nimsimd/avx2,
    ./base_operations


type
    SimdI16x8* = i16x8
    SimdI16x16* = i16x16
    SimdI16* = SimdI16x8 | SimdI16x16


template lanesI16*[T: SimdI16](): int =
    when T is SimdI16x16:
        16
    else:
        8


proc set1I16*[T: SimdI16](v: int16|uint16): T =
    ## v: scalar value to broadcast across lanes.
    when T is SimdI16x16:
        result = i16x16(mm256_set1_epi16(v))
    else:
        result = i16x8(mm_set1_epi16(v))


proc loadI16x8*[T: SimdI16x8](A: array[8, uint16]|array[8, int16]): T =
    ## A: input array with 8 elements.
    result = i16x8(mm_loadu_si128(cast[pointer](unsafeAddr A[0])))


proc loadI16x16*[T: SimdI16x16](A: array[16, uint16]|array[16, int16]): T =
    ## A: input array with 16 elements.
    result = i16x16(mm256_loadu_si256(cast[pointer](unsafeAddr A[0])))


proc storeI16x8*[T: SimdI16x8](A: T): array[8, uint16] =
    ## A: input SIMD vector.
    var
        R: array[8, uint16]
    mm_storeu_si128(cast[pointer](unsafeAddr R[0]), M128i(A))
    result = R


proc storeI16x16*[T: SimdI16x16](A: T): array[16, uint16] =
    ## A: input SIMD vector.
    var
        R: array[16, uint16]
    mm256_storeu_si256(cast[pointer](unsafeAddr R[0]), M256i(A))
    result = R


proc rotl16*[T: SimdI16](A: T, k: int32): T =
    ## A: input vector.
    ## k: rotation count.
    result = (A shl k) or (A shr (16 - k))


proc rotr16*[T: SimdI16](A: T, k: int32): T =
    ## A: input vector.
    ## k: rotation count.
    result = (A shr k) or (A shl (16 - k))


proc xor3*[T: SimdI16](A, B, C: T): T =
    ## A: left operand.
    ## B: middle operand.
    ## C: right operand.
    result = (A xor B) xor C


proc add3*[T: SimdI16](A, B, C: T): T =
    ## A: left operand.
    ## B: middle operand.
    ## C: right operand.
    result = (A + B) + C


proc andNot*[T: SimdI16](A, B: T): T =
    ## A: left operand.
    ## B: mask operand to invert.
    result = A and (not B)
