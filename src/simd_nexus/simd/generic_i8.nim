#============================================#
# simd/generic_i8.nim                        #
# <- Generic 8-bit SIMD traits and helpers.  #
#============================================#

import
    nimsimd/avx2,
    ./base_operations


type
    SimdI8x16* = i8x16
    SimdI8x32* = i8x32
    SimdI8* = SimdI8x16 | SimdI8x32


template lanesI8*[T: SimdI8](): int =
    when T is SimdI8x32:
        32
    else:
        16


proc set1I8*[T: SimdI8](v: int8|uint8): T =
    ## v: scalar value to broadcast across lanes.
    when T is SimdI8x32:
        result = i8x32(mm256_set1_epi8(v))
    else:
        result = i8x16(mm_set1_epi8(v))


proc loadI8x16*[T: SimdI8x16](A: array[16, uint8]|array[16, int8]): T =
    ## A: input array with 16 elements.
    result = i8x16(mm_loadu_si128(cast[pointer](unsafeAddr A[0])))


proc loadI8x32*[T: SimdI8x32](A: array[32, uint8]|array[32, int8]): T =
    ## A: input array with 32 elements.
    result = i8x32(mm256_loadu_si256(cast[pointer](unsafeAddr A[0])))


proc storeI8x16*[T: SimdI8x16](A: T): array[16, uint8] =
    ## A: input SIMD vector.
    var
        R: array[16, uint8]
    mm_storeu_si128(cast[pointer](unsafeAddr R[0]), M128i(A))
    result = R


proc storeI8x32*[T: SimdI8x32](A: T): array[32, uint8] =
    ## A: input SIMD vector.
    var
        R: array[32, uint8]
    mm256_storeu_si256(cast[pointer](unsafeAddr R[0]), M256i(A))
    result = R


proc xor3*[T: SimdI8](A, B, C: T): T =
    ## A: left operand.
    ## B: middle operand.
    ## C: right operand.
    result = (A xor B) xor C


proc add3*[T: SimdI8](A, B, C: T): T =
    ## A: left operand.
    ## B: middle operand.
    ## C: right operand.
    result = (A + B) + C


proc andNot*[T: SimdI8](A, B: T): T =
    ## A: left operand.
    ## B: mask operand to invert.
    result = A and (not B)
