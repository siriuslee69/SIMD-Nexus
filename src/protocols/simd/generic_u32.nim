#============================================#
# simd/generic_u32.nim                       #
# <- Generic 32-bit SIMD traits and helpers. #
#============================================#

import ../isa/x86_avx2
import
    nimsimd/avx2,
    nimsimd/neon,
    ./base_operations


type
    SimdU32x4* = M128i | i32x4 | uint32x4
    SimdU32x8* = M256i | i32x8
    SimdU32* = SimdU32x4 | SimdU32x8


template lanesU32*[T: SimdU32](): int =
    when T is SimdU32x8:
        8
    else:
        4


proc set1U32*[T: SimdU32](v: uint32): T =
    ## v: scalar value to broadcast across lanes.
    when T is i32x8:
        result = i32x8(mm256_set1_epi32(cast[int32](v)))
    elif T is M256i:
        result = mm256_set1_epi32(cast[int32](v))
    elif T is i32x4:
        result = i32x4(mm_set1_epi32(cast[int32](v)))
    elif T is M128i:
        result = mm_set1_epi32(cast[int32](v))
    else:
        result = vmovq_n_u32(v)


proc loadU32x4*[T: SimdU32x4](A: array[4, uint32]): T =
    ## A: input array with 4 elements.
    when T is uint32x4:
        result = vld1q_u32(cast[pointer](unsafeAddr A[0]))
    elif T is i32x4:
        result = i32x4(mm_setr_epi32(
            cast[int32](A[0]), cast[int32](A[1]), cast[int32](A[2]), cast[int32](A[3])
        ))
    else:
        result = mm_setr_epi32(
            cast[int32](A[0]), cast[int32](A[1]), cast[int32](A[2]), cast[int32](A[3])
        )


proc loadU32x8*[T: SimdU32x8](A: array[8, uint32]): T =
    ## A: input array with 8 elements.
    when T is i32x8:
        result = i32x8(mm256_setr_epi32(
            cast[int32](A[0]), cast[int32](A[1]), cast[int32](A[2]), cast[int32](A[3]),
            cast[int32](A[4]), cast[int32](A[5]), cast[int32](A[6]), cast[int32](A[7])
        ))
    else:
        result = mm256_setr_epi32(
            cast[int32](A[0]), cast[int32](A[1]), cast[int32](A[2]), cast[int32](A[3]),
            cast[int32](A[4]), cast[int32](A[5]), cast[int32](A[6]), cast[int32](A[7])
        )


proc storeU32x4*[T: SimdU32x4](A: T): array[4, uint32] =
    ## A: input SIMD vector.
    var
        R: array[4, uint32]
    when T is uint32x4:
        vst1q_u32(cast[pointer](unsafeAddr R[0]), A)
    elif T is i32x4:
        mm_storeu_si128(addr R[0], M128i(A))
    else:
        mm_storeu_si128(addr R[0], A)
    result = R


proc storeU32x8*[T: SimdU32x8](A: T): array[8, uint32] =
    ## A: input SIMD vector.
    var
        R: array[8, uint32]
    when T is i32x8:
        mm256_storeu_si256(addr R[0], M256i(A))
    else:
        mm256_storeu_si256(addr R[0], A)
    result = R


proc rotl32*[T: SimdU32](A: T, k: int32): T =
    ## A: input vector.
    ## k: rotation count.
    result = (A shl k) or (A shr (32 - k))


proc rotr32*[T: SimdU32](A: T, k: int32): T =
    ## A: input vector.
    ## k: rotation count.
    result = (A shr k) or (A shl (32 - k))


proc xor3*[T: SimdU32](A, B, C: T): T =
    ## A: left operand.
    ## B: middle operand.
    ## C: right operand.
    result = (A xor B) xor C


proc add3*[T: SimdU32](A, B, C: T): T =
    ## A: left operand.
    ## B: middle operand.
    ## C: right operand.
    result = (A + B) + C


proc andNot*[T: SimdU32](A, B: T): T =
    ## A: left operand.
    ## B: mask operand to invert.
    result = A and (not B)
