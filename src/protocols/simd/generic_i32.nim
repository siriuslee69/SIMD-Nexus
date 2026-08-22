#============================================#
# simd/generic_i32.nim                       #
# <- Generic 32-bit SIMD traits and helpers. #
#============================================#

import ../isa/x86_avx2
import
    nimsimd/avx2,
    nimsimd/neon,
    ./base_operations


type
    SimdI32x4* = i32x4 | uint32x4
    SimdI32x8* = i32x8
    SimdI32* = SimdI32x4 | SimdI32x8


template lanesI32*[T: SimdI32](): int =
    when T is SimdI32x8:
        8
    else:
        4


proc set1I32*[T: SimdI32](v: int32): T =
    ## v: scalar value to broadcast across lanes.
    when T is i32x8:
        result = i32x8(mm256_set1_epi32(v))
    elif T is i32x4:
        result = i32x4(mm_set1_epi32(v))
    else:
        result = vreinterpretq_u32_s32(vdupq_n_s32(v))


proc loadI32x4*[T: SimdI32x4](A: array[4, int32]): T =
    ## A: input array with 4 elements.
    when T is uint32x4:
        result = cast[uint32x4](vld1q_s32(cast[pointer](unsafeAddr A[0])))
    else:
        result = i32x4(mm_setr_epi32(A[0], A[1], A[2], A[3]))


proc loadI32x4At*[T: SimdI32x4](A: openArray[int32], off: int): T =
    ## A: input scalar slice.
    ## off: starting offset.
    when T is uint32x4:
        result = cast[uint32x4](vld1q_s32(cast[pointer](unsafeAddr A[off])))
    else:
        result = i32x4(mm_loadu_si128(cast[pointer](unsafeAddr A[off])))


proc loadI32x8*[T: SimdI32x8](A: array[8, int32]): T =
    ## A: input array with 8 elements.
    result = i32x8(mm256_setr_epi32(
        A[0], A[1], A[2], A[3],
        A[4], A[5], A[6], A[7]
    ))


proc storeI32x4*[T: SimdI32x4](A: T): array[4, int32] =
    ## A: input SIMD vector.
    var
        R: array[4, int32]
    when T is uint32x4:
        vst1q_s32(cast[pointer](unsafeAddr R[0]), vreinterpretq_s32_u32(A))
    else:
        mm_storeu_si128(addr R[0], M128i(A))
    result = R


proc storeI32x4At*[T: SimdI32x4](A: T, dst: var openArray[int32], off: int) =
    ## A: input SIMD vector.
    ## dst: destination scalar slice.
    ## off: starting offset.
    when T is uint32x4:
        vst1q_s32(cast[pointer](unsafeAddr dst[off]), vreinterpretq_s32_u32(A))
    else:
        mm_storeu_si128(cast[pointer](unsafeAddr dst[off]), M128i(A))


proc storeI32x8*[T: SimdI32x8](A: T): array[8, int32] =
    ## A: input SIMD vector.
    var
        R: array[8, int32]
    mm256_storeu_si256(addr R[0], M256i(A))
    result = R


proc rotl32*[T: SimdI32](A: T, k: int32): T =
    ## A: input vector.
    ## k: rotation count.
    result = (A shl k) or (A shr (32 - k))


proc rotr32*[T: SimdI32](A: T, k: int32): T =
    ## A: input vector.
    ## k: rotation count.
    result = (A shr k) or (A shl (32 - k))


proc xor3*[T: SimdI32](A, B, C: T): T =
    ## A: left operand.
    ## B: middle operand.
    ## C: right operand.
    result = (A xor B) xor C


proc add3*[T: SimdI32](A, B, C: T): T =
    ## A: left operand.
    ## B: middle operand.
    ## C: right operand.
    result = (A + B) + C


proc andNot*[T: SimdI32](A, B: T): T =
    ## A: left operand.
    ## B: mask operand to invert.
    result = A and (not B)
