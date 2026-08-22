#============================================#
# simd/generic_i16.nim                       #
# <- Generic 16-bit SIMD traits and helpers. #
#============================================#

import ../isa/x86_avx2
import
    nimsimd/avx2,
    nimsimd/neon,
    ./base_operations,
    ./generic_i32


type
    SimdI16x8* = i16x8 | uint16x8
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
    elif T is uint16x8:
        result = vmovq_n_u16(cast[uint16](v))
    else:
        result = i16x8(mm_set1_epi16(v))


proc loadI16x8*[T: SimdI16x8](A: array[8, uint16]|array[8, int16]): T =
    ## A: input array with 8 elements.
    when T is uint16x8:
        result = vld1q_u16(cast[pointer](unsafeAddr A[0]))
    else:
        result = i16x8(mm_loadu_si128(cast[pointer](unsafeAddr A[0])))


proc loadI16x8At*[T: SimdI16x8](A: openArray[uint16], off: int): T =
    ## A: input scalar slice.
    ## off: starting offset.
    when T is uint16x8:
        result = vld1q_u16(cast[pointer](unsafeAddr A[off]))
    else:
        result = i16x8(mm_loadu_si128(cast[pointer](unsafeAddr A[off])))


proc loadI16x8At*[T: SimdI16x8](A: openArray[int16], off: int): T =
    ## A: input scalar slice.
    ## off: starting offset.
    when T is uint16x8:
        result = vld1q_u16(cast[pointer](unsafeAddr A[off]))
    else:
        result = i16x8(mm_loadu_si128(cast[pointer](unsafeAddr A[off])))


proc loadI16x16*[T: SimdI16x16](A: array[16, uint16]|array[16, int16]): T =
    ## A: input array with 16 elements.
    result = i16x16(mm256_loadu_si256(cast[pointer](unsafeAddr A[0])))


proc storeI16x8*[T: SimdI16x8](A: T): array[8, uint16] =
    ## A: input SIMD vector.
    var
        R: array[8, uint16]
    when T is uint16x8:
        vst1q_u16(cast[pointer](unsafeAddr R[0]), A)
    else:
        mm_storeu_si128(cast[pointer](unsafeAddr R[0]), M128i(A))
    result = R


proc storeI16x8At*[T: SimdI16x8](A: T, dst: var openArray[uint16], off: int) =
    ## A: input SIMD vector.
    ## dst: destination scalar slice.
    ## off: starting offset.
    when T is uint16x8:
        vst1q_u16(cast[pointer](unsafeAddr dst[off]), A)
    else:
        mm_storeu_si128(cast[pointer](unsafeAddr dst[off]), M128i(A))


proc storeI16x8At*[T: SimdI16x8](A: T, dst: var openArray[int16], off: int) =
    ## A: input SIMD vector.
    ## dst: destination scalar slice.
    ## off: starting offset.
    when T is uint16x8:
        vst1q_u16(cast[pointer](unsafeAddr dst[off]), A)
    else:
        mm_storeu_si128(cast[pointer](unsafeAddr dst[off]), M128i(A))


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


proc mulLoI16*[T: SimdI16](A, B: T): T =
    ## A: left operand.
    ## B: right operand.
    ## Returns the low 16 bits of each lane product.
    when T is SimdI16x16:
        result = i16x16(mm256_mullo_epi16(A.M256i, B.M256i))
    elif T is uint16x8:
        var
            lo: uint32x4 = vmull_u16(vget_low_u16(A), vget_low_u16(B))
            hi: uint32x4 = vmull_u16(vget_high_u16(A), vget_high_u16(B))
            loWide: array[4, int32]
            hiWide: array[4, int32]
            packedLo: array[4, uint16]
            packedHi: array[4, uint16]
            i: int = 0
        ## ARM NEON narrowing shifts require an immediate in 1..16, so we cannot
        ## use `vshrn_n_u32(..., 0)` as a low-16 truncation shortcut here.
        ## Keep the explicit lane packing unless the bindings grow a direct
        ## `vmovn_u32`-style helper in the future.
        loWide = storeI32x4[uint32x4](lo)
        hiWide = storeI32x4[uint32x4](hi)
        i = 0
        while i < 4:
            packedLo[i] = uint16(uint32(loWide[i]) and 0xffff'u32)
            packedHi[i] = uint16(uint32(hiWide[i]) and 0xffff'u32)
            i = i + 1
        result = vcombine_u16(vld1_u16(cast[pointer](unsafeAddr packedLo[0])),
          vld1_u16(cast[pointer](unsafeAddr packedHi[0])))
    else:
        result = i16x8(mm_mullo_epi16(A.M128i, B.M128i))


proc andNot*[T: SimdI16](A, B: T): T =
    ## A: left operand.
    ## B: mask operand to invert.
    result = A and (not B)
