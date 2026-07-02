#============================================#
# simd/base_operations.nim                   #
# <- Core SIMD type aliases and operators.   #
#============================================#

import 
    nimsimd/sse2,
    nimsimd/avx2,
    nimsimd/neon
    
    
export
    avx2,
    neon
#Compiletime conversion
type
    i8x16* = distinct M128i
    i16x8* = distinct M128i
    i32x4* = distinct M128i
    u64x2* = distinct M128i
    
    i8x32* = distinct M256i
    i16x16* = distinct M256i
    i32x8* = distinct M256i
    u64x4* = distinct M256i
    f32x4* = M128
    f64x2* = M128d
    f32x8* = M256
    f64x4* = M256d

####-------------------- M128i integer templates ----------------------####
##.......M128iX extracts compile time.......##

template `[]`*(A: i8x16, k: uint32|int32 ): int8  =
    mm_extract_epi8(A.M128i, k).int8()

template `[]`*(A: i16x8, k: uint32|int32 ): int16  =
    mm_extract_epi16(A.M128i, k).int16()

template `[]`*(A: i32x4, k: uint32|int32 ): int32  =
    mm_extract_epi32(A.M128i, k).int32()

template `[]`*(A: u64x2, k: uint32|int32 ): uint64  =
    mm_extract_epi64(A.M128i, k).uint64()

###

template `[]`*[T: int8](A: M128i, k: uint32|int32, t: typedesc[T]): T  =
    T(mm_extract_epi8(A, k))

template `[]`*[T: int16](A: M128i, k: uint32|int32, t: typedesc[T]): T  =
    T(mm_extract_epi16(A, k))

template `[]`*[T: int32](A: M128i, k: uint32|int32, t: typedesc[T]): T  =
    T(mm_extract_epi32(A, k))

template `[]`*[T: int64](A: M128i, k: uint32|int32, t: typedesc[T]): T  =
    T(mm_extract_epi64(A, k))

template `[]`*[T: uint8](A: M128i, k: uint32|int32, t: typedesc[T]): T  =
    T(mm_extract_epi8(A, k))

template `[]`*[T: uint16](A: M128i, k: uint32|int32, t: typedesc[T]): T  =
    T(mm_extract_epi16(A, k))

template `[]`*[T: uint32](A: M128i, k: uint32|int32, t: typedesc[T]): T  =
    T(mm_extract_epi32(A, k))

template `[]`*[T: uint64](A: M128i, k: uint32|int32, t: typedesc[T]): T  =
    T(mm_extract_epi64(A, k))

template `i8`*(A: i8x16, k: int32): int8  =
    mm_extract_epi8(A.M128i, k)

template `i16`*(A: i16x8, k: int32): int16  =
    mm_extract_epi16(A.M128i, k)

template `i32`*(A: i32x4, k: int32): int32  =
    mm_extract_epi32(A.M128i, k)

template `u64`*(A: u64x2, k: int32): uint64  =
    mm_extract_epi64(A.M128i, k).uint64()

template `u8`*(A: M128i, k: int32): uint8  =
    mm_extract_epi8(A, k).uint8()

template `u16`*(A: M128i, k: int32): uint16  =
    mm_extract_epi16(A, k).uint16()

template `u32`*(A: M128i, k: int32): uint32  =
    mm_extract_epi32(A, k).uint32()

template `u64`*(A: M128i, k: int32): uint64  =
    mm_extract_epi64(A, k).uint64()

template `and`*(A, B: M128i): M128i  =
    mm_and_si128(A, B)

template `or`*(A, B: M128i): M128i  =
    mm_or_si128(A, B)

template `xor`*(A, B: M128i): M128i  =
    mm_xor_si128(A, B)

template `and`*[T: i32x4](A, B: T): T =
    ## A: left operand.
    ## B: right operand.
    T(mm_and_si128(A.M128i, B.M128i))

template `or`*[T: i32x4](A, B: T): T =
    ## A: left operand.
    ## B: right operand.
    T(mm_or_si128(A.M128i, B.M128i))

template `xor`*[T: i32x4](A, B: T): T =
    ## A: left operand.
    ## B: right operand.
    T(mm_xor_si128(A.M128i, B.M128i))

template `and`*[T: i8x16|i16x8|u64x2](A, B: T): T =
    ## A: left operand.
    ## B: right operand.
    T(mm_and_si128(A.M128i, B.M128i))

template `or`*[T: i8x16|i16x8|u64x2](A, B: T): T =
    ## A: left operand.
    ## B: right operand.
    T(mm_or_si128(A.M128i, B.M128i))

template `xor`*[T: i8x16|i16x8|u64x2](A, B: T): T =
    ## A: left operand.
    ## B: right operand.
    T(mm_xor_si128(A.M128i, B.M128i))

template `not`*(A: M128i): M128i  =
    ## A: input vector.
    mm_xor_si128(A, mm_set1_epi32(-1))

template `not`*[T: i32x4](A: T): T =
    ## A: input vector.
    T(mm_xor_si128(A.M128i, mm_set1_epi32(-1)))

template `not`*[T: i8x16|i16x8|u64x2](A: T): T =
    ## A: input vector.
    T(mm_xor_si128(A.M128i, mm_set1_epi32(-1)))

template `shl`*(A: M128i, k: int32|uint32): M128i  =
    ## A: input vector.
    ## k: shift amount.
    mm_slli_epi32(A, int32(k))

template `shl`*[T: i32x4](A: T, k: int32|uint32): T =
    ## A: input vector.
    ## k: shift amount.
    T(mm_slli_epi32(A.M128i, int32(k)))

template `shr`*(A: M128i, k: int32|uint32): M128i  =
    ## A: input vector.
    ## k: shift amount.
    mm_srli_epi32(A, int32(k))

template `shr`*[T: i32x4](A: T, k: int32|uint32): T =
    ## A: input vector.
    ## k: shift amount.
    T(mm_srli_epi32(A.M128i, int32(k)))

template `shl`*(A: i16x8, k: int32|uint32): i16x8 =
    ## A: input vector.
    ## k: shift amount.
    i16x8(mm_slli_epi16(A.M128i, int32(k)))

template `shr`*(A: i16x8, k: int32|uint32): i16x8 =
    ## A: input vector.
    ## k: shift amount.
    i16x8(mm_srli_epi16(A.M128i, int32(k)))

template `shl`*(A: u64x2, k: int32|uint32): u64x2 =
    ## A: input vector.
    ## k: shift amount.
    u64x2(mm_slli_epi64(A.M128i, int32(k)))

template `shr`*(A: u64x2, k: int32|uint32): u64x2 =
    ## A: input vector.
    ## k: shift amount.
    u64x2(mm_srli_epi64(A.M128i, int32(k)))

template `+`*[T: i8x16](A, B: T): T =
    T(mm_add_epi8(A.M128i, B.M128i))

template `+`*[T: i16x8](A, B: T): T =
    T(mm_add_epi16(A.M128i, B.M128i))

template `-`*[T: i16x8](A, B: T): T =
    T(mm_sub_epi16(A.M128i, B.M128i))

template `+`*[T: i32x4](A, B: T): T =
    T(mm_add_epi32(A.M128i, B.M128i))

template `-`*[T: i32x4](A, B: T): T =
    T(mm_sub_epi32(A.M128i, B.M128i))

template `+`*[T: u64x2](A, B: T): T =
    T(mm_add_epi64(A.M128i, B.M128i))

template `+`*(A, B: M128i): M128i  =
    mm_add_epi32(A, B)

template `-`*(A, B: M128i): M128i  =
    mm_sub_epi32(A, B)

template `*`*(A, B: M128i): M128i  =
    mm_mul_epi32(A, B)

template `rot_left`*(A: M128i, k: int32): M128i =
    mm_or_si128(mm_slli_epi32(A, k), mm_srli_epi32(A, 32 - k))  
    
template `perm_right_rot_by_one`*(A: M128i): M128i  =
    mm_shuffle_epi32(A, 0x93)

template `perm_swap_four`*(A: M128i): M128i  =
    mm_shuffle_epi32(A, 0x4E)

####-------------------- M128 32bit float templates ----------------------####

template f32*(A: M128, k: int32): float32 =
    mm_extract_ps(A, k)

template `and`*(A, B: M128): M128 =
    mm_and_ps(A, B)

template `or`*(A, B: M128): M128 =
    mm_or_ps(A, B)

template `xor`*(A, B: M128): M128 =
    mm_xor_ps(A, B)



template `+`*(A, B: M128): M128 =
    mm_add_ps(A, B)

template `-`*(A, B: M128): M128 =
    mm_sub_ps(A, B)

template `*`*(A, B: M128): M128 =
    mm_mul_ps(A, B)

template `/`*(A, B: M128): M128 =
    mm_div_ps(A, B)


####-------------------- M128d 64bit float templates ----------------------####


#doesnt work
template f64*(A: M128d, k: int32): float64 =
    mm_extract_pd(A, k)

template `+`*(A, B: M128d): M128d =
    ## A: left operand.
    ## B: right operand.
    mm_add_pd(A, B)

template `-`*(A, B: M128d): M128d =
    ## A: left operand.
    ## B: right operand.
    mm_sub_pd(A, B)

template `*`*(A, B: M128d): M128d =
    mm_mul_pd(A, B)

template `/`*(A, B: M128d): M128d =
    ## A: left operand.
    ## B: right operand.
    mm_div_pd(A, B)

####-------------------- M256i integer templates ----------------------####
#doesnt work
template `=`*(dst: var M128i, src: array[16, int8]) =
    dst = mm_setr_epi8(
        src[0], src[1], src[2], src[3],
        src[4], src[5], src[6], src[7],
        src[8], src[9], src[10], src[11],
        src[12], src[13], src[14], src[15]
    )
#doesnt work
template `=`*(dst: var M128i, src: array[8, int16]): untyped =
    dst = mm_setr_epi16(    
        src[0], src[1], src[2], src[3],
        src[4], src[5], src[6], src[7]
    )
#doesnt work
template `=` *(dst: var M128i, src: array[4, int32]) =
    dst = mm_setr_epi32(
        src[0], src[1], src[2], src[3]
    )


template set1i8*(k: int8|uint8): M256i =
    mm256_set1_epi8(k)

template set1i16*(k: int16|uint16): M256i =
    mm256_set1_epi16(k)

template set1i32*(k: int32|uint32): M256i =
    mm256_set1_epi32(k)

template set1u64*(k: int64|uint64): M256i =
    mm256_set1_epi64x(k)

template set1*(k: int8): M256i =
    mm256_set1_epi8(k)

template `[]`*(A: i8x32, k: uint32|int32 ): int8  =
    mm256_extract_epi8(A.M256i, k).int8()

template `[]`*(A: i16x16, k: uint32|int32 ): int16  =
    mm256_extract_epi16(A.M256i, k).int16()

template `[]`*(A: i32x8, k: uint32|int32 ): int32  =
    mm256_extract_epi32(A.M256i, k).int32()

template `[]`*(A: u64x4, k: uint32|int32 ): uint64  =
    mm256_extract_epi64(A.M256i, k).uint64()


template `[]`*[T: int8](A: M256i, k: uint32|int32, t: typedesc[T]): T  =
    T(mm256_extract_epi8(A, k))

template `[]`*[T: int16](A: M256i, k: uint32|int32, t: typedesc[T]): T  =
    T(mm256_extract_epi16(A, k))

template `[]`*[T: int32](A: M256i, k: uint32|int32, t: typedesc[T]): T  =
    T(mm256_extract_epi32(A, k))

template `[]`*[T: int64](A: M256i, k: uint32|int32, t: typedesc[T]): T  =
    T(mm256_extract_epi64(A, k))

template `[]`*[T: uint8](A: M256i, k: uint32|int32, t: typedesc[T]): T  =
    T(mm256_extract_epi8(A, k))

template `[]`*[T: uint16](A: M256i, k: uint32|int32, t: typedesc[T]): T  =
    T(mm256_extract_epi16(A, k))

template `[]`*[T: uint32](A: M256i, k: uint32|int32, t: typedesc[T]): T  =
    T(mm256_extract_epi32(A, k))

template `[]`*[T: uint64](A: M256i, k: uint32|int32, t: typedesc[T]): T  =
    T(mm256_extract_epi64(A, k))

template `i8`*(A: M256i, k: int32): int8  =
    mm256_extract_epi8(A, k)

template `i16`*(A: M256i, k: int32): int16  =
    mm256_extract_epi16(A, k)

template `i32`*(A: M256i, k: int32): int32  =
    mm256_extract_epi32(A, k)

template `i64`*(A: M256i, k: int32): int64  =
    mm256_extract_epi64(A, k)

template `and`*[T: i8x32|i16x16|i32x8|u64x4](A, B: T): T  =
    T(mm256_and_si256(A.M256i, B.M256i))

template `and`*(A, B: M256i): M256i  =
    mm256_and_si256(A, B)

template `or`*[T: i8x32|i16x16|i32x8|u64x4](A, B: T): T  =
    T(mm256_or_si256(A.M256i, B.M256i))

template `or`*(A, B: M256i): M256i  =
    mm256_or_si256(A, B)

template `xor`*[T: i8x32|i16x16|i32x8|u64x4](A, B: T): T  =
    T(mm256_xor_si256(A.M256i, B.M256i))

template `xor`*(A, B: M256i): M256i  =
    mm256_xor_si256(A, B)

template `not`*[T: i8x32|i16x16|i32x8|u64x4](A: T): T  =
        T(mm256_xor_si256(A.M256i, mm256_set1_epi32(-1)))

template `shl`*(A: i8x32, k: int32|uint32): i8x32 =
    mm256_slli_epi16(A.M256i, k).i8x32

template `shl`*(A: i16x16, k: int32|uint32): i16x16 =
    mm256_slli_epi16(A.M256i, k).i16x16

template `shr`*(A: i16x16, k: int32|uint32): i16x16 =
    ## A: input vector.
    ## k: shift amount.
    mm256_srli_epi16(A.M256i, int32(k)).i16x16

template `shl`*(A: i32x8, k: int32|uint32): i32x8 =
    mm256_slli_epi32(A.M256i, k).i32x8

template `shr`*(A: i32x8, k: int32|uint32): i32x8 =
    ## A: input vector.
    ## k: shift amount.
    mm256_srli_epi32(A.M256i, int32(k)).i32x8

template `shl`*(A: u64x4, k: int32|uint32): u64x4 =
    mm256_slli_epi64(A.M256i, k).u64x4

template `shr`*(A: u64x4, k: int32|uint32): u64x4 =
    ## A: input vector.
    ## k: shift amount.
    mm256_srli_epi64(A.M256i, int32(k)).u64x4

template `shl`*(A: M256i, k: int32|uint32): M256i  =
    mm256_slli_epi32(A, k)

template `shr`*(A: M256i, k: int32|uint32): M256i =
    mm256_srli_epi32(A,k)

template `+`*[T: i8x32](A, B: T): T  =
    T(mm256_add_epi8(A.M256i, B.M256i))

template `+`*[T: i16x16](A, B: T): T  =
    T(mm256_add_epi16(A.M256i, B.M256i))

template `-`*[T: i16x16](A, B: T): T  =
    T(mm256_sub_epi16(A.M256i, B.M256i))

template `+`*[T: i32x8](A, B: T): T  =
    T(mm256_add_epi32(A.M256i, B.M256i))

template `-`*[T: i32x8](A, B: T): T  =
    T(mm256_sub_epi32(A.M256i, B.M256i))

template `+`*[T: u64x4](A, B: T): T  =
    T(mm256_add_epi64(A.M256i, B.M256i))

template `+`*(A, B: M256i): M256i  =
    mm256_add_epi32(A, B)

template `-`*(A, B: M256i): M256i  =
    mm256_sub_epi32(A, B)

template `*`*(A, B: M256i): M256i  =
    mm256_mul_epi32(A, B)

template `rotLeft32`*(A: M256i, k: int32): M256i =
    mm256_or_si256(mm256_slli_epi32(A, k), mm256_srli_epi32(A, 32 - k))  

template `perm_right_rot_8_by_one`*(A: M256i): M256i  =
      A.mm256_permutevar8x32_epi32(mm256_set_epi32( 7, 0, 1, 2, 3, 4, 5, 6 ))

template `perm_right_rot_4x2_by_one`*(A: M256i): M256i  =
      A.mm256_permutevar8x32_epi32(mm256_set_epi32( 3, 0, 1, 2, 7, 4, 5, 6 ))

template `perm_swap_four`*(A: M256i): M256i  =
      A.mm256_permutevar8x32_epi32(mm256_set_epi32( 4, 5, 6, 7, 0, 1, 2, 3 ))


####-------------------- M256 float templates ----------------------####

template f32*(A: M256, k: int32): float32 =
    mm256_extract_ps(A, k)

template f64*(A: M256d, k: int32): float64 =
    mm256_extract_pd(A, k)

template `and`*(A, B: M256): M256 =
    mm256_and_ps(A, B)

template `or`*(A, B: M256): M256 =
    mm256_or_ps(A, B)

template `xor`*(A, B: M256): M256 =
    mm256_xor_ps(A, B)

template `+`*(A, B: M256): M256 =
    mm256_add_ps(A, B)

template `-`*(A, B: M256): M256 =
    mm256_sub_ps(A, B)

template `*`*(A, B: M256): M256 =
    mm256_mul_ps(A, B)

template `/`*(A, B: M256): M256 =
    mm256_div_ps(A, B)

template `+`*(A, B: M256d): M256d =
    ## A: left operand.
    ## B: right operand.
    mm256_add_pd(A, B)

template `-`*(A, B: M256d): M256d =
    ## A: left operand.
    ## B: right operand.
    mm256_sub_pd(A, B)

template `*`*(A, B: M256d): M256d =
    ## A: left operand.
    ## B: right operand.
    mm256_mul_pd(A, B)

template `/`*(A, B: M256d): M256d =
    ## A: left operand.
    ## B: right operand.
    mm256_div_pd(A, B)


####-------------------- NEON templates ----------------------####


template `and`*(A, B: uint16x8): uint16x8 =
    vandq_u16(A, B)

template `and`*(A, B: uint32x4): uint32x4  =
    vandq_u32(A, B)

template `and`*(A, B: uint64x2): uint64x2 =
    vandq_u64(A, B)

template `or`*(A, B: uint16x8): uint16x8 =
    vorrq_u16(A, B)

template `or`*(A, B: uint32x4): uint32x4  =
    vorrq_u32(A, B)

template `or`*(A, B: uint64x2): uint64x2 =
    vorrq_u64(A, B)

template `xor`*(A, B: uint16x8): uint16x8 =
    veorq_u16(A, B)

template `xor`*(A, B: uint32x4): uint32x4  =
    veorq_u32(A, B)

template `xor`*(A, B: uint64x2): uint64x2 =
    veorq_u64(A, B)

template `+`*(A, B: uint16x8): uint16x8 =
    vaddq_u16(A, B)

template `-`*(A, B: uint16x8): uint16x8 =
    vsubq_u16(A, B)

template `+`*(A, B: uint32x4): uint32x4 =
    vaddq_u32(A, B)

template `-`*(A, B: uint32x4): uint32x4 =
    vsubq_u32(A, B)

template `+`*(A, B: uint64x2): uint64x2 =
    vaddq_u64(A, B)

template `-`*(A, B: uint64x2): uint64x2 =
    vsubq_u64(A, B)

template `not`*(A: uint16x8): uint16x8 =
    veorq_u16(A, vmovq_n_u16(0xffff'u16))

template `shl`*(A, B: uint32x4): uint32x4  =
    vshlq_n_u32(A, B)

template `not`*(A: uint32x4): uint32x4 =
    ## A: input vector.
    vmvnq_u32(A)

template `not`*(A: uint64x2): uint64x2 =
    veorq_u64(A, vmovq_n_u64(0xffffffffffffffff'u64))

proc `shl`*(A: uint16x8, k: int32|uint32): uint16x8  =
    var
        src: array[8, uint16]
        dst: array[8, uint16]
        i: int = 0
    vst1q_u16(cast[pointer](unsafeAddr src[0]), A)
    while i < src.len:
        dst[i] = src[i] shl int(k)
        i = i + 1
    result = vld1q_u16(cast[pointer](unsafeAddr dst[0]))

proc `shr`*(A: uint16x8, k: int32|uint32): uint16x8  =
    var
        src: array[8, uint16]
        dst: array[8, uint16]
        i: int = 0
    vst1q_u16(cast[pointer](unsafeAddr src[0]), A)
    while i < src.len:
        dst[i] = src[i] shr int(k)
        i = i + 1
    result = vld1q_u16(cast[pointer](unsafeAddr dst[0]))

proc `shl`*(A: uint32x4, k: int32|uint32): uint32x4  =
    ## A: input vector.
    ## k: shift amount.
    var
        src: array[4, uint32]
        dst: array[4, uint32]
        i: int = 0
    vst1q_u32(cast[pointer](unsafeAddr src[0]), A)
    while i < src.len:
        dst[i] = src[i] shl int(k)
        i = i + 1
    result = vld1q_u32(cast[pointer](unsafeAddr dst[0]))

proc `shr`*(A: uint32x4, k: int32|uint32): uint32x4  =
    ## A: input vector.
    ## k: shift amount.
    var
        src: array[4, uint32]
        dst: array[4, uint32]
        i: int = 0
    vst1q_u32(cast[pointer](unsafeAddr src[0]), A)
    while i < src.len:
        dst[i] = src[i] shr int(k)
        i = i + 1
    result = vld1q_u32(cast[pointer](unsafeAddr dst[0]))

template `rot_left`*(A: uint32x4, k: int32): uint32x4 =
    (A shl k) or (A shr (32 - k))

proc `shl`*(A: uint64x2, k: int32|uint32): uint64x2 =
    var
        src: array[2, uint64]
        dst: array[2, uint64]
        i: int = 0
    vst1q_u64(cast[pointer](unsafeAddr src[0]), A)
    while i < src.len:
        dst[i] = src[i] shl int(k)
        i = i + 1
    result = vld1q_u64(cast[pointer](unsafeAddr dst[0]))

proc `shr`*(A: uint64x2, k: int32|uint32): uint64x2 =
    var
        src: array[2, uint64]
        dst: array[2, uint64]
        i: int = 0
    vst1q_u64(cast[pointer](unsafeAddr src[0]), A)
    while i < src.len:
        dst[i] = src[i] shr int(k)
        i = i + 1
    result = vld1q_u64(cast[pointer](unsafeAddr dst[0]))

template `rot_left`*(A: uint64x2, k: int32): uint64x2 =
    (A shl k) or (A shr (64 - k))
    
####-------------------- M128i Sequence templates ----------------------####


proc `and`*(A, B: seq[M128i]): seq[M128i] =
    let
        n = min(A.len, B.len)
    for i in 0..<n:
        result.add(A[i] and B[i])

proc `or`*(A, B: seq[M128i]): seq[M128i]  =
    let
        n = min(A.len, B.len)
    for i in 0..<n:
        result.add(A[i] or B[i])

proc `xor`*(A, B: seq[M128i]): seq[M128i]  =
    let
        n = min(A.len, B.len)
    for i in 0..<n:
        result.add(A[i] xor B[i])

proc `shl`*(A: seq[M128i], k: int32): seq[M128i]  =
    for el in A:
        result.add(el shl k)

proc `rot_left`*(A: seq[M128i], k: int32): seq[M128i] =
    for el in A:
        result.add(el.rot_left(k))
    
proc `+`*(A, B: seq[M128i]): seq[M128i]  =
    let
        n = min(A.len, B.len)
    for i in 0..<n:
        result.add(A[i] + B[i])

proc `-`*(A, B: seq[M128i]): seq[M128i]  =
    let
        n = min(A.len, B.len)
    for i in 0..<n:
        result.add(A[i] - B[i])

proc `*`*(A, B: seq[M128i]): seq[M128i]  =
    let
        n = min(A.len, B.len)
    for i in 0..<n:
        result.add(A[i] * B[i])
    
####-------------------- M256i sequence templates ----------------------####



proc `and`*(A, B: seq[M256i]): seq[M256i] =
    let
        n = min(A.len, B.len)
    for i in 0..<n:
        result.add(A[i] and B[i])

proc `or`*(A, B: seq[M256i]): seq[M256i]  =
    let
        n = min(A.len, B.len)
    for i in 0..<n:
        result.add(A[i] or B[i])

proc `xor`*(A, B: seq[M256i]): seq[M256i]  =
    let
        n = min(A.len, B.len)
    for i in 0..<n:
        result.add(A[i] xor B[i])

proc `shl`*(A: seq[M256i], k: int32): seq[M256i]  =
    for el in A:
        result.add(el shl k)

proc `rot_left`*(A: seq[M256i], k: int32): seq[M256i] =
    for el in A:
        result.add(rotLeft32(el, k))
    
proc `+`*(A, B: seq[M256i]): seq[M256i]  =
    let
        n = min(A.len, B.len)
    for i in 0..<n:
        result.add(A[i] + B[i])

proc `-`*(A, B: seq[M256i]): seq[M256i]  =
    let
        n = min(A.len, B.len)
    for i in 0..<n:
        result.add(A[i] - B[i])

proc `*`*(A, B: seq[M256i]): seq[M256i]  =
    let
        n = min(A.len, B.len)
    for i in 0..<n:
        result.add(A[i] * B[i])

