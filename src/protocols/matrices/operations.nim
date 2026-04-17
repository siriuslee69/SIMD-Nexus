#============================================#
# matrices/operations.nim                    #
# <- Matrix-oriented SIMD operations.        #
#============================================#

import 
    nimsimd/avx2,
    nimsimd/neon

let 
    mask_right: M256i = mm256_setr_epi32(0xFFFFFFFF'u32,0xFFFFFFFF'u32,0xFFFFFFFF'u32,0xFFFFFFFF'u32,0'u32,0'u32,0'u32,0'u32)
    mask_left: M256i = mm256_setr_epi32(0'u32,0'u32,0'u32,0'u32,0xFFFFFFFF'u32,0xFFFFFFFF'u32,0xFFFFFFFF'u32,0xFFFFFFFF'u32)

template `and`*(A, B: M128i): M128i  =
    mm_and_si128(A, B)

template `or`*(A, B: M128i): M128i  =
    mm_or_si128(A, B)

template `xor`*(A, B: M128i): M128i  =
    mm_xor_si128(A, B)

template `shl`*(A: M128i, k: int32): M128i  =
    mm_slli_epi32(A, k)

template `rot_left`*(A: M128i, k: int32): M128i =
    mm_or_si128(mm_slli_epi32(A, k), mm_srli_epi32(A, 32 - k))  
    
template `perm_right_rot_by_one`*(A: M128i): M128i  =
    mm_shuffle_epi32(A, 0x93)

template `perm_swap_four`*(A: M128i): M128i  =
    mm_shuffle_epi32(A, 0x4E)

####-------------------- M256i templates ----------------------####
 
template `and`*(A, B: M256i): M256i  =
    mm256_and_si256(A, B)

template `or`*(A, B: M256i): M256i  =
    mm256_or_si256(A, B)

template `xor`*(A, B: M256i): M256i  =
    mm256_xor_si256(A, B)

template `shl`*(A: M256i, k: int32): M256i  =
    mm256_slli_epi32(A, k)

template `rot_left`*(A: M256i, k: int32): M256i =
    mm256_or_si256(mm256_slli_epi32(A, k), mm256_srli_epi32(A, 32 - k))  
    
template `perm_right_rot_8_by_one`*(A: M256i): M256i  =
      A.mm256_permutevar8x32_epi32(mm256_set_epi32( 7, 0, 1, 2, 3, 4, 5, 6 ))

template `perm_right_rot_4x2_by_one`*(A: M256i): M256i  =
      A.mm256_permutevar8x32_epi32(mm256_set_epi32( 3, 0, 1, 2, 7, 4, 5, 6 ))

template `perm_swap_four`*(A: M256i): M256i  =
      A.mm256_permutevar8x32_epi32(mm256_set_epi32( 4, 5, 6, 7, 0, 1, 2, 3 ))

template `+`*(A, B: M256i): M256i  =
    mm256_add_epi32(A, B)

template `-`*(A, B: M256i): M256i  =
    mm256_sub_epi32(A, B)

template `*`*(A, B: M256i): M256i  =
    mm256_mul_epi32(A, B)

####-------------------- NEON templates ----------------------####

template `and`*(A, B: uint32x4): uint32x4  =
    vandq_u32(A, B)

template `or`*(A, B: uint32x4): uint32x4  =
    vorrq_u32(A, B)

template `xor`*(A, B: uint32x4): uint32x4  =
    veorq_u32(A, B)

template `shl`*(A, B: uint32x4): uint32x4  =
    vshlq_n_u32(A, B)

template `rot_left`*(A: uint32x4, k: int32): uint32x4 =
    vorrq_u32(vshlq_n_u32(A, k), vshrq_n_u32(A, 32 - k))  
    
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
        result.add(el.rot_left(k))
