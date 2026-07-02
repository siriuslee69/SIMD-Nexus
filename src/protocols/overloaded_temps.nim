#============================================#
# overloaded_temps.nim                       #
# <- Legacy SIMD helpers and experiments.    #
#============================================#

import 
    nimsimd/avx2,
    nimsimd/neon

let 
    mask_right: M256i = mm256_setr_epi32(0xFFFFFFFF'u32,0xFFFFFFFF'u32,0xFFFFFFFF'u32,0xFFFFFFFF'u32,0'u32,0'u32,0'u32,0'u32)
    mask_left: M256i = mm256_setr_epi32(0'u32,0'u32,0'u32,0'u32,0xFFFFFFFF'u32,0xFFFFFFFF'u32,0xFFFFFFFF'u32,0xFFFFFFFF'u32)

template asM128i(A: array[4, uint32]|array[4, int32]): M128i =
    ## A: input array with 4 elements.
    mm_setr_epi32(int32(A[0]), int32(A[1]), int32(A[2]), int32(A[3]))

proc toM128i(A: array[4, uint32]|array[4,int32]): M128i =
    ## A: input array with 4 elements.
    result = mm_setr_epi32(int32(A[0]), int32(A[1]), int32(A[2]), int32(A[3]))

proc toM128iSeqImpl[T](A: openArray[T]): seq[M128i] =
    ## A: input scalar array.
    let 
        n = A.len
        numFullVectors = n div 4
        remainder = n mod 4
    var 
        lastVec: array[4, int32]
    result = newSeqOfCap[M128i](if remainder > 0: numFullVectors + 1 else: numFullVectors)
    
    for i in 0 ..< numFullVectors:
        let offset = i * 4
        result.add(mm_setr_epi32(
            int32(A[offset]), int32(A[offset + 1]), int32(A[offset + 2]), int32(A[offset + 3])
        ))
    
    if remainder > 0:
        for i in 0 ..< remainder:
            lastVec[i] = int32(A[numFullVectors * 4 + i])
        result.add(mm_setr_epi32(lastVec[0], lastVec[1], lastVec[2], lastVec[3]))

proc toM128iSeq(A: openArray[uint32]|openArray[int32]): seq[M128i] =
    ## A: input scalar array.
    result = toM128iSeqImpl(A)

template asM256i(A: array[8, uint32]|array[8, int32]): M256i =
    ## A: input array with 8 elements.
    mm256_setr_epi32(
        int32(A[0]), int32(A[1]), int32(A[2]), int32(A[3]),
        int32(A[4]), int32(A[5]), int32(A[6]), int32(A[7])
    )

template asM256i(A: array[4, uint32]|array[4, int32]): M256i =
    ## A: input array with 4 elements (upper lanes filled with zeros).
    mm256_setr_epi32(
        int32(A[0]), int32(A[1]), int32(A[2]), int32(A[3]),
        0, 0, 0, 0
    )

proc toM256i(A: array[8, uint32]|array[8,int32]): M256i =
    ## A: input array with 8 elements.
    result = mm256_setr_epi32(
        int32(A[0]), int32(A[1]), int32(A[2]), int32(A[3]),
        int32(A[4]), int32(A[5]), int32(A[6]), int32(A[7])
    )

proc toM256i(A: array[4, uint32]|array[4,int32]): M256i =
    ## A: input array with 4 elements (upper lanes filled with zeros).
    result = mm256_setr_epi32(
        int32(A[0]), int32(A[1]), int32(A[2]), int32(A[3]),
        0, 0, 0, 0
    )

proc toM256iSeqImpl[T](A: openArray[T]): seq[M256i] =
    ## A: input scalar array.
    let 
        n = A.len
        numFullVectors = n div 8
        remainder = n mod 8
    var 
        lastVec: array[8, int32]
    result = newSeqOfCap[M256i](if remainder > 0: numFullVectors + 1 else: numFullVectors)
    
    for i in 0 ..< numFullVectors:
        let offset = i * 8
        result.add(mm256_setr_epi32(
            int32(A[offset]), int32(A[offset + 1]), int32(A[offset + 2]), int32(A[offset + 3]),
            int32(A[offset + 4]), int32(A[offset + 5]), int32(A[offset + 6]), int32(A[offset + 7])
        ))
    
    if remainder > 0:
        for i in 0 ..< remainder:
            lastVec[i] = int32(A[numFullVectors * 8 + i])
        result.add(mm256_setr_epi32(
            lastVec[0], lastVec[1], lastVec[2], lastVec[3],
            lastVec[4], lastVec[5], lastVec[6], lastVec[7]
        ))

proc toM256iSeq(A: openArray[uint32]|openArray[int32]): seq[M256i] =
    ## A: input scalar array.
    result = toM256iSeqImpl(A)


template `+`*(A, B: M128i): M128i  =
    mm_add_epi32(A, B)

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
      A.mm256_permutevar8x32_epi32(mm256_setr_epi32( 7, 0, 1, 2, 3, 4, 5, 6 ))

template `perm_right_rot_4x2_by_one`*(A: M256i): M256i  =
      A.mm256_permutevar8x32_epi32(mm256_setr_epi32( 3, 0, 1, 2, 7, 4, 5, 6 ))

template `perm_swap_four`*(A: M256i): M256i  =
      A.mm256_permutevar8x32_epi32(mm256_setr_epi32( 4, 5, 6, 7, 0, 1, 2, 3 ))

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
    
proc `add`*(A, B: uint32x4): uint32x4 {.inline.} =
    return vaddq_u32(A, B)

proc `rotate_left`*(A: uint32x4, k: int32): uint32x4 {.inline.} =
    return vorrq_u32(vshlq_n_u32(A, k), vshrq_n_u32(A, 32 - k))  
    
proc `rotate_left`*(A: M256i, k: int32): M256i {.inline.} =
    return mm256_or_si256(mm256_slli_epi32(A, k), mm256_srli_epi32(A, 32 - k))  
    
proc `swap`*(A,B: var M256i): void {.inline.} =
    A = mm256_xor_si256(A, B)
    B = mm256_xor_si256(A, B)
    A = mm256_xor_si256(A, B)

proc avx_small_swap_32x4x2*(A: var M256i): void {.inline.} =
    A = A.mm256_permutevar8x32_epi32(mm256_setr_epi32( 6, 7, 4, 5, 2, 3, 0, 1 ))

proc avx_big_swap_32x4x2*(A: var M256i): void {.inline.} =
    A = A.mm256_permutevar8x32_epi32(mm256_setr_epi32( 4, 5, 6, 7, 0, 1, 2, 3 ))

proc avx_small_block_swap_128x2x2*(A: var M256i): void {.inline.} =
    A = A.mm256_permutevar8x32_epi32(mm256_setr_epi32( 3, 2, 1, 0, 7, 6, 5, 4 ))

proc avx_big_block_swap_128x2x2*(A,B: var M256i): void {.inline.} =
    B.avx_small_block_swap_128x2x2()
    A = A.mm256_xor_si256(B)
    B = B.mm256_xor_si256(A)
    A = A.mm256_xor_si256(B)
