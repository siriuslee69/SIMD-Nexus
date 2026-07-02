#============================================#
# array_overloaded.nim                       #
# <- Array convenience overloads.            #
#============================================#

import 
    nimsimd/sse2,
    nimsimd/avx2,
    nimsimd/neon
import 
    ./simd/converters

let 
    mask_right: M256i = mm256_setr_epi32(0xFFFFFFFF'u32,0xFFFFFFFF'u32,0xFFFFFFFF'u32,0xFFFFFFFF'u32,0'u32,0'u32,0'u32,0'u32)
    mask_left: M256i = mm256_setr_epi32(0'u32,0'u32,0'u32,0'u32,0xFFFFFFFF'u32,0xFFFFFFFF'u32,0xFFFFFFFF'u32,0xFFFFFFFF'u32)


template `and`*(A, B: array[8, uint32]): M256i =
    asM256i(A) and asM256i(B)
