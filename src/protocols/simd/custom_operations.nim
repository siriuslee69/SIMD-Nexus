#============================================#
# simd/custom_operations.nim                 #
# <- Higher-level SIMD helpers and formatters#
#============================================#

import 
    base_operations,
    converters


proc rotLeft32S(A: M256i, c: uint32): M256i =
    ## A: input vector.
    ## c: rotation count (masked to 0..63).
    ## Safe version, will make sure that c is within range.
    let 
        k = 0b0000_0000_0000_0000_0000_0000_0011_1111 and c
        s = 32 - k
    result = (A shl k) or (A shr s)

proc rotLeft32F*(A: M256i, k: uint32, anti_k: uint32): M256i =
    ## A: input vector.
    ## k: rotation count.
    ## anti_k: precomputed 32-k (unchecked).
    ## Allows for faster calculation when in a loop, anti_k has to be 32-k, potentially unsafe.
    result = (A shl k) or (A shr anti_k)


proc `$`*(A: i8x32): string =
    result = $[
            A[0], A[1], A[2], A[3],  
            A[4], A[5], A[6], A[7], 
            A[8], A[9], A[10], A[11], 
            A[12], A[13], A[14], A[15], 
            A[16], A[17], A[18], A[19], 
            A[20], A[21], A[22], A[23], 
            A[24], A[25], A[26], A[27], 
            A[28], A[29], A[30], A[31],                       
            ]

proc `$`*(A: i16x16): string =    
    result = $[
            A[0], A[1], A[2], A[3],  
            A[4], A[5], A[6], A[7], 
            A[8], A[9], A[10], A[11], 
            A[12], A[13], A[14], A[15], 
            ]

proc `$`*(A: i32x8): string =
     result = $[
            A[0], A[1], A[2], A[3],  
            A[4], A[5], A[6], A[7], 
            ]

proc `$`*(A: u64x4): string =
     result = $[
            A[0], A[1], A[2], A[3],  
            ]
