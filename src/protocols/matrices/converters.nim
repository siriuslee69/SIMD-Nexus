#============================================#
# matrices/converters.nim                    #
# <- Matrix-oriented SIMD conversions.       #
#============================================#

import ../isa/x86_avx2
import 
    nimsimd/avx2

type Matrix = seq[M128i]

template `Matrix`(A: array[4, uint32]|array[4, int32]): Matrix =
    ## A: input array with 4 elements.
    @[toM128i(A)]

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

template `M256i`(A: array[8, uint32]|array[8, int32]): M256i =
    ## A: input array with 8 elements.
    mm256_setr_epi32(
        int32(A[0]), int32(A[1]), int32(A[2]), int32(A[3]),
        int32(A[4]), int32(A[5]), int32(A[6]), int32(A[7])
    )

template `M256i`(A: array[4, uint32]|array[4, int32]): M256i =
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
