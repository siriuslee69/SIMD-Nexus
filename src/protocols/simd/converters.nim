#============================================#
# simd/converters.nim                        #
# <- Scalar/array to SIMD conversion helpers #
#============================================#

import 
    nimsimd/avx2,
    base_operations


template fillM128i*(a: uint8|int8 ): i8x16 =
    i8x16(mm_set1_epi8(a))

template fillM128i*(a: uint16|int16 ): i16x8 =
    i16x8(mm_set1_epi16(a))

template fillM128i*(a: uint32|int32 ): i32x4 =
    i32x4(mm_set1_epi32(a))

template set1M128i*(a: uint8|int8 ): i8x16 =
    i8x16(mm_set1_epi8(a))

template set1M128i*(a: uint16|int16 ): i16x8 =
    i16x8(mm_set1_epi16(a))

template set1M128i*(a: uint32|int32 ): i32x4 =
    i32x4(mm_set1_epi32(a))

template asM128i*(A: array[16, uint8]|array[16, int8]): i8x16 =
    i8x16(mm_setr_epi8(
        A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7],
        A[8], A[9], A[10], A[11], A[12], A[13], A[14], A[15]
    ))

template asM128i*(A: array[8, uint16]|array[8, int16]): i16x8 =
    i16x8(mm_setr_epi16(A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7]))

template asM128i*(A: array[4, uint32]|array[4,int32]): i32x4 =
    i32x4(mm_setr_epi32(A[0], A[1], A[2], A[3]))

#Runtime conversion
proc toM128i*(A: array[4, uint32]|array[4,int32]): i32x4 =
    result = i32x4(mm_setr_epi32(A[0], A[1], A[2], A[3]))

proc toi8x16*(A: M128i): i8x16 =
    cast[i8x16](A)

proc toi16x8*(A: M128i): i16x8 =
    cast[i16x8](A)

proc toi32x4*(A: M128i): i32x4 =
    cast[i32x4](A)

proc toM128iSeqImpl[T](A: openArray[T]): seq[i32x4] =
    let 
        n = A.len
        numFullVectors = n div 4
        remainder = n mod 4
    var lastVec: array[4, int32]
    result = newSeqOfCap[i32x4](if remainder > 0: numFullVectors + 1 else: numFullVectors)
    
    for i in 0 ..< numFullVectors:
        let offset = i * 4
        result.add(i32x4(mm_setr_epi32(
            int32(A[offset]), int32(A[offset + 1]), int32(A[offset + 2]), int32(A[offset + 3])
        )))
    
    if remainder > 0:
        for i in 0 ..< remainder:
            lastVec[i] = int32(A[numFullVectors * 4 + i])
        result.add(i32x4(mm_setr_epi32(lastVec[0], lastVec[1], lastVec[2], lastVec[3])))

proc toM128iSeq*(A: openArray[uint32]): seq[i32x4] =
    toM128iSeqImpl(A)

proc toM128iSeq*(A: openArray[int32]): seq[i32x4] =
    toM128iSeqImpl(A)


template asM256i*(A: array[4, uint64]|array[4,int64]): M256i =
    mm256_setr_epi64x(A[0], A[1], A[2], A[3])

#Compile time conversion
template asM256i*(A: array[8, uint32]|array[8,int32]): M256i =
    mm256_setr_epi32(A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7])

#Runtime conversion
proc toM256i*(A: array[8, uint32]|array[8,int32]): M256i =
    result = mm256_setr_epi32(A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7])


proc toM256iSeqImpl[T](A: openArray[T]): seq[M256i] =
    let 
        n = A.len
        numFullVectors = n div 8
        remainder = n mod 8
    var lastVec: array[8, int32]
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

proc toM256iSeq*(A: openArray[uint32]): seq[M256i] =
    toM256iSeqImpl(A)

proc toM256iSeq*(A: openArray[int32]): seq[M256i] =
    toM256iSeqImpl(A)

proc toi8x32*(A: M256i): i8x32 =
    cast[i8x32](A)

proc toi16x16*(A: M256i): i16x16 =
    cast[i16x16](A)

proc toi32x8*(A: M256i): i32x8 =
    cast[i32x8](A)

proc toSeqU32*(A: openArray[M256i]): seq[uint32] =
    for el in A:
        result.add( el[0, uint32] )
        result.add( el[1, uint32] )
        result.add( el[2, uint32] )
        result.add( el[3, uint32] )
        result.add( el[4, uint32] )
        result.add( el[5, uint32] )
        result.add( el[6, uint32] )
        result.add( el[7, uint32] )

proc toSeqU32*(A: openArray[M128i]): seq[uint32] =
    for el in A:
        result.add( el[0, uint32] )
        result.add( el[1, uint32] )
        result.add( el[2, uint32] )
        result.add( el[3, uint32] )
