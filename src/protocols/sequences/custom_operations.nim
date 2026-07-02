#============================================#
# sequences/custom_operations.nim            #
# <- Search helpers across SIMD sequences.   #
#============================================#

import
    nimsimd/sse2,
    ../simd/base_operations

proc compressMask16To8(m: uint16): uint8 =
    ## m: movemask with one bit per byte.
    var
        tMask: uint8 = 0
        i: int = 0
    while i < 8:
        if (m and (1'u16 shl (i * 2))) != 0:
            tMask = tMask or (1'u8 shl i)
        i = i + 1
    result = tMask

proc compressMask16To4(m: uint16): uint8 =
    ## m: movemask with one bit per byte.
    var
        tMask: uint8 = 0
        i: int = 0
    while i < 4:
        if (m and (1'u16 shl (i * 4))) != 0:
            tMask = tMask or (1'u8 shl i)
        i = i + 1
    result = tMask

proc compressMask16To2(m: uint16): uint8 =
    ## m: movemask with one bit per byte.
    var
        tMask: uint8 = 0
        i: int = 0
    while i < 2:
        if (m and (1'u16 shl (i * 8))) != 0:
            tMask = tMask or (1'u8 shl i)
        i = i + 1
    result = tMask

proc buildMaskM128iCmp8(ms: seq[M128i], v: int8): uint64 =
    ## ms: input SIMD vectors.
    ## v: scalar value to compare against.
    var
        tMask: uint64 = 0
        i: int = 0
        j: int = 0
        tCmp: M128i
        tMask16: uint16 = 0
    let
        vVec = mm_set1_epi8(v)
    for el in ms:
        tCmp = mm_cmpeq_epi8(el, vVec)
        tMask16 = uint16(mm_movemask_epi8(tCmp))
        j = 0
        while j < 16 and i < 64:
            if (tMask16 and (1'u16 shl j)) != 0:
                tMask = tMask or (1'u64 shl i)
            i = i + 1
            j = j + 1
        if i >= 64:
            break
    result = tMask

proc buildMaskM128iCmp16(ms: seq[M128i], v: int16): uint64 =
    ## ms: input SIMD vectors.
    ## v: scalar value to compare against.
    var
        tMask: uint64 = 0
        i: int = 0
        j: int = 0
        tCmp: M128i
        tMask8: uint8 = 0
    let
        vVec = mm_set1_epi16(v)
    for el in ms:
        tCmp = mm_cmpeq_epi16(el, vVec)
        tMask8 = compressMask16To8(uint16(mm_movemask_epi8(tCmp)))
        j = 0
        while j < 8 and i < 64:
            if (tMask8 and (1'u8 shl j)) != 0:
                tMask = tMask or (1'u64 shl i)
            i = i + 1
            j = j + 1
        if i >= 64:
            break
    result = tMask

proc buildMaskM128iCmp32(ms: seq[M128i], v: int32): uint64 =
    ## ms: input SIMD vectors.
    ## v: scalar value to compare against.
    var
        tMask: uint64 = 0
        i: int = 0
        j: int = 0
        tCmp: M128i
        tMask4: uint8 = 0
    let
        vVec = mm_set1_epi32(v)
    for el in ms:
        tCmp = mm_cmpeq_epi32(el, vVec)
        tMask4 = compressMask16To4(uint16(mm_movemask_epi8(tCmp)))
        j = 0
        while j < 4 and i < 64:
            if (tMask4 and (1'u8 shl j)) != 0:
                tMask = tMask or (1'u64 shl i)
            i = i + 1
            j = j + 1
        if i >= 64:
            break
    result = tMask

proc buildMaskM128iCmp64(ms: seq[M128i], v: int64): uint64 =
    ## ms: input SIMD vectors.
    ## v: scalar value to compare against.
    var
        tMask: uint64 = 0
        i: int = 0
        j: int = 0
        tCmp32: M128i
        tCmp64: M128i
        tMask2: uint8 = 0
    let
        vVec = mm_set1_epi64x(v)
    for el in ms:
        tCmp32 = mm_cmpeq_epi32(el, vVec)
        tCmp64 = mm_and_si128(tCmp32, mm_shuffle_epi32(tCmp32, 0xB1))
        tMask2 = compressMask16To2(uint16(mm_movemask_epi8(tCmp64)))
        j = 0
        while j < 2 and i < 64:
            if (tMask2 and (1'u8 shl j)) != 0:
                tMask = tMask or (1'u64 shl i)
            i = i + 1
            j = j + 1
        if i >= 64:
            break
    result = tMask


proc firstSetBitIndex(m: uint64): uint64 =
    ## m: bitmask to scan.
    var
        tMask: uint64 = m
        i: uint64 = 0
    while tMask != 0:
        if (tMask and 1'u64) != 0:
            return i
        tMask = tMask shr 1
        i = i + 1
    result = high(uint64)

proc buildMaskM128Cmp(ms: seq[M128], v: float32): uint32 =
    ## ms: input SIMD vectors.
    ## v: scalar value to compare against.
    var
        tMask: uint32 = 0
        i: int = 0
        j: int = 0
        tCmp: M128
        tMask4: uint32 = 0
    let
        vVec = mm_set1_ps(v)
    for el in ms:
        tCmp = mm_cmpeq_ps(el, vVec)
        tMask4 = uint32(mm_movemask_ps(tCmp))
        j = 0
        while j < 4 and i < 32:
            if (tMask4 and (1'u32 shl j)) != 0:
                tMask = tMask or (1'u32 shl i)
            i = i + 1
            j = j + 1
        if i >= 32:
            break
    result = tMask


proc firstSetBitIndex(m: uint32): uint32 =
    ## m: bitmask to scan.
    var
        tMask: uint32 = m
        i: uint32 = 0
    while tMask != 0:
        if (tMask and 1'u32) != 0:
            return i
        tMask = tMask shr 1
        i = i + 1
    result = high(uint32)

proc find*(ms: seq[M128i], v: SomeInteger): uint64 =
    ## ms: input SIMD vectors.
    ## v: scalar value to compare against.
    type Tv = typeof(v)
    when sizeof(Tv) == 1:
        result = firstSetBitIndex(buildMaskM128iCmp8(ms, int8(v)))
    elif sizeof(Tv) == 2:
        result = firstSetBitIndex(buildMaskM128iCmp16(ms, int16(v)))
    elif sizeof(Tv) == 4:
        result = firstSetBitIndex(buildMaskM128iCmp32(ms, int32(v)))
    elif sizeof(Tv) == 8:
        result = firstSetBitIndex(buildMaskM128iCmp64(ms, int64(v)))
    else:
        result = high(uint64)

proc find*(ms: seq[M128], v: SomeFloat): uint32 =
    ## ms: input SIMD vectors.
    ## v: scalar value to compare against.
    result = firstSetBitIndex(buildMaskM128Cmp(ms, float32(v)))

proc findAll*(ms: seq[M128i], v: SomeInteger): uint64 =
    ## ms: input SIMD vectors.
    ## v: scalar value to compare against.
    type Tv = typeof(v)
    when sizeof(Tv) == 1:
        result = buildMaskM128iCmp8(ms, int8(v))
    elif sizeof(Tv) == 2:
        result = buildMaskM128iCmp16(ms, int16(v))
    elif sizeof(Tv) == 4:
        result = buildMaskM128iCmp32(ms, int32(v))
    elif sizeof(Tv) == 8:
        result = buildMaskM128iCmp64(ms, int64(v))
    else:
        result = 0
