#============================================#
# simd/iterators.nim                         #
# <- SIMD range iterators with vector masks. #
#============================================#

import ../isa/x86_avx2
import
    nimsimd/avx2,
    nimsimd/sse2,
    ./base_operations,
    ./generic_u32,
    ./generic_i8,
    ./generic_i16,
    ./generic_u64,
    ./generic_f32,
    ./generic_f64


proc simdIndexU32*[T: SimdU32](s: uint32): T =
    ## s: start value.
    var
        i: int = 0
    when T is SimdU32x8:
        var
            As: array[8, uint32]
        i = 0
        while i < 8:
            As[i] = s + uint32(i)
            i = i + 1
        result = loadU32x8[T](As)
    else:
        var
            As: array[4, uint32]
        i = 0
        while i < 4:
            As[i] = s + uint32(i)
            i = i + 1
        result = loadU32x4[T](As)


proc simdMaskU32*[T: SimdU32](r: int): T =
    ## r: number of valid lanes.
    var
        t: int = r
        i: int = 0
    if t < 0:
        t = 0
    when T is SimdU32x8:
        if t > 8:
            t = 8
        var
            ms: array[8, uint32]
        i = 0
        while i < 8:
            if i < t:
                ms[i] = 0xFFFFFFFF'u32
            else:
                ms[i] = 0'u32
            i = i + 1
        result = loadU32x8[T](ms)
    else:
        if t > 4:
            t = 4
        var
            ms: array[4, uint32]
        i = 0
        while i < 4:
            if i < t:
                ms[i] = 0xFFFFFFFF'u32
            else:
                ms[i] = 0'u32
            i = i + 1
        result = loadU32x4[T](ms)


proc simdIndexI8*[T: SimdI8](s: int8): T =
    ## s: start value.
    var
        i: int = 0
    when T is SimdI8x32:
        var
            As: array[32, int8]
        i = 0
        while i < 32:
            As[i] = int8(int32(s) + i)
            i = i + 1
        result = loadI8x32[T](As)
    else:
        var
            As: array[16, int8]
        i = 0
        while i < 16:
            As[i] = int8(int32(s) + i)
            i = i + 1
        result = loadI8x16[T](As)


proc simdMaskI8*[T: SimdI8](r: int): T =
    ## r: number of valid lanes.
    var
        t: int = r
        i: int = 0
    if t < 0:
        t = 0
    when T is SimdI8x32:
        if t > 32:
            t = 32
        var
            ms: array[32, uint8]
        i = 0
        while i < 32:
            if i < t:
                ms[i] = 0xFF'u8
            else:
                ms[i] = 0'u8
            i = i + 1
        result = loadI8x32[T](ms)
    else:
        if t > 16:
            t = 16
        var
            ms: array[16, uint8]
        i = 0
        while i < 16:
            if i < t:
                ms[i] = 0xFF'u8
            else:
                ms[i] = 0'u8
            i = i + 1
        result = loadI8x16[T](ms)


proc simdIndexI16*[T: SimdI16](s: int16): T =
    ## s: start value.
    var
        i: int = 0
    when T is SimdI16x16:
        var
            As: array[16, int16]
        i = 0
        while i < 16:
            As[i] = int16(int32(s) + i)
            i = i + 1
        result = loadI16x16[T](As)
    else:
        var
            As: array[8, int16]
        i = 0
        while i < 8:
            As[i] = int16(int32(s) + i)
            i = i + 1
        result = loadI16x8[T](As)


proc simdMaskI16*[T: SimdI16](r: int): T =
    ## r: number of valid lanes.
    var
        t: int = r
        i: int = 0
    if t < 0:
        t = 0
    when T is SimdI16x16:
        if t > 16:
            t = 16
        var
            ms: array[16, uint16]
        i = 0
        while i < 16:
            if i < t:
                ms[i] = 0xFFFF'u16
            else:
                ms[i] = 0'u16
            i = i + 1
        result = loadI16x16[T](ms)
    else:
        if t > 8:
            t = 8
        var
            ms: array[8, uint16]
        i = 0
        while i < 8:
            if i < t:
                ms[i] = 0xFFFF'u16
            else:
                ms[i] = 0'u16
            i = i + 1
        result = loadI16x8[T](ms)


proc simdIndexU64*[T: SimdU64](s: uint64): T =
    ## s: start value.
    var
        i: int = 0
    when T is SimdU64x4:
        var
            As: array[4, uint64]
        i = 0
        while i < 4:
            As[i] = s + uint64(i)
            i = i + 1
        result = loadU64x4[T](As)
    else:
        var
            As: array[2, uint64]
        i = 0
        while i < 2:
            As[i] = s + uint64(i)
            i = i + 1
        result = loadU64x2[T](As)


proc simdMaskU64*[T: SimdU64](r: int): T =
    ## r: number of valid lanes.
    var
        t: int = r
        i: int = 0
    if t < 0:
        t = 0
    when T is SimdU64x4:
        if t > 4:
            t = 4
        var
            ms: array[4, uint64]
        i = 0
        while i < 4:
            if i < t:
                ms[i] = 0xFFFFFFFFFFFFFFFF'u64
            else:
                ms[i] = 0'u64
            i = i + 1
        result = loadU64x4[T](ms)
    else:
        if t > 2:
            t = 2
        var
            ms: array[2, uint64]
        i = 0
        while i < 2:
            if i < t:
                ms[i] = 0xFFFFFFFFFFFFFFFF'u64
            else:
                ms[i] = 0'u64
            i = i + 1
        result = loadU64x2[T](ms)


proc simdIndexF32*[T: SimdF32](s: float32): T =
    ## s: start value.
    var
        i: int = 0
    when T is SimdF32x8:
        var
            As: array[8, float32]
        i = 0
        while i < 8:
            As[i] = s + float32(i)
            i = i + 1
        result = loadF32x8[T](As)
    else:
        var
            As: array[4, float32]
        i = 0
        while i < 4:
            As[i] = s + float32(i)
            i = i + 1
        result = loadF32x4[T](As)


proc simdMaskF32*[T: SimdF32](r: int): T =
    ## r: number of valid lanes.
    var
        t: int = r
        i: int = 0
    if t < 0:
        t = 0
    when T is SimdF32x8:
        if t > 8:
            t = 8
        var
            b: M256i
            ms: array[8, uint32]
        i = 0
        while i < 8:
            if i < t:
                ms[i] = 0xFFFFFFFF'u32
            else:
                ms[i] = 0'u32
            i = i + 1
        b = mm256_loadu_si256(cast[pointer](unsafeAddr ms[0]))
        result = mm256_castsi256_ps(b)
    else:
        if t > 4:
            t = 4
        var
            a: M128i
            ms: array[4, uint32]
        i = 0
        while i < 4:
            if i < t:
                ms[i] = 0xFFFFFFFF'u32
            else:
                ms[i] = 0'u32
            i = i + 1
        a = mm_loadu_si128(cast[pointer](unsafeAddr ms[0]))
        result = mm_castsi128_ps(a)


proc simdIndexF64*[T: SimdF64](s: float64): T =
    ## s: start value.
    var
        i: int = 0
    when T is SimdF64x4:
        var
            As: array[4, float64]
        i = 0
        while i < 4:
            As[i] = s + float64(i)
            i = i + 1
        result = loadF64x4[T](As)
    else:
        var
            As: array[2, float64]
        i = 0
        while i < 2:
            As[i] = s + float64(i)
            i = i + 1
        result = loadF64x2[T](As)


proc simdMaskF64*[T: SimdF64](r: int): T =
    ## r: number of valid lanes.
    var
        t: int = r
        i: int = 0
    if t < 0:
        t = 0
    when T is SimdF64x4:
        if t > 4:
            t = 4
        var
            b: M256i
            ms: array[4, uint64]
        i = 0
        while i < 4:
            if i < t:
                ms[i] = 0xFFFFFFFFFFFFFFFF'u64
            else:
                ms[i] = 0'u64
            i = i + 1
        b = mm256_loadu_si256(cast[pointer](unsafeAddr ms[0]))
        result = mm256_castsi256_pd(b)
    else:
        if t > 2:
            t = 2
        var
            a: M128i
            ms: array[2, uint64]
        i = 0
        while i < 2:
            if i < t:
                ms[i] = 0xFFFFFFFFFFFFFFFF'u64
            else:
                ms[i] = 0'u64
            i = i + 1
        a = mm_loadu_si128(cast[pointer](unsafeAddr ms[0]))
        result = mm_castsi128_pd(a)


iterator simdRangeU32*[T: SimdU32](s: uint32, l: int): tuple[i: T, mask: T] =
    ## s: start value.
    ## l: number of elements to cover.
    var
        n: int = l
        lanes: int = 0
        s0: T
        s1: T
        s2: T
    if n > 0:
        when T is SimdU32x8:
            lanes = 8
        else:
            lanes = 4
        s0 = simdIndexU32[T](s)
        s1 = simdMaskU32[T](lanes)
        while n >= lanes:
            yield (s0, s1)
            s0 = s0 + set1U32[T](uint32(lanes))
            n = n - lanes
        if n > 0:
            s2 = simdMaskU32[T](n)
            yield (s0, s2)


iterator simdRangeI8*[T: SimdI8](s: int8, l: int): tuple[i: T, mask: T] =
    ## s: start value.
    ## l: number of elements to cover.
    var
        n: int = l
        lanes: int = 0
        s0: T
        s1: T
        s2: T
    if n > 0:
        when T is SimdI8x32:
            lanes = 32
        else:
            lanes = 16
        s0 = simdIndexI8[T](s)
        s1 = simdMaskI8[T](lanes)
        while n >= lanes:
            yield (s0, s1)
            s0 = s0 + set1I8[T](int8(lanes))
            n = n - lanes
        if n > 0:
            s2 = simdMaskI8[T](n)
            yield (s0, s2)


iterator simdRangeI16*[T: SimdI16](s: int16, l: int): tuple[i: T, mask: T] =
    ## s: start value.
    ## l: number of elements to cover.
    var
        n: int = l
        lanes: int = 0
        s0: T
        s1: T
        s2: T
    if n > 0:
        when T is SimdI16x16:
            lanes = 16
        else:
            lanes = 8
        s0 = simdIndexI16[T](s)
        s1 = simdMaskI16[T](lanes)
        while n >= lanes:
            yield (s0, s1)
            s0 = s0 + set1I16[T](int16(lanes))
            n = n - lanes
        if n > 0:
            s2 = simdMaskI16[T](n)
            yield (s0, s2)


iterator simdRangeU64*[T: SimdU64](s: uint64, l: int): tuple[i: T, mask: T] =
    ## s: start value.
    ## l: number of elements to cover.
    var
        n: int = l
        lanes: int = 0
        s0: T
        s1: T
        s2: T
    if n > 0:
        when T is SimdU64x4:
            lanes = 4
        else:
            lanes = 2
        s0 = simdIndexU64[T](s)
        s1 = simdMaskU64[T](lanes)
        while n >= lanes:
            yield (s0, s1)
            s0 = s0 + set1U64[T](uint64(lanes))
            n = n - lanes
        if n > 0:
            s2 = simdMaskU64[T](n)
            yield (s0, s2)


iterator simdRangeF32*[T: SimdF32](s: float32, l: int): tuple[i: T, mask: T] =
    ## s: start value.
    ## l: number of elements to cover.
    var
        n: int = l
        lanes: int = 0
        s0: T
        s1: T
        s2: T
    if n > 0:
        when T is SimdF32x8:
            lanes = 8
        else:
            lanes = 4
        s0 = simdIndexF32[T](s)
        s1 = simdMaskF32[T](lanes)
        while n >= lanes:
            yield (s0, s1)
            s0 = s0 + set1F32[T](float32(lanes))
            n = n - lanes
        if n > 0:
            s2 = simdMaskF32[T](n)
            yield (s0, s2)


iterator simdRangeF64*[T: SimdF64](s: float64, l: int): tuple[i: T, mask: T] =
    ## s: start value.
    ## l: number of elements to cover.
    var
        n: int = l
        lanes: int = 0
        s0: T
        s1: T
        s2: T
    if n > 0:
        when T is SimdF64x4:
            lanes = 4
        else:
            lanes = 2
        s0 = simdIndexF64[T](s)
        s1 = simdMaskF64[T](lanes)
        while n >= lanes:
            yield (s0, s1)
            s0 = s0 + set1F64[T](float64(lanes))
            n = n - lanes
        if n > 0:
            s2 = simdMaskF64[T](n)
            yield (s0, s2)
