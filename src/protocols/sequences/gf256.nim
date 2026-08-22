#============================================#
# sequences/gf256.nim                        #
# <- GF(256) byte arithmetic for erasure     #
#    codes, table-driven under SIMD.         #
#============================================#

when defined(amd64) or defined(i386):
    import ../isa/x86
    import nimsimd/sse2
    import nimsimd/ssse3

when defined(arm64) or defined(aarch64):
    import nimsimd/neon

when defined(simdNexusEnableAvx2) or defined(eirEnableAvx2):
    import ../isa/x86_avx2
    import nimsimd/avx2


const
    gf256Poly* = 0x1d'u8
        ## Reduction tail of x^8 + x^4 + x^3 + x^2 + 1, the polynomial used by
        ## every mainstream Reed-Solomon implementation.
    gf256LaneWidth* = 16
    gf256AvxLaneWidth* = 32
    gf256NibbleMask: array[16, uint8] = [
        0x0f'u8, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f,
        0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f]

    gf256Ascii* = """
One coefficient c becomes two 16-entry tables. Every byte is then two
lookups and one xor, which a single SIMD shuffle does 16 or 32 lanes wide.

  byte b = 0xA7
          +--------+--------+
          |  hi=A  |  lo=7  |
          +--------+--------+
              |        |
     hi[A] ---+        +--- lo[7]
              |        |
              +-- xor --+
                   |
              b * c in GF(256)

  lo[i] = i * c            (i = 0..15)
  hi[i] = (i shl 4) * c    (i = 0..15)
"""


type
    ## Gf256Tables: one coefficient split into low and high nibble products.
    Gf256Tables* = object
        ## lo: products of the low nibble, indexed 0..15.
        lo*: array[16, uint8]
        ## hi: products of the high nibble already shifted into place.
        hi*: array[16, uint8]


proc gf256Mul*(a, b: uint8): uint8 =
    ## a: left field element.
    ## b: right field element.
    ## Returns the carry-less product reduced by gf256Poly.
    var
        x: uint8 = a
        y: uint8 = b
        t: uint8 = 0'u8
        carry: uint8 = 0'u8
        i: int = 0
    while i < 8:
        if (y and 1'u8) != 0'u8:
            t = t xor x
        carry = x and 0x80'u8
        x = x shl 1
        if carry != 0'u8:
            x = x xor gf256Poly
        y = y shr 1
        i = i + 1
    result = t


proc gf256Pow*(a: uint8, e: int): uint8 =
    ## a: base field element.
    ## e: non-negative exponent.
    ## Returns a raised to e by square-and-multiply.
    var
        base: uint8 = a
        power: int = e
        t: uint8 = 1'u8
    while power > 0:
        if (power and 1) != 0:
            t = gf256Mul(t, base)
        base = gf256Mul(base, base)
        power = power shr 1
    result = t


proc gf256Inv*(a: uint8): uint8 =
    ## a: field element to invert; zero has no inverse and returns zero.
    ## Returns a^254, which is a^-1 for every non-zero a.
    if a == 0'u8:
        return 0'u8
    result = gf256Pow(a, 254)


proc gf256Tables*(c: uint8): Gf256Tables =
    ## c: coefficient whose nibble product tables are built once and reused.
    var
        i: int = 0
    while i < 16:
        result.lo[i] = gf256Mul(uint8(i), c)
        result.hi[i] = gf256Mul(uint8(i) shl 4, c)
        i = i + 1


proc gf256MulAddScalar(dst: var openArray[uint8], A: openArray[uint8],
    t: Gf256Tables, first, last: int) =
    ## dst: accumulator mutated between first and last.
    ## A: source bytes read over the same span.
    ## t: split tables of the coefficient.
    ## first/last: half-open byte range still to process.
    var
        i: int = first
    while i < last:
        dst[i] = dst[i] xor t.lo[int(A[i] and 0x0f'u8)] xor t.hi[int(A[i] shr 4)]
        i = i + 1


proc gf256AddScalar(dst: var openArray[uint8], A: openArray[uint8],
    first, last: int) =
    ## dst: accumulator mutated between first and last.
    ## A: source bytes read over the same span.
    ## first/last: half-open byte range still to process.
    var
        i: int = first
    while i < last:
        dst[i] = dst[i] xor A[i]
        i = i + 1


when defined(simdNexusEnableAvx2) or defined(eirEnableAvx2):
    proc gf256MulAddAvx2(dst: var openArray[uint8], A: openArray[uint8],
        t: Gf256Tables, first, last: int): int =
        ## Returns the offset just past the last whole 32-byte lane.
        var
            vlo: M256i = mm256_broadcastsi128_si256(
                mm_loadu_si128(cast[pointer](unsafeAddr t.lo[0])))
            vhi: M256i = mm256_broadcastsi128_si256(
                mm_loadu_si128(cast[pointer](unsafeAddr t.hi[0])))
            mask: M256i = mm256_broadcastsi128_si256(
                mm_loadu_si128(cast[pointer](unsafeAddr gf256NibbleMask[0])))
            va: M256i
            vd: M256i
            low: M256i
            high: M256i
            i: int = first
        while i + gf256AvxLaneWidth <= last:
            va = mm256_loadu_si256(cast[pointer](unsafeAddr A[i]))
            vd = mm256_loadu_si256(cast[pointer](unsafeAddr dst[i]))
            low = mm256_shuffle_epi8(vlo, mm256_and_si256(va, mask))
            high = mm256_shuffle_epi8(vhi,
                mm256_and_si256(mm256_srli_epi64(va, 4), mask))
            mm256_storeu_si256(cast[pointer](unsafeAddr dst[i]),
                mm256_xor_si256(vd, mm256_xor_si256(low, high)))
            i = i + gf256AvxLaneWidth
        result = i


    proc gf256AddAvx2(dst: var openArray[uint8], A: openArray[uint8],
        first, last: int): int =
        ## Returns the offset just past the last whole 32-byte lane.
        var
            va: M256i
            vd: M256i
            i: int = first
        while i + gf256AvxLaneWidth <= last:
            va = mm256_loadu_si256(cast[pointer](unsafeAddr A[i]))
            vd = mm256_loadu_si256(cast[pointer](unsafeAddr dst[i]))
            mm256_storeu_si256(cast[pointer](unsafeAddr dst[i]),
                mm256_xor_si256(vd, va))
            i = i + gf256AvxLaneWidth
        result = i


when defined(amd64) or defined(i386):
    proc gf256MulAddSse(dst: var openArray[uint8], A: openArray[uint8],
        t: Gf256Tables, first, last: int): int =
        ## Returns the offset just past the last whole 16-byte lane.
        var
            vlo: M128i = mm_loadu_si128(cast[pointer](unsafeAddr t.lo[0]))
            vhi: M128i = mm_loadu_si128(cast[pointer](unsafeAddr t.hi[0]))
            mask: M128i = mm_loadu_si128(
                cast[pointer](unsafeAddr gf256NibbleMask[0]))
            va: M128i
            vd: M128i
            low: M128i
            high: M128i
            i: int = first
        while i + gf256LaneWidth <= last:
            va = mm_loadu_si128(cast[pointer](unsafeAddr A[i]))
            vd = mm_loadu_si128(cast[pointer](unsafeAddr dst[i]))
            low = mm_shuffle_epi8(vlo, mm_and_si128(va, mask))
            high = mm_shuffle_epi8(vhi,
                mm_and_si128(mm_srli_epi64(va, 4), mask))
            mm_storeu_si128(cast[pointer](unsafeAddr dst[i]),
                mm_xor_si128(vd, mm_xor_si128(low, high)))
            i = i + gf256LaneWidth
        result = i


    proc gf256AddSse(dst: var openArray[uint8], A: openArray[uint8],
        first, last: int): int =
        ## Returns the offset just past the last whole 16-byte lane.
        var
            va: M128i
            vd: M128i
            i: int = first
        while i + gf256LaneWidth <= last:
            va = mm_loadu_si128(cast[pointer](unsafeAddr A[i]))
            vd = mm_loadu_si128(cast[pointer](unsafeAddr dst[i]))
            mm_storeu_si128(cast[pointer](unsafeAddr dst[i]),
                mm_xor_si128(vd, va))
            i = i + gf256LaneWidth
        result = i


when defined(arm64) or defined(aarch64):
    proc gf256MulAddNeon(dst: var openArray[uint8], A: openArray[uint8],
        t: Gf256Tables, first, last: int): int =
        ## Returns the offset just past the last whole 16-byte lane.
        var
            vlo: uint8x16 = vld1q_u8(cast[pointer](unsafeAddr t.lo[0]))
            vhi: uint8x16 = vld1q_u8(cast[pointer](unsafeAddr t.hi[0]))
            mask: uint8x16 = vld1q_u8(
                cast[pointer](unsafeAddr gf256NibbleMask[0]))
            va: uint8x16
            vd: uint8x16
            low: uint8x16
            high: uint8x16
            i: int = first
        while i + gf256LaneWidth <= last:
            va = vld1q_u8(cast[pointer](unsafeAddr A[i]))
            vd = vld1q_u8(cast[pointer](unsafeAddr dst[i]))
            low = vqtbl1q_u8(vlo, vandq_u8(va, mask))
            high = vqtbl1q_u8(vhi, vshrq_n_u8(va, 4))
            vst1q_u8(cast[pointer](unsafeAddr dst[i]),
                veorq_u8(vd, veorq_u8(low, high)))
            i = i + gf256LaneWidth
        result = i


    proc gf256AddNeon(dst: var openArray[uint8], A: openArray[uint8],
        first, last: int): int =
        ## Returns the offset just past the last whole 16-byte lane.
        var
            va: uint8x16
            vd: uint8x16
            i: int = first
        while i + gf256LaneWidth <= last:
            va = vld1q_u8(cast[pointer](unsafeAddr A[i]))
            vd = vld1q_u8(cast[pointer](unsafeAddr dst[i]))
            vst1q_u8(cast[pointer](unsafeAddr dst[i]), veorq_u8(vd, va))
            i = i + gf256LaneWidth
        result = i


proc gf256MulAdd*(dst: var openArray[uint8], A: openArray[uint8],
    t: Gf256Tables) =
    ## dst: accumulator mutated in place to dst[i] xor (A[i] * c).
    ## A: source bytes; the shorter of the two buffers bounds the work.
    ## t: split tables built once from the coefficient c.
    var
        n: int = min(dst.len, A.len)
        i: int = 0
    if n <= 0:
        return
    when defined(simdNexusEnableAvx2) or defined(eirEnableAvx2):
        i = gf256MulAddAvx2(dst, A, t, i, n)
    when defined(amd64) or defined(i386):
        i = gf256MulAddSse(dst, A, t, i, n)
    when defined(arm64) or defined(aarch64):
        i = gf256MulAddNeon(dst, A, t, i, n)
    gf256MulAddScalar(dst, A, t, i, n)


proc gf256MulAdd*(dst: var openArray[uint8], A: openArray[uint8], c: uint8) =
    ## dst: accumulator mutated in place to dst[i] xor (A[i] * c).
    ## A: source bytes.
    ## c: coefficient used for a single pass; build tables yourself to reuse one.
    gf256MulAdd(dst, A, gf256Tables(c))


proc gf256AddInto*(dst: var openArray[uint8], A: openArray[uint8]) =
    ## dst: accumulator mutated in place to dst[i] xor A[i].
    ## A: source bytes; the shorter of the two buffers bounds the work.
    var
        n: int = min(dst.len, A.len)
        i: int = 0
    if n <= 0:
        return
    when defined(simdNexusEnableAvx2) or defined(eirEnableAvx2):
        i = gf256AddAvx2(dst, A, i, n)
    when defined(amd64) or defined(i386):
        i = gf256AddSse(dst, A, i, n)
    when defined(arm64) or defined(aarch64):
        i = gf256AddNeon(dst, A, i, n)
    gf256AddScalar(dst, A, i, n)
