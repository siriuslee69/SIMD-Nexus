#============================================#
# sequences/byte_streams.nim                 #
# <- Packed SIMD byte streams (SSE2/AVX2).  #
#============================================#

import
    nimsimd/sse2

when defined(neon) or defined(arm64) or defined(aarch64):
    import nimsimd/neon

when defined(simdNexusEnableAvx2) or defined(eirEnableAvx2):
    import ../isa/x86_avx2
    import nimsimd/avx2


const
    sseByteLaneWidth* = 16
    avxByteLaneWidth* = 32


type
    ## SseByteStream: byte stream represented by packed M128i lanes.
    SseByteStream* = object
        ## lanes: 16-byte SIMD lanes.
        lanes*: seq[M128i]
        ## len: logical byte length (tail bytes in last lane may be padding).
        len*: int


proc xorBytes16Into*(dst: var openArray[uint8], dstOffset: int,
    A: openArray[uint8], aOffset: int, B: openArray[uint8], bOffset: int) =
    ## dst: destination byte buffer with at least 16 writable bytes at dstOffset.
    ## dstOffset: destination byte offset.
    ## A: left byte buffer.
    ## aOffset: left byte offset.
    ## B: right byte buffer.
    ## bOffset: right byte offset.
    when defined(amd64) or defined(i386):
        var
            va: M128i = mm_loadu_si128(cast[pointer](unsafeAddr A[aOffset]))
            vb: M128i = mm_loadu_si128(cast[pointer](unsafeAddr B[bOffset]))
        mm_storeu_si128(cast[pointer](unsafeAddr dst[dstOffset]), mm_xor_si128(va, vb))
    elif defined(neon) or defined(arm64) or defined(aarch64):
        var
            va: uint8x16 = vld1q_u8(cast[pointer](unsafeAddr A[aOffset]))
            vb: uint8x16 = vld1q_u8(cast[pointer](unsafeAddr B[bOffset]))
        vst1q_u8(cast[pointer](unsafeAddr dst[dstOffset]), veorq_u8(va, vb))
    else:
        var
            i: int = 0
        while i < sseByteLaneWidth:
            dst[dstOffset + i] = A[aOffset + i] xor B[bOffset + i]
            i = i + 1


proc xorBytes16InPlace*(dst: var openArray[uint8], dstOffset: int,
    A: openArray[uint8], aOffset: int) =
    ## dst: destination byte buffer mutated in place.
    ## dstOffset: destination byte offset.
    ## A: source byte buffer.
    ## aOffset: source byte offset.
    when defined(amd64) or defined(i386):
        var
            vd: M128i = mm_loadu_si128(cast[pointer](unsafeAddr dst[dstOffset]))
            va: M128i = mm_loadu_si128(cast[pointer](unsafeAddr A[aOffset]))
        mm_storeu_si128(cast[pointer](unsafeAddr dst[dstOffset]), mm_xor_si128(vd, va))
    elif defined(neon) or defined(arm64) or defined(aarch64):
        var
            vd: uint8x16 = vld1q_u8(cast[pointer](unsafeAddr dst[dstOffset]))
            va: uint8x16 = vld1q_u8(cast[pointer](unsafeAddr A[aOffset]))
        vst1q_u8(cast[pointer](unsafeAddr dst[dstOffset]), veorq_u8(vd, va))
    else:
        var
            i: int = 0
        while i < sseByteLaneWidth:
            dst[dstOffset + i] = dst[dstOffset + i] xor A[aOffset + i]
            i = i + 1


proc sadBytes16*(A: openArray[uint8], aOffset: int,
    B: openArray[uint8], bOffset: int): int =
    ## A/B: byte buffers with at least 16 readable bytes at their offsets.
    ## Returns the unsigned sum of absolute byte differences.
    when defined(amd64) or defined(i386):
        var
            va: M128i = mm_loadu_si128(cast[pointer](unsafeAddr A[aOffset]))
            vb: M128i = mm_loadu_si128(cast[pointer](unsafeAddr B[bOffset]))
            sums: array[2, uint64]
        mm_storeu_si128(cast[pointer](unsafeAddr sums[0]), mm_sad_epu8(va, vb))
        result = int(sums[0] + sums[1])
    else:
        var i: int = 0
        while i < sseByteLaneWidth:
            result += abs(int(A[aOffset + i]) - int(B[bOffset + i]))
            i += 1


proc toSseByteStream*(bs: openArray[uint8]): SseByteStream =
    ## bs: scalar byte stream.
    ## Returns packed SSE2 stream with zero-padded final lane.
    var
        laneCount: int = 0
        laneIndex: int = 0
        laneBase: int = 0
        j: int = 0
        t: array[sseByteLaneWidth, uint8]
    if bs.len <= 0:
        return SseByteStream(lanes: @[], len: 0)
    laneCount = (bs.len + sseByteLaneWidth - 1) div sseByteLaneWidth
    result.lanes = newSeq[M128i](laneCount)
    result.len = bs.len
    laneIndex = 0
    while laneIndex < laneCount:
        laneBase = laneIndex * sseByteLaneWidth
        j = 0
        while j < sseByteLaneWidth:
            t[j] = 0'u8
            if laneBase + j < bs.len:
                t[j] = bs[laneBase + j]
            j = j + 1
        result.lanes[laneIndex] = mm_loadu_si128(cast[pointer](unsafeAddr t[0]))
        laneIndex = laneIndex + 1


proc laneToBytes*(lane: M128i): array[sseByteLaneWidth, uint8] =
    ## lane: packed SSE2 lane.
    ## Returns scalar bytes from the SIMD lane.
    mm_storeu_si128(cast[pointer](unsafeAddr result[0]), lane)


proc toByteSeq*(bs: SseByteStream): seq[uint8] =
    ## bs: packed SSE2 stream.
    ## Returns flattened scalar bytes.
    var
        outBs: seq[uint8] = newSeq[uint8](bs.len)
        laneIndex: int = 0
        laneBase: int = 0
        valid: int = 0
        j: int = 0
        laneBytes: array[sseByteLaneWidth, uint8]
    while laneIndex < bs.lanes.len:
        laneBase = laneIndex * sseByteLaneWidth
        valid = sseByteLaneWidth
        if laneBase + valid > bs.len:
            valid = bs.len - laneBase
        laneBytes = laneToBytes(bs.lanes[laneIndex])
        j = 0
        while j < valid:
            outBs[laneBase + j] = laneBytes[j]
            j = j + 1
        laneIndex = laneIndex + 1
    result = outBs


proc xorByteStreams*(a: SseByteStream, b: SseByteStream): SseByteStream =
    ## a: first packed stream.
    ## b: second packed stream.
    ## Returns packed XOR stream with length truncated to min(a.len, b.len).
    var
        n: int = min(a.len, b.len)
        laneCount: int = 0
        i: int = 0
    result.len = n
    if n <= 0:
        result.lanes = @[]
        return
    laneCount = (n + sseByteLaneWidth - 1) div sseByteLaneWidth
    result.lanes = newSeq[M128i](laneCount)
    while i < laneCount:
        result.lanes[i] = mm_xor_si128(a.lanes[i], b.lanes[i])
        i = i + 1


when defined(simdNexusEnableAvx2) or defined(eirEnableAvx2):
    type
        ## AvxByteStream: byte stream represented by packed M256i lanes.
        AvxByteStream* = object
            ## lanes: 32-byte SIMD lanes.
            lanes*: seq[M256i]
            ## len: logical byte length (tail bytes in last lane may be padding).
            len*: int


    proc xorBytes32InPlace*(dst: var openArray[uint8], dstOffset: int,
        A: openArray[uint8], aOffset: int) =
        ## dst/A: buffers with at least 32 readable/writable bytes at offsets.
        var
            vd: M256i = mm256_loadu_si256(cast[pointer](unsafeAddr dst[dstOffset]))
            va: M256i = mm256_loadu_si256(cast[pointer](unsafeAddr A[aOffset]))
        mm256_storeu_si256(cast[pointer](unsafeAddr dst[dstOffset]), mm256_xor_si256(vd, va))


    proc toAvxByteStream*(bs: openArray[uint8]): AvxByteStream =
        ## bs: scalar byte stream.
        ## Returns packed AVX2 stream with zero-padded final lane.
        var
            laneCount: int = 0
            laneIndex: int = 0
            laneBase: int = 0
            j: int = 0
            t: array[avxByteLaneWidth, uint8]
        if bs.len <= 0:
            return AvxByteStream(lanes: @[], len: 0)
        laneCount = (bs.len + avxByteLaneWidth - 1) div avxByteLaneWidth
        result.lanes = newSeq[M256i](laneCount)
        result.len = bs.len
        laneIndex = 0
        while laneIndex < laneCount:
            laneBase = laneIndex * avxByteLaneWidth
            j = 0
            while j < avxByteLaneWidth:
                t[j] = 0'u8
                if laneBase + j < bs.len:
                    t[j] = bs[laneBase + j]
                j = j + 1
            result.lanes[laneIndex] = mm256_loadu_si256(cast[pointer](unsafeAddr t[0]))
            laneIndex = laneIndex + 1


    proc laneToBytes*(lane: M256i): array[avxByteLaneWidth, uint8] =
        ## lane: packed AVX2 lane.
        ## Returns scalar bytes from the SIMD lane.
        mm256_storeu_si256(cast[pointer](unsafeAddr result[0]), lane)


    proc toByteSeq*(bs: AvxByteStream): seq[uint8] =
        ## bs: packed AVX2 stream.
        ## Returns flattened scalar bytes.
        var
            outBs: seq[uint8] = newSeq[uint8](bs.len)
            laneIndex: int = 0
            laneBase: int = 0
            valid: int = 0
            j: int = 0
            laneBytes: array[avxByteLaneWidth, uint8]
        while laneIndex < bs.lanes.len:
            laneBase = laneIndex * avxByteLaneWidth
            valid = avxByteLaneWidth
            if laneBase + valid > bs.len:
                valid = bs.len - laneBase
            laneBytes = laneToBytes(bs.lanes[laneIndex])
            j = 0
            while j < valid:
                outBs[laneBase + j] = laneBytes[j]
                j = j + 1
            laneIndex = laneIndex + 1
        result = outBs


    proc xorByteStreams*(a: AvxByteStream, b: AvxByteStream): AvxByteStream =
        ## a: first packed stream.
        ## b: second packed stream.
        ## Returns packed XOR stream with length truncated to min(a.len, b.len).
        var
            n: int = min(a.len, b.len)
            laneCount: int = 0
            i: int = 0
        result.len = n
        if n <= 0:
            result.lanes = @[]
            return
        laneCount = (n + avxByteLaneWidth - 1) div avxByteLaneWidth
        result.lanes = newSeq[M256i](laneCount)
        while i < laneCount:
            result.lanes[i] = mm256_xor_si256(a.lanes[i], b.lanes[i])
            i = i + 1


    proc sadBytes32*(A: openArray[uint8], aOffset: int,
        B: openArray[uint8], bOffset: int): int =
        ## A/B: byte buffers with at least 32 readable bytes at their offsets.
        ## Returns the unsigned sum of absolute byte differences.
        var
            va: M256i = mm256_loadu_si256(cast[pointer](unsafeAddr A[aOffset]))
            vb: M256i = mm256_loadu_si256(cast[pointer](unsafeAddr B[bOffset]))
            sums: array[4, uint64]
        mm256_storeu_si256(cast[pointer](unsafeAddr sums[0]), mm256_sad_epu8(va, vb))
        result = int(sums[0] + sums[1] + sums[2] + sums[3])
