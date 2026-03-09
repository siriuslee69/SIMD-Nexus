#============================================#
# sequences/byte_streams.nim                 #
# <- Packed SIMD byte streams (SSE2/AVX2).  #
#============================================#

import
    nimsimd/sse2

when defined(simdNexusEnableAvx2) or defined(eirEnableAvx2):
    {.passC: "-mavx2".}
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
