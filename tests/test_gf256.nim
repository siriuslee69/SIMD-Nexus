#============================================#
# tests/test_gf256.nim                       #
# <- GF(256) table multiply validation.      #
#============================================#

import std/unittest
import ../src/protocols/sequences/gf256

proc referenceMulAdd(dst: var seq[uint8], A: seq[uint8], c: uint8) =
    ## dst/A/c: scalar reference the SIMD lanes must agree with byte for byte.
    var
        i: int = 0
    while i < A.len:
        dst[i] = dst[i] xor gf256Mul(A[i], c)
        i = i + 1

proc rampBytes(n: int, seed: uint8): seq[uint8] =
    ## n/seed: buffer length and starting byte for a deterministic ramp.
    var
        i: int = 0
        t: uint8 = seed
    result = newSeq[uint8](n)
    while i < n:
        result[i] = t
        t = t + 37'u8
        i = i + 1

suite "gf256 field":
    test "multiplication identities":
        var
            i: int = 0
        while i < 256:
            check gf256Mul(uint8(i), 0'u8) == 0'u8
            check gf256Mul(uint8(i), 1'u8) == uint8(i)
            i = i + 1

    test "multiplication commutes":
        var
            a: int = 0
            b: int = 0
        while a < 256:
            b = 0
            while b < 256:
                check gf256Mul(uint8(a), uint8(b)) == gf256Mul(uint8(b), uint8(a))
                b = b + 17
            a = a + 13

    test "every non-zero element has an inverse":
        var
            i: int = 1
        while i < 256:
            check gf256Mul(uint8(i), gf256Inv(uint8(i))) == 1'u8
            i = i + 1
        check gf256Inv(0'u8) == 0'u8

    test "tables reproduce the scalar product":
        var
            t: Gf256Tables
            c: int = 0
            b: int = 0
        while c < 256:
            t = gf256Tables(uint8(c))
            b = 0
            while b < 256:
                check (t.lo[b and 0x0f] xor t.hi[b shr 4]) ==
                    gf256Mul(uint8(b), uint8(c))
                b = b + 1
            c = c + 29

suite "gf256 buffers":
    test "mul-add matches the scalar reference at every length":
        var
            n: int = 0
            src: seq[uint8] = @[]
            got: seq[uint8] = @[]
            want: seq[uint8] = @[]
        while n <= 200:
            src = rampBytes(n, 3'u8)
            got = rampBytes(n, 91'u8)
            want = got
            referenceMulAdd(want, src, 0xb7'u8)
            gf256MulAdd(got, src, 0xb7'u8)
            check got == want
            n = n + 1

    test "mul-add by one is a plain xor":
        var
            src: seq[uint8] = rampBytes(133, 11'u8)
            got: seq[uint8] = rampBytes(133, 200'u8)
            want: seq[uint8] = got
            i: int = 0
        while i < src.len:
            want[i] = want[i] xor src[i]
            i = i + 1
        gf256MulAdd(got, src, 1'u8)
        check got == want

    test "mul-add by zero leaves the accumulator alone":
        var
            src: seq[uint8] = rampBytes(77, 5'u8)
            got: seq[uint8] = rampBytes(77, 17'u8)
            want: seq[uint8] = got
        gf256MulAdd(got, src, 0'u8)
        check got == want

    test "add-into matches a scalar xor at every length":
        var
            n: int = 0
            src: seq[uint8] = @[]
            got: seq[uint8] = @[]
            want: seq[uint8] = @[]
            i: int = 0
        while n <= 100:
            src = rampBytes(n, 61'u8)
            got = rampBytes(n, 7'u8)
            want = got
            i = 0
            while i < n:
                want[i] = want[i] xor src[i]
                i = i + 1
            gf256AddInto(got, src)
            check got == want
            n = n + 1

    test "applying a coefficient twice cancels through the inverse":
        var
            src: seq[uint8] = rampBytes(160, 23'u8)
            acc: seq[uint8] = newSeq[uint8](160)
            back: seq[uint8] = newSeq[uint8](160)
        gf256MulAdd(acc, src, 0x8d'u8)
        gf256MulAdd(back, acc, gf256Inv(0x8d'u8))
        check back == src
