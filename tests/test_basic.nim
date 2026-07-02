#============================================#
# tests/test_basic.nim                       #
# <- Basic SIMD expansion validation tests.  #
#============================================#

import std/unittest
import nimsimd/sse2
import ../src/protocols/simd/base_operations
import ../src/protocols/simd/converters
import ../src/protocols/simd/generic_u32
import ../src/protocols/simd/generic_i8
import ../src/protocols/simd/generic_i16
import ../src/protocols/simd/generic_u64
import ../src/protocols/simd/generic_f32
import ../src/protocols/simd/generic_f64
import ../src/protocols/simd/iterators
import ../src/protocols/sequences/custom_operations
import ../src/protocols/gpu/dispatch

suite "simd_nexus basic":
  test "i32x4 add/extract":
    let x: i32x4 = [15'u32, 12'u32, 56'u32, 84'u32].asM128i()
    let sum = x + x
    let extracted = [sum[0], sum[1], sum[2], sum[3]]
    check extracted == [30'i32, 24, 112, 168]

  test "i8x16 add/extract":
    let x: i8x16 = [1'u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].asM128i()
    let sum = x + x
    check sum[0] == 2'i8
    check sum[15] == 32'i8

  when defined(simdNexusEnableAvx2):
    test "M256i add/extract":
      let x: M256i = [15'u32, 12, 56, 84, 51, 22, 65, 568].asM256i()
      let sum = x + x
      let extracted = [
        sum[0, int32], sum[1, int32], sum[2, int32], sum[3, int32],
        sum[4, int32], sum[5, int32], sum[6, int32], sum[7, int32]
      ]
      check extracted == [30'i32, 24, 112, 168, 102, 44, 130, 1136]

  test "toM128iSeq packs 4-wide vectors":
    let data = [1'u32, 2, 3, 4, 5, 6, 7]
    let vecs = toM128iSeq(data)
    check vecs.len == 2
    check vecs[0][0] == 1'i32
    check vecs[0][3] == 4'i32
    check vecs[1][0] == 5'i32

suite "simd_nexus searches":
  test "find/findAll int8":
    var
      ms: seq[M128i] = @[
        mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        mm_setr_epi8(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)
      ]
      idx: uint64 = 0
      mask: uint64 = 0
    idx = find(ms, 5'i8)
    check idx == 5'u64
    mask = findAll(ms, 19'i8)
    check mask == (1'u64 shl 19)

  test "find/findAll int16":
    var
      ms: seq[M128i] = @[
        mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8),
        mm_setr_epi16(9, 10, 11, 12, 13, 14, 15, 16)
      ]
      idx: uint64 = 0
      mask: uint64 = 0
    idx = find(ms, 14'i16)
    check idx == 13'u64
    mask = findAll(ms, 3'i16)
    check mask == (1'u64 shl 2)

  test "find int32":
    var
      ms: seq[M128i] = @[
        mm_setr_epi32(100, 200, 300, 400),
        mm_setr_epi32(500, 600, 700, 800)
      ]
      idx: uint64 = 0
    idx = find(ms, 700'i32)
    check idx == 6'u64

  test "find/findAll int64":
    var
      ms: seq[M128i] = @[
        mm_set_epi64x(20, 10),
        mm_set_epi64x(40, 30)
      ]
      idx: uint64 = 0
      mask: uint64 = 0
    idx = find(ms, 30'i64)
    check idx == 2'u64
    mask = findAll(ms, 20'i64)
    check mask == (1'u64 shl 1)

  test "find float32":
    var
      ms: seq[M128] = @[
        mm_set_ps(4.0'f32, 3.0'f32, 2.0'f32, 1.0'f32),
        mm_set_ps(8.0'f32, 7.0'f32, 6.0'f32, 5.0'f32)
      ]
      idx: uint32 = 0
    idx = find(ms, 6.0'f32)
    check idx == 5'u32
    idx = find(ms, 9.0'f32)
    check idx == high(uint32)

suite "simd_nexus generics":
  test "lanesU32 reports width":
    check lanesU32[M128i]() == 4
    when defined(simdNexusEnableAvx2):
      check lanesU32[M256i]() == 8

  test "load/store U32x4 roundtrip":
    let
      A = [1'u32, 2'u32, 3'u32, 4'u32]
      v = loadU32x4[M128i](A)
      B = storeU32x4[M128i](v)
    check B == A

  when defined(simdNexusEnableAvx2):
    test "load/store U32x8 roundtrip":
      let
        A = [1'u32, 2'u32, 3'u32, 4'u32, 5'u32, 6'u32, 7'u32, 8'u32]
        v = loadU32x8[M256i](A)
        B = storeU32x8[M256i](v)
      check B == A

  test "rotl32 matches scalar":
    let
      A = [0x11223344'u32, 0xA5A5A5A5'u32, 0x80000001'u32, 0x01020304'u32]
      v = loadU32x4[M128i](A)
      r = rotl32(v, 8)
      B = storeU32x4[M128i](r)
    check B[0] == 0x22334411'u32
    check B[1] == 0xA5A5A5A5'u32
    check B[2] == 0x00000180'u32
    check B[3] == 0x02030401'u32

suite "simd_nexus generics other types":
  test "i8 lanes and roundtrip":
    let
      A = [0'u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
      v = loadI8x16[i8x16](A)
      B = storeI8x16[i8x16](v)
    check lanesI8[i8x16]() == 16
    check B == A
    when defined(simdNexusEnableAvx2):
      let
        C = [0'u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
             16'u8, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        v2 = loadI8x32[i8x32](C)
        D = storeI8x32[i8x32](v2)
      check lanesI8[i8x32]() == 32
      check D == C

  test "i16 lanes and roundtrip":
    let
      A = [1'u16, 2, 3, 4, 5, 6, 7, 8]
      v = loadI16x8[i16x8](A)
      B = storeI16x8[i16x8](v)
    check lanesI16[i16x8]() == 8
    check B == A
    when defined(simdNexusEnableAvx2):
      let
        C = [1'u16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        v2 = loadI16x16[i16x16](C)
        D = storeI16x16[i16x16](v2)
      check lanesI16[i16x16]() == 16
      check D == C

  test "u64 lanes and roundtrip":
    let
      A = [0x0102030405060708'u64, 0x1112131415161718'u64]
      v = loadU64x2[u64x2](A)
      B = storeU64x2[u64x2](v)
    check lanesU64[u64x2]() == 2
    check B == A
    when defined(simdNexusEnableAvx2):
      let
        C = [1'u64, 2'u64, 3'u64, 4'u64]
        v2 = loadU64x4[u64x4](C)
        D = storeU64x4[u64x4](v2)
      check lanesU64[u64x4]() == 4
      check D == C

  test "f32 lanes and roundtrip":
    let
      A = [1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32]
      v = loadF32x4[f32x4](A)
      B = storeF32x4[f32x4](v)
    check lanesF32[f32x4]() == 4
    check B == A
    when defined(simdNexusEnableAvx2):
      let
        C = [1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32, 5.0'f32, 6.0'f32, 7.0'f32, 8.0'f32]
        v2 = loadF32x8[f32x8](C)
        D = storeF32x8[f32x8](v2)
      check lanesF32[f32x8]() == 8
      check D == C

  test "f64 lanes and roundtrip":
    let
      A = [1.0'f64, 2.0'f64]
      v = loadF64x2[f64x2](A)
      B = storeF64x2[f64x2](v)
    check lanesF64[f64x2]() == 2
    check B == A
    when defined(simdNexusEnableAvx2):
      let
        C = [1.0'f64, 2.0'f64, 3.0'f64, 4.0'f64]
        v2 = loadF64x4[f64x4](C)
        D = storeF64x4[f64x4](v2)
      check lanesF64[f64x4]() == 4
      check D == C

suite "simd_nexus iterators":
  test "simdRangeU32 masks":
    var
      counts: int = 0
      idx0s: array[4, uint32]
      mask0s: array[4, uint32]
      idx1s: array[4, uint32]
      mask1s: array[4, uint32]
    for (i, mask) in simdRangeU32[M128i](0'u32, 6):
      if counts == 0:
        idx0s = storeU32x4[M128i](i)
        mask0s = storeU32x4[M128i](mask)
      elif counts == 1:
        idx1s = storeU32x4[M128i](i)
        mask1s = storeU32x4[M128i](mask)
      counts = counts + 1
    check counts == 2
    check idx0s == [0'u32, 1, 2, 3]
    check mask0s == [0xFFFFFFFF'u32, 0xFFFFFFFF'u32, 0xFFFFFFFF'u32, 0xFFFFFFFF'u32]
    check idx1s == [4'u32, 5, 6, 7]
    check mask1s == [0xFFFFFFFF'u32, 0xFFFFFFFF'u32, 0'u32, 0'u32]

  test "simdRangeI16 masks":
    var
      counts: int = 0
      mask1s: array[8, uint16]
    for (i, mask) in simdRangeI16[i16x8](0'i16, 10):
      if counts == 1:
        mask1s = storeI16x8[i16x8](mask)
      counts = counts + 1
    check counts == 2
    check mask1s == [0xFFFF'u16, 0xFFFF'u16, 0'u16, 0'u16, 0'u16, 0'u16, 0'u16, 0'u16]

  when defined(simdNexusEnableAvx2):
    test "simdRangeU64 masks":
      var
        counts: int = 0
        idx1s: array[4, uint64]
      for (i, mask) in simdRangeU64[u64x4](0'u64, 6):
        if counts == 1:
          idx1s = storeU64x4[u64x4](i)
        counts = counts + 1
      check counts == 2
      check idx1s == [4'u64, 5, 6, 7]

  test "simdRangeF32 masks":
    var
      counts: int = 0
      m0: int32 = 0
      m1: int32 = 0
    for (i, mask) in simdRangeF32[f32x4](0.0'f32, 6):
      if counts == 0:
        m0 = mm_movemask_ps(mask)
      elif counts == 1:
        m1 = mm_movemask_ps(mask)
      counts = counts + 1
    check counts == 2
    check m0 == 0xF'i32
    check m1 == 0x3'i32

  test "simdRangeF64 masks":
    var
      counts: int = 0
      m0: int32 = 0
      m1: int32 = 0
    for (i, mask) in simdRangeF64[f64x2](0.0'f64, 3):
      if counts == 0:
        m0 = mm_movemask_pd(mask)
      elif counts == 1:
        m1 = mm_movemask_pd(mask)
      counts = counts + 1
    check counts == 2
    check m0 == 0x3'i32
    check m1 == 0x1'i32

suite "simd_nexus gpu dispatch":
  test "array converters preserve or change element type":
    var
      g: GpuDevice
      A: array[4, int32]
      B: array[3, uint8]
      x: GpuArray[int32]
      y: GpuArray[float32]
      z: GpuArray[uint8]
    g = getGpu()[0]
    A = [1'i32, 2'i32, 3'i32, 4'i32]
    B = [5'u8, 6'u8, 7'u8]
    x = toGpuArray(A, g)
    y = toGpuArrayAs(A, float32, g)
    z = dispatch(B, g)
    check x.toSeq() == @[1'i32, 2'i32, 3'i32, 4'i32]
    check y.toSeq() == @[1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32]
    check z.toSeq() == @[5'u8, 6'u8, 7'u8]

  test "cpu fallback executes native-looking vector ops":
    var
      g: GpuDevice
      x: GpuArray[float32]
      y: GpuArray[float32]
      z: GpuArray[float32]
      r: GpuScalar[float32]
    g = getGpu()[0]
    x = dispatch(@[1.0'f32, 2.0'f32, 3.0'f32], g)
    y = dispatch(@[4.0'f32, 5.0'f32, 6.0'f32], g)
    z = dispatch(x + y, g)
    check z.toSeq() == @[5.0'f32, 7.0'f32, 9.0'f32]
    z = dispatch(scale(x, -2.0'f32), g)
    z = dispatch(relu(z), g)
    check z.toSeq() == @[0.0'f32, 0.0'f32, 0.0'f32]
    r = dispatch(dot(x, y), g)
    check r.value == 32.0'f32
