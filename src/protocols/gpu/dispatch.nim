## ---------------------------------------------------------------
## GPU Dispatch <- high-level CPU/OpenCL-style numeric operations
## ---------------------------------------------------------------

import std/math

type
  GpuBackendKind* = enum
    gbkCpuFallback,
    gbkOpenCl

  GpuDevice* = object
    id*: int
    ordinal*: int
    backend*: GpuBackendKind
    name*: string
    vendor*: string
    computeUnits*: int
    globalMemBytes*: uint64
    platformHandle*: pointer
    deviceHandle*: pointer

  GpuArray*[T] = object
    device*: GpuDevice
    values*: seq[T]

  GpuOpKind* = enum
    gokAdd,
    gokSub,
    gokMul,
    gokDiv,
    gokScale,
    gokRelu,
    gokSigmoid,
    gokTanh

  GpuOp*[T] = object
    kind*: GpuOpKind
    a*: GpuArray[T]
    b*: GpuArray[T]
    scalar*: T

  GpuReduceKind* = enum
    grkSum,
    grkDot

  GpuReduceOp*[T] = object
    kind*: GpuReduceKind
    a*: GpuArray[T]
    b*: GpuArray[T]

  GpuScalar*[T] = object
    device*: GpuDevice
    value*: T

when defined(simdNexusOpenCL):
  when defined(windows):
    const openClLib = "OpenCL.dll"
  elif defined(macosx):
    const openClLib = "/System/Library/Frameworks/OpenCL.framework/OpenCL"
  else:
    const openClLib = "libOpenCL.so"

  type
    ClInt = cint
    ClUint = cuint
    ClUlong = culonglong
    ClBool = ClUint
    ClPlatformId = pointer
    ClDeviceId = pointer
    ClContext = pointer
    ClCommandQueue = pointer
    ClMem = pointer
    ClProgram = pointer
    ClKernel = pointer
    ClEvent = pointer

  const
    clSuccess = 0.ClInt
    clTrue = 1.ClBool
    clDeviceTypeGpu = 1.ClUlong shl 2
    clDeviceName = 0x102B.ClUint
    clDeviceVendor = 0x102C.ClUint
    clDeviceMaxComputeUnits = 0x1002.ClUint
    clDeviceGlobalMemSize = 0x101F.ClUint
    clMemReadOnly = 1.ClUlong shl 2
    clMemWriteOnly = 1.ClUlong shl 1

  proc clGetPlatformIDs(n: ClUint, ps: ptr ClPlatformId,
      count: ptr ClUint): ClInt {.importc: "clGetPlatformIDs",
      dynlib: openClLib.}
  proc clGetDeviceIDs(p: ClPlatformId, kind: ClUlong, n: ClUint,
      ds: ptr ClDeviceId, count: ptr ClUint): ClInt {.importc: "clGetDeviceIDs",
      dynlib: openClLib.}
  proc clGetDeviceInfo(d: ClDeviceId, name: ClUint, size: csize_t,
      value: pointer, sizeRet: ptr csize_t): ClInt {.importc: "clGetDeviceInfo",
      dynlib: openClLib.}
  proc clCreateContext(props: pointer, n: ClUint, ds: ptr ClDeviceId,
      notify: pointer, user: pointer, err: ptr ClInt): ClContext {.
      importc: "clCreateContext", dynlib: openClLib.}
  proc clCreateCommandQueue(c: ClContext, d: ClDeviceId, props: ClUlong,
      err: ptr ClInt): ClCommandQueue {.importc: "clCreateCommandQueue",
      dynlib: openClLib.}
  proc clCreateBuffer(c: ClContext, flags: ClUlong, size: csize_t,
      host: pointer, err: ptr ClInt): ClMem {.importc: "clCreateBuffer",
      dynlib: openClLib.}
  proc clEnqueueWriteBuffer(q: ClCommandQueue, b: ClMem, blocking: ClBool,
      offset: csize_t, size: csize_t, ptrData: pointer, events: ClUint,
      waitList: pointer, event: ptr ClEvent): ClInt {.
      importc: "clEnqueueWriteBuffer", dynlib: openClLib.}
  proc clCreateProgramWithSource(c: ClContext, count: ClUint,
      strings: ptr cstring, lengths: ptr csize_t, err: ptr ClInt): ClProgram {.
      importc: "clCreateProgramWithSource", dynlib: openClLib.}
  proc clBuildProgram(p: ClProgram, count: ClUint, ds: ptr ClDeviceId,
      opts: cstring, notify: pointer, user: pointer): ClInt {.
      importc: "clBuildProgram", dynlib: openClLib.}
  proc clCreateKernel(p: ClProgram, name: cstring, err: ptr ClInt): ClKernel {.
      importc: "clCreateKernel", dynlib: openClLib.}
  proc clSetKernelArg(k: ClKernel, idx: ClUint, size: csize_t,
      value: pointer): ClInt {.importc: "clSetKernelArg", dynlib: openClLib.}
  proc clEnqueueNDRangeKernel(q: ClCommandQueue, k: ClKernel, dims: ClUint,
      offset: ptr csize_t, global: ptr csize_t, local: ptr csize_t,
      events: ClUint, waitList: pointer, event: ptr ClEvent): ClInt {.
      importc: "clEnqueueNDRangeKernel", dynlib: openClLib.}
  proc clEnqueueReadBuffer(q: ClCommandQueue, b: ClMem, blocking: ClBool,
      offset: csize_t, size: csize_t, ptrData: pointer, events: ClUint,
      waitList: pointer, event: ptr ClEvent): ClInt {.
      importc: "clEnqueueReadBuffer", dynlib: openClLib.}
  proc clFinish(q: ClCommandQueue): ClInt {.importc: "clFinish",
      dynlib: openClLib.}
  proc clReleaseMemObject(m: ClMem): ClInt {.importc: "clReleaseMemObject",
      dynlib: openClLib.}
  proc clReleaseKernel(k: ClKernel): ClInt {.importc: "clReleaseKernel",
      dynlib: openClLib.}
  proc clReleaseProgram(p: ClProgram): ClInt {.importc: "clReleaseProgram",
      dynlib: openClLib.}
  proc clReleaseCommandQueue(q: ClCommandQueue): ClInt {.
      importc: "clReleaseCommandQueue", dynlib: openClLib.}
  proc clReleaseContext(c: ClContext): ClInt {.importc: "clReleaseContext",
      dynlib: openClLib.}

proc cpuFallbackDevice(): GpuDevice =
  ## Returns a deterministic CPU fallback device.
  result.id = 0
  result.ordinal = 0
  result.backend = gbkCpuFallback
  result.name = "cpu-fallback"
  result.vendor = "SIMD-Nexus"

when defined(simdNexusOpenCL):
  proc readOpenClString(d: ClDeviceId, p: ClUint): string =
    ## Reads an OpenCL string device property.
    var
      size: csize_t = 0
      buf: seq[char] = @[]
      code: ClInt = clSuccess
    code = clGetDeviceInfo(d, p, 0, nil, addr size)
    if code != clSuccess or size == 0:
      return ""
    buf.setLen(int(size))
    code = clGetDeviceInfo(d, p, size, addr buf[0], nil)
    if code != clSuccess:
      return ""
    result = $cast[cstring](addr buf[0])

  proc readOpenClUint(d: ClDeviceId, p: ClUint): int =
    ## Reads an OpenCL uint device property.
    var
      v: ClUint = 0
      code: ClInt = clSuccess
    code = clGetDeviceInfo(d, p, csize_t(sizeof(v)), addr v, nil)
    if code == clSuccess:
      result = int(v)

  proc readOpenClUlong(d: ClDeviceId, p: ClUint): uint64 =
    ## Reads an OpenCL ulong device property.
    var
      v: ClUlong = 0
      code: ClInt = clSuccess
    code = clGetDeviceInfo(d, p, csize_t(sizeof(v)), addr v, nil)
    if code == clSuccess:
      result = uint64(v)

  proc enumerateOpenClGpus(): seq[GpuDevice] =
    ## Enumerates OpenCL GPU devices.
    var
      platformCount: ClUint = 0
      deviceCount: ClUint = 0
      platforms: seq[ClPlatformId] = @[]
      devices: seq[ClDeviceId] = @[]
      p: int = 0
      d: int = 0
      outId: int = 0
      code: ClInt = clSuccess
      gpu: GpuDevice
    code = clGetPlatformIDs(0, nil, addr platformCount)
    if code != clSuccess or platformCount == 0:
      return @[]
    platforms.setLen(int(platformCount))
    code = clGetPlatformIDs(platformCount, addr platforms[0], nil)
    if code != clSuccess:
      return @[]
    while p < platforms.len:
      deviceCount = 0
      code = clGetDeviceIDs(platforms[p], clDeviceTypeGpu, 0, nil,
        addr deviceCount)
      if code == clSuccess and deviceCount > 0:
        devices.setLen(int(deviceCount))
        code = clGetDeviceIDs(platforms[p], clDeviceTypeGpu, deviceCount,
          addr devices[0], nil)
        if code == clSuccess:
          d = 0
          while d < devices.len:
            gpu = GpuDevice()
            gpu.id = outId
            gpu.ordinal = outId
            gpu.backend = gbkOpenCl
            gpu.name = readOpenClString(devices[d], clDeviceName)
            gpu.vendor = readOpenClString(devices[d], clDeviceVendor)
            gpu.computeUnits = readOpenClUint(devices[d],
              clDeviceMaxComputeUnits)
            gpu.globalMemBytes = readOpenClUlong(devices[d],
              clDeviceGlobalMemSize)
            gpu.platformHandle = platforms[p]
            gpu.deviceHandle = devices[d]
            result.add(gpu)
            outId.inc
            d.inc
      p.inc

proc getGpu*(): seq[GpuDevice] =
  ## Returns GPU candidates. A CPU fallback is always returned as the last item.
  when defined(simdNexusOpenCL):
    result = enumerateOpenClGpus()
  else:
    result = @[]
  result.add(cpuFallbackDevice())

proc selectGpu*(ordinal: int = 0): GpuDevice =
  ## Selects a GPU by ordinal, falling back to CPU when out of range.
  ## ordinal: preferred device ordinal.
  var
    gs: seq[GpuDevice] = @[]
  gs = getGpu()
  if ordinal >= 0 and ordinal < gs.len:
    return gs[ordinal]
  result = cpuFallbackDevice()

proc toGpuArray*[T](xs: openArray[T], g: GpuDevice): GpuArray[T] =
  ## xs: host array/seq values to copy into a dispatch array.
  ## g: target device.
  var
    i: int = 0
  result.device = g
  result.values.setLen(xs.len)
  while i < xs.len:
    result.values[i] = xs[i]
    i.inc

proc toGpuArrayAs*[T, U](xs: openArray[T], dest: typedesc[U],
    g: GpuDevice): GpuArray[U] =
  ## xs: host array/seq values to convert and copy into a dispatch array.
  ## dest: destination element type.
  ## g: target device.
  var
    i: int = 0
  result.device = g
  result.values.setLen(xs.len)
  while i < xs.len:
    result.values[i] = U(xs[i])
    i.inc

proc gpuArray*[T](xs: openArray[T], g: GpuDevice): GpuArray[T] =
  ## xs: host array/seq values to copy into a dispatch array.
  ## g: target device.
  result = toGpuArray(xs, g)

proc gpuArrayAs*[T, U](xs: openArray[T], dest: typedesc[U],
    g: GpuDevice): GpuArray[U] =
  ## xs: host array/seq values to convert and copy into a dispatch array.
  ## dest: destination element type.
  ## g: target device.
  result = toGpuArrayAs(xs, dest, g)

proc dispatch*[T](xs: openArray[T], g: GpuDevice): GpuArray[T] =
  ## xs: host array/seq values to copy into a dispatch array.
  ## g: target device.
  result = toGpuArray(xs, g)

proc dispatch*[T, U](xs: openArray[T], dest: typedesc[U],
    g: GpuDevice): GpuArray[U] =
  ## xs: host array/seq values to convert and copy into a dispatch array.
  ## dest: destination element type.
  ## g: target device.
  result = toGpuArrayAs(xs, dest, g)

proc dispatch*[T](x: GpuArray[T], g: GpuDevice): GpuArray[T] =
  ## x: existing dispatch array.
  ## g: target device.
  result = x
  result.device = g

proc toSeq*[T](x: GpuArray[T]): seq[T] =
  ## x: dispatch array to read back.
  result = x.values

proc len*[T](x: GpuArray[T]): int =
  ## x: dispatch array.
  result = x.values.len

proc sameLen[T](a, b: GpuArray[T]) =
  ## a: first array.
  ## b: second array.
  if a.values.len != b.values.len:
    raise newException(ValueError, "GPU arrays must have equal length")

proc buildOp[T](k: GpuOpKind, a, b: GpuArray[T], s: T): GpuOp[T] =
  ## k: operation kind.
  ## a: first operand.
  ## b: second operand.
  ## s: scalar operand.
  result.kind = k
  result.a = a
  result.b = b
  result.scalar = s

proc `+`*[T: SomeNumber](a, b: GpuArray[T]): GpuOp[T] =
  ## a: left operand.
  ## b: right operand.
  sameLen(a, b)
  result = buildOp(gokAdd, a, b, T(0))

proc `-`*[T: SomeNumber](a, b: GpuArray[T]): GpuOp[T] =
  ## a: left operand.
  ## b: right operand.
  sameLen(a, b)
  result = buildOp(gokSub, a, b, T(0))

proc `*`*[T: SomeNumber](a, b: GpuArray[T]): GpuOp[T] =
  ## a: left operand.
  ## b: right operand.
  sameLen(a, b)
  result = buildOp(gokMul, a, b, T(0))

proc `/`*[T: SomeNumber](a, b: GpuArray[T]): GpuOp[T] =
  ## a: left operand.
  ## b: right operand.
  sameLen(a, b)
  result = buildOp(gokDiv, a, b, T(0))

proc scale*[T: SomeNumber](a: GpuArray[T], s: T): GpuOp[T] =
  ## a: input array.
  ## s: scalar multiplier.
  result = buildOp(gokScale, a, GpuArray[T](), s)

proc relu*(a: GpuArray[float32]): GpuOp[float32] =
  ## a: input array.
  result = buildOp(gokRelu, a, GpuArray[float32](), 0.0'f32)

proc sigmoid*(a: GpuArray[float32]): GpuOp[float32] =
  ## a: input array.
  result = buildOp(gokSigmoid, a, GpuArray[float32](), 0.0'f32)

proc tanhAct*(a: GpuArray[float32]): GpuOp[float32] =
  ## a: input array.
  result = buildOp(gokTanh, a, GpuArray[float32](), 0.0'f32)

proc cpuDispatchOp[T: SomeNumber](op: GpuOp[T], g: GpuDevice): GpuArray[T] =
  ## op: operation to execute.
  ## g: target device metadata.
  var
    i: int = 0
    x: T
  result.device = g
  result.values.setLen(op.a.values.len)
  while i < op.a.values.len:
    case op.kind
    of gokAdd:
      result.values[i] = op.a.values[i] + op.b.values[i]
    of gokSub:
      result.values[i] = op.a.values[i] - op.b.values[i]
    of gokMul:
      result.values[i] = op.a.values[i] * op.b.values[i]
    of gokDiv:
      result.values[i] = op.a.values[i] / op.b.values[i]
    of gokScale:
      result.values[i] = op.a.values[i] * op.scalar
    of gokRelu:
      if op.a.values[i] > T(0):
        result.values[i] = op.a.values[i]
      else:
        result.values[i] = T(0)
    of gokSigmoid:
      x = op.a.values[i]
      result.values[i] = T(1) / (T(1) + T(exp(-float(x))))
    of gokTanh:
      x = op.a.values[i]
      result.values[i] = T(tanh(float(x)))
    i.inc

when defined(simdNexusOpenCL):
  proc openClSource(k: GpuOpKind): string =
    ## k: operation kind to compile.
    var
      expr: string = ""
    case k
    of gokAdd:
      expr = "a[i] + b[i]"
    of gokSub:
      expr = "a[i] - b[i]"
    of gokMul:
      expr = "a[i] * b[i]"
    of gokDiv:
      expr = "a[i] / b[i]"
    of gokScale:
      expr = "a[i] * scalar"
    of gokRelu:
      expr = "fmax(a[i], 0.0f)"
    of gokSigmoid:
      expr = "1.0f / (1.0f + exp(-a[i]))"
    of gokTanh:
      expr = "tanh(a[i])"
    result = "__kernel void nx_op(__global const float* a, " &
      "__global const float* b, __global float* out, const float scalar, " &
      "const uint n) { uint i = get_global_id(0); if (i < n) out[i] = " &
      expr & "; }"

  proc openClDispatchF32(op: GpuOp[float32], g: GpuDevice): GpuArray[float32] =
    ## op: operation to execute on OpenCL.
    ## g: OpenCL device.
    var
      err: ClInt = clSuccess
      ctx: ClContext
      queue: ClCommandQueue
      bufA: ClMem
      bufB: ClMem
      bufOut: ClMem
      program: ClProgram
      kernel: ClKernel
      source: string
      csource: cstring
      length: csize_t = 0
      n: cuint = 0
      nsize: csize_t = 0
      bytes: csize_t = 0
      scalar: float32 = 0.0'f32
      empty: seq[float32] = @[]
      dev: ClDeviceId
      zeroOffset: csize_t = 0
    if g.backend != gbkOpenCl or g.deviceHandle == nil:
      return cpuDispatchOp(op, g)
    result.device = g
    result.values.setLen(op.a.values.len)
    if op.a.values.len == 0:
      return
    dev = cast[ClDeviceId](g.deviceHandle)
    bytes = csize_t(op.a.values.len * sizeof(float32))
    n = cuint(op.a.values.len)
    nsize = csize_t(op.a.values.len)
    scalar = op.scalar
    source = openClSource(op.kind)
    csource = source.cstring
    length = csize_t(source.len)
    ctx = clCreateContext(nil, 1, addr dev, nil, nil, addr err)
    if err != clSuccess or ctx == nil:
      return cpuDispatchOp(op, g)
    queue = clCreateCommandQueue(ctx, dev, 0, addr err)
    if err != clSuccess or queue == nil:
      discard clReleaseContext(ctx)
      return cpuDispatchOp(op, g)
    bufA = clCreateBuffer(ctx, clMemReadOnly, bytes, nil, addr err)
    bufB = clCreateBuffer(ctx, clMemReadOnly, bytes, nil, addr err)
    bufOut = clCreateBuffer(ctx, clMemWriteOnly, bytes, nil, addr err)
    if err != clSuccess or bufA == nil or bufB == nil or bufOut == nil:
      discard clReleaseCommandQueue(queue)
      discard clReleaseContext(ctx)
      return cpuDispatchOp(op, g)
    discard clEnqueueWriteBuffer(queue, bufA, clTrue, 0, bytes,
      unsafeAddr op.a.values[0], 0, nil, nil)
    if op.b.values.len == op.a.values.len:
      discard clEnqueueWriteBuffer(queue, bufB, clTrue, 0, bytes,
        unsafeAddr op.b.values[0], 0, nil, nil)
    else:
      empty.setLen(op.a.values.len)
      discard clEnqueueWriteBuffer(queue, bufB, clTrue, 0, bytes,
        addr empty[0], 0, nil, nil)
    program = clCreateProgramWithSource(ctx, 1, addr csource, addr length,
      addr err)
    if err == clSuccess:
      err = clBuildProgram(program, 1, addr dev, nil, nil, nil)
    if err == clSuccess:
      kernel = clCreateKernel(program, "nx_op", addr err)
    if err == clSuccess:
      discard clSetKernelArg(kernel, 0, csize_t(sizeof(ClMem)), addr bufA)
      discard clSetKernelArg(kernel, 1, csize_t(sizeof(ClMem)), addr bufB)
      discard clSetKernelArg(kernel, 2, csize_t(sizeof(ClMem)), addr bufOut)
      discard clSetKernelArg(kernel, 3, csize_t(sizeof(float32)), addr scalar)
      discard clSetKernelArg(kernel, 4, csize_t(sizeof(cuint)), addr n)
      err = clEnqueueNDRangeKernel(queue, kernel, 1, addr zeroOffset,
        addr nsize, nil, 0, nil, nil)
    if err == clSuccess:
      discard clFinish(queue)
      err = clEnqueueReadBuffer(queue, bufOut, clTrue, 0, bytes,
        addr result.values[0], 0, nil, nil)
    if kernel != nil:
      discard clReleaseKernel(kernel)
    if program != nil:
      discard clReleaseProgram(program)
    discard clReleaseMemObject(bufA)
    discard clReleaseMemObject(bufB)
    discard clReleaseMemObject(bufOut)
    discard clReleaseCommandQueue(queue)
    discard clReleaseContext(ctx)
    if err != clSuccess:
      result = cpuDispatchOp(op, g)

proc dispatch*[T: SomeNumber](op: GpuOp[T], g: GpuDevice): GpuArray[T] =
  ## op: numeric operation to execute.
  ## g: target device.
  when defined(simdNexusOpenCL):
    when T is float32:
      if g.backend == gbkOpenCl:
        return openClDispatchF32(op, g)
  result = cpuDispatchOp(op, g)

proc dispatch*[T: SomeNumber](op: GpuOp[T]): GpuArray[T] =
  ## op: numeric operation to execute on its source device.
  result = dispatch(op, op.a.device)

proc sum*[T: SomeNumber](a: GpuArray[T]): GpuReduceOp[T] =
  ## a: input array.
  result.kind = grkSum
  result.a = a

proc dot*[T: SomeNumber](a, b: GpuArray[T]): GpuReduceOp[T] =
  ## a: left vector.
  ## b: right vector.
  sameLen(a, b)
  result.kind = grkDot
  result.a = a
  result.b = b

proc dispatch*[T: SomeNumber](op: GpuReduceOp[T],
    g: GpuDevice): GpuScalar[T] =
  ## op: reduction to execute.
  ## g: target device.
  var
    i: int = 0
    total: T = T(0)
  result.device = g
  case op.kind
  of grkSum:
    while i < op.a.values.len:
      total = total + op.a.values[i]
      i.inc
  of grkDot:
    while i < op.a.values.len:
      total = total + op.a.values[i] * op.b.values[i]
      i.inc
  result.value = total
