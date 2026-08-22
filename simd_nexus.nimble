import std/[os, strutils]

version       = "0.1.0"
author        = "siriuslee69"
description   = "SIMD helper types and operations."
license       = "UNLICENSED"
srcDir        = "src"
requires "nim >= 1.6.0"
requires "nimsimd >= 1.3.2"


proc consumerCheck(name, body: string) =
  ## Compile one consumer OUTSIDE the repo, so this repo's nim.cfg cannot
  ## supply the instruction-set flags. Every module must declare the flags its
  ## own intrinsics need, or importing simd_nexus as a library fails in gcc
  ## with a target-mismatch error the repo's own build never sees.
  let
    dir = getTempDir() / "simd_nexus_consumer"
    src = dir / (name & ".nim")
  mkDir(dir)
  writeFile(src, body)
  exec "nim c --hints:off --path:" & quoteShell(thisDir() / "src") &
    " -o:" & quoteShell(dir / name) & " " & quoteShell(src)

task testConsumer, "Compile outside-the-repo consumers of the public modules":
  consumerCheck("use_gf256", """
import simd_nexus/sequences/gf256
var
  parity = newSeq[uint8](64)
  data = newSeq[uint8](64)
gf256MulAdd(parity, data, gf256Tables(0x1b'u8))
gf256AddInto(parity, data)
doAssert gf256Mul(gf256Inv(7'u8), 7'u8) == 1'u8
""")
  consumerCheck("use_streams", """
import simd_nexus/sequences/byte_streams
var data = newSeq[uint8](64)
doAssert toByteSeq(toSseByteStream(data)).len == 64
""")
  consumerCheck("use_all", """
import simd_nexus
var
  data = newSeq[uint8](64)
  parity = newSeq[uint8](64)
  x: i32x4 = [1'u32, 2, 3, 4].asM128i()
  w: M256i = [1'u32, 2, 3, 4, 5, 6, 7, 8].asM256i()
gf256MulAdd(parity, data, gf256Tables(3'u8))
doAssert (x + x)[0] == 2'i32
doAssert (w + w)[7, int32] == 16'i32
doAssert toByteSeq(toSseByteStream(data)).len == 64
doAssert getGpu().len > 0
""")

task test, "Run unit tests":
  exec "nim c -r tests/test_basic.nim"
  exec "nim c -r tests/test_gf256.nim"
  exec "nim c -r -d:simdNexusEnableAvx2 --out:build/test_gf256_avx2 tests/test_gf256.nim"
  testConsumerTask()

task buildLib, "Build the simd_nexus module":
  exec "nim c src/simd_nexus.nim"

task autopush, "Add, commit, and push with message from .iron/PROGRESS.md":
  let progressCandidates = @[".iron/PROGRESS.md", ".iron/progress.md", "progress.md"]
  var path = ""
  for candidate in progressCandidates:
    if fileExists(candidate):
      path = candidate
      break
  var msg = ""
  if path.len > 0 and fileExists(path):
    let content = readFile(path)
    for line in content.splitLines:
      if line.startsWith("Commit Message:"):
        msg = line["Commit Message:".len .. ^1].strip()
        break
  if msg.len == 0:
    msg = "No specific commit message given."
  exec "git add -A ."
  exec "git commit -m \" " & msg & "\""
  exec "git push"

task find, "Use local clones for submodules in parent folder":
  let modulesPath = ".gitmodules"
  if not fileExists(modulesPath):
    echo "No .gitmodules found."
  else:
    let root = parentDir(getCurrentDir())
    var current = ""
    for line in readFile(modulesPath).splitLines:
      let s = line.strip()
      if s.startsWith("[submodule"):
        let start = s.find('"')
        let stop = s.rfind('"')
        if start >= 0 and stop > start:
          current = s[start + 1 .. stop - 1]
      elif current.len > 0 and s.startsWith("path"):
        let parts = s.split("=", maxsplit = 1)
        if parts.len == 2:
          let subPath = parts[1].strip()
          let tail = splitPath(subPath).tail
          let localDir = joinPath(root, tail)
          if dirExists(localDir):
            let localUrl = localDir.replace('\\', '/')
            exec "git config -f .gitmodules submodule." & current & ".url " & localUrl
            exec "git config submodule." & current & ".url " & localUrl
    exec "git submodule sync --recursive"



