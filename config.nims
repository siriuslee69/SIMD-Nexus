switch("path", "src")
# begin Nimble config (version 2)
when withDir(thisDir(), system.fileExists("nimble.paths")):
  include "nimble.paths"
# end Nimble config

# Monorepo fallback: workspace root may set `--noNimblePath`, so add local
# nimsimd install path if available.
when not declared(simdNexusNoNimbleFallback):
  import std/[os, strutils]
  var
    p: string = ""
    line: string = ""
    eirNimblePaths: string = joinPath("..", "Eir-CompressionAndECC", "nimble.paths")
    fallback: string = joinPath(getHomeDir(), ".nimble", "pkgs2", "nimsimd-1.3.2-5202ce48d46eaf593da54e884774cdb2a884e717")
  if fileExists(eirNimblePaths):
    for rawLine in readFile(eirNimblePaths).splitLines:
      line = rawLine.strip()
      if line.startsWith("--path:") and line.contains("nimsimd-"):
        p = line["--path:".len .. ^1].strip(chars = {'"', '\''})
        if dirExists(p):
          switch("path", p)
  if dirExists(fallback):
    switch("path", fallback)
