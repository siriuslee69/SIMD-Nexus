#============================================#
# isa/x86.nim                                #
# <- Baseline x86 instruction sets.          #
#============================================#

## Modules that compile SSSE3 or SSE4.1 intrinsics import this so the compiler
## flags travel with the import. Putting them in a repo-local nim.cfg only
## works while SIMD-Nexus is the project being built; anyone importing it as a
## library got a target-mismatch error from gcc instead.
##
## `passC` is global, so importing this once from a module is enough for that
## module's C output. Import only what you use and you get only the flags that
## code needs -- there is no per-function `when` facade to maintain, because
## Nim already drops procs nothing references.

const
  isaX86Baseline* = "ssse3 sse4.1"
    ## Named so importers reference a symbol and the import is never flagged
    ## as unused.

when defined(amd64) or defined(i386):
  {.passC: "-mssse3 -msse4.1".}
