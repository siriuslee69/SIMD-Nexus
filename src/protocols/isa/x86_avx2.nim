#============================================#
# isa/x86_avx2.nim                           #
# <- AVX2 for modules with 256-bit lanes.    #
#============================================#

## Modules whose types or operations are 256 bits wide import this. AVX2
## implies AVX, SSE4.2, SSE4.1, SSSE3, SSE3 and SSE2, so importing it covers
## everything `isa/x86` does as well.
##
## Import it only from modules that genuinely compile `M256i` work. A consumer
## reaching for a 128-bit byte-stream helper should not be handed an AVX2-only
## binary as a side effect of touching the package at all.

const
  isaX86Avx2* = "avx2"
    ## Named so importers reference a symbol and the import is never flagged
    ## as unused.

when defined(amd64) or defined(i386):
  {.passC: "-mavx2".}
