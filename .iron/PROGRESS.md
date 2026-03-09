Commit Message: add simd iterators with vector masks

Features Planned:
- AVX2 compare/mask intrinsics for wider search helpers.
- Expanded matrix operations and dedicated tests.

Features Implemented:
- M128i/M256i conversion helpers in simd and matrices modules.
- Base SIMD arithmetic/logic templates for M128i/M256i.
- Sequence search helpers with SIMD compare/movemask paths.
- Expanded unit tests for search helpers.
- Generic u32 SIMD traits/helpers (lanes, set1, load/store, rotate, xor/add helpers).
- Generic i8/i16/u64 and f32/f64 SIMD helpers (lanes, set1, load/store, rotate, xor/add helpers).
- Added missing shifts/bitwise ops for i8/i16/u64 and f64 operators for SSE/AVX.
- SIMD iterators with vector masks for u32/i8/i16/u64/f32/f64.
- Added iron template folder and full README conventions.

Features In Progress:
- (none)

Last Big Change/Problem:
- Needed SIMD iterators with mask tails for generic loops.

Fix Attempt/Outcome:
- Added iterator module and tests to validate indices and masks.
