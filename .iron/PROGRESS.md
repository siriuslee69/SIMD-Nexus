Commit Message: fix simd nexus autopush path and repo hygiene

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
- Removed tracked Windows build artifacts from `src/` and `tests/`.
- Updated `autopush` to resolve `.iron/PROGRESS.md` (with compatibility fallbacks).
- Added `CONTRIBUTING.md` and README issue playbook coverage.

Features In Progress:
- (none)

Last Big Change/Problem:
- First-pass audit found stale tracked binaries, autopush metadata-path mismatch, and missing maintainer docs.

Fix Attempt/Outcome:
- Cleaned binary artifacts, fixed `autopush` progress resolution, added contributor guidance, and documented known issue workarounds.

### 2026-04-04 first-pass audit
Readiness: Not yet production ready-the SIMD helpers build but the repo still lacks documentation and automation hygiene required for a reliable handoff.
Findings:
- [High] `src/simd_nexus.exe` and `tests/test_basic.exe` remain tracked even though `*.exe` is supposed to be ignored, so clones ship stale Windows binaries that Linux builders cannot reproduce or verify.
- [Medium] `simd_nexus.nimble`'s `autopush` task reads `iron/progress.md` while this repo stores `.iron/PROGRESS.md`, so the recorded commit message never reaches `git commit` and the task fails on fresh clones.
- [Low] There is no `CONTRIBUTING.md` and the README lacks an issue-playbook appendix, leaving maintainers without documented repo boundaries, review checklists, commands, or known mitigation steps.
Next fixes:
- [x] Remove `src/simd_nexus.exe` and `tests/test_basic.exe` from version control and confirm `.gitignore` excludes generated binaries so every clone starts from the Nim sources.
- [x] Update `simd_nexus.nimble` to read `.iron/PROGRESS.md` (with a lowercase fallback) so `autopush` uses the recorded message instead of the default text.
- [x] Add `CONTRIBUTING.md` and append an issue-playbook section to `README.md` describing repo intent, review requirements, available commands/tests, and outstanding limitations.

### 2026-04-04 implementation pass
Readiness: Improved, but not fully production-ready until the remaining roadmap items are completed.
Implemented:
- Removed `src/simd_nexus.exe` and `tests/test_basic.exe` from the worktree and reinforced ignore coverage for generated executables.
- Fixed `simd_nexus.nimble` `autopush` to look up `.iron/PROGRESS.md` first, then compatibility fallbacks.
- Added `CONTRIBUTING.md` with scope, review checklist, commands, and known caveats.
- Added an issue playbook section to `README.md` and aligned progress/autopush references to `.iron/PROGRESS.md`.
Verification:
- `nimble test` fails in this sandbox due Nimble metadata write failure at `C:\Users\n1ght\.nimble\nimbledata2.json` and fallback cache paths.
- `nimble build` fails in this sandbox (`Nothing to build` plus the same Nimble metadata write failure).
- `nim c -r --nimcache:._tmp_nimcache_test tests/test_basic.nim` passed.
- `nim c --nimcache:._tmp_nimcache_build src/simd_nexus.nim` passed.
Remaining blocker:
- Environment-level Nimble write permissions under `%USERPROFILE%\.nimble` are outside this repo and prevent clean `nimble` task execution in this sandbox.
