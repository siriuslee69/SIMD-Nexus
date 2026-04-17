# Contributing to SIMD-Nexus

## Purpose
This repository provides reusable SIMD helpers for Nim. Changes here should improve shared vector operations, conversions, sequence helpers, and matrix helpers without adding application-specific behavior.

## What belongs here
- Generic SIMD helper code in `src/protocols/simd/`.
- SIMD-backed sequence helpers in `src/protocols/sequences/`.
- SIMD-backed matrix helpers in `src/protocols/matrices/`.
- Public exports and package wiring in `src/simd_nexus.nim`.
- Unit tests for behavior and regressions in `tests/`.

## What does not belong here
- Frontend/UI code.
- Service-specific business logic from application repos.
- Generated binaries (`*.exe`) or local machine state.
- Hard-coded local paths in tracked files.

## Files and modules to read first
- `src/simd_nexus.nim`
- `src/protocols/simd/base_operations.nim`
- `src/protocols/simd/converters.nim`
- `src/protocols/simd/iterators.nim`
- `src/protocols/sequences/custom_operations.nim`
- `src/protocols/matrices/operations.nim`
- `tests/test_basic.nim`

## Review checklist
1. Keep behavior cross-platform and backend-safe (SSE/AVX/NEON paths must compile or gate correctly).
2. Prefer shared generic helpers over duplicated lane-width-specific code.
3. Add or update tests for changed public behavior.
4. Keep generated binaries out of source control.
5. Update `README.md` and `.iron/PROGRESS.md` when behavior or workflow changes.

## Commands
- `nimble test`
- `nimble build`
- `nim c -r tests/test_basic.nim`

## Progress and autopush
- Record the current commit message and work log in `.iron/PROGRESS.md`.
- `nimble autopush` reads commit text from `.iron/PROGRESS.md` (with compatibility fallbacks).

## Known environment caveat
Some environments fail to write Nimble metadata under `%USERPROFILE%\\.nimble` (for example `nimbledata2.json`). If that happens, run direct `nim c` commands to validate code paths while preserving the failure note in `.iron/PROGRESS.md`.
