# simd_nexus

SIMD helper types, conversions, and operations for Nim with AVX2/SSE/NEON backends.

## Structure

- `simd/`: core SIMD types, conversions, and operations (including generic traits/helpers and SIMD iterators with masks).
- `matrices/`: matrix-oriented SIMD helpers.
- `sequences/`: SIMD-aware sequence utilities.
- `tests/`: unit tests for core behavior.

## Quick Usage

```nim
import simd_nexus

let x: i32x4 = [1'u32, 2, 3, 4].asM128i()
let y = x + x
echo y[0]

let v = loadU32x4[M128i]([1'u32, 2'u32, 3'u32, 4'u32])
let r = rotl32(v, 8)
echo storeU32x4[M128i](r)[0]

for (i, mask) in simdRangeU32[M128i](0'u32, 6):
  let idxs = storeU32x4[M128i](i)
  let masks = storeU32x4[M128i](mask)
  echo idxs, " ", masks
```

## Coding Conventions (Short)

- Prefer clarity and modularity over micro-optimizations.
- Keep modules layered (helpers/types at top levels; deeper modules depend upward).
- Avoid nested functions; build helpers and call them from high-level procs.
- Use concise parameter names based on meaning; document each parameter with `##`.
- Declare variables at the top of procs; initialize immediately when possible.
- Add SIMD helpers in shared locations to avoid near-duplicate implementations.
- Keep `.iron/PROGRESS.md` updated with commit message, features, and recent work notes.
- Add nimble tasks for tests/builds and an `autopush` task using `.iron/PROGRESS.md`.
- Exclude `builds/` and `*.exe` in `.gitignore`.

## Coding Conventions (Full)

### Function Structure
- Keep functions short and avoid nesting. Prefer small helpers that are called by high-level procs.

Example:
```nim
proc myFunc1(): void =
  ...

proc myFunc2(): void =
  ...

proc highLevelFunc(): int =
  myFunc1()
  myFunc2()
```

### Function Syntax
- Call functions as `funcX(param1, param2)` or `param1.funcX(param2)`.
- Avoid the `funcX:` block call syntax unless absolutely needed and explain why in a comment above it.

### Naming and Parameter Rules
- Parameter names use the first letter of what they represent.
- Explain parameter meaning directly below the function declaration with `##` doc comments.
- Arrays, sequences, openArrays, and tables end with `s`.
- State objects that will be mutated use `s` (or `s0`, `s1`, `s2`, ...).
- Math-heavy functions use `a,b,c` or `x,y,z` (then `x1`, `x2`, ...).
- Arrays/lists in math functions use uppercase letters like `A,B,C` or `X,Y,Z`.
- `t` is reserved for temporary variables inside functions.
- `i,j,k` are indices; `l,m,n` are lengths.
- Use `while` for complex loops and `for` for simple one-call loops.
- If a function has only one parameter, you may use its first letter unless it collides with index identifiers.

### Result Variables
- It is OK to assign to a temporary variable and set `result` at the end for clarity.

Example:
```nim
proc myProc(a, b: uint8): uint8 =
  var
    veryImportantNumber: uint8
  veryImportantNumber = callSomeOtherFunc(a, b)
  veryImportantNumber = veryImportantNumber + callYetAnotherFunc(a)
  result = veryImportantNumber
```

### Declarations and Formatting
- Declare variables at the start of the proc, not mid-block or inside loops.
- Always indent `var`, `let`, `const`, and `type` when declaring multiple values.
- Use `const` whenever possible; otherwise use `var`, and assign immediately if the value is known.

### Project Layout
- The actual project belongs in `src`. Create it if missing.
- Submodules can live outside `src`.
- Every repo must include a `.iron/` folder next to `src/` for repo-coordination metadata.
- Every module (`.nim` file) must have a description at the top explaining what it does.
- Organize modules by dependency levels (helpers/types at top; deeper modules depend upward).

Example structure:
```
src/utils.nim
src/types.nim
src/level1/module1.nim
src/level1/module2.nim
src/level1/level2/module3.nim
```

### Reuse and Compression
- If you write three similar helpers across modules, move them into `utils` and overload or use generics (`when`/`case`) instead.

### Documentation
- Update the README when making bigger project changes.
- Keep this full conventions section at the bottom of the README.

### Tools and Tests
- Add a `tools` folder when needed (submodule builders or other pre-compile utilities).
- Always include a `tests` folder with unit tests for important functions.
- After changing code or dependencies, run tests and fix errors.

### iron Folder (Repo Coordination)
- Use `Proto-RepoTemplate/.iron/` as the template source.
- The local submodule override file is `.iron/.local.gitmodules.toml` and must be ignored by git.

### Dependencies and External Projects
- If you need an entirely different project as a dependency, ask before starting a new sibling repo.
- Prefer Nim and nimble only. Do not add Python, bash, or PowerShell build tools.

### C Bindings (cNimWrapper)
- The shared cNimWrapper repo can be used to generate bindings for C libraries when needed.

### Shared Utils (Fylgia-Utils)
- Generic helper functions may be added to `Fylgia-Utils` (https://github.com/siriuslee69/fylgia-utils).

### Nimsuggest
- Do not write pre-compile-time import statements that prevent nimsuggest from checking functions.

### .iron/PROGRESS.md
- Track current commit message, features planned/implemented/in progress, and recent changes/problems.

### .nimble Tasks
- Include tasks for tests and builders.
- Include an `autopush` task that reads the commit message from `.iron/PROGRESS.md`.

### Git
- Add `builds/` and `*.exe` to `.gitignore`.

### Repo Examples (App vs Library)
- Libraries do not need a frontend (at most a CLI).
- Avoid frontend/backend splits in library repos.

## Issue Playbook

- Symptom: `nimble test` or `nimble build` fails with Nimble metadata write errors under `%USERPROFILE%\\.nimble` (for example `nimbledata2.json`).
  Workaround: run direct Nim commands such as `nim c -r tests/test_basic.nim` and `nim c src/simd_nexus.nim`, then record the environment issue in `.iron/PROGRESS.md`.
- Symptom: stale `*.exe` files appear in `src/` or `tests/` after local runs.
  Workaround: remove generated binaries before committing; `*.exe` is intentionally ignored and only Nim sources should be tracked.
- Symptom: `nimble autopush` uses a generic commit message.
  Workaround: ensure `.iron/PROGRESS.md` contains a line starting with `Commit Message:` and rerun `nimble autopush`.
