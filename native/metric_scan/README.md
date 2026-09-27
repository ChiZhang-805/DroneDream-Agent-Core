# Exact CPU metric-scan kernel

This optional C++17 CPython extension accelerates **scan preparation only**. It
does not decide where to fly, modify the live map, renew source timestamps, or
weaken the flight-controller safety checks. No GPU, network or runtime compiler
is required. The Python implementation is retained as the supported reference
on platforms without a compiled extension, not as an exception-swallowing retry.

## Contract

`exact-scan-max-hit-first-v1` accepts frozen scalar rays. It preserves the original
sampling step count and arithmetic order, ordered adjacent-key deduplication,
maximum evidence per voxel per frame, and hit-over-free precedence. Python still
performs sensor, bounds, timestamp and frame work-budget validation. Native input
validation also rejects malformed direct calls and limits total sampled work to
2,000,000 points. All evidence is temporary until the existing map transaction
and freshness checks allow a commit. The native function releases the GIL only
after copying all Python inputs into private storage. Concurrent calls share no
mutable map or accumulation state.

The optional `all-pixels-nearest-radial-v1` depth entry point inspects every
source pixel in immutable little-endian float32 data. It preserves row padding,
nearest radial hit per tile, original pixel directions, row-major ties, invalid
pixels as unknown, explicit far-clip semantics and valid-pixel counts. One GIL
release covers the entire reduction instead of repeated NumPy thread handoffs.
No frame, pose, source timestamp or collision rule changes. Old scan-only wheels
use the portable NumPy depth reference; a declared incompatible depth entry point
fails visibly. Native errors are not swallowed by a fallback.

Windows builds explicitly use UTF-8 source decoding, independent of the host
code page. Verify both `CONTRACT` and `DEPTH_CONTRACT` and the loaded binary hash;
optional setuptools compilation can exit successfully without producing a binary.
`test_native_depth_projection.py` compares both implementations on Linux and
Windows, malformed direct calls and concurrent independent calls.

Do **not** enable fast-math, fused multiply-add contraction, approximate voxel
hash equality, early ray termination, or subsampling without a new, independently
validated numerical contract. Hash collisions currently use full three-axis
equality and do not merge different voxels.

## Build and deployment

The extension is declared in the repository's `setup.py`. Building a wheel with a
C++17 compiler and Python development headers includes the native module. Wheels
are Python-ABI/platform specific; a Linux CPython 3.12 wheel is not a Windows or
CPython 3.11 wheel. Optional compilation may produce a reference-only build on
unsupported systems, so a successful wheel build alone is **not** proof that the
accelerator is present. Import and verify the installed artifact before rollout.

During flight startup the depth worker records `metric-scan-kernel.json`:
native protocol plus binary SHA-256, or explicitly `numpy-python`. The diagnostic
bench independently records its backend; the comparison tool rejects mismatched
parent/worker identities. A missing module permits the reference implementation;
an installed incompatible module or a missing internal dependency fails visibly.

Development build outputs belong under
`Q:/DroneDream-Workspace/Build/<explicit-native-build>`, not the application-managed
Runtime installation. Install verification uses a separate target directory and
does not change the user's EXE, WSL service or product model package.

## Verification

- `tests/test_native_metric_scan.py`: randomized exact-value/order comparison,
  float-neighbor voxel boundaries, long rays, malformed inputs, bounded work and
  concurrent call isolation. This suite skips explicitly when no extension exists.
- `tests/test_metric_scan_loader.py`: absent versus incompatible installation and
  nested-dependency failures, also runnable without a native compiler.
- Existing scan, voxel, free-commit, perception and navigation tests verify
  transaction isolation and original source-time rejection.
- `scripts/benchmark_metric_fusion.py`: full synthetic fusion comparison at
  3, 10 and 25 meters, including repeated navigation snapshots and exact per-frame
  evidence digests. This is not a flight performance guarantee or training data.
- Native PX4/Gazebo diagnostics must retain rejected/failed runs and ground
  confirmation; safe holds are not counted as learned motion.

## Research basis

[Voxblox's merged integrator](https://voxblox.readthedocs.io/en/latest/api/classvoxblox_1_1MergedTsdfIntegrator.html)
shows how redundant ray processing can dominate reconstruction work.
[Its fast integrator](https://voxblox.readthedocs.io/en/latest/api/classvoxblox_1_1FastTsdfIntegrator.html)
uses approximation and early stopping, which this occupancy contract does not
adopt. [NVIDIA nvblox](https://github.com/nvidia-isaac/nvblox) provides a GPU-based
reconstruction approach; this narrow CPU optimization does not add a CUDA runtime
requirement. The extension uses the
[official CPython extension interface](https://docs.python.org/3/extending/extending.html).
