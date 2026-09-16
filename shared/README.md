# Shared Runtime Base

AUTONOMY reuses the same `DroneDreamRuntime` installer core as the four public
DroneDream desktop products. The pinned source lives in the
`shared/dronedream-runtime-source` Git submodule; `runtime-base.contract.json`
binds its commit, selected source files, hashes, release manifest, and active
Ed25519 verification key.

The desktop imports the upstream installer modules directly. This preserves
the reviewed manifest verification, resumable range download, archive hashing,
WSL import, health checking, upgrade rollback, diagnostics, and cross-product
operation lease instead of maintaining an AUTONOMY-only copy.

Initialize a new checkout with:

```powershell
git submodule update --init --recursive
python scripts/verify_shared_runtime_base.py
```

Updating the pin is a reviewed operation: update the gitlink, recompute the
contract hashes, run the source verifier, Rust checks, frontend tests, and a
Windows installer build. Never advance the submodule implicitly during a
normal application build.
