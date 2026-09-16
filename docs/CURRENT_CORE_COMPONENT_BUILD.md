# Current Core component builds

The desktop product builds this public repository at an exact commit. A clean
checkout by itself is not evidence that pre-existing ignored binaries came from
that checkout. The builder regenerates components; `core_build_receipt.py` binds
the resulting bytes to the selected source. Product staging verifies that receipt
before replacing generated output and verifies copied bytes again afterwards.
This receipt is not a compiler attestation or a flight qualification.
The staging caller freezes the receipt bytes first and supplies
`--expected-receipt-sha256` together with `--verify`. Verification decodes and
hashes the same bounded byte buffer, so a subsequently replaced receipt cannot
silently authorize the caller's earlier copy.

## Inputs

- Current native sensor runtime, validated against `native/gazebo_sensors`.
- A complete ten-expert ONNX package using the current feature contract and
  normalized body-velocity control, not an old candidate-only policy.
- Its independently produced standard-simulation admission. Historical receipts
  cannot be relabeled or transferred to newly trained weights.
- `assembly-receipt.json` and the ten content-bound `training-evidence/*.json`
  files. Production preflight validates each expert's lineage and independent
  spatial holdout groups, then loads the actual ONNX interfaces on CPU.
- A maintainer-reviewed distribution-license JSON matching the exact package.
  It uses `schema_version: dronedream.model-distribution-licenses.v1`, the actual
  `package_sha256`, and one `artifacts` entry per expert. Each entry supplies
  `role`, the actual weight `sha256`, `license_id`, `source`, full `license_text`,
  and `redistribution_approved: true`. This records an actual licensing decision;
  the flag does not itself grant rights. First-party MIT permission does not
  automatically cover pretrained third-party weights.

`stage_local_policy_runtime.py --check-only --verify-onnx` accepts the same
`--package`, `--simulation-admission` and `--distribution-licenses` arguments as
the staging operation, without an output directory. It does not write output,
train models, grant qualification, or initiate flight. Structural-only checks
without `--verify-onnx` are diagnostic and are not the production build gate.

## Windows components

`scripts/build-autonomy-windows.ps1 -StageOnly` takes explicit
`-LocalPolicyPackage`, `-LocalPolicySimulationAdmission`,
`-LocalPolicyDistributionLicenses`, and `-NativeSensorRuntime`. It builds the
Core sidecar, plugin isolator, official plugins, current Runtime and default
assets without redundantly building the standalone Core frontend installer.
Model/native preflight runs before generated component directories are reset.

The generated `artifacts/desktop/core-components-build.json` binds the two
sidecars and all Runtime, plugin, and default-asset manifest members. Model
training evidence and license records are retained inside the Runtime and thus
included in its manifest. Only generated files inside this checkout are used.

The product repository owns the five-edition packaging, formal signing and
GitHub update-channel publication. A component build or a passing unit test
does not imply a signed installer or successful autonomous flight.
