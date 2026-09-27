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
- Current native camera-clock runtime, validated against `native/camera_clock`.
  `scripts/stage_native_camera_clock.py --source <built-directory> --check-only`
  verifies source and binary identities without changing deployment. The
  `camera-clock/camera-clock-runtime.json` receipt and
  `camera-clock/libdronedream-camera-clock.so` library are mandatory Runtime
  manifest members. The relay binds pixels to exact native simulation ticks;
  it does not supply pose truth or grant flight qualification.
- Current detached-parcel placement runtime, source-bound to `native/payload_placement`.
  Build it inside the supported Gazebo Harmonic Linux environment with
  `scripts/build_payload_placement_runtime.py --output <new-build-directory>`.
  On Windows, `--stage-from <built-directory> --check-only` validates the
  receipt and current source without compiling or moving simulation objects.
  The library and receipt are both mandatory Runtime manifest members.
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
`-LocalPolicyDistributionLicenses`, `-NativeSensorRuntime`, and
`-PayloadPlacementRuntime`, and `-CameraClockRuntime`. It builds the
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

Learned yaw composition is explicit per role. Historical precision packages
retain `precision_heading_context_sha256`; a composed cruise actor additionally
declares `navigation_heading_context_sha256`. Neither declaration enables the
input for recovery or risk experts. Every composed actor retains frozen source
graphs, role-matched training provenance, split isolation and an independent
four-axis evaluation requirement. A new graph cannot inherit the old actor's
admission or flight receipts.

The parcel placement service models a person placing a **detached** parcel:
it resets that parcel's pose and velocity for one simulation step, then removes
its velocity command. Gravity remains active. It rejects attached parcels and
never controls the aircraft. Runtime copies of payload SDFs retain original
physics and add only this verified plugin; deployment evidence records both
the original and derived asset digests. This component does not replace real
hardware attachment sensors or recipient verification.
