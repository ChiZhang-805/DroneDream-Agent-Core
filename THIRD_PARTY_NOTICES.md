# Third-party notices

The root MIT license covers DroneDream's first-party source only. It does not
relicense any dependency, imported asset, model weight, or dataset. Dependencies
are obtained from their upstream distributions, which must retain their license
files and required notices when bundled. A package name alone is not a complete
license inventory: the release inventory must cover the exact resolved versions,
transitive dependencies, and native libraries in the final binary.

School Map and My Drone have a separate [first-party MIT asset grant](docs/default-assets-license.md).
My Drone references the upstream PX4 `x500_depth` model without relicensing it.

## Python and native dependencies

`pyproject.toml` declares the supported Python dependencies and optional training
extras. JavaScript versions are locked in `app/*/package-lock.json`; Rust versions
are locked in `app/desktop/src-tauri/Cargo.lock`. The Runtime source contract pins
the separate DroneDream product sources and their third-party notices. Preserve
the licenses delivered by cryptography, HTTPX, jsonschema, NumPy, OpenAI's SDK,
Pydantic, Pillow, pypdf, FastAPI, python-multipart, Uvicorn, ONNX, ONNX Runtime,
PyTorch, torchvision, and their dependencies. PyInstaller's licensing exception
must be retained when distributing a frozen application.

Do not present this list as approval of an uninspected binary or model archive.
The public product's `runtime/THIRD_PARTY_NOTICES.md` covers its pinned PX4,
Gazebo, ROS, and supporting Runtime components, not all possible user imports.

## Model weights and data

`local_vision_training.py` can explicitly request torchvision's
`MobileNet_V3_Large_Weights.IMAGENET1K_V2`. The code's license does not by itself
grant rights to that dataset or every derivative checkpoint. Pretrained download
remains opt-in; no third-party pretrained weights are included in the source-only
export. Distribution of trained artifacts requires separate data/weight provenance
and license verification in addition to technical model admission.

## Lobe Icons

AUTONOMY includes selected AI and model-provider SVG marks from
[`@lobehub/icons-static-svg`](https://github.com/lobehub/lobe-icons), version
1.94.0, distributed under the MIT License.

Provider names and logos remain trademarks of their respective owners. Their
display identifies a configured service and does not imply endorsement of
DroneDream.
