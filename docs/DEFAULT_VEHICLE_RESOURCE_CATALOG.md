# Default aircraft resource catalog

DroneDream exposes a reviewed, on-demand aircraft catalog in the **Aircraft**
page. Catalog admission is deliberately narrower than flight qualification: a
listed model has a pinned source, reviewed license, valid Gazebo metadata, and a
deterministic structural analysis. It is not automatically safe in every map.

## Source and license

- Source: `PX4/PX4-gazebo-models`
- Pinned commit: `5577035667afb4b63fe1f966fb1a58bbb05d905b`
- License: BSD-3-Clause
- Runtime target: PX4 SITL with Gazebo Harmonic

The catalog uses deterministic ZIP hashes for each model directory. Nested model
dependencies are resolved only from the same pinned source revision. Downloaded
plugins remain disabled until admitted by the packaged runtime.

## Included models and intended tasks

| Model | Primary use | Important limitation |
| --- | --- | --- |
| X500 Depth | Indoor navigation, doorways, stairs, depth avoidance | Requires X500, X500 Base, and OAK-D Lite dependencies |
| X500 Optical Flow | GPS-denied position hold and landing | Requires textured ground, adequate lighting, and valid range data |
| X500 Downward Lidar | Ground clearance, altitude hold, and landing | Does not detect forward obstacles |
| X500 Forward Lidar | Doorway, corridor, and approach avoidance | Does not provide full surround coverage |
| X500 Forward Monocular Camera | Detection, tracking, and visual landmarks | Monocular imagery alone is not metric depth |
| X500 Downward Monocular Camera | Landing markers and visual odometry | Does not detect forward obstacles |
| X500 2D Lidar | Corridor clearance and horizontal obstacle ranging | Does not provide full vertical depth coverage |
| X500 Vision | Visual odometry and external-vision fusion | The odometry interface is not an object detector |
| X500 Gimbal | Inspection, camera pointing, and target tracking | Needs an additional depth/range source for tight autonomous flight |
| X500 | Basic control, hover, and open-area reference flights | Not an indoor autonomy configuration by itself |
| Standard VTOL | Outdoor long-range transit with vertical takeoff | Not suitable for stairs, doors, or indoor corridors |
| Tiltrotor | Outdoor transition-flight research | Experimental configuration; not an indoor default |
| Quad Tailsitter | Outdoor VTOL and transition-flight research | Experimental configuration; not an indoor default |

Omnicopter and Omniquad remain excluded from the default catalog in this
revision. Their research value is real, but their additional actuation and
airframe-binding requirements are not justified for a general-purpose default.

## Readiness states

1. **Preparsed**: SDF, sensors, plugins, model references, license, and task fit
   are known.
2. **Dependencies resolved**: every nested model and runtime plugin is present at
   the reviewed version.
3. **Simulation ready**: PX4 airframe binding and Gazebo spawning pass.
4. **Flight qualified**: the exact aircraft content hash has passed the exact map
   content hash, route, sensor, collision, and safe-landing checks.

Changing a model or map produces a new content identity and invalidates only the
affected pair qualification. A successful run for one pair never grants another
pair permission to fly.

## Reproducing the source verification

Run:

```powershell
.venv\Scripts\python.exe scripts\verify_vehicle_resource_catalog.py `
  --checkout Q:\DroneDream-Workspace\Caches\ThirdParty\PX4-gazebo-models-5577035667af `
  --report Q:\DroneDream-Workspace\TestRuns\vehicle-resource-catalog-20260928\verification.json
```
