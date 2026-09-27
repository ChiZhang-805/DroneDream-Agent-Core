# Default map resource catalog

The desktop map page exposes a reviewed, on-demand catalog. Catalog inclusion
means that the source and license were reviewed and that a pinned revision has a
precompiled structural summary. It does **not** mean that the map is a Gazebo
world, that every referenced model is installed, or that flight has been
qualified with an aircraft.

## Included revision

- Repository: `open-rmf/rmf_demos`
- Commit: `7851a5792d19a037833292a3e2a823b0f9e0c111`
- License: Apache-2.0
- Acquisition: isolated HTTPS Git fetch from an immutable commit, restricted to
  one reviewed map directory, with an exact SHA-256 for the deterministic ZIP
  containing its `.building.yaml` and companion floor-plan images
- Execution boundary: source files remain declarative data; imported repository
  code is never executed

| Resource | Levels | Doors | Lifts | Models | Lanes | Walls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Airport Terminal | 1 | 5 | 0 | 282 | 223 | 464 |
| Clinic | 2 | 132 | 4 | 706 | 169 | 619 |
| Hotel | 3 | 19 | 2 | 175 | 122 | 254 |
| Office | 1 | 3 | 0 | 79 | 30 | 33 |
| Campus | 1 | 0 | 0 | 1 | 154 | 0 |
| Open Test Arena | 1 | 0 | 0 | 0 | 20 | 32 |
| Triple-H Corridor Test | 1 | 0 | 0 | 0 | 22 | 28 |

The structural values are deterministic data extracted from the pinned RMF
building maps. They do not require a model API. An LLM may later enrich semantic
place names, task affordances, and preferred UAV airspace after conversion, but
its output stays advisory and content-hash-bound.

## Installation and readiness states

1. `preparsed`: metadata is available immediately in the map repository.
2. `conversion_required`: RMF Traffic Editor must convert the building source in
   an isolated local boundary and Gazebo dependencies must resolve.
3. `simulation_ready`: the normalized world loads and its declared dependencies
   are available. This state is not granted by the catalog.
4. `flight qualified`: the exact map content hash and aircraft content hash pass
   recorded PX4/Gazebo qualification. This state is never inferred from a
   repository license or model interpretation.

RMF navigation lanes are ground-robot traffic lanes. They may inform topology,
but they are not UAV corridors and are never copied directly into flight control.

## Reviewed but excluded from the default catalog

- DARPA SubT practice world: the repository is Apache-2.0, but the selected SDF
  references multiple separately hosted Fuel models. It remains excluded until
  every transitive model license, immutable revision, and current Gazebo
  compatibility are closed.
- HSSD, HM3D, and GRScenes: non-commercial or academic-use terms do not meet the
  default commercial-product requirement.
- Paid marketplace scenes and assets requiring an account grant: these remain
  user-supplied optional imports and are never purchased or redistributed by the
  product without a separate authorization and license record.
- Archived or unmaintained sample worlds: archival source is not sufficient
  evidence of current Runtime compatibility.
