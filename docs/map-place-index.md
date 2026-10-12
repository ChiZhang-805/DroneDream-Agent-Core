# Precomputed map places

The desktop now loads a compact place index by the exact semantic-file SHA-256.
The bundled Kumpula index covers every shipped building component and is supplied
to both asset interpretation and the intent parser. A different map version never
inherits Kumpula names. Index parsing is cached; missions do not download or parse
city/OSM data again. The map resource API includes the same compact context.

The current source review contains 41 municipal building records covering 42
geometry components, 37 records with source names/addresses, 4 unnamed ancillary
records labelled by their source type and ID, 58 documented entrance records for
27 buildings, and 57 exterior handoff candidates covering all 41 records. Public
sources do not establish a complete entrance/exit census. Fourteen records have
no documented entrance. Missing information remains unknown, not zero entrances.

The distinction between municipal building identity and facility is intentional:
Exactum, Physicum and the campus library share one municipal geometry record.
They retain separate facility names and their source-linked entrances. Street
addresses shared by different buildings remain ambiguous until a building or
door reference disambiguates them. Address discrepancies between municipal and
OSM sources remain aliases rather than being silently erased.

## Rebuild

`scripts/build_map_place_index.py` accepts the shipped map package, a source
snapshot directory, and an output path. The source directory contains
`city-buildings.geojson`, `osm-map.xml`, `entrances.json`, `local-units.json` and
`facility-annotations.json`. The reviewed facility annotations are also retained
in `scripts/map_sources/kumpula-facilities.json`. Install `pyproj` and `shapely`
only in the build environment. The application does not depend on them.

Inputs and their hashes are recorded in the generated index. City footprints are
matched by source ID; OSM building outlines are matched by overlap. Entrance
membership uses OSM topology and nearest city footprint when one OSM building
covers several municipal records. City Service Map entrances supplement the
OSM nodes, merging only nearby duplicate records within a single building.
Entrance points outside the clipped scene are retained and explicitly flagged.

Sources: [Helsinki building WFS](https://kartta.hel.fi/ws/geoserver/avoindata/wfs),
[OpenStreetMap](https://www.openstreetmap.org/copyright),
[City of Helsinki Service Map API](https://www.hel.fi/palvelukarttaws/restpages/ver4.html),
[University campus information](https://www.helsinki.fi/en/about-us/visit-us/campuses/kumpula-campus).
Attribution is carried in the index. Item associations are authored suggestions
based on documented uses, not statements about actual item stock or location.

## Geometry and execution scope

Handoff candidates lie on mapped pedestrian paths outside the current simulation
building collision bodies, with at least 2.5 m horizontal building clearance.
They are deliberately identified as geometry candidates, not surveyed doorsteps,
approved landing sites or flight-authorized nodes. Ground height, door-to-point
access, people, obstacles outside the building dataset and live flight clearance
have not been verified. Current coarse collision boxes sometimes place a candidate
tens of metres from a real entrance; those candidates need scene review.

This update does not rewrite the qualified map geometry, reference route, or
qualification receipts and does not add unqualified nodes to the execution graph.
The missing entrances and route qualification remain actual open work; a successful
index test is not evidence that every building can already receive a mission.

## Validation

Tests cover complete geometry ownership, source identity, multiple entrances,
service aliases, candidate clearance, immutable cache use, intent-model visibility
and the separation between factual context and executable route nodes. Existing
asset, map resource, interpretation-cache and orchestration tests also run.
