"""Build an offline place index from city footprints, OSM and public entrance records.

Build-time dependencies: shapely and pyproj. No network or geospatial dependencies
are needed when the desktop reads the generated index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

from pyproj import Transformer
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box, mapping, shape
from shapely.ops import nearest_points, transform, unary_union


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tags(element) -> dict:
    return {entry.get("k"): entry.get("v") for entry in element.findall("tag")}


def build_index(package: Path, sources: Path) -> dict:
    with zipfile.ZipFile(package) as archive:
        raw = archive.read("normalized/map/gazebo/semantic.json")
        semantic = json.loads(raw)
    origin = semantic["origin"]
    projection = Transformer.from_crs(4326, semantic["real_world_crs"], always_xy=True)

    def xy(lon, lat):
        east, north = projection.transform(float(lon), float(lat))
        return east - origin["easting_m"], north - origin["northing_m"]

    def local_geometry(geometry):
        projected = transform(projection.transform, shape(geometry))
        return affinity.translate(projected, -origin["easting_m"], -origin["northing_m"])

    city = json.loads((sources / "city-buildings.geojson").read_bytes())
    city = {str(f["properties"]["tietopalvelu_id"]): f for f in city["features"]}
    osm = ElementTree.fromstring((sources / "osm-map.xml").read_bytes())
    nodes = {n.get("id"): n for n in osm.findall("node")}
    positions = {nid: xy(n.get("lon"), n.get("lat")) for nid, n in nodes.items()}
    ways, paths = [], []
    for way in osm.findall("way"):
        attributes = tags(way)
        ids = [n.get("ref") for n in way.findall("nd")]
        if any(nid not in positions for nid in ids):
            continue  # OSM bbox responses may contain incomplete outside relations.
        points = [positions[nid] for nid in ids]
        if "building" in attributes and len(points) >= 4:
            polygon = Polygon(points)
            if polygon.is_valid:
                ways.append((way.get("id"), polygon, attributes, ids))
        if (
            attributes.get("highway") in {"footway", "path", "pedestrian"}
            and len(points) > 1
            and attributes.get("access") not in {"no", "private"}
        ):
            paths.append(LineString(points))
    walkways = unary_union(paths)
    components = [p for p in semantic["collision_primitives"] if p["semantic"] == "building"]
    collision_shapes = []
    by_city = {}
    for component in components:
        cid = component["name"].split("-")[1]
        by_city.setdefault(cid, []).append(component)
        rectangle = box(
            -component["size_x"] / 2,
            -component["size_y"] / 2,
            component["size_x"] / 2,
            component["size_y"] / 2,
        )
        rectangle = affinity.rotate(rectangle, component["yaw_rad"], use_radians=True)
        collision_shapes.append(
            affinity.translate(rectangle, component["center_x"], component["center_y"])
        )
    occupied = unary_union(collision_shapes)
    extent = box(
        -semantic["bounds_m"]["x"] / 2,
        -semantic["bounds_m"]["y"] / 2,
        semantic["bounds_m"]["x"] / 2,
        semantic["bounds_m"]["y"] / 2,
    )
    footprints = {cid: local_geometry(city[cid]["geometry"]) for cid in by_city}
    matches = {cid: [] for cid in by_city}
    for wid, polygon, attributes, ids in ways:
        for cid, footprint in footprints.items():
            overlap = polygon.intersection(footprint).area / min(polygon.area, footprint.area)
            if overlap >= 0.65:
                matches[cid].append((wid, polygon, attributes, ids, overlap))

    type_names = {
        "Asuinrakennus": "住宅楼",
        "Varastorakennus": "仓储附属建筑",
        "Talousrakennus": "附属建筑",
        "Muu rakennus": "建筑",
        "Paikoitustalo": "停车楼",
        "Opetusrakennus": "教学楼",
        "Päiväkoti": "幼儿园",
        "Parakki": "临时用房",
    }
    buildings, entrances, services = [], {}, []
    unit_records = json.loads((sources / "local-units.json").read_bytes())
    public_entrances = json.loads((sources / "entrances.json").read_bytes())
    units_to_buildings = {}
    for unit in unit_records:
        main = next(
            (
                e
                for e in public_entrances
                if e["unit_id"] == unit["id"] and e["is_main_entrance"] == "Y"
            ),
            None,
        )
        location = main or unit
        point = Point(xy(location["longitude"], location["latitude"]))
        distances = sorted((p.distance(point), cid) for cid, p in footprints.items())
        if distances[0][0] <= 20:
            units_to_buildings[unit["id"]] = distances[0][1]

    for cid in sorted(by_city, key=int):
        prop = city[cid]["properties"]
        address = " ".join(str(prop[k]) for k in ("katunimi_suomi", "osoitenumero") if prop[k])
        osm_matches = matches[cid]
        aliases = [address] if address else []
        named = []
        for wid, _polygon, attributes, ids, _overlap in osm_matches:
            named.extend(attributes[k] for k in ("name", "name:en", "int_name") if k in attributes)
            osm_address = " ".join(
                attributes.get(k, "") for k in ("addr:street", "addr:housenumber")
            ).strip()
            if osm_address:
                aliases.append(osm_address)
            for nid in ids:
                nt = tags(nodes[nid])
                if "entrance" not in nt and nt.get("emergency") != "exit":
                    continue
                if nt.get("entrance") == "no":
                    continue
                point = Point(positions[nid])
                # One OSM outline may cover multiple city buildings. Assign a door
                # only to its closest city footprint, not to every overlapping one.
                owners = [other for other in by_city if any(m[0] == wid for m in matches[other])]
                owner = min(owners, key=lambda other: footprints[other].distance(point))
                if owner != cid:
                    continue
                eid = f"osm-door-{nid}"
                entrances[eid] = {
                    "entrance_id": eid,
                    "building_id": f"city-{cid}",
                    "name": nt.get("ref", nt.get("name", nt.get("entrance", "exit"))),
                    "kind": "exit" if nt.get("emergency") == "exit" else nt.get("entrance"),
                    "access": nt.get("access", "unspecified"),
                    "level": nt.get("level"),
                    "position_xy_m": [round(v, 3) for v in positions[nid]],
                    "source_refs": [f"https://www.openstreetmap.org/node/{nid}"],
                    "osm_building_id": wid,
                    "inside_scene_bounds": extent.covers(point),
                    "height_status": "not-measured",
                    "flight_target_available": False,
                }
        units = [u for u in unit_records if units_to_buildings.get(u["id"]) == cid]
        named.extend(u.get("name_en", u["name_fi"]) for u in units)
        for u in units:
            services.append(
                {
                    "service_id": f"hel-service-{u['id']}",
                    "building_id": f"city-{cid}",
                    "name": u.get("name_en", u["name_fi"]),
                    "name_fi": u["name_fi"],
                    "source_refs": [f"https://www.hel.fi/palvelukarttaws/rest/v4/unit/{u['id']}"],
                }
            )
        aliases = list(dict.fromkeys([*named, *aliases]))
        if not address:
            address = next((a for a in aliases if any(char.isdigit() for char in a)), "")
        label = (
            named[0]
            if named
            else address or f"{type_names.get(prop['tyyppi'], prop['tyyppi'])} {cid}"
        )
        buildings.append(
            {
                "building_id": f"city-{cid}",
                "name": label,
                "aliases": aliases or [label],
                "label_basis": "source-name"
                if named
                else "source-address"
                if address
                else "type-and-source-id",
                "address": address or None,
                "use": type_names.get(prop["tyyppi"], prop["tyyppi"]),
                "registry_id": prop["c_vtj_prt"],
                "geometry_component_ids": [c["name"] for c in by_city[cid]],
                "osm_building_ids": [m[0] for m in osm_matches],
                "footprint": mapping(footprints[cid]),
                "source_refs": [f"city-buildings.geojson#Rakennukset_alue_rekisteritiedot.{cid}"]
                + [f"https://www.openstreetmap.org/way/{m[0]}" for m in osm_matches],
            }
        )
    # Municipal entrance records supplement OSM. Close records for the same door
    # are merged with both references; distinct accessible/service doors survive.
    for entry in public_entrances:
        cid = units_to_buildings.get(entry["unit_id"])
        if cid is None or not entry.get("longitude") or not entry.get("latitude"):
            continue
        position = xy(entry["longitude"], entry["latitude"])
        if footprints[cid].distance(Point(position)) > 20:
            continue
        reference = f"https://www.hel.fi/palvelukarttaws/rest/v4/entrance/{entry['id']}"
        nearby = [
            door
            for door in entrances.values()
            if door["building_id"] == f"city-{cid}"
            and math.dist(door["position_xy_m"], position) <= 4
        ]
        if nearby:
            door = min(nearby, key=lambda d: math.dist(d["position_xy_m"], position))
            door["source_refs"].append(reference)
            door.setdefault("service_ids", []).append(f"hel-service-{entry['unit_id']}")
        else:
            eid = f"hel-door-{entry['id']}"
            entrances[eid] = {
                "entrance_id": eid,
                "building_id": f"city-{cid}",
                "name": entry.get(
                    "name_en", "Main entrance" if entry["is_main_entrance"] == "Y" else "Entrance"
                ),
                "kind": "main" if entry["is_main_entrance"] == "Y" else "yes",
                "access": "unspecified",
                "level": None,
                "position_xy_m": [round(v, 3) for v in position],
                "source_refs": [reference],
                "service_ids": [f"hel-service-{entry['unit_id']}"],
                "inside_scene_bounds": extent.covers(Point(position)),
                "height_status": "not-measured",
                "flight_target_available": False,
            }

    facilities = []
    annotations = json.loads((sources / "facility-annotations.json").read_bytes())
    for building in buildings:
        cid = building["building_id"].removeprefix("city-")
        for wid, _polygon, attributes, _ids, _overlap in matches[cid]:
            if not attributes.get("name") and wid not in annotations:
                continue
            authored = annotations.get(wid, {})
            names = list(
                dict.fromkeys(
                    [attributes[k] for k in ("name", "name:en", "int_name") if k in attributes]
                    + authored.get("aliases", [])
                )
            )
            facilities.append(
                {
                    "facility_id": f"osm-facility-{wid}",
                    "building_id": building["building_id"],
                    "footprint": mapping(_polygon),
                    "name": attributes.get("name", names[0]),
                    "aliases": names,
                    "uses": authored.get("uses", [attributes.get("building")]),
                    "item_associations": authored.get("item_associations", []),
                    "association_authority": "candidate-context-only; not actual item inventory",
                    "entrance_ids": [
                        d["entrance_id"]
                        for d in entrances.values()
                        if d.get("osm_building_id") == wid
                    ],
                    "source_refs": [f"https://www.openstreetmap.org/way/{wid}"]
                    + authored.get("source_refs", []),
                }
            )
            building["aliases"] = list(dict.fromkeys(building["aliases"] + names))

    # Named amenities are separate from building identity (e.g. a library inside
    # Physicum). Only documented services are recorded, never invented stock/items.
    for nid, node in nodes.items():
        attributes = tags(node)
        if not attributes.get("name") or not ("amenity" in attributes or "office" in attributes):
            continue
        point = Point(positions[nid])
        owners = [cid for cid, footprint in footprints.items() if footprint.covers(point)]
        if len(owners) == 1:
            services.append(
                {
                    "service_id": f"osm-service-{nid}",
                    "building_id": f"city-{owners[0]}",
                    "name": attributes.get("name:en", attributes["name"]),
                    "category": attributes.get("amenity", attributes.get("office")),
                    "source_refs": [f"https://www.openstreetmap.org/node/{nid}"],
                }
            )

    # Precompute exterior handoff candidates. These are simulation design points,
    # NOT surveyed entrances or approved landing sites. Geometry is checked once.
    handoffs = []
    free_walkways = walkways.difference(occupied.buffer(2.5, join_style="mitre")).intersection(
        extent.buffer(-2.5)
    )
    for building in buildings:
        bid = building["building_id"]
        doors = [d for d in entrances.values() if d["building_id"] == bid]
        building["entrance_ids"] = [d["entrance_id"] for d in doors]
        building["entrance_coverage"] = "documented-subset" if doors else "not-documented"
        building["total_entrance_count"] = None
        building["service_ids"] = [s["service_id"] for s in services if s["building_id"] == bid]
        if free_walkways.is_empty:
            building["handoff_candidate_ids"] = []
            continue
        footprint = footprints[bid.removeprefix("city-")]
        origins = [
            (d["entrance_id"], Point(d["position_xy_m"]))
            for d in doors
            if d["access"] not in {"no", "employees"} and d["kind"] != "exit"
        ]
        if not origins:
            origins = [(None, footprint)]
        selected = []
        for eid, start in origins:
            target = nearest_points(start, free_walkways)[1]
            if any(target.distance(Point(p["position_xy_m"])) < 2 for p in selected):
                continue
            hid = f"{bid}-handoff-{len(selected) + 1}"
            candidate = {
                "handoff_id": hid,
                "building_id": bid,
                "entrance_id": eid,
                "position_xy_m": [round(target.x, 3), round(target.y, 3)],
                "status": "geometry-candidate",
                "flight_target_available": False,
                "basis": "mapped-walkway-outside-simulation-building-collisions",
                "horizontal_building_clearance_m": round(occupied.distance(target), 3),
                "distance_to_reference_m": round(start.distance(target), 3),
                "height_status": "not-measured",
                "entrance_connection_verified": False,
            }
            selected.append(candidate)
        handoffs.extend(selected)
        building["handoff_candidate_ids"] = [p["handoff_id"] for p in selected]

    source_urls = {
        "city-buildings.geojson": "https://kartta.hel.fi/ws/geoserver/avoindata/wfs",
        "osm-map.xml": "https://api.openstreetmap.org/api/0.6/map?bbox=24.956,60.201,24.969,60.208",
        "entrances.json": "https://www.hel.fi/palvelukarttaws/rest/v4/entrance/",
        "local-units.json": "https://www.hel.fi/palvelukarttaws/rest/v4/unit/",
        "facility-annotations.json": "https://www.helsinki.fi/en/about-us/visit-us/campuses/kumpula-campus",
    }
    return {
        "schema_version": "dronedream.map-place-index.v1",
        "scene_id": semantic["scene_id"],
        "semantic_sha256": hashlib.sha256(raw).hexdigest(),
        "map_package_sha256": hashlib.sha256(package.read_bytes()).hexdigest(),
        "coordinate_frame": "ENU",
        "generated_from": [
            {
                "file": filename,
                "url": url,
                "sha256": hashlib.sha256((sources / filename).read_bytes()).hexdigest(),
            }
            for filename, url in source_urls.items()
        ],
        "attribution": (
            "City of Helsinki CC BY 4.0; OpenStreetMap contributors ODbL 1.0; "
            "City of Helsinki Service Map"
        ),
        "buildings": buildings,
        "facilities": facilities,
        "entrances": list(entrances.values()),
        "services": services,
        "handoff_candidates": handoffs,
        "coverage": {
            "geometry_components": len(components),
            "city_building_records": len(buildings),
            "label_basis": dict(Counter(b["label_basis"] for b in buildings)),
            "documented_entrances": len(entrances),
            "buildings_with_documented_entrances": sum(bool(b["entrance_ids"]) for b in buildings),
            "buildings_without_documented_entrances": [
                b["building_id"] for b in buildings if not b["entrance_ids"]
            ],
            "handoff_candidates": len(handoffs),
            "entrance_inventory_complete": False,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = build_index(args.package, args.sources)
    write_json(args.output, index)
    print(json.dumps(index["coverage"]))


if __name__ == "__main__":
    main()
