"""Scientific overlay of map-ground evidence in its actual metric frame."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

parser = argparse.ArgumentParser()
parser.add_argument("candidate", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--bounds", type=float, nargs=4)
parser.add_argument("--route", type=Path)
args = parser.parse_args()
semantic = json.loads((args.candidate / "gazebo/semantic.json").read_text("utf-8"))
graph = json.loads((args.candidate / "navigation-graph.json").read_text("utf-8"))
fig, ax = plt.subplots(figsize=(14, 14))
ax.imshow(
    plt.imread(args.candidate / "gazebo/materials/textures/kumpula-orthophoto.jpg"),
    extent=(-250, 250, -250, 250),
)
layer = semantic["traversability"]
for item in layer["areas"]:
    ax.add_patch(Polygon(item["polygon_xy_m"], facecolor="#2dd4bf", edgecolor="#14b8a6", alpha=0.3))
for item in layer["exclusions"]:
    ax.add_patch(Polygon(item["polygon_xy_m"], facecolor="#ef4444", edgecolor="#f87171", alpha=0.2))
for item in layer["corridors"]:
    points = item["centerline_xy_m"]
    ax.plot([p[0] for p in points], [p[1] for p in points], color="#a78bfa", linewidth=1)
for node in graph["nodes"]:
    if node["semantic"] == "corridor":
        continue
    p = node["position_m"]
    ax.plot(p["x"], p["y"], "o", color="#22d3ee", markersize=4)
    ax.annotate(
        node["node_id"],
        (p["x"], p["y"]),
        fontsize=5,
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1},
    )
if args.route:
    route = json.loads(args.route.read_text("utf-8"))
    positions = route["positions_m"]
    ax.plot(
        [p["x"] for p in positions],
        [p["y"] for p in positions],
        color="#ffdd00",
        linewidth=3,
        marker="o",
        markersize=4,
    )
ax.set(
    xlabel="East (m)",
    ylabel="North (m)",
    title="Candidate ground surfaces — cyan: named access; purple: survey; red: exclusions",
)
ax.set_aspect("equal")
if args.bounds:
    ax.set_xlim(*args.bounds[:2])
    ax.set_ylim(*args.bounds[2:])
fig.tight_layout()
fig.savefig(args.output, dpi=150)
