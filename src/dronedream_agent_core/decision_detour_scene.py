"""Looped maps with a real blocked passage and a geometrically verified detour.

These are experiment materials, not executed replan demonstrations. A tree has
no alternate path after removing an edge; collecting more such trees cannot
supply the missing successful replan class.
"""

import copy
import hashlib
import math
from collections import deque

from .decision_scene_topology import scene_documents, topology


# 功能：在显式邻接图上求最短房间路径；输入：无向边、节点及禁用边；输出：节点列表或None。
# 不读取未来执行结果，不能把图上有路等同于实际飞行成功。
def shortest_path(size, edges, start, goal, *, blocked=()):
    blocked = {tuple(sorted(edge)) for edge in blocked}
    adjacency = {i: [] for i in range(size * size)}
    for a, b in edges:
        if tuple(sorted((a, b))) not in blocked:
            adjacency[a].append(b)
            adjacency[b].append(a)
    parents, queue = {start: None}, deque([start])
    while queue:
        node = queue.popleft()
        if node == goal:
            path = []
            while node is not None:
                path.append(node)
                node = parents[node]
            return path[::-1]
        for neighbor in sorted(adjacency[node]):
            if neighbor not in parents:
                parents[neighbor] = node
                queue.append(neighbor)
    return None


# 功能：旋转/编号不改变布局族；输入：完整连通邻接；输出：保守WL拓扑桶，碰撞只会合并族。
# WL不能证明同构；同桶始终放同一集合，绝不把同桶当不同独立布局。
def graph_family(size, edges):
    neighbors = {i: [] for i in range(size * size)}
    for a, b in edges:
        neighbors[a].append(b)
        neighbors[b].append(a)
    colors = {i: str(len(values)) for i, values in neighbors.items()}
    for _ in range(size * size):
        colors = {
            i: hashlib.sha256(
                (colors[i] + "|" + "|".join(sorted(colors[n] for n in values))).encode()
            ).hexdigest()
            for i, values in neighbors.items()
        }
    return (
        "native-loop-graph-v1:"
        + hashlib.sha256("|".join(sorted(colors.values())).encode()).hexdigest()
    )


# 功能：从树布局增加真实通道，在一条旧通道放实体挡板并生成可绕行路线。
# 输入：已验证树配方、新开的相邻房间边、机型净空；输出：前后两份真实几何、旧/新路线与族。
# 本模块仅产生材料；必须另有观察到阻塞、采用新路线、实际恢复的回执才能授予正式标签。
def detour_documents(recipe, extra_edges, vehicle_clearance):
    semantic, _, parent_family = scene_documents(recipe, vehicle_clearance)
    size = recipe["size"]
    topology(size, recipe["edges"])
    edges = {tuple(edge) for edge in recipe["edges"]}
    if type(extra_edges) is not list or not 1 <= len(extra_edges) <= 8:
        raise ValueError("DECISION_DETOUR_EXTRA_EDGES_INVALID")
    opened = set()
    for edge in extra_edges:
        if (
            type(edge) is not list
            or len(edge) != 2
            or any(type(n) is not int or not 0 <= n < size * size for n in edge)
        ):
            raise ValueError("DECISION_DETOUR_EXTRA_EDGES_INVALID")
        a, b = sorted(edge)
        if (
            (a, b) in edges
            or (a, b) in opened
            or (abs(a % size - b % size) + abs(a // size - b // size) != 1)
        ):
            raise ValueError("DECISION_DETOUR_EXTRA_EDGES_INVALID")
        opened.add((a, b))
    edges |= opened
    removed = {
        f"wall-x-{b % size}-{a // size}" if b - a == 1 else f"wall-y-{a % size}-{b // size}"
        for a, b in opened
    }
    primitives = [p for p in semantic["collision_primitives"] if p["name"] not in removed]
    if len(semantic["collision_primitives"]) - len(primitives) != len(opened):
        raise ValueError("DECISION_DETOUR_OPENED_WALL_NOT_FOUND")
    semantic["collision_primitives"] = semantic["runtime_collision_primitives"] = primitives
    semantic["name"] = "Decision looped rooms"
    # 选目标和阻塞边都在采集前确定，不按哪个物理实验成功来挑选。
    choices = []
    for goal in range(1, size * size):
        route = shortest_path(size, edges, 0, goal)
        for a, b in zip(route, route[1:], strict=False):
            alternate = shortest_path(size, edges, 0, goal, blocked=[(a, b)])
            if alternate is not None and len(alternate) > len(route):
                choices.append((len(route), goal, a, b, route, alternate))
    if not choices:
        raise ValueError("DECISION_DETOUR_NO_ALTERNATIVE")
    _, goal, a, b, before, after = max(choices)
    cell, z = recipe["cell_m"], recipe["flight_z_m"]

    # 功能：节点路径转ENU三维航点及米制长度；输入：图路径；输出：未宣称飞行验证的GraphRoute。
    def route_document(path):
        points = [
            {"x": float(n % size * cell), "y": float(n // size * cell), "z": float(z)} for n in path
        ]
        return {
            "schema_version": "dronedream.graph-route.v1",
            "start_node": "room-0",
            "goal_node": f"room-{goal}",
            "node_ids": [f"room-{n}" for n in path],
            "edge_ids": [f"door-{u}-{v}" for u, v in zip(path, path[1:], strict=False)],
            "positions_m": points,
            "route_length_m": float((len(path) - 1) * cell),
            "all_edges_flight_verified": False,
        }

    blocked = copy.deepcopy(semantic)
    center = ((a % size + b % size) * cell / 2, (a // size + b // size) * cell / 2, 2.0)
    dimensions = (0.2, cell + 0.2, 3.6) if abs(a - b) == 1 else (cell + 0.2, 0.2, 3.6)
    barrier = {
        "shape": "box",
        "name": "closed-passage",
        **dict(zip(("center_" + k for k in "xyz"), center, strict=True)),
        **dict(zip(("size_" + k for k in "xyz"), dimensions, strict=True)),
        "yaw_rad": 0.0,
        "pitch_rad": 0.0,
        "roll_rad": 0.0,
    }
    blocked["collision_primitives"].append(barrier)
    # 深拷贝保留原来别名，显式赋值确保发布前两种几何契约一致。
    blocked["runtime_collision_primitives"] = blocked["collision_primitives"]
    return {
        "before": semantic,
        "blocked": blocked,
        "initial_route": route_document(before),
        "replacement_route": route_document(after),
        "blocked_edge": sorted([a, b]),
        "family": graph_family(size, sorted(edges)),
        "parent_family": parent_family,
        "edges": sorted(map(list, edges)),
    }


# 功能：用真实机体包络验证旧路确被阻塞、新路具有足够净空；输入：材料及米制边界；输出：摘要。
def verify_detour_geometry(documents, *, radius_m, height_m, clearance_m):
    from .training.swept_geometry import SweptMapGeometry

    if any(
        type(n) not in (float, int) or not math.isfinite(n) or n <= 0
        for n in (radius_m, height_m, clearance_m)
    ):
        raise ValueError("DECISION_DETOUR_ENVELOPE_INVALID")
    geometry = SweptMapGeometry(
        documents["blocked"]["collision_primitives"], radius_m=radius_m, half_height_m=height_m / 2
    )
    values = {}
    for name in ("initial_route", "replacement_route"):
        vectors = [tuple(p[k] for k in "xyz") for p in documents[name]["positions_m"]]
        values[name] = geometry.clearance(vectors)
    if values["initial_route"] >= clearance_m or values["replacement_route"] < clearance_m:
        raise ValueError("DECISION_DETOUR_GEOMETRY_NOT_CONFIRMED")
    return {
        "initial_route_clearance_m": values["initial_route"],
        "replacement_route_clearance_m": values["replacement_route"],
        "physically_executed": False,
        "formal_training_additions": 0,
    }
