"""Reproducible room/corridor trees; topology, not random seed, identifies a layout."""

import hashlib
import math
import random
from collections import deque

from dronedream_plugin_sdk.protocol import encode_json


# 功能：生成连通无环的房间连接；随机种子只选择布局，不被当作独立布局的证据。
# 输入：3～5格的正方形、整数种子；输出：按房间编号排序的相邻连接，无运行或训练资格。
def generate_edges(size, seed):
    if type(size) is not int or not 3 <= size <= 5 or type(seed) is not int:
        raise ValueError("DECISION_TOPOLOGY_GENERATOR_INPUT_INVALID")
    candidates = [
        (i, j)
        for i in range(size * size)
        for j in (i + 1, i + size)
        if j < size * size and (j == i + size or i // size == j // size)
    ]
    random.Random(seed).shuffle(candidates)
    parents = list(range(size * size))

    # 功能：查询并压缩并查集路径；输入：房间编号；输出：连接分量编号。
    def root(node):
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    edges = []
    for left, right in candidates:
        a, b = root(left), root(right)
        if a != b:
            parents[a] = b
            edges.append([left, right])
    return sorted(edges)


# 功能：校验原始连通树并忽略编号/旋转/镜像构造族身份；不将同形树改名当成独立地图。
# 输入：格数和全部相邻边；输出：邻接表、无根树的规范化字符串。
def topology(size, edges):
    if (
        type(size) is not int
        or not 3 <= size <= 5
        or type(edges) is not list
        or len(edges) != size * size - 1
    ):
        raise ValueError("DECISION_TOPOLOGY_INVALID")
    adjacency = [[] for _ in range(size * size)]
    seen = set()
    for pair in edges:
        if (
            type(pair) is not list
            or len(pair) != 2
            or any(type(n) is not int or not 0 <= n < size * size for n in pair)
        ):
            raise ValueError("DECISION_TOPOLOGY_EDGE_INVALID")
        a, b = pair
        if a >= b or (a, b) in seen or abs(a // size - b // size) + abs(a % size - b % size) != 1:
            raise ValueError("DECISION_TOPOLOGY_EDGE_INVALID")
        seen.add((a, b))
        adjacency[a].append(b)
        adjacency[b].append(a)
    visited, pending = {0}, [0]
    while pending:
        for node in adjacency[pending.pop()]:
            if node not in visited:
                visited.add(node)
                pending.append(node)
    if len(visited) != size * size:
        raise ValueError("DECISION_TOPOLOGY_DISCONNECTED")

    # 功能：树形结构递归规范化，先验边数与连通性已排除环；输出：与编号无关的结构。
    def rooted(node, parent):
        return "(" + "".join(sorted(rooted(n, node) for n in adjacency[node] if n != parent)) + ")"

    return adjacency, min(rooted(node, -1) for node in range(size * size))


# 功能：从同一配方构造语义几何及有转弯的真实路径，后续SDF直接使用这些碰撞基元。
# 输入：树配方、房间宽度/高度米和机体净空规格；输出：地图、路线、布局族。
# 只是离线几何材料，不宣称定位、模型控制或物理仿真已通过。
def scene_documents(recipe, vehicle_clearance):
    if (
        type(recipe) is not dict
        or set(recipe) != {"size", "edges", "cell_m", "flight_z_m"}
        or type(recipe["cell_m"]) not in (int, float)
        or not math.isfinite(recipe["cell_m"])
        or not 3 <= recipe["cell_m"] <= 5
        or type(recipe["flight_z_m"]) not in (int, float)
        or not 1.2 <= recipe["flight_z_m"] <= 2.2
    ):
        raise ValueError("DECISION_SCENE_RECIPE_INVALID")
    size, cell, z = recipe["size"], recipe["cell_m"], recipe["flight_z_m"]
    adjacency, canonical = topology(size, recipe["edges"])
    family = "native-cell-tree-v1:" + hashlib.sha256(canonical.encode()).hexdigest()
    primitives = []

    # 功能：生成ENU米制盒体；输入：名称、中心和尺寸；输出：追加同源碰撞/视觉对象。
    def box(name, center, dimensions):
        primitives.append(
            {
                "shape": "box",
                "name": name,
                **dict(zip(("center_" + k for k in "xyz"), center, strict=True)),
                **dict(zip(("size_" + k for k in "xyz"), dimensions, strict=True)),
                "yaw_rad": 0.0,
                "pitch_rad": 0.0,
                "roll_rad": 0.0,
            }
        )

    midpoint, extent = (size - 1) * cell / 2, size * cell
    box("floor", (midpoint, midpoint, 0.11), (extent + 0.4, extent + 0.4, 0.22))
    box("ceiling", (midpoint, midpoint, 3.9), (extent + 0.4, extent + 0.4, 0.2))
    edges = {tuple(edge) for edge in recipe["edges"]}
    for y in range(size):
        for x in range(size + 1):
            a, b = y * size + x - 1, y * size + x
            if x in (0, size) or (a, b) not in edges:
                box(f"wall-x-{x}-{y}", ((x - 0.5) * cell, y * cell, 2.0), (0.2, cell + 0.2, 3.6))
    for x in range(size):
        for y in range(size + 1):
            a, b = (y - 1) * size + x, y * size + x
            if y in (0, size) or (a, b) not in edges:
                box(f"wall-y-{x}-{y}", (x * cell, (y - 0.5) * cell, 2.0), (cell + 0.2, 0.2, 3.6))
    # 各房间有真实梁体提供三维特征；梁底3米，不以绘制纹理替代几何定位证据。
    for node in range(size * size):
        box(
            f"beam-{node}",
            ((node % size) * cell + 0.65, (node // size) * cell, 3.35),
            (0.3, cell - 0.2, 0.7),
        )
    parents, distance, queue = {0: None}, {0: 0}, deque([0])
    while queue:
        node = queue.popleft()
        for other in sorted(adjacency[node]):
            if other not in parents:
                parents[other], distance[other] = node, distance[node] + 1
                queue.append(other)
    goal = max(distance, key=lambda n: (distance[n], n))
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = parents[node]
    path.reverse()
    points = [
        {"x": float((n % size) * cell), "y": float((n // size) * cell), "z": float(z)} for n in path
    ]
    semantic = {
        "schema_version": "dronedream.map-semantic.v1",
        "name": "Decision room tree",
        "coordinate_frame": "ENU",
        "collision_primitives": primitives,
        "runtime_collision_primitives": primitives,
        "vehicle_clearance": vehicle_clearance,
        "runtime_bindings": {
            "schema_version": "dronedream.map-runtime-bindings.v1",
            "simulator": "gazebo-harmonic",
            "coordinate_frame": "ENU",
            "mission_launch_waypoint": points[0],
            "vehicle_spawn": {"x": 0.0, "y": 0.0, "z": 0.209},
        },
        "qualification_granted": False,
    }
    route = {
        "schema_version": "dronedream.graph-route.v1",
        "start_node": "room-0",
        "goal_node": f"room-{goal}",
        "node_ids": [f"room-{n}" for n in path],
        "edge_ids": [f"door-{a}-{b}" for a, b in zip(path, path[1:], strict=False)],
        "positions_m": points,
        "route_length_m": (len(points) - 1) * cell,
        "all_edges_flight_verified": False,
    }
    return semantic, route, family


# 功能：重新生成语义地图并核对真实文件字节摘要，拒绝任意修改的独立布局声明。
# 输入：场景收据、语料地图摘要；输出：由拓扑重算的族身份，不接受自填family字符串。
def verified_scene_family(receipt, map_sha):
    if (
        receipt.get("scene_family") != "room-tree-generator-v1"
        or receipt.get("distinct_layout_claim") is not True
    ):
        raise ValueError("DECISION_LAYOUT_GENERATOR_NOT_VERIFIED")
    semantic, _, family = scene_documents(receipt["recipe"], receipt["vehicle_clearance"])
    actual = hashlib.sha256(encode_json(semantic).encode("utf-8")).hexdigest()
    if actual != map_sha or receipt.get("semantic_sha256") != map_sha:
        raise ValueError("DECISION_LAYOUT_GENERATOR_MAP_MISMATCH")
    return family
