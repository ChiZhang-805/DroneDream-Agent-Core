"""Conservative, bounded geometry from measured occupied cells.

These helpers never declare space free. Exact adjacent-cell merging reduces
work; if the output budget is still exceeded, a labelled bounding volume
covers the remainder rather than discarding it. Coarsening may stop motion.
"""

import math
from collections import defaultdict
from collections.abc import Mapping

import numpy as np

Cell = tuple[int, int, int]
CellBox = tuple[int, int, int, int, int, int]


# 功能：
#   提取可用于包含证明的有限水平旋转盒；未知形状、倾斜或非法数值均不参与证据删除。
# 输入：
#   box：待解析的几何映射。
# 输出：
#   result：中心、尺寸、偏航角组成的数值行；不能证明时为 None。
def _box_row(box: Mapping) -> tuple[float, ...] | None:
    result = None
    fields = ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z")
    try:
        if not isinstance(box, Mapping) or box.get("shape", "box") != "box":
            return result
        raw = tuple(box[key] for key in fields) + tuple(
            box.get(key, 0.0) for key in ("yaw_rad", "roll_rad", "pitch_rad"))
        if any(type(value) not in (int, float) for value in raw):
            return result
        row = tuple(float(value) for value in raw[:6])
        yaw, roll, pitch = (float(box.get(key, 0.0)) for key in
                            ("yaw_rad", "roll_rad", "pitch_rad"))
        if (not all(math.isfinite(value) for value in (*row, yaw, roll, pitch))
                or any(size <= 0.0 for size in row[3:]) or roll != 0.0 or pitch != 0.0):
            return result
    except (KeyError, TypeError, ValueError, OverflowError):
        return result
    result = (*row, yaw)
    return result


# 功能：
#   证明候选盒完全位于外盒内部；按外盒坐标投影所有角点并向内留量，接触边界不删证据。
# 输入：
#   candidate：拟检查冗余的候选盒。
#   enclosing：已有的包围盒。
# 输出：
#   contained：能够证明完整包含时为 True，否则为 False。
def box_fully_inside(candidate: Mapping, enclosing: Mapping) -> bool:
    contained = False
    inner, outer = _box_row(candidate), _box_row(enclosing)
    if inner is None or outer is None:
        return contained
    dx, dy, dz = (inner[axis] - outer[axis] for axis in range(3))
    cosine, sine = math.cos(outer[6]), math.sin(outer[6])
    local_x, local_y = cosine * dx + sine * dy, -sine * dx + cosine * dy
    relative_yaw = inner[6] - outer[6]
    if not math.isfinite(relative_yaw):
        return contained
    cos_delta, sin_delta = abs(math.cos(relative_yaw)), abs(math.sin(relative_yaw))
    projected_x = (cos_delta * inner[3] + sin_delta * inner[4]) / 2.0
    projected_y = (sin_delta * inner[3] + cos_delta * inner[4]) / 2.0
    guard = 1e-9
    contained = (abs(local_x) + projected_x + guard <= outer[3] / 2.0
            and abs(local_y) + projected_y + guard <= outer[4] / 2.0
            and abs(dz) + inner[5] / 2.0 + guard <= outer[5] / 2.0)
    return contained


# 功能：
#   1. 按固定大小的双向分块筛选包含候选，避免构造无界的全量两两比较张量。
#   2. 每次删除仍使用标量完整包含证明；粗筛漏选只保留额外证据，不生成自由空间。
# 输入：
#   candidates：需要检查的候选几何列表。
#   enclosing：已知外包围几何列表。
# 输出：
#   removed：与候选顺序一致的冗余标记。
def fully_contained_box_mask(candidates: list[Mapping], enclosing: list[Mapping]) -> list[bool]:
    inner_rows = [(index, row) for index, box in enumerate(candidates)
                  if (row := _box_row(box)) is not None]
    outer_rows = [(index, row) for index, box in enumerate(enclosing)
                  if (row := _box_row(box)) is not None]
    removed = [False] * len(candidates)
    for first in range(0, len(inner_rows), 96):
        inner_block = inner_rows[first:first + 96]
        inner = np.asarray([row for _, row in inner_block], dtype=np.float64)[:, None, :]
        for second in range(0, len(outer_rows), 128):
            outer_block = outer_rows[second:second + 128]
            outer = np.asarray([row for _, row in outer_block], dtype=np.float64)[None, :, :]
            # Finite inputs may still overflow subtraction at extreme numeric
            # magnitudes; those unprovable pairs must simply remain present.
            with np.errstate(over="ignore", invalid="ignore"):
                delta = inner[:, :, :3] - outer[:, :, :3]
                cosine, sine = np.cos(outer[:, :, 6]), np.sin(outer[:, :, 6])
                x = cosine * delta[:, :, 0] + sine * delta[:, :, 1]
                y = -sine * delta[:, :, 0] + cosine * delta[:, :, 1]
                yaw = inner[:, :, 6] - outer[:, :, 6]
                c, s = np.abs(np.cos(yaw)), np.abs(np.sin(yaw))
                broad = ((np.abs(x) + (c * inner[:, :, 3] + s * inner[:, :, 4]) / 2
                          <= outer[:, :, 3] / 2)
                         & (np.abs(y) + (s * inner[:, :, 3] + c * inner[:, :, 4]) / 2
                            <= outer[:, :, 4] / 2)
                         & (np.abs(delta[:, :, 2]) + inner[:, :, 5] / 2
                            <= outer[:, :, 5] / 2))
            for i, j in zip(*np.nonzero(broad), strict=True):
                candidate_index, enclosing_index = inner_block[i][0], outer_block[j][0]
                if not removed[candidate_index] and box_fully_inside(
                    candidates[candidate_index], enclosing[enclosing_index]
                ):
                    removed[candidate_index] = True
    return removed


# 功能：
#   将相邻整数占用格精确合并为半开区间盒，不跨越孔洞、角接触或对角接触填充空间。
# 输入：
#   keys：可由双精度数精确表示边界的三维整数格索引列表。
# 输出：
#   result：顺序稳定、并集与输入格完全相同的盒列表。
def merged_occupied_cell_boxes(keys: list[Cell]) -> list[CellBox]:
    # +1 后的上边界也必须精确可表示，不能静默截断分数格或把 True 当成格索引。
    if any(not isinstance(key, tuple) or len(key) != 3 or any(
        type(value) is not int or abs(value) > 2**53 - 2 for value in key
    ) for key in keys):
        raise ValueError("occupied cell index must contain three representable integers")
    boxes = [(*key, *(value + 1 for value in key)) for key in sorted(set(keys))]
    for axis in range(3):
        groups = defaultdict(list)
        others = [index for index in range(6) if index not in (axis, axis + 3)]
        for box in boxes:
            groups[tuple(box[index] for index in others)].append(box)
        merged = []
        for group in groups.values():
            pending = None
            for box in sorted(group, key=lambda item: item[axis]):
                if pending is not None and pending[axis + 3] == box[axis]:
                    pending[axis + 3] = box[axis + 3]
                else:
                    if pending is not None:
                        merged.append(tuple(pending))
                    pending = list(box)
            if pending is not None:
                merged.append(tuple(pending))
        boxes = merged
    result = sorted(boxes)
    return result


# 功能：
#   优先保留近处精确占用盒，数量超限时用一个明确标记的粗包围体覆盖全部剩余格。
# 输入：
#   keys：三维整数占用格索引。
#   center_in_cells：用于排序的查询中心，单位为格。
#   limit：允许返回的最大盒数量。
# 输出：
#   result：占用盒及是否被保守粗化的配对列表。
def bounded_occupied_cell_boxes(
    keys: list[Cell], *, center_in_cells: tuple[float, float, float], limit: int,
) -> list[tuple[CellBox, bool]]:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("local occupancy primitive limit must be a positive integer")
    try:
        if len(center_in_cells) != 3 or any(type(x) not in (int, float)
                or not math.isfinite(x) for x in center_in_cells):
            raise ValueError("local occupancy center must be finite")
    except (TypeError, OverflowError) as error:
        raise ValueError("local occupancy center must be finite") from error

    # 功能：
    #   计算查询中心到闭包围体的最短距离，使用缩放范数避免平方中间量溢出。
    # 输入：
    #   box：已验证的整数格包围体。
    # 输出：
    #   distance：用于确定性排序的非负距离。
    def distance_to_box(box: CellBox) -> float:
        distance = math.hypot(*(max(box[axis] - center_in_cells[axis], 0.0,
                                   center_in_cells[axis] - box[axis + 3]) for axis in range(3)))
        return distance

    boxes = sorted(merged_occupied_cell_boxes(keys), key=lambda box: (distance_to_box(box), box))
    if len(boxes) <= limit:
        result = [(box, False) for box in boxes]
        return result
    retained, remainder = boxes[:limit - 1], boxes[limit - 1:]
    overflow = tuple(min(box[axis] for box in remainder) for axis in range(3)) + tuple(
        max(box[axis + 3] for box in remainder) for axis in range(3))
    # 粗包围体可能包含空区域，但不能删掉真实占用格，也不能伪称它是精确重建。
    result = [*((box, False) for box in retained), (overflow, True)]
    return result
