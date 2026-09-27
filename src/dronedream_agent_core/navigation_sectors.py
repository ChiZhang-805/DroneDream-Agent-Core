"""Bounded numeric reduction of existing voxel evidence; never creates free space."""

import math
from itertools import islice

import numpy as np

SECTOR_BATCH_SIZE = 4096


# 功能：
#   分块筛选与局部球体相交的已有体素，球面附近按原标量距离复算，不生成空地证据。
# 输入：
#   members：已有体素键；minimum、resolution：已经验证的地图几何。
#   current、radius：球心与已包含体素外接球余量的半径。
# 输出：
#   selected：通过原距离条件的体素键集合。
def select_navigation_keys(members, minimum, resolution, current, radius):
    selected = set()
    iterator = iter(members)
    while batch := list(islice(iterator, SECTOR_BATCH_SIZE)):
        centers = (np.asarray(batch, dtype=np.float64) + .5) * resolution
        centers += minimum
        delta = centers - current
        # hypot 避免平方在大坐标范围溢出；临界值仍由 math.dist 最终裁决。
        distances = np.hypot(np.hypot(delta[:, 0], delta[:, 1]), delta[:, 2])
        boundary = np.flatnonzero(np.abs(distances - radius) <= max(1e-9, radius * 1e-12))
        for index in boundary:
            distances[index] = math.dist(current, tuple(centers[index]))
        selected.update(batch[index] for index in np.flatnonzero(distances <= radius))
    return selected


# 功能：
#   1. 分块汇总已有体素的相对方向、空地数量与最近障碍，不增加、删减或续期感知证据。
#   2. 半径和扇区分界附近回到标量计算，保持原先边界、银行家舍入及毫米显示规则。
#   3. 精确半径筛选后才计算方位，空地分组一次计数；不缓存旧帧结果或改变证据寿命。
# 输入：
#   classified：按原顺序提供体素键与分类；1 为已知空地，2 为占用。
#   minimum、resolution：地图最小坐标与分辨率，已经由地图入口验证。
#   current、heading、radius：已经验证的查询位置、目标方向与范围。
# 输出：
#   summaries：八个方向对应的空地数、占用数和最近障碍净空。
def reduce_navigation_sectors(classified, minimum, resolution, current, heading, radius):
    iterator = iter(classified)

    # 功能：
    #   将一次性分类流拆成原大小的不可变数字块，共用缓存路径的归约实现。
    # 输入：
    #   无：读取外层分类迭代器和地图几何。
    # 输出：
    #   block：每次迭代返回一个中心坐标与分类数组二元组。
    def blocks():
        while batch := list(islice(iterator, SECTOR_BATCH_SIZE)):
            yield pack_navigation_sector_block(batch, minimum, resolution)

    summaries = reduce_navigation_sector_blocks(blocks(), resolution, current, heading, radius)
    return summaries


# 功能：
#   将一个有界证据桶转换为不可写中心坐标和分类；只保留当前分类，不携带或更新观测时间。
# 输入：
#   classified：最多 4096 个体素键与分类；minimum、resolution：原地图几何。
# 输出：
#   block：字节缓冲区支持的只读中心数组和分类数组。
def pack_navigation_sector_block(classified, minimum, resolution):
    batch = list(islice(iter(classified), SECTOR_BATCH_SIZE + 1))
    if len(batch) > SECTOR_BATCH_SIZE:
        raise ValueError('NAVIGATION_SECTOR_BLOCK_CAPACITY')
    keys = np.asarray([item[0] for item in batch], dtype=np.float64).reshape(-1, 3)
    classes = np.asarray([item[1] for item in batch], dtype=np.int8)
    # 与未缓存路径保留相同浮点运算次序；不可变 bytes 阻止调用方重新打开写权限。
    centers = (keys + .5) * resolution
    centers += minimum
    block = (np.frombuffer(centers.tobytes(), dtype=np.float64).reshape(-1, 3),
             np.frombuffer(classes.tobytes(), dtype=np.int8))
    return block


# 功能：
#   按原顺序将零碎空间桶合成最多 4096 个体素的计算批次，避免每个小桶分别启动数值归约。
#   不改变缓存、分类或体素坐标；临时内存由单批容量限制，空桶不制造观测。
# 输入：
#   blocks：地图所有者提供的当前中心坐标和分类数组流。
# 输出：
#   batch：每次迭代返回一批中心坐标及对应分类，不跨调用保留结果。
def _coalesced_sector_blocks(blocks):
    centers_parts, class_parts = [], []
    count = 0
    for centers, classes in blocks:
        if centers.shape != (len(classes), 3) or classes.ndim != 1:
            raise ValueError("NAVIGATION_SECTOR_BLOCK_SHAPE_INVALID")
        offset = 0
        while offset < len(classes):
            taken = min(SECTOR_BATCH_SIZE - count, len(classes) - offset)
            centers_parts.append(centers[offset:offset + taken])
            class_parts.append(classes[offset:offset + taken])
            count += taken
            offset += taken
            if count == SECTOR_BATCH_SIZE:
                yield (centers_parts[0], class_parts[0]) if len(class_parts) == 1 else (
                    np.concatenate(centers_parts), np.concatenate(class_parts))
                centers_parts, class_parts, count = [], [], 0
    if count:
        yield (centers_parts[0], class_parts[0]) if len(class_parts) == 1 else (
            np.concatenate(centers_parts), np.concatenate(class_parts))


# 功能：
#   对当前分类的数字块重新计算相对位置、球面筛选及八方向统计，不缓存依赖位置的结论。
#   空间桶先按有界批次合并，边界标量复核和全部体素计数保持不变。
# 输入：
#   blocks：与当前地图分类一致的只读中心与分类数组流。
#   resolution、current、heading、radius：经地图入口校验的查询参数。
# 输出：
#   summaries：八个方向的空地数、占用数与最近障碍净空。
def reduce_navigation_sector_blocks(blocks, resolution, current, heading, radius):
    summaries = [[0, 0, None] for _ in range(8)]
    voxel_radius = math.sqrt(3.0) * resolution / 2.0
    for centers, classes in _coalesced_sector_blocks(blocks):
        if not len(classes):
            continue
        delta = centers - current
        distances = np.sqrt(np.sum(delta * delta, axis=1))
        boundary = np.flatnonzero(np.abs(distances - radius) <= max(1e-9, radius * 1e-12))
        for index in boundary:
            distances[index] = math.dist(current, tuple(centers[index]))
        eligible = np.flatnonzero(distances <= radius)
        if not eligible.size:
            continue
        # 球面边界已按原公式复算；球外体素不再参与三角函数和八个方向的计数。
        centers, delta = centers[eligible], delta[eligible]
        distances, classes = distances[eligible], classes[eligible]
        relative = (np.arctan2(delta[:, 1], delta[:, 0]) - heading + math.pi) % (2 * math.pi) - math.pi
        units = relative / (math.pi / 4)
        sectors = np.rint(units).astype(np.int8) % 8
        # atan2 的实现可在半格附近产生末位差异；只对这些点执行原公式。
        boundary = np.flatnonzero(np.abs(np.abs(units - np.rint(units)) - .5) <= 1e-10)
        for index in boundary:
            angle = (math.atan2(float(delta[index, 1]), float(delta[index, 0]))
                     - heading + math.pi) % (2 * math.pi) - math.pi
            sectors[index] = int(round(angle / (math.pi / 4))) % 8
        free_counts = np.bincount(sectors[classes == 1], minlength=8)
        occupied_indices = np.flatnonzero(classes == 2)
        occupied_sectors = sectors[occupied_indices]
        for sector in range(8):
            summaries[sector][0] += int(free_counts[sector])
            occupied = occupied_indices[occupied_sectors == sector]
            summaries[sector][1] += len(occupied)
            if occupied.size:
                approximate_minimum = float(np.min(distances[occupied]))
                # 近似最短的并列点全部复算，防止最后一位误差改变最近障碍或毫米舍入。
                nearest = occupied[distances[occupied] <= approximate_minimum + max(1e-9, approximate_minimum * 1e-12)]
                clearance = round(max(0., min(math.dist(current, tuple(centers[index]))
                                              for index in nearest) - voxel_radius), 3)
                previous = summaries[sector][2]
                summaries[sector][2] = clearance if previous is None else min(previous, clearance)
    return summaries
