import math
import random

import numpy as np
import pytest
from test_navigation_sectors import scalar_sectors

from dronedream_agent_core import navigation_sectors as sectors


# 功能：
#   构造跨多个桶的固定体素及空桶，不改变位置与分类，专门比较分桶方式的影响。
# 输入：
#   items：体素及分类；size：每桶数量；minimum、resolution：地图坐标约定。
# 输出：
#   blocks：同一体素序列的只读数字块列表。
def split_blocks(items, size, minimum, resolution):
    blocks = [sectors.pack_navigation_sector_block(items[i:i + size], minimum, resolution)
              for i in range(0, len(items), size)]
    blocks.insert(1, sectors.pack_navigation_sector_block([], minimum, resolution))
    return blocks


# 功能：
#   零碎桶、整批和尾批必须保持体素顺序、完整数量和最多 4096 个的内存边界。
# 输入：
#   size：原空间桶大小。
# 输出：
#   None：断言合并前后数字完全一致且原始缓存未被修改。
@pytest.mark.parametrize("size", [1, 7, 64, 4095, 4096])
def test_coalescing_preserves_all_values_and_bounded_batches(size):
    items = [((i, 0, 0), 1 + i % 2) for i in range(8201)]
    blocks = split_blocks(items, size, (-1.0, 0.0, 0.0), 0.25)
    before = [(centers.tobytes(), classes.tobytes()) for centers, classes in blocks]
    batches = list(sectors._coalesced_sector_blocks(iter(blocks)))
    assert [len(classes) for _, classes in batches] == [4096, 4096, 9]
    np.testing.assert_array_equal(np.concatenate([b[0] for b in batches]), np.concatenate([b[0] for b in blocks]))
    np.testing.assert_array_equal(np.concatenate([b[1] for b in batches]), np.concatenate([b[1] for b in blocks]))
    assert before == [(centers.tobytes(), classes.tobytes()) for centers, classes in blocks]


# 功能：
#   用独立标量公式及未合并的数值块双重对照，保证精确半径、方向、净空和统计均不漂移。
# 输入：
#   seed：可复现的几何种子；monkeypatch：临时禁用批次合并以对照旧路径。
# 输出：
#   None：断言不同位置和朝向的结果严格相等。
@pytest.mark.parametrize("seed", range(12))
def test_coalesced_reduction_matches_original_and_scalar(seed, monkeypatch):
    rng = random.Random(seed)
    items = [(tuple(rng.randrange(-32, 32) for _ in range(3)), rng.choice([1, 2])) for _ in range(8701)]
    minimum, resolution = (-0.5, -0.5, -0.5), 0.25
    current, heading, radius = (0.0, 0.0, 0.0), math.pi / 8, 8.0
    blocks = split_blocks(items, 17, minimum, resolution)
    actual = sectors.reduce_navigation_sector_blocks(iter(blocks), resolution, current, heading, radius)
    expected = scalar_sectors(items, minimum, resolution, current, heading, radius)
    assert actual == expected
    monkeypatch.setattr(sectors, "_coalesced_sector_blocks", lambda stream: stream)
    assert actual == sectors.reduce_navigation_sector_blocks(iter(blocks), resolution, current, heading, radius)


# 功能：
#   半径和方向半格附近的体素拆入不同小桶后仍执行原标量边界裁决，不因重新分批改变结果。
# 输入：
#   无：固定边界体素和相邻浮点数。
# 输出：
#   None：断言边界结果逐值一致。
def test_coalesced_boundaries_keep_original_rounding():
    items = [((x, y, z), 2) for x, y, z in [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (1, 1, 0), (-1, -1, 0)]]
    blocks = split_blocks(items, 1, (-0.5, -0.5, -0.5), 1.0)
    for angle in (-math.pi / 8, math.pi / 8, 7 * math.pi / 8):
        for heading in (math.nextafter(angle, -math.inf), angle, math.nextafter(angle, math.inf)):
            for radius in (math.nextafter(1.0, 0.0), 1.0, math.nextafter(1.0, 2.0), math.sqrt(2)):
                actual = sectors.reduce_navigation_sector_blocks(iter(blocks), 1.0, (0.0, 0.0, 0.0), heading, radius)
                assert actual == scalar_sectors(items, (-0.5, -0.5, -0.5), 1.0, (0.0, 0.0, 0.0), heading, radius)


# 功能：
#   空桶不创建额外点，损坏的中心和分类配对不能被截断为合法数据。
# 输入：
#   无：空流以及不匹配的数组形状。
# 输出：
#   None：断言空结果与输入拒绝行为。
def test_coalescing_empty_and_mismatched_blocks():
    assert list(sectors._coalesced_sector_blocks([])) == []
    with pytest.raises(ValueError, match="SHAPE_INVALID"):
        list(sectors._coalesced_sector_blocks([(np.zeros((2, 3)), np.ones(1, dtype=np.int8))]))
