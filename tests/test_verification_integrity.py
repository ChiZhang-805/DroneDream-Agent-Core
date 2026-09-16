"""验证义务必须绑定完整、合法且相互一致的输入，而不是仅散列任意内容。"""

import pytest
from test_mission_verification import _artifacts, _build

from dronedream_agent_core.verification import MissionVerificationPlanError


# 功能：
#   拒绝构造后被改写成非法结构的任务图和净空报告，即使相应摘要被重新计算。
# 输入：
#   defect：布尔净空、重复任务或非法坐标。
# 输出：
#   None：非法输入不能编译出可接受的验证计划。
@pytest.mark.parametrize("defect", ["clearance-bool", "duplicate-task", "nan-position"])
def test_verification_revalidates_mutable_inputs(defect):
    from dronedream_agent_core.hashing import sha256_json

    values = list(_artifacts())
    if defect == "clearance-bool":
        values[6] = values[6].model_copy(update={"accepted": 1})
    elif defect == "duplicate-task":
        values[2].nodes.append(values[2].nodes[0].model_copy(deep=True))
        values[9].task_graph_sha256 = sha256_json(values[2])
    else:
        values[7].points[0] = values[7].points[0].model_copy(update={"x": float("nan")})
    with pytest.raises(MissionVerificationPlanError):
        _build(values)


# 功能：
#   拒绝未授权的动作目录或用户没有授权的任务，防止模型借用新目录扩展权限。
# 输入：
#   defect：目录摘要或动作授权集合。
# 输出：
#   None：编译应拒绝未绑定权限。
@pytest.mark.parametrize("defect", ["catalog", "authorization"])
def test_verification_binds_contract_action_authority(defect):
    values = _artifacts()
    if defect == "catalog":
        values[0].action_catalog_sha256 = "f" * 64
    else:
        values[0].authorized_actions.remove("navigate")
    with pytest.raises(MissionVerificationPlanError):
        _build(values)


# 功能：
#   拒绝不属于实际飞行段的检查点，包括伪造索引、节点、任务和重复标识。
# 输入：
#   defect：选中的检查点绑定损坏方式。
# 输出：
#   None：不能把仅名称存在的检查点当作任务到达证据。
@pytest.mark.parametrize("defect", ["index", "node", "segment", "task", "duplicate"])
def test_verification_requires_checkpoint_geometry_and_identity(defect):
    values = _artifacts()
    checkpoint = values[8].checkpoints[0]
    if defect == "index":
        checkpoint.track_point_index = 999
    elif defect == "node":
        checkpoint.target_node = "office"
    elif defect == "segment":
        checkpoint.segment_id = "segment-002"
    elif defect == "task":
        values[8].checkpoints.append(
            checkpoint.model_copy(
                update={
                    "checkpoint_id": "checkpoint-003",
                    "task_id": "ghost",
                }
            )
        )
    else:
        values[8].checkpoints.append(checkpoint.model_copy(deep=True))
    with pytest.raises(MissionVerificationPlanError):
        _build(values)


# 功能：
#   拒绝额外飞行段、断裂的段内端点及飞行计划与净空验收路线不一致的几何。
# 输入：
#   defect：需要破坏的几何绑定。
# 输出：
#   None：验证计划编译必须拒绝不一致几何。
@pytest.mark.parametrize("defect", ["extra-segment", "endpoint", "route-position"])
def test_verification_rejects_unbound_flight_geometry(defect):
    values = _artifacts()
    if defect == "extra-segment":
        values[4].segments.append(
            values[4]
            .segments[0]
            .model_copy(
                update={
                    "segment_id": "segment-003",
                    "task_id": "ghost",
                }
            )
        )
    elif defect == "endpoint":
        values[4].segments[0].path[0].node_id = "wrong-origin"
    else:
        values[4].segments[0].path[1].position_m.x += 1
    with pytest.raises(MissionVerificationPlanError):
        _build(values)
