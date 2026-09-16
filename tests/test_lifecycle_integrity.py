"""任务、计划和持久化身份必须成对匹配，损坏记录不能获得执行或替换资格。"""

import sqlite3

import pytest

from dronedream_agent_core.lifecycle import LifecycleTransitionError, MissionLifecycleStore


# 功能：
#   提供真实 SQLite 生命周期存储；每项用例结束关闭连接，不访问账户或飞控。
# 输入：
#   无。
# 输出：
#   store：独立内存数据库上的生命周期存储。
@pytest.fixture
def store():
    connection = sqlite3.connect(":memory:")
    try:
        yield MissionLifecycleStore(connection)
    finally:
        connection.close()


# 功能：
#   创建待用户确认的真实计划，统一后续身份篡改用例的有效基线。
# 输入：
#   store：目标生命周期存储。
#   conversation_id：独立任务的会话名称。
# 输出：
#   binding：已写入数据库的任务与计划绑定。
def _proposal(store, conversation_id="mission-a"):
    binding = store.record_plan_revision(
        conversation_id=conversation_id,
        contract_id="contract-a",
        prepared_mission_sha256="a" * 64,
        source_message_sha256="b" * 64,
    )
    return binding


# 功能：
#   拒绝内部身份不一致的边车，且失败不留下任何半条导入记录。
# 输入：
#   store：生成有效边车的原存储。
#   field：需要破坏的计划身份字段。
#   value：另一个格式合法但不属于此任务的身份。
# 输出：
#   None：断言异常及空数据库；不产生业务返回值。
@pytest.mark.parametrize(
    "field,value",
    [
        ("mission_id", "mission-" + "f" * 32),
        ("conversation_id", "different-conversation"),
        ("plan_revision_id", "plan-" + "f" * 32),
    ],
)
def test_sidecar_identity_must_match_before_import(store, field, value):
    binding = _proposal(store)
    setattr(binding.plan_revision, field, value)
    connection = sqlite3.connect(":memory:")
    try:
        destination = MissionLifecycleStore(connection)
        with pytest.raises(LifecycleTransitionError, match="BINDING"):
            destination.import_binding(binding)
        assert connection.execute("SELECT count(*) FROM task_threads").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM plan_revisions").fetchone()[0] == 0
    finally:
        connection.close()


# 功能：
#   拒绝计划状态与线程状态矛盾，以及没有执行身份的执行中边车。
# 输入：
#   store：有效计划的原存储。
#   state：待注入的线程状态。
#   status：待注入的计划状态。
# 输出：
#   None：要求导入失败且不写入矛盾状态。
@pytest.mark.parametrize(
    "state,status",
    [
        ("awaiting_confirmation", "executing"),
        ("executing", "executing"),
        ("completed", "failed"),
        ("planning", "proposed"),
    ],
)
def test_sidecar_states_must_match(store, state, status):
    binding = _proposal(store)
    binding.thread.state = state
    binding.plan_revision.status = status
    connection = sqlite3.connect(":memory:")
    try:
        destination = MissionLifecycleStore(connection)
        with pytest.raises(LifecycleTransitionError, match="BINDING"):
            destination.import_binding(binding)
    finally:
        connection.close()


# 功能：
#   验证数据库索引列与记录 JSON 的身份必须一致，不可按索引读取另一任务内容。
# 输入：
#   store：当前测试存储。
#   kind：损坏任务或计划的 JSON 身份。
# 输出：
#   None：读取损坏记录必须拒绝。
@pytest.mark.parametrize("kind", ["thread", "revision"])
def test_sql_identity_and_json_identity_must_match(store, kind):
    binding = _proposal(store)
    connection = store._connection
    if kind == "thread":
        binding.thread.conversation_id = "wrong-conversation"
        connection.execute(
            "UPDATE task_threads SET thread_json = ?", (binding.thread.model_dump_json(),)
        )
        connection.commit()
        with pytest.raises(LifecycleTransitionError, match="IDENTITY"):
            store.get_thread("mission-a")
    else:
        original_id = binding.plan_revision.plan_revision_id
        binding.plan_revision.revision = 9
        connection.execute(
            "UPDATE plan_revisions SET record_json = ?", (binding.plan_revision.model_dump_json(),)
        )
        connection.commit()
        with pytest.raises(LifecycleTransitionError, match="IDENTITY"):
            store.get_revision(original_id)


# 功能：
#   让本任务错误指向另一任务的合法计划，确认读取、重规划和结束执行均不能借用它。
# 输入：
#   store：隔离存储。
#   operation：读取绑定、发布替换计划或结束执行。
# 输出：
#   None：操作拒绝，另一任务仍为未确认状态。
@pytest.mark.parametrize("operation", ["binding", "replan", "finish"])
def test_current_revision_cannot_belong_to_another_task(store, operation):
    first = _proposal(store)
    second = _proposal(store, "mission-b")
    if operation == "finish":
        thread, _, execution_id = store.confirm_execution(
            conversation_id="mission-a",
            plan_revision_id=first.plan_revision.plan_revision_id,
            contract_id="contract-a",
            prepared_mission_sha256="a" * 64,
        )
    else:
        thread = first.thread
    thread.current_plan_revision_id = second.plan_revision.plan_revision_id
    store._connection.execute(
        "UPDATE task_threads SET thread_json = ? WHERE conversation_id = ?",
        (thread.model_dump_json(), "mission-a"),
    )
    store._connection.commit()
    with pytest.raises(LifecycleTransitionError, match="BINDING"):
        if operation == "binding":
            store.binding("mission-a")
        elif operation == "replan":
            _proposal(store)
        else:
            store.set_execution_state(
                conversation_id="mission-a", execution_id=execution_id, state="completed"
            )
    assert store.get_revision(second.plan_revision.plan_revision_id).status == "proposed"
    assert store.get_thread("mission-b") == second.thread


# 功能：
#   丢失当前计划时拒绝建立新计划，不能悄悄截断原任务的历史关系。
# 输入：
#   store：包含一个有效计划的数据库。
# 输出：
#   None：重规划失败后原任务身份不变。
def test_missing_parent_rejects_new_proposal(store):
    original = _proposal(store)
    store._connection.execute("DELETE FROM plan_revisions")
    store._connection.commit()
    with pytest.raises(LifecycleTransitionError, match="MISSING"):
        _proposal(store)
    assert store.get_thread("mission-a") == original.thread


# 功能：
#   正常边车导入返回独立快照，随后修改原输入不应改写返回的执行身份。
# 输入：
#   store：原始存储。
# 输出：
#   None：断言返回值和数据库均不随原输入变化。
def test_import_returns_detached_binding(store):
    binding = _proposal(store)
    imported = store.import_binding(binding)
    binding.plan_revision.contract_id = "mutated"
    assert imported.plan_revision.contract_id == "contract-a"
    assert store.binding("mission-a") == imported


# 功能：
#   初始化存储不能隐式提交调用方尚未完成的事务。
# 输入：
#   无。
# 输出：
#   None：要求初始化拒绝，并证明调用方仍能回滚。
def test_initialization_does_not_commit_owner_transaction():
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE TABLE owner_data (value INTEGER)")
        connection.execute("INSERT INTO owner_data VALUES (1)")
        with pytest.raises(LifecycleTransitionError, match="TRANSACTION"):
            MissionLifecycleStore(connection)
        assert connection.in_transaction
        connection.rollback()
        assert connection.execute("SELECT count(*) FROM owner_data").fetchone()[0] == 0
    finally:
        connection.close()
