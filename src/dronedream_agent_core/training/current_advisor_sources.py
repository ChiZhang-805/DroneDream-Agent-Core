"""Read current executed-teacher observations for offline advisor supervision.

The original recorder bytes remain unchanged. This adapter grants neither
model admission nor permission to treat a teacher trajectory as learned flight.
"""

import hashlib
from pathlib import Path

from .advisor_sources import RecordedSource
from .demonstrations import collect_demonstrations


# 功能：
#   1. 复用当前示范读取器，核验物理运行、教师协议、真实执行回执及空间分组。
#   2. 按核验时的摘要冻结原始观测，不转换或伪造历史模型调用记录。
# 输入：
#   root：包含当前 learning-observations 的仿真运行目录。
# 输出：
#   receipt：绑定当前来源与空间分组的辅助专家数据来源记录。
#   observations：按已核验摘要冻结的原始观测字节。
def read_current_advisor_source(root: Path) -> tuple[dict, RecordedSource]:
    corpus = collect_demonstrations([root], require_visual=False, allow_nonvisual_history=True)
    source = corpus.receipts[0]
    observations = RecordedSource.read(root / "learning-observations.jsonl")
    digest = hashlib.sha256(observations.content).hexdigest()
    if digest != source["files"]["observations"]:
        raise ValueError("ADVISOR_CURRENT_OBSERVATIONS_CHANGED_AFTER_VALIDATION")
    receipt = {
        "run_directory": str(observations.path.parent),
        "source_kind": "executed-teacher-learning-observations",
        "mission_status": "verified",
        "mission_evidence_sha256": source["mission_evidence_sha256"],
        "snapshot_file_sha256": digest,
        "cycle_file_sha256": None,
        "depth_safety_history_sha256": None,
        "multimodal_dataset_records_sha256": None,
        "semantic_sha256": source["mission_split"]["semantic_sha256"],
        "mission_split": source["mission_split"],
        "verification": "verified-executed-teacher-source",
        "teacher_contract_sha256": source["teacher_contract_sha256"],
        "verified_source_files_sha256": source["files"],
        "flight_qualification_granted": False,
    }
    return receipt, observations
