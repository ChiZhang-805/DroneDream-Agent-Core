"""Single-writer offline training sessions, not a product model admission path."""

from __future__ import annotations

import hashlib
import math
import os
import time
from contextlib import contextmanager, suppress
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4
from zipfile import ZipFile

from dronedream_agent_core.plugin_files import check_plain_plugin_path, read_plugin_file
from dronedream_agent_core.runtime_control_io import publish_runtime_json, read_runtime_object
from dronedream_plugin_sdk.protocol import encode_json

MAX_CHECKPOINT_BYTES = 256 * 1024**2
MAX_SESSION_CHECKPOINT_BYTES = 8 * 1024**3
SESSION_SCHEMA = "dronedream.vision-training-session.v1"


class TrainingBudgetReached(RuntimeError):
    """A completed batch was saved; no model was exported or admitted."""


# 功能：
#   递归复制张量到 CPU，检查有限数值及总大小，不保存可执行的模型实例。
# 输入：
#   value：仅由张量、容器和基本类型组成的训练状态。
# 输出：
#   snapshot：独立 CPU 快照，后续训练不会改变其内容。
def cpu_snapshot(value):
    import torch

    budget = [0, 0]

    # 功能：
    #   逐节点复制状态树，限制深度、节点数和张量字节，拒绝任意对象。
    # 输入：
    #   item、depth：当前状态节点及递归深度。
    # 输出：
    #   copied：经预算和类型检查的节点副本。
    def copy_node(item, depth):
        budget[1] += 1
        if depth > 12 or budget[1] > 100_000:
            raise ValueError("VISION_CHECKPOINT_STRUCTURE_BUDGET_EXCEEDED")
        if isinstance(item, torch.Tensor):
            budget[0] += item.numel() * item.element_size()
            if (item.layout != torch.strided or budget[0] > MAX_CHECKPOINT_BYTES
                    or not torch.isfinite(item).all()):
                raise ValueError("VISION_CHECKPOINT_TENSOR_INVALID")
            copied = item.detach().cpu().clone()
        elif type(item) is dict:
            if any(type(key) not in (str, int) for key in item):
                raise ValueError("VISION_CHECKPOINT_KEY_INVALID")
            copied = {key: copy_node(child, depth + 1) for key, child in item.items()}
        elif type(item) in (list, tuple):
            copied = type(item)(copy_node(child, depth + 1) for child in item)
        elif item is None or type(item) in (str, bool, int, float):
            if isinstance(item, str) and len(item) > 16_384:
                raise ValueError("VISION_CHECKPOINT_STRING_OVERSIZED")
            if type(item) is float and not math.isfinite(item):
                raise ValueError("VISION_CHECKPOINT_NUMBER_NOT_FINITE")
            copied = item
        else:
            raise ValueError("VISION_CHECKPOINT_OBJECT_NOT_ALLOWED")
        return copied

    # state_dict 通常是 OrderedDict；顶层调用方将其显式转换为普通字典。
    snapshot = copy_node(value, 0)
    return snapshot


# 功能：
#   完整检查点先写入独占临时文件并刷盘，再无覆盖发布；清理仅限本次临时文件。
# 输入：
#   directory、payload：本次作业目录及已检查的 CPU 训练快照。
# 输出：
#   record：不可变检查点的文件名和完整摘要。
def write_checkpoint(directory: Path, payload: dict) -> dict:
    import torch

    check_plain_plugin_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"checkpoint-{uuid4().hex}.pt"
    temporary = None
    identity = None
    try:
        with NamedTemporaryFile(dir=directory, prefix=".checkpoint-", delete=False) as stream:
            temporary = Path(stream.name)
            identity = os.fstat(stream.fileno())
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        raw = read_plugin_file(temporary, limit=MAX_CHECKPOINT_BYTES)
        if not os.path.samestat(identity, temporary.stat()):
            raise ValueError("VISION_CHECKPOINT_STAGING_CHANGED")
        check_plain_plugin_path(destination)
        os.link(temporary, destination)
        record = {"file": destination.name, "sha256": hashlib.sha256(raw).hexdigest()}
        return record
    finally:
        if temporary is not None and identity is not None:
            with suppress(OSError, ValueError):
                check_plain_plugin_path(temporary)
                if os.path.samestat(identity, temporary.stat()):
                    temporary.unlink()


class VisionTrainingSession:
    """Explicit run identity, immutable recovery points, and held-out model selection."""

    # 功能：
    #   固定作业数据与配置身份，设置保存周期和有界训练时间；已有作业须显式恢复。
    # 输入：
    #   directory、binding：检查点目录及数据、架构、配置、初始化来源绑定。
    #   resume：是否读取该目录的最后完整检查点。
    #   checkpoint_batches、max_seconds、patience：保存间隔、批次边界预算和早停轮数。
    # 输出：
    #   None：初始化作业状态，不启动 GPU、训练或计费资源。
    def __init__(self, directory, binding, *, resume=False, checkpoint_batches=100,
                 max_seconds=3600.0, patience=5):
        if (type(resume) is not bool or type(checkpoint_batches) is not int
                or not 1 <= checkpoint_batches <= 10_000 or type(patience) is not int
                or not 1 <= patience <= 1000 or type(max_seconds) not in (int, float)
                or not math.isfinite(max_seconds) or not 1 <= max_seconds <= 604_800):
            raise ValueError("VISION_SESSION_OPTIONS_INVALID")
        self.directory = Path(directory).absolute()
        check_plain_plugin_path(self.directory)
        self.binding = deepcopy(binding)
        self.digest = hashlib.sha256(encode_json(binding).encode("utf-8")).hexdigest()
        self.resume = resume
        self.checkpoint_batches = checkpoint_batches
        self.max_seconds = max_seconds
        self.patience = patience
        self.started = time.monotonic()
        self.steps = 0
        self.next_epoch = 0
        self.next_batch = 0
        self.best_score = None
        self.best_state = None
        self.stale_epochs = 0
        self.history = []
        self.last_record = None
        if not resume and self.directory.exists() and any(self.directory.iterdir()):
            raise FileExistsError("VISION_SESSION_DIRECTORY_NOT_EMPTY")

    # 功能：
    #   持有操作系统级单写者锁，进程退出自动释放，防止重复启动互相覆盖恢复指针。
    # 输入：
    #   self：明确的新建或恢复作业。
    # 输出：
    #   self：在上下文期间独占写入的会话；竞争作业直接失败。
    @contextmanager
    def writer(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / ".writer.lock"
        check_plain_plugin_path(path)
        with path.open("a+b") as stream:
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                if not self.resume and (self.directory / "latest.json").exists():
                    raise FileExistsError("VISION_SESSION_ALREADY_STARTED")
                yield self
            finally:
                stream.seek(0)
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    # 功能：
    #   校验最后回执和完整检查点，恢复模型、优化器、随机状态和验证历史。
    # 输入：
    #   model、optimizer：与绑定配置一致的已构造网络和优化器。
    # 输出：
    #   None：就地恢复状态；不匹配或损坏时拒绝继续训练。
    def restore(self, model, optimizer):
        import torch

        if not self.resume:
            return
        record = read_runtime_object(self.directory / "latest.json")
        name = record.get("file")
        if (record.get("schema") != SESSION_SCHEMA or record.get("binding_sha256") != self.digest
                or type(name) is not str or Path(name).name != name
                or not name.startswith("checkpoint-") or not name.endswith(".pt")):
            raise ValueError("VISION_SESSION_BINDING_MISMATCH")
        raw = read_plugin_file(self.directory / name, limit=MAX_CHECKPOINT_BYTES)
        if hashlib.sha256(raw).hexdigest() != record.get("sha256"):
            raise ValueError("VISION_CHECKPOINT_HASH_MISMATCH")
        with ZipFile(BytesIO(raw)) as archive:
            entries = archive.infolist()
            if (len(entries) > 10_000
                    or sum(entry.file_size for entry in entries) > MAX_CHECKPOINT_BYTES):
                raise ValueError("VISION_CHECKPOINT_ARCHIVE_OVERSIZED")
        state = cpu_snapshot(torch.load(BytesIO(raw), map_location="cpu", weights_only=True))
        if state.get("schema") != SESSION_SCHEMA or state.get("binding") != self.binding:
            raise ValueError("VISION_SESSION_BINDING_MISMATCH")
        for key in ("next_epoch", "next_batch", "steps", "stale_epochs"):
            if type(state.get(key)) is not int or not 0 <= state[key] <= 1_000_000_000:
                raise ValueError("VISION_CHECKPOINT_PROGRESS_INVALID")
        if any(record.get(key) != state[key] or type(record.get(key)) is not int
               for key in ("next_epoch", "next_batch", "steps")):
            raise ValueError("VISION_CHECKPOINT_RECEIPT_PROGRESS_MISMATCH")
        history = state.get("history")
        if (type(history) is not list or len(history) != state["next_epoch"]
                or state["stale_epochs"] > len(history)
                or any(type(row) is not dict or type(row.get("epoch")) is not int
                       or row["epoch"] != index for index, row in enumerate(history))):
            raise ValueError("VISION_CHECKPOINT_VALIDATION_HISTORY_INVALID")
        from dronedream_agent_core.local_vision_training import LocalVisionTrainingMetrics

        scores = [LocalVisionTrainingMetrics.model_validate(row.get("metrics")).mean_loss
                  for row in history]
        score, best_state = state.get("best_score"), state.get("best_state")
        if (bool(scores) != (best_state is not None) or bool(scores) != (score is not None)
                or (scores and (type(score) not in (int, float)
                                or abs(min(scores) - score) > 1e-6))):
            raise ValueError("VISION_CHECKPOINT_BEST_STATE_INCONSISTENT")
        # 不修补缺少/多余层；新旧模型结构不匹配必须重新确认训练配置。
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng_cpu"])
        if state["rng_cuda"]:
            if not torch.cuda.is_available():
                raise ValueError("VISION_CHECKPOINT_CUDA_REQUIRED")
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        for key in ("next_epoch", "next_batch", "steps", "stale_epochs", "best_score",
                    "best_state", "history"):
            setattr(self, key, state[key])
        model.initialization_record = state["initialization"]
        self.last_record = record

    # 功能：
    #   将批次游标与实际数据批数、已完成优化步数对应，防止错误跳过训练窗口。
    # 输入：
    #   batch_count、epoch_count：当前配置的每轮批数和最大轮数。
    # 输出：
    #   None：恢复进度能精确对应当前数据时通过。
    def validate_progress(self, batch_count, epoch_count):
        if (self.next_epoch > epoch_count or self.next_batch > batch_count
                or (self.next_epoch == epoch_count and self.next_batch != 0)
                or self.steps != self.next_epoch * batch_count + self.next_batch):
            raise ValueError("LOCAL_VISION_CHECKPOINT_PROGRESS_OUT_OF_RANGE")

    # 功能：
    #   保存完整批次之后的可恢复状态，再发布最新指针，保留先前完整检查点。
    # 输入：
    #   model、optimizer、reason：训练状态及保存原因。
    # 输出：
    #   record：已发布检查点与作业身份、进度组成的回执。
    def save(self, model, optimizer, reason):
        import torch

        check_plain_plugin_path(self.directory)
        if self.directory.exists():
            existing_bytes = 0
            for path in self.directory.glob("checkpoint-*.pt"):
                check_plain_plugin_path(path)
                existing_bytes += path.stat().st_size
            if existing_bytes + MAX_CHECKPOINT_BYTES > MAX_SESSION_CHECKPOINT_BYTES:
                # 保留最近完整恢复点，不自动删除历史或冒险把磁盘写满。
                raise ValueError("VISION_SESSION_STORAGE_BUDGET_EXCEEDED")
        state = cpu_snapshot({
            "schema": SESSION_SCHEMA, "binding": self.binding,
            "model": dict(model.state_dict()), "optimizer": optimizer.state_dict(),
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [],
            "initialization": model.initialization_record,
            "next_epoch": self.next_epoch, "next_batch": self.next_batch, "steps": self.steps,
            "best_score": self.best_score, "best_state": self.best_state,
            "stale_epochs": self.stale_epochs, "history": self.history,
        })
        record = {**write_checkpoint(self.directory, state), "schema": SESSION_SCHEMA,
                  "binding_sha256": self.digest, "reason": reason, "steps": self.steps,
                  "next_epoch": self.next_epoch, "next_batch": self.next_batch,
                  "flight_qualification_granted": False}
        publish_runtime_json(self.directory / "latest.json", record)
        self.last_record = record
        return record

    # 功能：
    #   在已完成批次处更新恢复游标，到期保存；预算耗尽时保存后停止，禁止导出半成品。
    # 输入：
    #   model、optimizer、epoch、next_batch：已完成优化后的状态与下个批次位置。
    # 输出：
    #   None：更新内部状态；达到预算时抛出专用停止信号。
    def batch_completed(self, model, optimizer, epoch, next_batch):
        self.steps += 1
        self.next_epoch, self.next_batch = epoch, next_batch
        expired = time.monotonic() - self.started >= self.max_seconds
        if expired or self.steps % self.checkpoint_batches == 0:
            self.save(model, optimizer, "time-budget" if expired else "batch-interval")
        if expired:
            raise TrainingBudgetReached("VISION_TRAINING_BUDGET_SAVED")

    # 功能：
    #   使用验证集损失选择最佳轮次并执行早停；不使用最终测试集调参。
    # 输入：
    #   model、optimizer、epoch、metrics：完整轮次及独立验证指标。
    # 输出：
    #   stop：连续多个完整轮次没有改善时为真。
    def epoch_completed(self, model, optimizer, epoch, metrics):
        score = metrics.mean_loss
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError("VISION_VALIDATION_SCORE_INVALID")
        self.history.append({"epoch": epoch, "metrics": metrics.model_dump(mode="json")})
        if self.best_score is None or score < self.best_score - 1e-6:
            self.best_score = score
            self.best_state = cpu_snapshot(dict(model.state_dict()))
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        self.next_epoch, self.next_batch = epoch + 1, 0
        stop = self.stale_epochs >= self.patience
        self.save(model, optimizer, "early-stop" if stop else "epoch-complete")
        print(f"vision epoch={epoch + 1} loss={score:.6f} steps={self.steps}", flush=True)
        return stop

    # 功能：
    #   恢复验证集最优权重用于后续独立评估，训练恢复文件仍保留最后优化器状态。
    # 输入：
    #   model：训练结束的网络。
    # 输出：
    #   None：只替换网络权重，不将检查点注册为产品模型。
    def select_best(self, model):
        if self.best_state is None:
            raise ValueError("VISION_SESSION_HAS_NO_VALIDATED_EPOCH")
        model.load_state_dict(self.best_state, strict=True)
