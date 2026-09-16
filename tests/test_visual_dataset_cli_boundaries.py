"""Offline visual dataset boundaries; synthetic images and actual tiny ONNX graphs."""

import hashlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import onnxruntime as ort
import pytest
from PIL import Image
from test_local_visual_policy_dataset import _encoder

from dronedream_agent_core.local_policy_training import LocalPolicyTrainingSample
from scripts import encode_local_policy_visual_features as encoding


# 功能：
#   创建单线程真实测试会话，避免小图测试为每个用例占用整机线程池。
# 输入：
#   model：本测试固定的像素均值图路径。
# 输出：
#   session：只使用 CPU 的真实 ONNX 会话。
def _real_session(model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(model.read_bytes(), options, providers=["CPUExecutionProvider"])
    return session


# 功能：
#   创建白色 PNG、求像素均值的真实小图和单条已绑定样本，不模拟产品视觉识别能力。
# 输入：
#   tmp_path：测试输入输出根目录。
#   monkeypatch：设置离线编码命令行的工具。
# 输出：
#   case：固定来源路径、输出路径及原参数。
@pytest.fixture
def visual_case(tmp_path, monkeypatch):
    frames = tmp_path / "frames"
    frames.mkdir()
    frame = frames / "forward.png"
    with Image.new("RGB", (32, 32), "white") as picture:
        picture.save(frame)
    model = tmp_path / "encoder.onnx"
    _encoder(model)
    sample = LocalPolicyTrainingSample(
        source_snapshot_sha256="a" * 64,
        source_visual_sha256=hashlib.sha256(frame.read_bytes()).hexdigest(),
        state_features=[0.0] * 46,
        candidate_features=[[0.0] * 15 for _ in range(8)],
        candidate_mask=[1.0] + [0.0] * 7,
        target_action_index=0,
        risk_target=0.0,
    )
    source = tmp_path / "source.jsonl"
    source.write_text(sample.model_dump_json() + "\n", encoding="utf-8")
    output, receipt = tmp_path / "encoded.jsonl", tmp_path / "receipt.json"
    argv = [
        "encode",
        "--policy-data",
        str(source),
        "--frame-root",
        str(frames),
        "--perception-encoder",
        str(model),
        "--width",
        "32",
        "--height",
        "32",
        "--visual-feature-count",
        "1",
        "--normalization",
        "zero-to-one",
        "--output",
        str(output),
        "--receipt",
        str(receipt),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    case = SimpleNamespace(
        source=source,
        frames=frames,
        frame=frame,
        model=model,
        output=output,
        receipt=receipt,
        argv=argv,
    )
    return case


# 功能：
#   编码入口必须使用严格样本读取，不能跳过空白行或让最后一个重复字段覆盖原标签。
# 输入：
#   visual_case：有效单条训练数据和真实测试编码器。
#   kind：空白行或重复风险标签。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["blank", "duplicate"])
def test_encoding_rejects_ambiguous_samples(visual_case, kind):
    case = visual_case
    original = case.source.read_bytes()
    content = original + b"\n" if kind == "blank" else b'{"risk_target":1.0,' + original[1:]
    case.source.write_bytes(content)
    with pytest.raises(ValueError):
        encoding.main()
    assert not case.output.exists()


# 功能：
#   即使元素总数相同，也不能接受错误批次秩、布尔或字符串作为 float32 视觉向量。
# 输入：
#   visual_case：真实编码器及源图像。
#   monkeypatch：仅替换实际会话返回张量的工具。
#   output：后端非法输出。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "output", [np.ones((1, 1, 1), dtype=np.float32), np.ones((1,), dtype=bool), np.array(["1"])]
)
def test_encoding_rejects_wrong_tensor_shape_and_type(visual_case, monkeypatch, output):
    real = _real_session(visual_case.model)
    session = Mock(wraps=real)
    session.run.return_value = [output]
    monkeypatch.setattr(ort, "InferenceSession", Mock(return_value=session))
    with pytest.raises((ValueError, RuntimeError), match="OUTPUT_INVALID"):
        encoding.main()
    assert not visual_case.output.exists()


# 功能：
#   视觉来源目录先检查链接再规范化，不能把链接指向的外部目录当成直接来源。
# 输入：
#   visual_case：测试图像目录。
#   tmp_path：仅创建本测试可恢复链接的位置。
# 输出：
#   None：不返回业务数据。
def test_frame_index_rejects_root_link(visual_case, tmp_path):
    linked = tmp_path / "linked-frames"
    try:
        linked.symlink_to(visual_case.frames, target_is_directory=True)
    except OSError:
        pytest.skip("host does not permit creating this test symlink")
    with pytest.raises(ValueError, match="LINK"):
        encoding._frame_index([linked])


# 功能：
#   图像读取必须执行编码入口字节预算，不能在检查身份前无界读取。
# 输入：
#   visual_case：有效 PNG 和原摘要。
#   monkeypatch：缩小测试字节预算的工具。
# 输出：
#   None：不返回业务数据。
def test_tensor_enforces_file_byte_budget(visual_case, monkeypatch):
    monkeypatch.setattr(encoding, "MAXIMUM_IMAGE_BYTES", 8, raising=False)
    with pytest.raises((ValueError, RuntimeError)):
        encoding._tensor(visual_case.frame, width=32, height=32, normalization="zero-to-one")


# 功能：
#   零百分位必须返回最小观测，不能由负下标错误取到最大延迟。
# 输入：
#   无：使用三条确定毫秒观测。
# 输出：
#   None：不返回业务数据。
def test_visual_latency_zero_percentile_is_minimum():
    assert encoding._percentile([1.0, 2.0, 3.0], 0.0) == 1.0


# 功能：
#   百分位统计拒绝空样本、负值、布尔值和非有限数，不能把错误时钟写入编码证据。
# 输入：
#   values：非法延迟观测。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("values", [[], [-1.0], [True], [float("nan")]])
def test_visual_latency_rejects_invalid_evidence(values):
    with pytest.raises(ValueError):
        encoding._percentile(values, 99.0)


# 功能：
#   推理期间目标被其他操作创建时，入口必须保留其内容，不覆盖并宣告本次成功。
# 输入：
#   visual_case：尚未发布的离线编码任务。
#   monkeypatch：注入真实推理后的目标竞争。
#   target：最终数据或回执。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("target", ["output", "receipt"])
def test_encoding_preserves_competing_outputs(visual_case, monkeypatch, target):
    case = visual_case
    real = _real_session(case.model)

    # 功能：
    #   执行真实像素均值推理后，让另一方占用本次最终路径。
    # 输入：
    #   names：本次请求的输出名。
    #   inputs：实际预处理图像张量。
    # 输出：
    #   result：真实 ONNX 输出。
    def occupy_after_run(names, inputs):
        result = real.run(names, inputs)
        getattr(case, target).write_bytes(b"foreign content")
        return result

    session = Mock(wraps=real)
    session.run.side_effect = occupy_after_run
    monkeypatch.setattr(ort, "InferenceSession", Mock(return_value=session))
    with pytest.raises(FileExistsError):
        encoding.main()
    assert getattr(case, target).read_bytes() == b"foreign content"


# 功能：
#   推理后的来源文件变化不改变已读取字节的身份，回执必须绑定真正参与编码的样本。
# 输入：
#   visual_case：有效来源与输出。
#   monkeypatch：注入推理后来源变化的工具。
# 输出：
#   None：不返回业务数据。
def test_encoding_receipt_binds_consumed_source(visual_case, monkeypatch):
    case = visual_case
    original = case.source.read_bytes()
    real = _real_session(case.model)

    # 功能：
    #   实际编码后修改本测试来源，模拟后续同名文件已经不代表本次输入。
    # 输入：
    #   names：请求的视觉输出名。
    #   inputs：本次实际输入张量。
    # 输出：
    #   result：真实 ONNX 输出。
    def mutate_source(names, inputs):
        result = real.run(names, inputs)
        case.source.write_bytes(b"different source")
        return result

    session = Mock(wraps=real)
    session.run.side_effect = mutate_source
    monkeypatch.setattr(ort, "InferenceSession", Mock(return_value=session))
    assert encoding.main() == 0
    result = json.loads(case.receipt.read_bytes())
    assert result["source_policy_data_sha256"] == hashlib.sha256(original).hexdigest()


# 功能：
#   图中已声明的错误类型、维数或外部权重应在创建推理会话之前拒绝。
# 输入：
#   visual_case：本测试可修改的编码器图。
#   monkeypatch：监视是否创建会话的工具。
#   kind：错误静态尺寸、特征类型或外部初始化权重。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["input-size", "output-type", "external-weights"])
def test_encoding_validates_graph_before_session(visual_case, monkeypatch, kind):
    import onnx

    case = visual_case
    document = onnx.load_model_from_string(case.model.read_bytes())
    if kind == "input-size":
        document.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 33
    elif kind == "output-type":
        document.graph.output[0].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    else:
        weight = onnx.TensorProto(name="external", data_type=onnx.TensorProto.FLOAT, dims=[1])
        weight.data_location = onnx.TensorProto.EXTERNAL
        weight.external_data.add(key="location", value="not-bound.bin")
        document.graph.initializer.append(weight)
    case.model.write_bytes(document.SerializeToString())
    constructor = Mock(side_effect=AssertionError("invalid graph reached inference"))
    monkeypatch.setattr(ort, "InferenceSession", constructor)
    with pytest.raises(ValueError):
        encoding.main()
    constructor.assert_not_called()
    assert not case.output.exists()


# 功能：
#   无论 PNG 个数还是总扫描字节越界，都必须终止并显式报错。
# 输入：
#   visual_case：可读取的图像目录。
#   monkeypatch：将扫描预算缩到一张图以下的工具。
#   budget：条目或字节预算名称。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("budget", ["MAXIMUM_FRAME_ENTRIES", "MAXIMUM_FRAME_SCAN_BYTES"])
def test_frame_index_enforces_scan_budgets(visual_case, monkeypatch, budget):
    monkeypatch.setattr(encoding, budget, 0)
    with pytest.raises(ValueError, match="SCAN_LIMIT"):
        encoding._frame_index([visual_case.frames])


# 功能：
#   无法遍历图像目录时保留原错误，不把不完整扫描当成完整来源索引。
# 输入：
#   visual_case：有效原目录。
#   monkeypatch：模拟文件系统遍历异常的工具。
# 输出：
#   None：不返回业务数据。
def test_frame_index_does_not_hide_walk_errors(visual_case, monkeypatch):
    # 功能：
    #   通过 os.walk 错误回调模拟目录不可读。
    # 输入：
    #   root：扫描根目录。
    #   followlinks：原遍历是否允许跟随链接。
    #   onerror：调用方显式提供的异常处理器。
    # 输出：
    #   None：回调抛出原遍历异常。
    def unreadable_walk(root, *, followlinks, onerror):
        onerror(PermissionError("unreadable frame directory"))

    monkeypatch.setattr(encoding.os, "walk", unreadable_walk)
    with pytest.raises(PermissionError, match="unreadable"):
        encoding._frame_index([visual_case.frames])


# 功能：
#   数据和回执不能使用相同路径或互为父子路径，避免把应为文件的输出创建成目录。
# 输入：
#   visual_case：有效来源与待设置输出。
#   monkeypatch：设置冲突输出参数并监视推理的工具。
#   nesting：同一路径或任一方向的父子嵌套。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("nesting", ["same", "receipt-child", "data-child"])
def test_encoding_rejects_aliasing_outputs_before_session(visual_case, monkeypatch, nesting):
    case = visual_case
    argv = list(case.argv)
    if nesting == "same":
        argv[-1] = str(case.output)
    elif nesting == "receipt-child":
        argv[-1] = str(case.output / "receipt.json")
    else:
        argv[argv.index("--output") + 1] = str(case.receipt / "data.jsonl")
    monkeypatch.setattr(sys, "argv", argv)
    constructor = Mock(side_effect=AssertionError("aliasing outputs reached inference"))
    monkeypatch.setattr(ort, "InferenceSession", constructor)
    with pytest.raises(ValueError):
        encoding.main()
    constructor.assert_not_called()
    assert not case.output.exists() and not case.receipt.exists()
