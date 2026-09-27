# 竖直运动与出入口约束（2026-09-17）

## 当前链路

`原生定位／深度 → 同周期地图与负载背景 → 连续速度候选 → 三维碰撞预测 + VerticalMotionGuard → 再验收获选候选 → 短租期命令 → 执行器`

云端选择任务与路线策略，本地控制专家输出机体相对速度和偏航速度；安全层不会凭文字创造“安全高度”、改写模型目标或把摄像头没有覆盖的方向当成空地。运行入口是 `scripts/runtime_depth_safety_worker.py`，不是另一条演示路径。

| 条件 | 实现约束 |
| --- | --- |
| 室内、楼梯、不同楼层 | 采用世界 ENU 的机体碰撞中心；局部地面与顶面分别约束。离地高度不是相对起飞点高度。楼梯踏步、斜楼板及墙体仍进入完整扫掠碰撞检查。 |
| 门口、屋檐 | 顶部边界查询包含整个机体半径及净空。顶部空间收紧立即生效；放宽需连续新观测达到 0.4 秒，重复帧不能累计时间。允许先保持高度通过，再前进并爬升。 |
| 上下方未知 | 静态已资格化走廊与原始测距空体素分开处理。前者已有机体膨胀，后者检查整个机体包围范围；不能把单条射线当作整片可通行空间。当前命令获选后还会再次检查。 |
| 室外车辆、树枝、架空物 | 不是“室外统一升到某个高度”。静态几何、当前移动目标及停止扫掠都需允许；无法安全通过时拒绝动作，交回既有重规划机制。 |
| 携带负载 | 检查实测质量、最大总重、装载稳定状态、实际电压、姿态及执行器归一化余量。包络中包含负载后的保守碰撞尺寸、爬升／下降速度、指令加速度和独立制动下界；静态通道同时按扩大后的碰撞尺寸重算。 |
| 制动 | 反应延迟、感知年龄、延迟期间继续加速、当前动量与候选速度均计入；最大加速度不能替代制动下界。停止范围同时检查静态和移动目标的扫掠空间。超过有界预测时域则拒绝放行。 |

已替换 `known_map_planner._primitive_bounds` 内只处理水平旋转的旧实现；现在复用 `collision.primitive_bounds`，避免斜楼板的规划索引与碰撞计算不一致。公开名称仍保留，现有调用不需要转回旧公式。

## 身份、时效与解释背景

- `RuntimeLocalSafetyObservation.motion_context_sha256` 绑定本周期背景；`evaluate_runtime_local_safety` 强制匹配保护器、位置、几何与净空。
- 背景包括当前地图/路线及体素观测版本摘要、几何摘要、负载摘要和动力学包络摘要。地图版本计数只用于来源绑定，不当作可通行证明。
- 当前运行总是构造保护器；保留旧回执反序列化时省略空字段的兼容逻辑，不能借此让新运行跳过保护。
- 过期模型意图转入制动的分支同样携带保护器，不会在第二次计算中丢失背景。
- 背景同时进入战略地图、负载上下文和既有训练记录。未擅自改变冻结模型的张量宽度；它不代表旧权重已经学会新的负载控制。
- `hold` 仍是带惯性预测的保护请求，不承诺已经静止、已安全落地或不存在风险。

## 动力学资格的装载方式

运行入口增加 `--flight-dynamics-envelope <JSON>`。装载前核对原始机型 SHA-256 和适用域；仿真包络不能用于 `hardware`。

文件布局：

```text
flight-dynamics-envelope.json
evidence/
  <envelope.evidence_sha256>.json
  calibration-trajectory.jsonl
  independent-validation-trajectory.jsonl
```

验收文件必须包含 `schema_version: dronedream.flight-dynamics-qualification.v1`、`status: accepted`、`independent_validation_passed: true`，以及与包络逐字段一致的 `limits`（不含 `evidence_sha256`）。`measurement_sources` 各项只有 `file`、`sha256`、`split`；至少分别有 calibration 与 validation 来源，不能用同一文件内容冒充独立验证。文件名不得越出 evidence 目录；逐文件读取有大小和总量上限。

这些完整性检查**不会自动签发物理资格**。包络应来自通过独立验收的连续物理测量，维护者不得把任意数据配上 accepted 标签来绕过验收。本次没有生成伪造的产品动力学资格文件。没有对应负载测量包络时，已装载运动返回 `LOADED_MOTION_DYNAMICS_UNQUALIFIED`；空载仍沿用已有的机型预测限制，但不据此声称已得到实测制动保证。

首批数据沿用现有 `--development-payload-collection` 流程：必须是仅仿真准入、本地策略和多模态记录，运行结果明确不授予飞行资格。此模式可以暂缺待测的包络，但质量上限、装载稳定、新鲜遥测、执行器归一化、空间覆盖及碰撞否决仍有效；新增矢量速度上限 0.2 m/s、指令加速度上限 0.2 m/s²。它们是隔离实验的限制，不是实测安全保证。每周期记录 `measurement_only: true`，不能拿采集运行冒充产品验收。`hardware` 不接受此模式，也不接受缺失的动力学包络。

覆盖检查对静态配置空间额外计算延迟漂移的膨胀，不能因为基础通道已资格化就忽略制动新增的空间要求。

需要的测量数据：实际源时间、同一机型和载荷身份、重量、位姿和速度、执行器确认的控制量、动作后的轨迹、电压、姿态，以及可追溯的场景/试验分组。最低覆盖爬升、下降、水平制动、上升制动、下降制动、携载转弯和装卸稳定。按完整轨迹及场景划分校准与留出验证，不按相邻帧随机切分。坡面、屋檐和动态车辆场景也应覆盖。

## 验证边界

- 单元和集成用例位于 `tests/test_vertical_navigation.py`，另运行现有动态安全、地图、时效和运行入口回归。
- `scripts/verify_vertical_motion_gazebo.py` 和 `tests/fixtures/vertical_exit.sdf` 是隔离的 **Gazebo 速度接口组件探针**：输入真实 Gazebo 位姿，执行候选检查后的速度，验证屋檐退出和爬升。
- 该探针刻意关闭重力、使用测试速度提议，不调用云端或本地神经模型，不生成训练照片，不构成 PX4 闭环、负载动力学校准或真实任务验收。完整任务仍使用产品的既有执行链，不能把探针替换进软件。
- 数据集不变：原始 13,870 组；保留 13,714 组，训练 10,959／验证 1,130／测试 1,625。静态 RGB／语义照片不代替上述动力学数据。

实现参考：[PX4 碰撞预防的覆盖与延迟边界](https://docs.px4.io/main/en/computer_vision/collision_prevention)。探针使用 [Gazebo VelocityControl](https://gazebosim.org/api/sim/8/classgz_1_1sim_1_1systems_1_1VelocityControl.html) 与 [PosePublisher](https://gazebosim.org/api/sim/8/classgz_1_1sim_1_1systems_1_1PosePublisher.html)，不将它们的直接速度设置误认为动力学训练。
