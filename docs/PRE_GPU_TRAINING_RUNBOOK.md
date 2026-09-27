# 租卡前准备与单卡训练操作

日期：2026-09-19。范围：当前公开 Core 工作树的训练准备；不是完整模型包或软件发布回执。

## 当前训练准备状态

首轮视觉训练已经完成，权重已下载，用户已停止 GPU。当前不自动重启或租用 Pod。逐批采集进度、失败记录、数据索引和候选对照保存在工作区的独立测试证据目录；本操作手册不混入不断叠加且互相覆盖的实验轮次说明。

- 正式视觉集为 **13,714 组**（原始 13,870），独立测试语义 mIoU 为 0.85417。
- 当前视觉权重 SHA-256 为 `52a476fc65481071a636931c689c1c6e72d76354330b863b4302f6f1b355e6ca`；所有因果控制视觉编码必须绑定此身份或重新明确冻结的新身份。
- 当前已核验控制语料为 **25,256 条实际动作／41,110 条观测／8,910 个完整窗口**。恢复窗口 153，其中训练 77、验证 43、测试 33；重叠窗口不能当作独立事件。新增留出运行仍属于预先划分的留出区域，不冒充新增地图。
- 当前因果专家每个 128,265 参数，本机 CPU 可完成小规模训练与导出对照；不能因照片数量增加就决定租卡。当前主要缺口仍是连续恢复覆盖和独立路线泛化。
- 保留 16 帧因果历史和不超过 250 ms 的来源间隔，不插帧、不改原始时间。缺图的数值历史可保留，但不生成视觉动作标签。
- 教师采集必须绑定实际执行回执，动态恢复必须绑定真实避让起因；最近物体、纯时效刹停或静态截图不能冒充恢复动作。
- 使用 `assemble_causal_control_data.py` 组装冻结分片，`train_causal_control_role.py` 训练单专家，`validate_causal_control_role.py` 核对实际 Torch／ONNX 输出及来源。三者均包含在当前上传清单中。
- 搬运已经组装的控制数据后，先以另存的组装回执摘要执行 `verify_causal_control_data.py --dataset DATASET --sha256 ASSEMBLY_SHA`。该入口无需 Torch，仅核验十份固定输入的实际字节，不访问原机器的绝对来源路径；文件完整不代表覆盖充分或模型合格。训练入口仍复核时间、视觉身份、分区隔离及实际张量。
- 先完成独立验证与候选评估；数据准备、教师飞行、离线误差和导出通过均不能代替学习策略的闭环验收。后续需要 GPU 时另行确定实际工作负载、卡型与预算。

### 固定控制训练配方

`training/control/baseline.json` 与 `training/control/regularization.json` 随包发布：16 帧、139 维视觉、128,265 参数、120 轮、batch 64、种子 805，路线均衡与 0.35 视觉块屏蔽。`seed-806.json`、`seed-807.json` 仅改变种子；`smoke.json` 仅将基准轮次减为一轮，用于先检查真实训练与导出链路，不作为正式候选。它们是预先声明的对照配置，不代表最优参数或飞行资格，不根据最终测试表现更换配方。

控制训练显式选择 `--device cpu` 或 `--device cuda`。完整数据与固定随机采样保留在 CPU，每次只搬运当前批次到 GPU；CUDA 不可用或实际内核探针失败时立即报错，不静默回退。训练结束后返回 CPU，进行验证、检查点保存与 ONNX 导出。训练入口不读取最终测试集合。本机 CPU 两线程可运行这些小策略；云端是否更快要看实际吞吐，不因增加模型数量就增加卡数。本机 CPU 验证不能替代租卡后的真实 CUDA 检查。

如需跨专家迁移，只使用 `--encoder-policy` 和配套 `--encoder-training-receipt`，与整模型 `--base-policy` 互斥。入口检查源角色、权重摘要、视觉/历史契约及全部祖先训练分组；仅复制传感器编码和时序层，目标专家的全部输出头重新初始化。新回执记录迁移源、实际复制键及累积训练分组，不能把旧导航模型改名成恢复模型。

## 固定边界

- 用户已停止首轮 GPU：不自动创建或重启 Pod，不发起付款，不猜测新的 SSH 地址，不修改已安装 Runtime。
- 下方单卡建议只保留为首轮历史方案。下一轮先通过控制数据就绪检查，再依据实际工作量说明卡型与预算；不让 GPU 空等数据准备。
- 静态相机渲染只供视觉监督，不能作为动作执行、负载响应或成功飞行证据。
- 控制仍是云端高层意图/规划＋本地连续专家＋Harness 约束＋独立飞控内环，不改成逐帧等待云端。
- 更新视觉权重后，必须用新视觉摘要重新编码控制训练数据、检查因果历史和重训/验证相关策略；不能只替换包内文件。

## 已实现的链路

1. `prepare_school_vision_source.py` 校验明确指定的 DDPKG，读取作者语义，逐可见面生成标签。未知类别拒绝，不默认当地面。
2. `capture_labelled_vision_views.py` 在隔离 Gazebo 分区移动无动力相机架。使用实际相机内参，RGB 与标签必须同一仿真时间，相机实测姿态稳定后才采集。
3. 保留原始 RGB、原始语义、派生样本、姿态、资源摘要和受控增强参数。曝光/模糊/遮挡增强不覆盖原始数据。遮挡后不继续监督隐藏的对象。
4. `assemble_vision_dataset.py --curate --cap-training-group-share 0.25` 汇总明确指定的完成批次，保存原图与逐视角来源、固定筛选决定及训练空间组损失权重。验证/测试不改权重。`verify_vision_dataset.py` 在上传后逐文件核对；`preflight_vision_data.py` 实际解码三个集合，检查文件摘要、语义覆盖、各辅助头正负样本、整任务/空间组隔离和图片查重，包括 224×128 模型输入尺寸下的可见类别。
5. `train_local_vision.py` 加载完整分割预训练权重，替换目标类别头。辅助任务经过 128 维嵌入反向传播，冻结阶段也固定骨干 BatchNorm。显式 CUDA 请求不回退 CPU。
6. 检查点保存模型、AdamW、CPU/CUDA 随机状态、完整批次游标、最佳验证权重和历史。恢复必须数据、代码、架构、超参数一致。单写者锁防止重复运行相互覆盖。
7. 验证集选择最佳轮次；`evaluate_frozen_vision.py` 用实际导出的 ONNX 在另一个测试集合上评估，不更新权重。该工具退出零只代表评估完成，不代表指标通过或飞行资格。
8. 现有因果策略、辅助专家、动作风险、DAgger/PPO 和完整包组装入口保留。因果控制策略支持显式 CPU/CUDA；这不表示其他辅助训练器自动获得 CUDA 支持。控制训练与视觉微调分开，不为每个专家各租一张卡。

动作模仿数据只使用 `build_executed_policy_dataset.py`：保留完整因果历史和飞控实际确认的动作。可选 `--test-run` 将预先留出的独立物理运行写为 `test.jsonl` 和 `test-observations.jsonl`；同时检查训练／验证、训练／测试、验证／测试三对来源，任何一对冲突都拒绝发布。测试集只用于冻结权重后的评估，不传入训练入口的 `--train` 或 `--validation`，不用于挑选训练轮次。旧 `build_local_policy_dataset.py` 的按行/旧候选标签提取工具不进入上传包；仓库中保留它用于历史证据回归测试，不能代替当前因果训练入口。

场景组标识必须在整份数据集内唯一命名。同一区域的不同光照、人物和增强版本保留同一组，不能通过改变世界摘要逃过跨集隔离。

## 当前数据扩充与来源

截至本轮正式组装回执：原始 13,870 组，筛选保留 13,714 组（训练 10,959、验证 1,130、测试 1,625），剔除 156 组且保留原始数据。下述视角计划数不能覆盖这一实际完成清点。

全图 UAV Corridor、高度软偏好、云端背景、实时上下文及真实空间视图契约见 `docs/UAV_PREFERRED_AIRSPACE.md`。空间查询不改变已冻结网络输入宽度，也不自动赋予旧控制权重新的能力。静态层与 Runtime 层障碍由规划和独立路线复核共同检查。

- 当前源码内 School Map：资产内容 `bc88b79228ec7ae4c974957b6b692d7941607885c5489fc1d9f5f925dba10f2d`；世界 `ef9d4e976dbd2b128f80aa2ff11f41a1cfffb8ccb58e763d206d9f9163ce7e8c`。
- 3,930 个可见面已对应作者语义：16 地面/起飞面，3,673 障碍，52 入口结构，84 楼梯，104 玻璃，1 取件标记。背景可来自天空；此静态地图没有行人模型。
- “入口”标签表示入口结构，不表示该像素前方可以直接飞行。地面/标记比例也不是障碍距离；距离、体积净空及制动仍由几何与安全链判断。
- `training/data/school-preview-views.json` 是六个视角的采集检查计划，不是训练集规模、独立测试集或完整任务成绩。
- 此前六张 School 预览和三张 T 姿态人物检查图仅作管线试验，明确排除本轮正式候选；不能沿用这九张图的数据就绪状态。
- 本轮固定 School Map 路线视角 8,238 张，整张地图训练专用；另有 36 个独立几何布局、4,608 张计划视角（20 个训练、8 个验证、8 个测试布局）。目录为 `Build/Training-Data-20260917/school-route-v1` 和 `native-scenarios-v2`。这些是采集计划数，最终完成数量及剔除数量须读取正式组装回执。
- 再加四个仅测试布局、512 张计划视角，使用训练/验证从未出现的第二个人物身份，目录 `identity-holdout-v1`。最终既报告原身份独立几何测试，也单独报告新身份测试，不能只看混合总分。
- 帧由原生 RGB 与同时间同位姿语义相机产生。严重过曝/欠曝只监督质量头，不让模型猜测不可见场景；遮挡区标为未知。普通模糊是明确记录的高斯扰动，不代表已覆盖全部真实运动模糊。
- 固定筛选政策去除完全重复 RGB 和无空间信息的近单色单类别图；原始文件和拒绝原因保留。模型尺寸每类至少 16 像素才计可见图片；最低 20 张、2 组只检查覆盖下限。
- School Map 数量多于其他地图，因此本轮组装限制单个训练空间组累计样本权重占比最多 25%，不复制少数样本、不改标签、不更换分组、不调整留出数据。它是预先声明的损失权重政策，不是已证明最佳的比例；实际图像采样顺序仍按原训练随机顺序。
- 历史最近五份非空 RGB 采集共 3,159 条记录，均缺少语义掩码。不能伪造标签或拿这些原料直接宣称分割训练已就绪。
- 本轮是静态视觉补齐；控制仍需连续物理仿真中的动作与后果，不能用静态渲染代替。第二个人物身份留出也不等于覆盖广泛人群、动态跟踪或真实相机成像。
- 预检最小样本数、每类像素数只是结构下限，不是统计可靠性保证。预检没有通过时不要开始长时间计费训练。

本轮进度和最终新数据路径写入 `Documents/Code-Handoff/TRAINING_DATA_EXPANSION_20260917.md`。只有实际完整采集、组装、传输校验和三集合预检回执均存在，才使用本轮数据开始长时间 GPU 训练；计划、部分 PNG 和成功的 CLI 帮助检查不能替代这些回执。

### 后续数据扩充清单

| 数据部分 | 需要补充的内容 | 不能替代的证据 |
| --- | --- | --- |
| 静态空间语义 | 多个独立区域的门口、楼梯、玻璃、取件标记，不同距离/朝向/光照 | 不能只复制同一取件点，改世界摘要后放到三个集合 |
| 人形与质量 | 不同身份/姿态/距离，保留有来源的曝光、模糊、遮挡正负样本 | T 姿态静态图不代表动态跟踪，也不代表真实相机所有模糊形式 |
| 连续控制 | 三个动作职责各自的正常动作、近障制动、精细机动与恢复；完整观测—动作—后果 | 静态相机架移动不作为无人机执行成功 |
| 负载与异常 | 已知仿真质量/惯量变化、传感器延迟/偏置/断流、正常对照 | 名义计算标签不称为物理实测或校准概率 |
| 独立评估 | 空间组/完整任务先分组，训练前固定最终测试集合 | 不根据最终测试表现反复选权重或调整门槛 |

先按这一清单扩充并复核数据。租卡前重复预检；数据缺口不通过增加 GPU 数量解决。采集计划扩展位置需要检查实际相机视图和几何，不能无条件沿网格穿墙生成“有效场景”。

## 权重及环境

使用官方 LRASPP MobileNetV3 Large COCO/VOC 分割预训练文件：

`lraspp_mobilenet_v3_large-d234d4ea.pth`

完整 SHA-256：`d234d4eae9d55d5f76de18b77cf0dc62c66fe5c5482758209d00f950c92bb280`，13,097,061 字节。

只将同义 `person` 分类行迁移到新类别表；其余不兼容分类行及自定义头需要训练。不能把 ImageNet 分类权重称为已经训练过我们的八类分割。

准备环境：Linux x86-64，Python 3.11–3.13，Torch 2.13.0、Torchvision 0.28.0、CUDA 12.6 wheel。版本来自[PyTorch 官方安装列表](https://pytorch.org/get-started/previous-versions/)；实际 Pod 的驱动、CUDA 可用性和依赖解算仍须登录后验证。

`requirements.txt` 固定主要依赖，不是跨平台传递依赖哈希锁。安装后执行 `pip check` 并保存 `pip freeze --all`；Windows CPU 环境的成功不冒充 Linux CUDA 成功。

预训练文件来自[官方 Torchvision 分割权重](https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth)。Torchvision 代码许可证与上游训练数据许可分别对待；该内部训练上传清单不授予将第三方权重改为 MIT 或公开再分发的权限。

当前自然站立人物来自 Gazebo Fuel：

- OpenRobotics [Standing person](https://fuel.gazebosim.org/1.0/OpenRobotics/models/Standing%20person)，第 3 版，作者 Marina Kollmitz，CC0；网格摘要 `bc20dd2cd005d70a35627e38cb52d4c84f3d66034b196efa28e11b10b3295c15`。
- 专用身份测试使用 [Casual female](https://fuel.gazebosim.org/1.0/OpenRobotics/models/Casual%20female)，第 4 版，作者 Rohit Salem，CC0；网格摘要 `b0f47228b4a1cfa5026f24f5955d5f2b7cc7f7a770888951041a787b3e51d994`。

完整资源元信息、许可和纹理摘要保存在每个场景源中；只使用数据资产，不运行下载模型中的插件。此前 Mingfei actor 的 CC BY 4.0 原始试验资源仍保留归属，但不进入本轮正式候选。第三方资源不自动作为项目 MIT 代码再授权。

训练 bundle 使用显式文件白名单，并逐字节比对 wheel 中的所有 Core Python 模块与当前源码。冻结后再次修改源码必须重新构建新 bundle，不能沿用旧 wheel。数据汇总独立于代码 bundle，不把用于链路测试的少量样本自动作为正式数据上传。

## 用户租卡后

先取得 Pod 的 SSH 主机、端口、用户名及其展示的连接方式。核对主机指纹；使用用户已有公钥，私钥和 API Key 不写日志、不打包。

上传单独准备的训练 bundle 和通过预检的数据目录。不要上传整个工作区、账户数据库、浏览器资料、`.env`、Supabase 密钥或历史聊天。

### 当前控制专家阶段

当前阶段使用已冻结的视觉特征，不自动重新训练视觉模型。三个动作专家顺序使用一张卡即可，先用一轮 `smoke` 实际训练验证，再按固定三种种子运行；每个角色、配方及迁移方案使用不同的新目录。示例中的路径与摘要必须取本次独立交接清单：

```bash
python3 -I -B BUNDLE/scripts/verify_training_bundle.py --bundle BUNDLE --sha256 MANIFEST_SHA
bash BUNDLE/training/cloud/bootstrap.sh BUNDLE MANIFEST_SHA ENVIRONMENT
ENVIRONMENT/venv/bin/python -B BUNDLE/scripts/verify_causal_control_data.py \
  --dataset CONTROL_DATA --sha256 CONTROL_ASSEMBLY_SHA
bash BUNDLE/training/cloud/run_control.sh \
  BUNDLE CONTROL_DATA NEW_SMOKE_RUN ENVIRONMENT MANIFEST_SHA CONTROL_ASSEMBLY_SHA \
  local-navigation-policy smoke
bash BUNDLE/training/cloud/run_control.sh \
  BUNDLE CONTROL_DATA NEW_RECOVERY_RUN ENVIRONMENT MANIFEST_SHA CONTROL_ASSEMBLY_SHA \
  recovery-policy baseline NAVIGATION_ENCODER
```

`NAVIGATION_ENCODER` 只包含原始 `local-navigation-policy.pt` 和绑定的 `training-receipt.json`。不提供该目录即为独立初始化；精细/恢复专家可以迁移编码器，导航专家不能在此入口再次导入导航编码器。所有迁移均执行前述契约与祖先分组检查。

每个控制作业最多训练 3,600 秒，随后导出核验最多 600 秒。控制训练目前不支持中途保存优化器并续训：超时作业保留日志和部分文件，不作为成功候选；重试必须使用新目录。每次只运行一个作业，不自动续跑、不修改产品包、不停止 Pod 计费。先测一轮实际耗时，再确认完整作业适合此上限。

当前动作语料只支持已执行的飞行控制标签；没有正例的风险/拒绝动作不能因为输出头存在就声称已经学会。独立安全约束继续负责未获资格的分支。最终测试只在配置与候选选择冻结后执行，不用于反复选权重。

候选选定后先在独立交接记录保存其 `training-receipt.json` 摘要，再使用 `evaluate_frozen_causal_role.py --candidate CANDIDATE --dataset CONTROL_DATA --training-receipt-sha256 FROZEN_RECEIPT_SHA --assembly-sha256 CONTROL_ASSEMBLY_SHA --output NEW_TEST_REPORT`。入口绑定实际检查点和 ONNX、原始训练数据、最终测试视觉身份，检查当前及祖先训练/调参分组，报告明确标为 `test`。该入口不参与 `run_control.sh`，不能为了挑选候选而提前运行。

### 已完成视觉阶段的复现入口

以下仅用于有明确需要时复现首轮视觉训练，不是当前控制准备的必跑步骤：

```bash
# MANIFEST_SHA、DATA_SHA 分别使用上传前另外保存的代码清单和数据组装回执摘要。
python3 -I -B BUNDLE/scripts/verify_training_bundle.py --bundle BUNDLE --sha256 MANIFEST_SHA
bash BUNDLE/training/cloud/bootstrap.sh BUNDLE MANIFEST_SHA ENVIRONMENT
ENVIRONMENT/venv/bin/python -B BUNDLE/scripts/verify_vision_dataset.py \
  --dataset DATASET --sha256 DATA_SHA
ENVIRONMENT/venv/bin/python -B BUNDLE/scripts/probe_training_device.py \
  --device cuda --batch-size 16 --steps 10 \
  --weights BUNDLE/weights/lraspp_mobilenet_v3_large-d234d4ea.pth \
  --sha256 d234d4eae9d55d5f76de18b77cf0dc62c66fe5c5482758209d00f950c92bb280 \
  --receipt GPU_PROBE.json
bash BUNDLE/training/cloud/run_vision.sh BUNDLE DATASET NEW_RUN ENVIRONMENT MANIFEST_SHA DATA_SHA
# 明确恢复同一数据/配置/实现，不在另一个版本上续用优化器。
bash BUNDLE/training/cloud/run_vision.sh BUNDLE DATASET NEW_RUN ENVIRONMENT MANIFEST_SHA DATA_SHA resume
```

先看显存余量、实际训练步吞吐、数据等待比例与预算，再启动后续轮次。合成随机输入探针不包括数据加载耗时，也不是精度评估。调整 batch size 会改变检查点绑定，需要新作业。

BUNDLE 保持只读输入用途；虚拟环境、缓存、数据、作业输出使用互不嵌套的独立目录。安装前及每次训练前重新核对完整上传目录，清单外旧代码会被拒绝；训练前还逐模块比对已安装 Core 与本次 wheel，不能选旧环境运行新脚本。使用 `-B` 防止在上传脚本目录生成缓存。视觉重编码与策略离线评估入口也包含在准备包中。

视觉训练默认在完整批次边界最多约一小时后保存退出，或验证早停；不自动续跑。退出码 2 表示已保存暂停，不是成功模型。一个视觉作业检查点累计预算 8 GiB，到达前停止，不自动删除旧状态。需要搬走历史恢复点时先备份，保持 latest.json 对应文件与最佳状态完整。此恢复机制不适用于上面的控制作业。

正式云配方每 500 批保存一次，每轮结束及时间预算到期另外保存。按 10,959 张训练图、batch 16、30 轮计算，共 20,550 步，常规保存最多 71 份；含已分配 AdamW 和最佳权重的实测单份约 54.0 MB，约占 3.84 GB，另留暂停恢复余量。旧 100 批间隔可能产生 235 份、约 12.7 GB，超过本作业 8 GiB 上限，因此不再用于正式云配方。此估计不是无限次暂停的容量保证，磁盘上限仍保持生效。

导出前以三种固定输入比较 PyTorch 与 ONNX 的全部五个输出，拒绝非有限、维度错误或数值不一致的产物。关键类别 IoU 门槛包含可通行面和普通障碍，不能只检查特殊目标。`local-vision-training` 安装项明确包含 ONNX Runtime，以支持导出数值复核及部署时延测量。

**训练进程退出不会停止 RunPod 计费。**检查点/回执/模型下载并校验后，再按用户授权停止或终止 Pod；终止前核对持久卷保留方式。详见[RunPod 连接说明](https://docs.runpod.io/pods/connect-to-a-pod)和[存储说明](https://docs.runpod.io/pods/storage/types)。

## 冻结与交付顺序

视觉冻结 → 独立测试与部署硬件时延 → 新视觉重编码 → 三个控制专家及辅助/风险训练 → 契约与来源绑定 → 完整包组装/运行端检查 → 仿真闭环 → 构建/签名/更新渠道 → 人工软件验收。

GPU 训练、代码回归、ONNX 导出、飞行成绩、安装签名是不同证据；不得相互替代。当前准备包不包含可立即发布的产品权重，也不改变当前 EXE。
