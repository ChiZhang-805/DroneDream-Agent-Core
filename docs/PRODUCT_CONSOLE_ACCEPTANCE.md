# 产品命令行验收

`dronedream-console` 调用桌面共用的账户、任务和 Runtime HTTP 接口。它不启动桌面操作自动化，不调用开发教师飞行脚本，也不自动确认计划。真实模型请求可能消耗账户额度。

## 会话

由受信任的产品启动流程提供以下环境变量，或使用 `--session-stdin` 输入单行 JSON 会话：

- `DRONEDREAM_CONSOLE_CORE_URL`：固定本机 `http://127.0.0.1:端口`。
- `DRONEDREAM_CONSOLE_SUPABASE_URL`：账户所属 Supabase 项目 HTTPS 根地址。
- `DRONEDREAM_CONSOLE_LOCAL_TOKEN`：当前本地 Core 会话令牌。
- `DRONEDREAM_CONSOLE_IDENTITY_TOKEN`：当前账户身份令牌。
- `DRONEDREAM_CONSOLE_PUBLISHABLE_KEY`：项目公开客户端密钥。
- `DRONEDREAM_CONSOLE_EDITION`：所使用的产品版本，例如 `autonomy`。

不得把秘密放进命令行参数、提交到 Git，或打印进验收日志。命令不会搜索浏览器或桌面数据库来猜测登录凭据。没有有效会话时应完成产品授权，不能用开发 profile 冒充已登录产品用户。

## 同一任务的工作流

下面的尖括号是需要从实际响应中选择的值，不是可直接执行的字面参数。

```text
dronedream-console catalog
dronedream-console runtime
dronedream-console create --model <目录中的模型ID> --title 办公室取餐往返
dronedream-console prepare <thread-ID> --text "帮我去拿一下外卖，再带回办公室。" --map <地图ID> --map-sha256 <地图内容摘要> --vehicle <飞机ID> --vehicle-sha256 <飞机内容摘要>
dronedream-console status <thread-ID>
dronedream-console execute <thread-ID> --confirm-plan <确认过的plan-ID>
dronedream-console live-sources <thread-ID>
dronedream-console telemetry <thread-ID>
dronedream-console frame <thread-ID> --output <新建的PNG路径>
dronedream-console evidence <thread-ID>
```

`prepare` 不执行飞行。`execute` 会再次核对账户、最新计划修订、资源和运行准入；不得把旧计划 ID 自动替换为新计划。写请求超时表示结果未知，应先查询任务状态，不能直接重复提交计费或执行请求。

## 图像与实际动作

`frame` 保存共享 Runtime 的原始俯视 PNG，并报告文件摘要；不覆盖已有文件，不跟随 HTTP 重定向，拒绝损坏或越界媒体。这个视角用于观察场景，不是无人机前视模型输入。前视图仍须从同次运行的机载相机记录核对，并与来源时钟、动作回执关联。

下载成功不证明画面新鲜、没有碰撞或任务成功。验收还必须核对：真实模型调用、本地模型动作被飞控接纳、路线及取餐动作完成、碰撞与净空证据、返航及安全落地。仅出现路径、成功 HTTP 响应或好看的截图，均不足以认定完整飞行通过。

规划的路线采用经过地图与机体包络验证的有序航点和分段约束。本地模型根据实时观测决定连续速度、升降和偏航；航点不是电机指令，也不能代替局部避障和动作安全约束。
