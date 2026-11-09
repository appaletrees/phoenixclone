# Piper 常驻策略推理

`tools/run_piper_policy.py` 是独立的部署入口，提供同步动作块、异步动作块和推理期 RTC 三种模式。旧入口 `tools/run_piper_inference.py` 及其默认行为保留，旧命令不自动切换到这里。

截至 2026-10-28，训练产物已恢复到 `postcheckpoints/piper_clean_table_lerobot_v3` 和 `postcheckpoints/piper_set_placemat_lerobot_v3`。`--run-dir` 必须指向含匹配 `config.yaml`、`dataset_stats.json` 和 checkpoint 的具体训练目录；冻结视觉编码器和任务文本缓存也必须可用。只复制一个 `.pt` 不能替代对应的运行配置与归一化；资源缺失时入口会在连接硬件前预检失败。

所有相对路径都相对 `--project-root`；该参数默认 `.`，即启动命令时的当前目录。下面在项目根目录运行：

```bash
cd /home/midea/phoenix-mwa
```

重启或重插 USB 后，CAN 名称和 UP 状态可能复位。当前配置通过 `can_usb_serials` 将左臂绑定到 `001900305631511920313857`、右臂绑定到 `003100324148570C20343133`；即使接口叫 `can0/can1` 或名字交换，读取和控制仍按物理身份对应，反馈 ID 偏移也跟随对应的机械臂。入口在加载模型前只读检查设备是否存在及接口是否 UP，并打印 `PIPER_CAN_READY`。机器人连接前会再次检查，配置错误不会等到加载大模型后才发现。

如果接口处于 DOWN，需要先在本机终端初始化 CAN：

```bash
sudo /usr/bin/python3 ../Desktop/piper_tele/sink-bot-control_bk/example_agilex/pi05_checkpoint_inference/activate_hardware_can.py
```

该现有工具只按适配器序列号恢复接口名称、设置 1 Mbps 并启动网络接口，不调用 π 策略，也不使能电机或发送运动命令。sudo 密码在终端输入。CAN 已经 UP 时无需反复执行；Phoenix 不会在推理过程中自动重置 CAN。

## 模型与接口

策略继续严格加载所选 Phoenix 权重，使用该次训练的归一化、Piper 身份、图像预处理和当前帧条件。Joint96 仍联合生成未来视频 latent 与动作；部署入口仅跳过最后将未来 latent 解码为 RGB 的步骤。它不会把模型改成只运行 action expert，也不会用 π 的图像尺寸、动作维度或控制频率替代 Phoenix 配置。

每轮输出 32 个双臂绝对目标，顺序为左臂 6 关节、左夹爪、右臂 6 关节、右夹爪。关节单位为弧度，夹爪沿用训练数据的 0–1 表示。模型内部 canonical80 映射保持一致。现场 CAN、相机和动作时钟在独立硬件进程中运行，GPU 推理在父进程中运行；目标按数据对应的 30 Hz 下发。这不表示模型每秒完成 30 次完整预测，也不是硬实时保证，实际漏步需要查看统计。

入口不额外加入步间平滑或任意速度阈值。正确的 shape、有限数值、有效硬件反馈以及底层驱动的物理范围检查仍是接口的一部分。

## 三种执行方式

| `--mode` | 行为 | 适用的检查 |
| --- | --- | --- |
| `sync` | 观测、推理、执行前 K 步，再重新观测；默认 K=32 | 最直接的动作块基线；推理期间会有等待 |
| `async` | 独立动作时钟执行旧段，模型预测下一段；按时间对齐采用尚未过去的动作 | 去掉推理等待，观察普通动作块切换效果；不使用 RTC 引导 |
| `rtc` | 在异步执行上增加推理期 VJP 引导，让新预测参考旧段尚未执行的动作 | 兼容现有非 RTC 训练权重的本地 RTC 实验 |

普通 `async` 不保证新旧段边界连续。`rtc` 是推理时引导，不等同于原作者 Prometheus 中需要专门训练权重的 Training RTC。队列用尽时不会凭空生成后续动作；应从运行统计检查供给情况。同步模式遇到调度迟到时会拉长间隔，仍按顺序发送所选 K 个目标；异步模式沿时间轴跳过已经过去的目标。

RTC 引导仅覆盖对齐后的旧预测有效部分。例如旧段已经过去 14 步，则 `prefix_attention_horizon=18`；后面 14 步仅用于补齐张量形状，不参与引导。此前本地接入把引导范围固定为 32，错误地把补齐的末尾位置作为后续目标，已修正。这个范围对应 [PI RTC 参考实现](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/main/src/eval_flow.py) 中的 `action_chunk_size - execute_horizon`。当估计延迟超过剩余有效前缀时，只引导真实存在的部分，同时报告覆盖不足；这不会凭空延长动作队列。

还需满足时间预算。设动作长度为 H、相邻请求实际间隔为 R、下一次观测到发布的延迟为 D，连续供给需要 `R + D <= H`（统一使用控制步）。本实现串行推理时 R 至少为上一轮耗时；所以 H32、30 Hz 下的稳定延迟预算约为 0.53 秒，实际还需为抖动留余量。把 `--replan-steps` 从 10 降到 1，不能绕过约 1 秒推理所造成的供给不足。

## 命令

先用已保存观测离线推理，不连接机器人。该 NPZ 应包含 `state`、`image`、`wrist_image_l`、`wrist_image_r`；图像为 HWC RGB uint8，状态为上述 14 维物理量。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-piper-inference/bin/python -u tools/run_piper_policy.py \
  --mode sync \
  --input-npz runs/piper-inference/robot-recorded/input.npz \
  --run-dir postcheckpoints/piper_clean_table_lerobot_v3 \
  --checkpoint postcheckpoints/piper_clean_table_lerobot_v3/checkpoints/weights/slot_1.pt \
  --prompt "put the tableware into the basket and the trash into the trash bin" \
  --chunks 2 --record actions
```

现场同步执行；`--live --execute` 会使能并移动双臂。下面显式使用此前选定的 4 次去噪，持续运行直到结束进程：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-piper-inference/bin/python -u tools/run_piper_policy.py \
  --mode sync --live --execute \
  --run-dir postcheckpoints/piper_clean_table_lerobot_v3 \
  --checkpoint postcheckpoints/piper_clean_table_lerobot_v3/checkpoints/weights/slot_1.pt \
  --inference-steps 4 \
  --prompt "put the tableware into the basket and the trash into the trash bin" \
  --execute-steps 32 --chunks 0 \
  --record none --output-dir runs/piper-policy
```

只读取现场观测时去掉 `--execute`。切换普通异步模式时，将 `--mode sync` 改为 `--mode async`，并用 `--replan-steps 10` 替换 `--execute-steps 32`。切换本地推理期 RTC 时使用 `--mode rtc --replan-steps 10`；默认 `--attention-backend saved` 会在 RTC 模式自动选择 SDPA，也可以显式写 `--attention-backend sdpa`。这三种模式共用同一组 checkpoint、归一化和任务参数。

上述清桌训练的 `slot_0.pt` 为 step 5000，`slot_1.pt` 为 step 9087；文件名本身不能证明任务效果。离线 NPZ 也应来自匹配任务的观测。

## 可配置参数

### Ctrl+C 后返回姿态

`tools/run_piper_policy.py --live --execute` 默认在收到 Ctrl+C 后停止策略动作队列，
用 5 秒平滑插值返回本次使能前记录的双臂关节姿态，确认到位后失能并关闭硬件。
该初始姿态在模型预热完成、使能机械臂之前采集。支持 `sync`、`async`、`rtc`
以及 `joint`、`mit` 控制；返回期间保持既有夹爪目标，不自动打开或闭合夹爪。

在原命令后追加以下选项可以改变返回方式：

```bash
--return-on-interrupt initial --return-duration 5
```

| 参数 | 作用 |
| --- | --- |
| `--return-on-interrupt initial` | 默认值；回到本次使能前的双臂姿态 |
| `--return-on-interrupt zero` | 双臂十二个关节回到 0 度，夹爪目标保持不变 |
| `--return-on-interrupt none` | 关闭返回，Ctrl+C 直接停止并失能 |
| `--return-duration 8` | 将返回插值时间改为 8 秒，数值越大返回越慢 |

返回使用当前关节反馈作为起点，沿关节空间的五次时间曲线插值；不包含障碍物避让。
插值结束后最多再等待 5 秒，十二个关节均进入目标 2 度范围且保持 0.2 秒才报告完成。
反馈丢失或未到位会报告失败并执行关闭流程。**返回中再按一次 Ctrl+C 会取消返回并失能。**
返回不重新使能已经关闭的机械臂；只读、离线评测、模型预热阶段的中断、运行故障及
有限轮次正常结束不触发返回。旧入口 `tools/run_piper_inference.py` 保留原退出行为。

终端打印 `PIPER_RETURN_START`、`PIPER_RETURN_RESULT`，本次 `summary.json` 的
`return_on_interrupt` 记录目标、耗时、是否完成／取消及最终关节误差；返回轨迹不作为策略预测写入 LeRobot 数据。

### 可选 MIT 关节控制

两个推理入口 `tools/run_piper_policy.py`、`tools/run_piper_inference.py` 都支持
`--control-mode joint|mit`，默认 `joint`。`--mode sync|async|rtc` 选择动作调度，
`--control-mode` 选择底层关节发送方式，两者独立，可以组合。

MIT 发送适配来自本机控制项目的 `mit_dev` 分支（`3058b10f6aa0`）。
默认参数已核对并同步自远端 `teleop_inference` 分支（`798b5b65b44a`，2026-11-02），
两个推理入口和独立数据回放工具共用 `configs/inference/piper_mit_defaults.json`。
上游原文件保存在 `configs/inference/upstream/teleop_inference/inference_config_temp_smooth_MIT.yaml`，
其 SHA256 和来源提交记录在默认参数 JSON 中；原文件仅作为来源副本，其他应用配置不自动启用。
独立模块 `src/wam/inference/piper_control.py` 只替换关节发送：
`MotionCtrl_2(0x01, 0x04, speed_limit, 0xAD)` 加六次 `JointMitCtrl`。
不需要切换控制项目分支。原来的反馈 ID 映射、目标同步、夹爪控制、停止退出流程继续使用当前驱动；
模型采样、RTC、动作块和训练代码不因 MIT 开关改变。

在现有推理命令后追加 `--control-mode mit` 即使用这组默认参数。
需要显式指定时，等价选项如下：

```bash
  --control-mode mit \
  --mit-kp 15 --mit-kd 1 \
  --mit-vel-ref 0 --mit-t-ref 0 \
  --mit-speed-limit 60 --mit-robot-rate 210
```

| 参数 | 默认值 | 含义及输入范围 |
| --- | --- | --- |
| `--control-mode` | `joint` | `joint` 原位置控制；`mit` 阻抗控制 |
| `--mit-kp` | `15` | 位置误差增益；SDK 范围 0–500 |
| `--mit-kd` | `1` | 速度误差增益；SDK 范围 −5–5，负值为负阻尼 |
| `--mit-vel-ref` | `0` | 目标速度 rad/s；SDK 范围 −45–45 |
| `--mit-t-ref` | `0` | 前馈力矩 Nm；当前 SDK 实际编码范围 −8–8 |
| `--mit-speed-limit` | `60` | `MotionCtrl_2` 速度百分比字段，整数 0–100；沿用分支含义 |
| `--mit-robot-rate` | `210` | 驱动线程设定频率 Hz，1–1000；不是模型推理频率 |

这些标量对双臂全部六个关节使用相同值，夹爪仍用原 `GripperCtrl`。
目标位置仍是模型输出的绝对角度，模型弧度 → 驱动内部度 → MIT 包弧度；没有改为增量动作。
目标速度和前馈力矩是显式常量，不自动从相邻目标计算；默认 0 不额外叠加该项。
MIT 线程频率设为 210 Hz 时仍按原动作时钟 30 Hz 更新目标，其间重复发送当前目标，未新增插值或平滑。
实际线程频率包含 CAN 发送和系统调度开销，不保证精确达到设定值；速度百分比也不等于回放倍速。

不指定 MIT 时原 Joint 驱动继续设为 100 Hz。运行开始打印 `PIPER_CONTROL`，新入口在
`run.json.robot_control` 保存实际配置，旧入口额外写 `robot_control.json`。
只读推理仍不使能或发送动作；SDK 缺少 `JointMitCtrl` 时显式报错，不静默退回 Joint。
已验证 SDK 调用、单位与门控逻辑，实机 MIT 响应和是否改善抖动需现场比较，不能由软件测试推断。

### 推理与记录选项

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--mode` | `sync` | 选择 `sync`、`async` 或 `rtc` |
| `--chunks` | `2` | 预热后的预测轮数；`0` 为持续运行直到 Ctrl+C |
| `--execute-steps` | `32` | 同步模式每轮执行多少个预测目标，不能超过模型动作 horizon |
| `--replan-steps` | `10` | 异步/RTC 相邻预测请求的最小间隔，以 30 Hz 控制步计 |
| `--inference-steps` | 保存的推理配置，本次训练为 `10` | 去噪次数；显式设为 `4` 改变采样次数，任务效果需另行比较 |
| `--attention-backend` | `saved` | 同步/普通异步沿用保存配置，RTC 自动选 SDPA；也可显式选 `sdpa` / `flex`，RTC 不接受 `flex` |
| `--rtc-delay-margin` | `2` | RTC 估计延迟额外增加的控制步数 |
| `--rtc-guidance-weight` | `5` | 推理期 RTC 引导权重 |
| `--rtc-inference-delay` | `auto` | 强引导区域的步数：自动使用最近至多 32 次往返耗时的 P95 × 动作频率向上取整再加余量；也可指定 `0..32`，固定值不再叠加余量 |
| `--rtc-prefix-horizon` | `auto` | 旧块参与引导的最长前缀；自动使用全部剩余有效动作，也可限制为 `0..32`；始终不引导补齐的尾部 |
| `--rtc-prefix-schedule` | `exp` | `exp` 指数衰减、`linear` 线性衰减、`ones` 整个前缀同权、`zeros` 仅延迟区域有权重 |
| `--record` | `none` | 记录级别，详见下文 |
| `--record-queue-size` | `4` | 后台记录队列容量，避免记录任务无限占用内存 |
| `--record-fps` | `30` | LeRobot v2.1 导出时间轴 FPS；采样时间与重复观测年龄单独保存 |
| `--output-dir` | `runs/piper-policy` | 输出根目录；每次自动创建唯一运行子目录，避免覆盖既有结果 |
| `--run-dir` | `postcheckpoints` | 匹配的训练运行配置、归一化与 checkpoint 清单所在目录 |
| `--checkpoint` | 运行目录的默认选择 | 显式选择相对项目根目录的权重路径；应与 `--run-dir` 匹配 |
| `--seed` | `0` | 推理采样随机种子 |
| `--device` | `cuda:0` | 模型运行设备 |

不指定 `--checkpoint` 时读取运行目录的 `weights_manifest.json` 及其 latest 权重；显式指定权重时不要求存在该清单，但仍要求匹配的配置与统计文件。`--list-checkpoints` 可以只列出权重信息。未指定 `--live` 或 `--input-npz` 时，从已准备数据中选取与任务匹配的录制观测；这不是独立测试集评估。`--execute` 必须与 `--live` 同用，录制观测不能直接发送到硬件。

正数 `--chunks` 统计完成的预测轮数。同步模式逐轮执行 K 步；异步/RTC 在最后一次发布后再等待 `--replan-steps` 个时钟周期，然后结束，不保证把最后一个 32 步预测全部执行。

### 调整当前推理期 RTC

下面显式列出原默认 RTC 设置，可加到现有命令中：

```bash
--mode rtc --inference-steps 4 --replan-steps 10 \
--rtc-inference-delay auto --rtc-delay-margin 2 \
--rtc-prefix-horizon auto --rtc-prefix-schedule exp --rtc-guidance-weight 5
```

例如 `--rtc-inference-delay 8 --rtc-prefix-horizon 16` 表示旧块前 8 步用完整前缀权重、
其后直到第 16 步按曲线衰减；实际还会受旧块剩余有效步数限制。
这是**软引导权重**，并不强制动作相等，也不启用官方训练期 RTC 的固定队列调度。
这里的 `prefix-horizon` 不等于官方 `execution_horizon=s`；`replan-steps` 仍然控制请求间隔。
两个数均显式指定时，延迟步数不能超过前缀范围；默认 `auto/auto/exp` 保持此前行为。

`run.json` 保存所选命令参数，`summary.json.rtc_config` 保存实际配置（`null` 表示自动）。
逐块日志同时保存 `estimated_inference_delay_ticks`（实测 P95 的估计）、
`rtc_requested_delay_ticks`（自动估计或手动值）、`rtc_guidance_delay_ticks`（受有效前缀限制后的值）
和 `prefix_attention_horizon`。手动缩短延迟不会缩短模型耗时；前缀不足仍会记录在
`rtc_prefix_covers_delay` / `rtc_prefix_shortfall_chunks`，便于分析抖动和供给中断。

## 输出与运行统计

**保存目录可改，记录现在自动导出为 LeRobot v2.1。** 相对路径以项目根目录为基准，也可使用绝对路径：

```bash
# 加在现有推理命令中；替换原来的 --record / --output-dir
--record full --output-dir runs/lerobot-inference --record-fps 30
```

正常退出或 Ctrl+C 后，先完成上述返回／停止流程并关闭硬件，再写出本次运行目录的：

```text
<output-dir>/<时间-编号>/
  meta/info.json
  meta/tasks.jsonl
  meta/episodes.jsonl
  meta/episodes_stats.jsonl
  meta/stats.json
  meta/recording.json
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/{image,wrist_image_l,wrist_image_r}/episode_000000.mp4
```

每次运行是一份独立单 episode 数据集。字段沿用采集数据的 `state`、`actions`、`image`、
`wrist_image_l`、`wrist_image_r` 命名，含 LeRobot 索引和任务元数据；当前机器人状态/动作是 14 维。
MP4 为 H264，Parquet 为 Zstd。`actions` 是该观测下预测的第一个动作，完整预测保存在
`predicted_actions[32,14]`；**它们不是逐控制 tick 的实机执行日志**，`meta/recording.json`
会明确标注动作含义，并保留计划接纳状态。原始 NPZ/JSON 同时保留用于诊断和重导出。

标准 `timestamp` 使用均匀时间轴；对于低频推理观测，使用之前最近一次样本保持，
`source_timestamp`、`state_sample_timestamp`、`state_age_seconds`、`source_sample_index` 及
`camera_sample_timestamp`、`camera_age_seconds` 保留真实时序信息。30 FPS 输出不意味着 30 Hz 新观测。
仅 `--record actions` 时可生成无视频的 LeRobot 数据集；`--record none` 不导出数据集。
缺失时间戳或完整三路图像时导出报错并保留原始文件，不生成假视频。
查看 `summary.json.recording.lerobot.status`，`complete` 才确认导出成功。

已有记录也可单独转换（无硬件、无需模型）：

```bash
.venv-piper-inference/bin/python tools/export_piper_lerobot.py \
  --input-dir runs/<原运行目录> --output-dir runs/<新的数据集目录> \
  --kind policy --record full --fps 30
```

历史记录没有图像时使用 `--record actions`；缺少真实观测时间的旧日志不支持伪造时间后导出。

每次运行在 `--output-dir` 下创建带时间戳和随机后缀的唯一子目录。正常运行即使 `--record none`，也保存启动选项、模型审计信息的 `run.json` 和结束统计 `summary.json`，并在终端逐轮打印 `PIPER_POLICY_RESULT`。异常可能额外生成 `failure.json`；资源预检失败时尚未创建运行目录或连接硬件。

- `none`：不保存逐轮 NPZ 和 JSON。
- `actions`：保存 `chunk_XXXXXX.npz`，包含 `state`、`prompt`、`actions[32,14]`、`normalized_actions[32,80]`；对应 JSON 保存该轮计时、发布及控制统计，不保存图像。
- `full`：在上述 NPZ 中增加 `image`、`wrist_image_l`、`wrist_image_r` 三路 RGB，使用压缩保存。

要同时记录图像、状态和动作，将原命令中的 `--record actions` 改成 `--record full`，其他推理参数保持不变。
图像是**每次推理输入的原始观测**，频率随推理请求变化，不是连续 30 FPS 录像；预热轮次不记录。
`state[14]` 顺序为左臂 6 关节、左夹爪、右臂 6 关节、右夹爪；关节为 rad，夹爪为 0–1。
`actions[32,14]` 是完整预测段，不代表 32 步都已执行；结合 JSON 中的 request/publish tick、
丢弃的过期步数、计划是否接纳及控制统计分析实际使用范围。

真机 JSON 的 `sensor_timing` 记录相机帧号、设备时间、采集时间、反馈时间与反馈频率。
`sensor_timing.feedback.left/right.gripper_diagnostics` 额外保存缓存的 SDK 夹爪行程（mm）、
扭矩（Nm）、原始状态码、故障位、使能与回零位，以及独立时间戳。
此诊断只读取缓存，不发送 CAN 查询或改变控制；SDK 不支持时记录 `available=false` 与原因。
相机和关节按各自时间戳记录，不能视为硬件同步采样。

记录工作在后台有界队列中完成。队列满时丢弃本轮记录并累计 `recording.dropped`，不等待磁盘而阻塞控制。实际写盘错误会结束运行并报告错误；退出时先停止动作并关闭硬件，再等待记录线程结束。

`summary.json` 的主要字段：

| 字段 | 含义 |
| --- | --- |
| `chunks_requested` / `chunks_completed` | 请求与完成的预测轮数；请求值为 0 表示无限运行 |
| `completed` / `interrupted` | 请求轮数是否正常完成，以及是否收到 Ctrl+C；均不判断折毛巾任务是否成功 |
| `accepted_plans` / `rejected_plans` | 接纳/拒绝的动作段数；整段时间已经过去的新预测不再接纳 |
| `controller_ticks` / `sent_target_ticks` | 控制时钟计数及成功提交的目标数；只读预览也会检验和计数目标 |
| `queue_underrun_ticks` | 控制周期中无可用预测目标的次数 |
| `rtc_prefix_shortfall_chunks` | RTC 请求时，估计端到端延迟加 margin 超过旧段剩余有效步数的轮数；这是预算预警，是否实际断供还要看 underrun |
| `missed_control_ticks` | 时钟迟到或跳过周期的计数 |
| `model_latency_seconds_last_256` | 最近最多 256 轮策略调用耗时的 `sample_count`、`p50`、`p95`、`max`，单位秒 |
| `observation_to_publish_seconds_last_256` | 同一窗口中从观测请求到动作段发布的耗时分布，包含采集、进程通信和推理 |
| `warmup_seconds` | 执行动作前的观测与采样预热时间；不包含此前模型加载时间 |
| `recording` | `mode`、`submitted`、`written`、`dropped`、`queue_capacity` |
| `hardware_commands_sent` | 是否实际向硬件提交过目标；预览为 false，异常导致计数无法确认时可能为 null |
| `continuous_supply_observed` | 有限异步/RTC 运行正常结束、超过一轮、无拒绝段、无队列耗尽且无漏步时为 true |

`continuous_supply_observed=true` 只说明本次记录到连续的动作供给，不代表机械臂成功完成任务、预测动作正确或具备硬实时保证。同步模式该值为 false；无限运行按 Ctrl+C 结束时也为 false，应结合累计漏步、耗尽和逐轮指标判断此前的供给情况。逐轮 JSON/终端还报告 `capture_seconds`、`discarded_elapsed_actions`、`remaining_actions` 和 `sensor_timing`；传感器年龄用于观察实际时序，不设置额外年龄阈值。

RTC 逐轮日志中的 `estimated_inference_delay_ticks` 保留完整延迟估计，`rtc_guidance_delay_ticks` 是裁到有效旧前缀内的实际引导延迟；`prefix_attention_horizon` 是有效旧前缀长度，`rtc_prefix_covers_delay=false` 表示覆盖不足。排查动作衔接应选 `--record actions`，保留预测目标和观测状态以比较新旧段；`none` 只保留汇总计时，无法还原动作跳变。软件目标提交计数不等同于底层 CAN 成功发送或机械臂实际跟踪。

## 5090 验证与原作者版本

先比较相同权重、观测和采样设置下的动作结果，再比较预热后的端到端耗时、P95/P99、显存及队列耗尽次数。冷启动与编译耗时应单列；仅测模型吞吐不足以证明机器人连续收到动作。4 次去噪比 10 次更快也不直接说明任务成功率更高。

原作者 [Prometheus 部署指南](https://github.com/Anyverse-Dynamics/Phoenix-MWA/blob/823f00952b987845984641812413a59706436f48/docs/PHOENIX_RTC.md) 给出约 300 ms 的 RTX 5090 客户端/服务端往返参考，但其示例采用 H32、9 帧、10 次去噪和经过 Training RTC 训练的权重。该数值不是本入口或当前机器的实测成绩。其默认 d8 无法覆盖 300 ms；文档要求按实际延迟调整已承诺前缀的时间预算。

当前 Joint96 权重依赖每一步带噪未来视频的特征，原作者明确禁止它使用静态视频缓存的纯动作推理入口。可缓存不变的文本和固定张量，可研究保持计算等价的推理编译；不能直接删除未来视频去噪来声称等价加速。另外，原作者已有的 `mot_compile_mode` 路径在 eval 时回退 eager，单独设置它并不会启用完整推理编译。

相关实现：[联合采样](https://github.com/Anyverse-Dynamics/Phoenix-MWA/blob/823f00952b987845984641812413a59706436f48/src/wam/inference/frame_joint.py)、[静态缓存入口的 Joint96 检查](https://github.com/Anyverse-Dynamics/Phoenix-MWA/blob/823f00952b987845984641812413a59706436f48/src/wam/inference/online_action_policy.py)、[训练编译路径](https://github.com/Anyverse-Dynamics/Phoenix-MWA/blob/823f00952b987845984641812413a59706436f48/src/wam/model/modules/mot/compile.py)。
