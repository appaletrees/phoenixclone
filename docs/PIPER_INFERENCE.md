# Piper 离线推理与控制接口

2026-10-27 重新核对：官方已经在 [prometheus 分支](https://github.com/Anyverse-Dynamics/Phoenix-MWA/tree/prometheus) 发布部署框架（核对提交 `823f00952b987845984641812413a59706436f48`）。主分支 README 的 `rtc coming soon` 未反映该分支。本文件中的 `tools/run_piper_inference.py --rtc` 是兼容当前非 RTC Piper checkpoint 的本地 VJP 引导适配，**不等同于官方 Prometheus Training RTC 部署流程**。以下性能测试证明本地实现的数据供给和输出有效性，不能作为官方流程已经复现的证据。

在 `/home/midea/phoenix-mwa` 中运行，参数中的项目资源路径均相对该目录。下面的离线命令不连接硬件、不下发动作。

```bash
cd /home/midea/phoenix-mwa
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-piper-inference/bin/python tools/run_piper_inference.py \
  --input-npz runs/piper-inference/robot-recorded/input.npz \
  --output-dir runs/piper-inference/robot-recorded --chunks 2
```

不传 `--input-npz` 时，从已准备的数据集中选择匹配指令的首个 episode，用真实录制图像、状态进行推理，并对照该窗口的参考动作。它属于训练数据上的单窗口检查，不代表独立测试集精度或真机任务成功率。

默认任务为 `fold the towel and put it into the plate`。权重从 `postcheckpoints/checkpoints/weights/weights_manifest.json` 选择最新保存项；本次验证为 `slot_1.pt`、step 10000。严格恢复 1650 个 mot 张量和 3 个 proprio_encoder 张量，使用训练保存的 `dataset_stats.json`、Piper embodiment 14，以及原项目的 `infer_frame_joint_window` 联合视频/动作采样器和 `auto` 当前帧适配方式。本次配置为 10 次去噪、9 帧视频、32 步动作。

可以使用 `--checkpoint` 显式选择项目根目录下的相对权重路径。当前可选文件位于 `postcheckpoints/checkpoints/weights/`：

| 文件 | 文件中实际保存的 step |
| --- | --- |
| `slot_0.pt` | 5000 |
| `slot_1.pt` | 10000 |
| `slot_2.pt` | 15000 |
| `slot_3.pt` | 17445 |

现有清单未登记 slot_2/slot_3，因此不传 `--checkpoint` 时默认仍是清单指向的 step 10000。显式选择会读取权重内部的真实 step，并继续严格检查 Piper 身份和全部策略模块；不会改写清单。最高保存步数不代表已验证任务效果最好。选择另一个训练运行的权重时，`--run-dir` 也应指向该运行对应的 `config.yaml` 和 `dataset_stats.json` 所在目录。

列出权重和实际 step（不连接硬件、不加载 GPU 模型）：

```bash
.venv-piper-inference/bin/python tools/run_piper_inference.py --list-checkpoints
```

沿用本地 RTC/异步、4 次去噪，选择 step 17445 的真机命令（会使能并移动双臂）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-piper-inference/bin/python -u tools/run_piper_inference.py \
  --live --execute --rtc --inference-steps 4 \
  --checkpoint postcheckpoints/checkpoints/weights/slot_3.pt \
  --prompt "fold the towel and put it into the plate" \
  --output-dir runs/piper-rtc-slot3 --chunks 20 --execute-steps 10
```

输入 `.npz` 包含 `state`、`image`、`wrist_image_l`、`wrist_image_r`，可附带标量字符串 `prompt`。三路图像为 HWC RGB uint8；状态为 `[左臂6关节, 左夹爪, 右臂6关节, 右夹爪]`，12 个关节使用弧度，夹爪使用 0–1。可输入第 15 个辅助字段，但不送入策略关节维度。输出 `actions` 为 32×14 的绝对目标，关节仍为弧度；`normalized_actions` 为 32×80 的模型归一化输出。

输出目录保存观测、原始预测、严格加载记录和每次推理耗时。模型原始夹爪预测可能略超出 0–1；离线结果保留该数值。执行时仅由现有底层驱动限制物理关节范围和夹爪 0–1 范围。

执行入口为本项目自己的 `wam.inference.piper_robot.PhoenixPiperRobot`，不导入 pi0.5 客户端或协议。流程为读取三路相机/双臂反馈 → 原 Phoenix 推理和反归一化 → 直接下发绝对目标 → 再次观测。仅从用户提供的硬件目录导入 `PiperController` 和 `MultiCameraManager`，进行 CAN/RealSense 通信、BGR→RGB、反馈角度→弧度及预测弧度→驱动角度转换。

已移除旧入口的相机/反馈年龄阈值、2 秒观测过期中止、每步 2°和夹爪 0.05 裁剪、客户端 watchdog 以及每段结束后对实测位置重新 hold 的处理。`--max-camera-age`、`--max-feedback-age`、`--max-observation-age`、`--max-joint-delta-deg`、`--max-gripper-delta` 和 `--timeout` 不再是入口参数。启动时仍需收到真实的图像和关节/夹爪反馈，输入/输出必须有限且维度正确。底层驱动自身的使能同步、物理范围限制和退出时停止/失能保持原样。

不加 `--rtc` 时，Phoenix 每次生成 32 步，默认取前 10 步，以数据的 30 Hz 节奏交给驱动。推理和执行为串行；下一次推理期间驱动维持最后一个目标。先下发动作，再压缩保存图像；报告中的观测年龄仅作记录，不触发中止。

`--live` 读取现场观测但不使能或执行；加 `--execute` 才执行动作。硬件资源仍为独占，CAN 锁与其他现有控制程序互通，避免两个进程同时写双臂。硬件配置在 `configs/inference/piper_robot.json`，驱动目录默认是相对路径 `../Desktop/piper_tele/sink-bot-control_bk/example_agilex`。这是一层本地 Piper 硬件适配，与官方已经发布的 Prometheus 部署框架不同。

2026-10-27 现场确认右臂 `can_arm2` 的八条反馈报文为 `0x3A0..0x3A7`，比 SDK 默认 `0x2A1..0x2A8` 偏移 `0xFF`。配置 `feedback_id_offsets` 对该接口设置十进制 `255`，左臂为 `0`。Phoenix 仅在接收回调中复制报文并还原 ID，再交给原 SDK 解码；不修改固件、主从设置、发送 ID、数值或模型。映射涵盖状态、末端位姿、关节和夹爪，电机反馈与查询应答保持原样。若硬件反馈 ID 配置改变，应重新核对该配置；省略接口时偏移默认为 `0`。初始化失败会报告具体左右臂、反馈频率和时间戳。

真机执行命令（会使能并移动双臂，需已有控制程序释放硬件）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-piper-inference/bin/python tools/run_piper_inference.py \
  --live --execute --prompt "fold the towel and put it into the plate" \
  --output-dir runs/piper-phoenix-live --chunks 100 --execute-steps 10
```

本次独立入口通过模拟控制器测试和保存观测的离线模型验证，未自动启动真机。

## RTC 与异步执行

官方部署使用 `rtc.method=condition`：完成训练期 RTC 前缀训练的 checkpoint，在每次去噪中固定已承诺前缀（timestep=0），只生成新的后缀。模型服务、策略客户端和 ROS2/ZMQ 环境/动作调度器分开运行。默认 H32/s16/d8、9 帧、10 次去噪、30 Hz；首次提交 24 步，剩余 8 步时启动下一次预测，固定这 8 步，后续提交 16 步；调度器等待旧队列完成后再切新段。官方也检查完整推理往返加一个控制周期是否落在 d 的预算内，要求 `0<d<s`、`s+d<=32`，且 d 不超过训练前缀范围。d8 模板不能覆盖官方文档提到的约 300 ms 往返，必须按实测调整。详见 [官方部署指南](https://github.com/Anyverse-Dynamics/Phoenix-MWA/blob/prometheus/docs/PHOENIX_RTC.md)、[客户端协议](https://github.com/Anyverse-Dynamics/Phoenix-MWA/blob/prometheus/phoenix/protocol.py) 和 [模型加载](https://github.com/Anyverse-Dynamics/Phoenix-MWA/blob/prometheus/phoenix/model.py)。

当前 `slot_1.pt` 的训练契约是 Piper 后训练，保存配置 `training_time_rtc.enabled=false`。已将实际 checkpoint 契约交给官方 `validate_training_contract` 校验，结果被拒绝，缺少 `phoenix.training_rtc.v1` 契约。严格按官方部署，需要兼容性核对后的 RTC 前缀训练 checkpoint，以及 Piper 的 Prometheus Robot 适配和对应夹爪单位约定；官方自带硬件模板是 ARX5、夹爪单位 m，当前 Piper 数据是 0–1。不能只修改 enabled 或伪造训练标记，也不能把本地 `--inference-steps 4` 当成官方固定 10 次去噪配置。

加 `--rtc` 后启用作者 `RTCActionGuidance` 的推理期 VJP 引导，以及本项目的独立异步动作时钟。当前 checkpoint 的 `training_time_rtc.enabled=false` 保持原样，不使用要求 RTC 训练的硬前缀模式。默认引导权重为 5、exp 权重计划，延迟按实测推理时间换算成控制周期并增加 2 步余量。

模型权重、10 次去噪、32 步动作、9 帧联合视频/动作预测及归一化保持原样。RTC 在内存中的推理配置使用 SDPA 并关闭 Flex 专用批次选项，以便求动作输入梯度；不修改保存的训练配置。联合采样器每次去噪缓存同一份当前/带噪未来视频 K/V，视频仍正常去噪。关闭引导的缓存路径已与原联合路径做实际模型对照，动作逐值一致。RTC 不解码用于查看的预测视频，不影响动作与视频 latent 的联合生成。

默认仍采用保存配置中的 10 次去噪；可用 `--inference-steps N` 显式覆盖本次推理次数，不修改权重或磁盘上的训练配置。实际模型测试中：10 次 RTC 每轮约 0.99–1.05 秒，8 轮回放有 190 个队列耗尽周期；5 次约 0.53–0.56 秒；4 次约 0.44–0.49 秒，8 轮回放的 117 个控制周期均有动作，耗尽/漏步均为 0。4 次去噪改变推理采样次数，不能根据速度测试推断真机任务成功率。测试记录见 `runs/piper-inference/rtc-validation/latency_options.json` 和 `runs/piper-inference/rtc-async-replay-4steps/rtc_summary.json`。

控制线程按 30 Hz 读取当前动作段，GPU 在主线程预测下一段。RTC 使用请求时刻对齐后的旧段作为条件；新段返回时，丢弃其时间轴中已经过去的动作，只替换尚未执行的部分。`--execute-steps 10` 在 RTC 模式下表示相邻推理请求至少间隔 10 个控制周期；若推理更慢，下次请求会在前次完成后尽快发起。图像压缩放到独立记录线程，退出时先结束动作线程，再关闭硬件。

现场 RTC 默认通过 `PiperRobotProcess` 把 CAN/RealSense 驱动放到独立进程，GPU 推理留在主进程，避免相机与 SDK 的 Python 线程拖慢模型调度。每个控制周期通过本地管道传递 14 维目标；只读预览也以相同频率传递预览目标，但不会调用驱动下发或使能接口。退出时父进程先结束动作时钟，再请求硬件进程正常清理。分进程前的现场只读预览约 0.51–0.62 秒/次，有队列耗尽；分进程后 12 轮现场预览约 0.43–0.48 秒/次，172 个控制周期无耗尽、无时钟漏步。现场预览不代表折毛巾运动效果已验证。

真机 RTC 执行命令（会使能并移动双臂）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-piper-inference/bin/python tools/run_piper_inference.py \
  --live --execute --rtc --inference-steps 4 \
  --prompt "fold the towel and put it into the plate" \
  --output-dir runs/piper-rtc-live --chunks 100 --execute-steps 10
```

先用保存观测做实际 GPU 异步回放，不连接硬件：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-piper-inference/bin/python tools/run_piper_inference.py \
  --rtc --inference-steps 4 --input-npz runs/piper-inference/robot-recorded/input.npz \
  --output-dir runs/piper-inference/rtc-async-replay --chunks 8
```

`rtc_summary.json` 记录完成状态、实际发送周期、队列耗尽周期和控制时钟错过的周期；每段 JSON 记录估计延迟、实际经过的步数、剩余动作数和新段是否被接纳。队列耗尽时驱动维持最后一个目标，不外推新的运动；整段已过去的预测不再下发。这是 RTC 时间轴对齐，不是固定 2 秒中止。RTC 并不能消除模型计算耗时，32 步在 30 Hz 下只覆盖约 1.07 秒；需要结合实际回放中的耗尽次数判断是否能连续供给动作。

这套入口是作者公开推理期 RTC 引导组件在当前 Joint96/Piper 上的本地适配，与官方 Prometheus 的训练期前缀条件 RTC 路径不同。尚未自动进行真机 RTC 运动验证。

接口测试：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. \
  .venv-piper-inference/bin/python -m unittest discover \
  -s tests -p test_piper_inference.py -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:. \
  .venv-piper-inference/bin/python -m unittest discover \
  -s tests -p test_piper_rtc.py -v
```
