# Piper Replay Tool

本目录是 `phoenixclone` 内的网页回放工具源码，来自独立的
[`appaletrees/piper_replay_tool`](https://github.com/appaletrees/piper_replay_tool)。
默认自动识别上一级 Phoenix 项目；原始数据、准备后的接口元数据、录像和运行结果不包含在 Git 中。

Piper 双臂机器人的**原始采集数据回放工具**。直接读取数据集里未处理的 parquet 与三路视频，按 `action`（采集时的目标动作）或 `state`（采集时的实测关节状态）逐帧下发到真机，同时提供网页界面、实时相机画面和 12 关节目标/反馈对比图。

工具不加载策略权重，不运行 RTC 或任何模型推理，也不修改原始训练数据与机器人驱动。硬件模式只做"读取反馈 + 下发关节目标"。
目前共用 Phoenix 控制模块，其包初始化会导入部分模型代码，因此安装时仍需要 Phoenix 的 Python 依赖。

## 特性

- **原始数据直读**：直接解析 `datasets/` 下原始 parquet，不使用标准化数据或模型预测值。
- **完整回放时钟**：原采集频率 30 Hz，支持 7.5 / 15 / 30 / 60 Hz（¼× / ½× / 1× / 2×）及任意 (0, 100] Hz。
- **单帧定位**：可跳到任意原始帧并"移动到该帧并保持"，用于分析固定目标下的抖动与稳态误差。
- **起点对齐**：从当前姿态到首帧的可配置过渡时长；设为 0 则直接发送首帧，轨迹内部不插值、不平滑。
- **同步录像**：读取所选轨迹原始三路视频，与当前回放帧对齐。
- **实时相机**：外部、左腕、右腕三路 RealSense 画面。
- **关节曲线**：蓝线为回放目标，黑线为真实反馈，支持 rad / ° 切换、30 s / 60 s / 全部记录窗口、PNG 导出。
- **运行记录**：每次回放保存配置、每次下发动作、约 30 Hz 状态与夹爪诊断；可选 1–30 Hz 三路现场 RGB 图像，包含时间戳、缺失相机与漏录统计。
- **离线演示模式**：`--demo` 不导入、不连接任何硬件，黑线明确标注为"模拟反馈"。

## 安装

```bash
git clone https://github.com/appaletrees/phoenixclone.git
cd phoenixclone

python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install -r piper-dataset-replay/requirements.txt
cd piper-dataset-replay
```

`requirements.txt` 覆盖网页与数据读取（Flask / numpy / pyarrow / av / opencv）。**硬件模式**还需要机器人控制环境中已有的 Piper SDK、`scipy`、`pyrealsense2`、`python-can`，以及可用的 CAN 接口。
回放不需要加载模型权重或使用 GPU；Python 导入依赖由上面的 Phoenix 安装提供。

先按 [Piper 数据准备说明](../docs/PIPER_DATA.md) 把自己的原始数据放到仓库根目录的
`datasets/`，配置 `configs/data/piper_sources.json` 并生成
`resources/piper/dataset_v3/meta/piper_interface.json`。`--demo` 也读取这些原始采集数据，
不会自动生成示例轨迹。真机控制源码通过 `--interface-root` 指定，不包含在本仓库。

## 运行

离线演示（无需硬件）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -u app.py --demo --port 8766
```

真机模式：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -u app.py --hardware --port 8766
```

浏览器打开 <http://127.0.0.1:8766>。同一端口只运行一个实例，`Ctrl+C` 停止网页服务。

### 常用参数

| 参数 | 说明 |
| --- | --- |
| `--project-root` | 数据与配置所在的项目根目录，默认自动识别本仓库根目录 |
| `--interface-root` | 外部机器人控制目录，默认在项目的同级 `Desktop/piper_tele/sink-bot-control_bk/example_agilex`；其他布局请显式指定 |
| `--robot-config` | 机器人配置路径（相对 `--project-root`），默认 `configs/inference/piper_robot.json` |
| `--output-dir` | 保存根目录，可相对本工具目录或使用绝对路径，默认 `runs`；网页也可修改 |
| `--record` | `actions`（默认）保存状态、动作和诊断；`full` 另外保存三路现场 RGB |
| `--record-image-hz` | 图像采样频率 1–30 Hz，默认 10；独立于回放/驱动/网页预览频率 |
| `--record-fps` | LeRobot v2.1 导出 FPS，默认 30；与图像实际采样频率分别记录 |
| `--port` | 网页端口，默认 `8766` |
| `--demo` / `--hardware` | 离线模拟 / 真机执行 |

首次在自己机器上运行前，请按实际路径覆盖 `--project-root` 与 `--interface-root`。

## 使用流程

### 记录图像、状态与动作

```bash
# 在已安装并激活环境的仓库根目录运行
cd piper-dataset-replay
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -u app.py \
  --hardware --port 8766 --record full --record-image-hz 10 --output-dir runs
```

也可在网页「04 本次运行记录」输入保存根目录、选择内容与频率，再点「应用记录配置」。
相对路径以本工具所在目录为基准；绝对路径和 `~` 也支持。写入权限在使能之前检查。
开始回放或单帧移动时创建 `runs/<时间-编号>/`；暂停与单帧保持期间继续记录，停止、断开、失能或回放完成时结束。
图像队列在后台保存，保存期间仍能停止/失能/断开；保存完成后再开始新记录。
停止记录后，后台自动在该运行目录生成 **LeRobot v2.1**；页面显示导出状态，`complete` 才是完成。
原始诊断记录保留用于分析和恢复，不会覆盖以前的运行。
切换单帧目标前先停止当前完整记录。演示模式明确没有现场图像，不拿采集录像替代现场画面。
刷新页面不能升级已运行的后端，旧进程需先 Ctrl+C 正常结束并按上述命令重启。

| 文件 | 内容 |
| --- | --- |
| `meta/info.json` | LeRobot v2.1 schema、数据/视频路径、FPS、总帧数 |
| `meta/tasks.jsonl`、`meta/episodes.jsonl` | 任务文本和本次 episode 信息 |
| `meta/episodes_stats.jsonl`、`meta/stats.json` | 每个特征的均值、标准差、最小/最大值和计数 |
| `meta/recording.json` | 动作含义、原始采样数、时序对齐方式和来源 |
| `data/chunk-000/episode_000000.parquet` | `state`、`actions`、`driver_actions`、索引/任务/时间戳、采样年龄和原始帧 |
| `videos/chunk-000/{image,wrist_image_l,wrist_image_r}/episode_000000.mp4` | 完整模式的三路 H264 视频 |
| `run.json` | 数据集/episode/字段/源 parquet、控制模式与参数、单位、记录设置与 schema |
| `commands.jsonl` | 每次应用层目标下发与保持操作，含原始帧、单调/系统时间、请求目标与驱动接受目标；不是 CAN 报文抓包 |
| `feedback.jsonl` | 约 30 Hz 的状态、当前目标、驱动目标、动作序号、原始帧与第 15 维、反馈频率/时间、夹爪行程/扭矩/故障码 |
| `joints.csv` | 原有图表用目标/反馈/驱动目标表格，兼容旧工具 |
| `frames/frame_XXXXXXXX.npz` | 完整模式：RGB `image`、`wrist_image_l`、`wrist_image_r`（相机原分辨率），及对应 `state`、`action`、`driver_action` |
| `frames/frame_XXXXXXXX.json` | 对应状态序号、原始帧、采样时间、各相机帧号/设备时间/图像年龄、状态年龄和缺失相机列表 |
| `summary.json` | 运行结束统计、记录条数、保存数/队列丢弃/采样迟到/缺失相机/错误；`finalized=true` 表示后台收尾已完成，仍需查看 `error` |

关节均为 rad，夹爪为 0–1。14 维顺序：左臂 6 关节＋左夹爪＋右臂 6 关节＋右夹爪。
第 15 维作为 `auxiliary_15` 单独保留，不下发；物理状态和动作仍为 14 维。
相机和关节没有硬件同步。LeRobot 使用均匀时间轴，按此前最近一次记录保持；低频图像会重复，
不是插值生成新图像。`state_sample_timestamp`、`state_age_seconds`、`camera_sample_timestamp`、
`camera_age_seconds` 和 `source_sample_index` 保留采样时间/年龄，原始诊断保存每摄像头硬件时间。
例如相机记录 10 Hz、导出 30 FPS，视频会重复相邻帧，不能视为 30 Hz 独立测量。
原始 NPZ 为无损 RGB；MP4 使用 H264 CRF 18。磁盘较慢时最多排队 4 组，超出会记录 `dropped`，不等待图像写盘来调度动作。
缺失相机不填伪图：NPZ 缺少相应键，JSON 和页面列出缺失相机。完整模式没有任何完整三路图像组时，
LeRobot 导出明确失败并保留原始记录；演示模式可选 `actions` 导出不含视频的数据集。
保存错误显示在页面并写入汇总，运行中的错误会结束本轮回放。正常停止会排空已入队图像。
全量逐次动作请看 `commands.jsonl`，高于 30 Hz 的回放不能仅凭 30 Hz CSV 重建所有下发动作。

Phoenix 真机推理使用原命令的 `--record full`，保存位置在 Phoenix 的 `--output-dir/<本次运行>/`。
推理图像按观测请求保存，包含预测动作段；与本工具独立频率的现场图像记录含义不同。

### Joint / MIT 控制选择

在左侧「设备与执行」中选择 `Joint · 原有位置控制` 或 `MIT · 阻抗控制`。
选择 MIT 后可配置 Kp、Kd、目标速度、前馈力矩、速度百分比、驱动线程频率；
点击「应用控制配置」，再连接设备并开始回放或移动到指定帧。
已连接时先断开设备，再更改控制配置；刷新页面会读回服务端当前值，服务重启后恢复配置文件/默认值。
旧服务需要在原终端 Ctrl+C 后按原命令重启，单独刷新网页不会升级 Python 后端。

| 页面参数 | 默认值 | 含义 |
| --- | --- | --- |
| Kp | 15 | 位置误差增益，0–500 |
| Kd | 1 | 速度误差增益，SDK 范围 −5–5；负值是负阻尼 |
| 目标速度 | 0 rad/s | 共同速度参考，−45–45，不从数据自动求导 |
| 前馈力矩 | 0 Nm | 共同力矩参考，当前 SDK 编码范围 −8–8 |
| 速度百分比 | 60 | `MotionCtrl_2` 的百分比字段，整数 0–100 |
| 驱动线程频率 | 210 Hz | MIT 循环设定频率，1–1000；实际有发送/调度开销 |

参数对双臂全部关节统一生效。默认参数已于 2026-11-02 从控制项目 `teleop_inference`
分支（`798b5b65b44a`）同步，共用 Phoenix 的 `configs/inference/piper_mit_defaults.json`。
上游原文件副本和来源校验信息保存在该配置旁。发送适配继续采用原来的
`MotionCtrl_2(1, 4, speed_limit, 0xAD)` 与逐关节 `JointMitCtrl` 发送方式。
无需切换机器人控制项目的分支。默认 Joint 仍为 100 Hz，夹爪控制保持原实现。
共用 `--project-root/src/wam/inference/piper_control.py` 适配模块及默认参数 JSON（移植时一起复制），
演示模式也读取参数定义，但不连接 SDK 或加载模型权重。

回放频率独立决定快慢：30 Hz 原速、15 Hz 半速。MIT 的 210 Hz 是重复发送当前关节目标的驱动循环，
不代表每秒播放 210 个原始帧，也没有新增轨迹平滑。轨迹位置仍是绝对角，MIT 包使用弧度。
`runs/<本次运行>/run.json.robot_control` 保存实际模式与全部 MIT 参数；演示数据注明非真机反馈。

### 轨迹与执行

1. 选择数据集、原始 episode 编号（从 0 开始）与 `action` / `state`，点击加载。
2. 选择起始帧、目标更新频率和起点对齐时长。
3. 点击"连接设备与相机"——该步骤只读取反馈与画面，**不自动使能**。
4. 点击"开始真机回放"——会使能双臂，按设定时长移动到起始帧，再逐帧回放。
5. 暂停 / 停止回放会保持当前姿态；"停止并失能"停止两臂且不自动回零。
6. `Ctrl+C` 退出时同样执行停止、失能及无回零断开。

> 关闭或刷新网页不会结束服务端回放。结束请用停止按钮或服务端终端 `Ctrl+C`。
>
> 设备被占用时，请在原控制程序终端正常停止；本工具不会杀死其他控制程序。

## 数据约定

- 前 14 维依次为：左臂 6 关节、左夹爪、右臂 6 关节、右夹爪。**第 15 维只显示，绝不下发。**
- 关节统一按绝对弧度处理，发给驱动时转为度；夹爪保持原有 0–1 定义。
- 每个原始帧只下发一次，不删除迟到帧，不快速连续补发积压帧。
- 底层驱动运行于 100 Hz，页面同时显示设定频率与实际频率。
- 驱动的机械关节限位保持原样；若限位修正了目标，网页会计数，CSV 中同时保留原目标与驱动接受的目标。

## 输出格式

记录位于 `runs/<时间-编号>/`，不写入上游项目：

- `run.json` — 数据来源、字段、频率、模式、起点对齐参数
- `joints.csv` — 30 Hz 采样的目标、反馈、驱动接受目标、原始帧号、阶段、设定频率
- `summary.json` — 完成阶段、实际下发帧数、调度迟到、驱动限位计数

页面"导出关节 CSV"下载最近一次运行记录。

## 测试

```bash
python -m unittest -v test_replay
```

## 已验证范围

- 14 项自动检查通过；现有 546 条原始轨迹的 action / state 共 1092 个字段加载检查通过。
- 网页已验证选轮次、字段切换、暂停续播、7.5 / 15 / 30 / 60 Hz 设置，以及 CSV / PNG 下载。
- 上述验证均使用离线演示设备，**未连接或移动真机**；演示中的黑线为模拟反馈，不代表真机测量。

## 许可证

MIT，见 [LICENSE](LICENSE)。
