Piper 原始采集数据回放工具
==========================

本目录随 phoenixclone 发布，默认自动识别上一级 Phoenix 根目录；完整安装步骤见 README.md。
推理与回放共享 Phoenix 的轻量 piper_control.py 控制适配，迁移时一起复制。
工具不加载 Phoenix 权重，也不运行 RTC 或模型推理。

启动真机网页：
  cd piper-dataset-replay  # 从 phoenixclone 仓库根目录进入
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u app.py --hardware --port 8766

启动离线演示：
  cd piper-dataset-replay  # 从 phoenixclone 仓库根目录进入
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u app.py --demo --port 8766

浏览器打开 http://127.0.0.1:8766 。同一个端口只运行一个实例；Ctrl+C 停止网页服务。
已复用现有 Python 环境，无需另外安装依赖。迁移时可参考 requirements.txt。
硬件模式还需要机器人控制目录原有的 Piper SDK、scipy、pyrealsense2 和可用 CAN。

页面操作：
1. 选择数据集、原始 episode 编号（从 0 开始）以及 action / state，点击加载。
2. 选择起始帧、回放频率和起点对齐时长。
3. 在设备与执行中选择 Joint 或 MIT；MIT 的六项参数可修改，点击应用控制配置。
   已连接时先断开设备再修改。点击连接设备与相机只读取反馈和画面，不自动使能。
4. 点击开始真机回放。该按钮会使能双臂，按设定时长移动到起始帧，再逐帧回放。
5. 暂停/停止回放会保持当前姿态；停止并失能会停止两臂且不自动回零。
6. Ctrl+C 退出时也执行停止、失能及无回零断开。
关闭或刷新网页不会结束服务端回放；结束时使用停止按钮或服务端终端 Ctrl+C。

单帧定位（不连续回放）：
- 在进度条下输入原始帧号；“预览该帧”只切换录像位置，不移动机械臂。
- 连接设备后点“移动到该帧并保持”，按起点对齐时长到达该帧的 action/state 目标。
- 过渡结束后保持该固定目标，不进入下一帧；蓝线保持固定，黑线继续显示反馈并写入 CSV。
- “保持指定帧目标”表示目标已下发，并不宣称实际关节已经无误差到位。
- 可再次指定其他帧；“开始真机回放”将从当前选定帧开始连续回放。
- 暂停/停止仍使用当前反馈保持姿态，因此分析固定目标下的抖动时，直接观察单帧保持阶段。

频率与数据含义：
- 原采集频率为 30 Hz：15 Hz 为半速、30 Hz 为原速、60 Hz 为两倍速。
- 每个原始帧只下发一次，不删除迟到帧，不快速连续补发积压帧。
- Joint 驱动设定 100 Hz，MIT 默认 210 Hz；目标更新频率独立可选 (0,100] Hz。
  驱动实际频率包含 CAN 与调度开销，页面实际更新频率指原始帧的更新频率。
- action 是采集时的目标动作；state 是采集时测得的关节状态。
- 直接读取 datasets 下原始 parquet，不使用标准化数据或模型预测。
- 单位根据现有明确的源数据映射转换：关节统一为绝对弧度，发给驱动时转为度。
- 前 14 维为左臂 6 关节、左夹爪、右臂 6 关节、右夹爪。第 15 维只显示，绝不下发。
- 起点对齐是可配置的单独过渡；轨迹内部不增加平滑。设为 0 会直接发送首帧目标。
- 现有驱动的机械关节限位保持原样；若限位修正目标，网页计数且 CSV 同时保留原目标与驱动接受目标。

画面与图表：
- 采集录像：读取所选轨迹原始三路视频，与当前回放帧同步。
- 实时相机：显示外部、左腕、右腕三个 RealSense 画面。缺失画面会显示状态；视频不参与动作生成。
- 12 关节图：蓝线为回放目标，黑线和圆点为当前真实关节反馈，按本次运行时间对齐。
- 离线演示模式黑线明确标为“模拟反馈”，不代表真机测量。演示不导入或连接硬件。
- 图表可切换 rad / °、30 秒 / 60 秒 / 全部记录，并导出 PNG。

保存根目录可通过网页“保存根目录”或 --output-dir 修改，相对路径以本工具目录为基准，也支持绝对路径。
每次自动创建 <保存根目录>/<时间-编号>/，停止后后台导出 LeRobot v2.1：
- meta/info.json、tasks.jsonl、episodes.jsonl、episodes_stats.jsonl、stats.json、recording.json。
- data/chunk-000/episode_000000.parquet。
- videos/chunk-000/{image,wrist_image_l,wrist_image_r}/episode_000000.mp4（full 模式）。
每次运行是一份独立单 episode 数据集，字段沿用 state/actions/image/wrist_image_l/wrist_image_r。
页面显示导出状态，complete 才是成功。完整三摄像头缺失时不会生成假视频，原始诊断仍保留。
以下为同时保留的原始诊断文件：
- run.json：数据来源、字段、频率、模式、起点对齐参数。
- joints.csv：30 Hz 采样的目标、反馈、驱动接受目标、原始帧号、阶段与设定频率。
- commands.jsonl：每次应用层下发或保持操作的原始帧、时间、请求目标、驱动接受目标。
- feedback.jsonl：约 30 Hz 状态、动作、原始帧、第 15 维、时间及夹爪扭矩/故障位。
- frames/*.npz + *.json：full 模式三路现场 RGB、对应状态/目标及相机和反馈时间戳。
- summary.json：完成阶段、实际下发帧数、调度迟到、驱动限位计数。
页面“导出关节 CSV”下载最近一次运行记录。关节为 rad，夹爪为 0–1。

可选启动参数：
  --project-root ..
  --interface-root ../Desktop/piper_tele/sink-bot-control_bk/example_agilex
  --robot-config configs/inference/piper_robot.json   （相对 project-root）
  --output-dir runs                                 （相对本工具目录）
  --record full                                     （默认 actions 不保存图像）
  --record-image-hz 10                               （1–30 Hz，独立于回放频率）
  --record-fps 30                                    （LeRobot 导出时间轴 FPS）
  --port 8766

完整记录启动示例：
  cd piper-dataset-replay  # 从 phoenixclone 仓库根目录进入
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -u app.py --hardware --port 8766 --record full --record-image-hz 10
网页“04 本次运行记录”也可选内容与频率，再点“应用记录配置”。
并可输入保存根目录，例如 runs/lerobot-replay 或 /data/piper-recordings。
LeRobot 均匀时间轴使用过去最近一次采样保持；低频图像会重复，实际采样时间与年龄保留，不能当作新测量。
推理导出的 actions 为模型首步预测，predicted_actions 保留完整 32 步，明确标注为预测而非已执行动作。
暂停和单帧保持仍记录，停止/完成后结束。图像后台保存时仍能停止/失能；保存完成后再开始新一轮。
相机图像与状态按实际时间戳关联，不是硬件同步。演示模式无现场图像，缺失相机明确标记。
页面显示图像保存数量、丢弃、采样迟到、缺失和错误。summary.json.finalized 表示收尾完成，还需查看 error。
图像队列最多 4 组，慢磁盘可导致漏录，控制不等待图像写盘。正常退出会排空队列。
Phoenix 推理原命令将 --record actions 改成 --record full 即保存每轮输入的三路图像（按推理请求频率）。

运行验证：
  python -m unittest -v test_replay

MIT 参数（对双臂全部关节生效）：
- Kp=15：位置误差增益，SDK 范围 0..500。
- Kd=1：速度误差增益，SDK 范围 -5..5，负值为负阻尼。
- 目标速度=0 rad/s：范围 -45..45，为常量参考，不从轨迹求导。
- 前馈力矩=0 Nm：当前 SDK 实际编码范围 -8..8。
- 速度百分比=60：MotionCtrl_2 字段 0..100，与回放倍速无关。
- 驱动线程=210 Hz：可设 1..1000，重复当前目标，不新增插值。
默认值于 2026-11-02 同步自 teleop_inference 分支（798b5b65b44a），不需要切换控制仓库分支。
与 Phoenix 推理共用 configs/inference/piper_mit_defaults.json；移植时连同适配模块一起复制。
SDK 调用为 MotionCtrl_2(1,4,speed_limit,0xAD) 和每关节 JointMitCtrl（目标位置弧度）。
run.json.robot_control 记录实际模式、频率和参数。
旧进程在原终端 Ctrl+C 后按原启动命令重启，再刷新网页使用新选项。

已验证：14 项自动检查通过；现有 546 条原始轨迹的 action/state 共 1092 个字段加载检查通过。
网页已验证选轮次、字段切换、暂停续播、15/30/60 Hz 设置，以及 CSV/PNG 下载。
这些验证使用离线演示设备；没有连接或移动真机。示例黑线为模拟反馈。

设备被占用时请在原控制程序终端正常停止；本工具不会杀死其他控制程序。
本工具沿用驱动已有的 CAN 锁和按 USB 序列号绑定左右臂，保留右臂反馈 ID 偏移适配。
