# 用 VGGT Omega Warp 监督 VERA Jacobian

## 可行性与边界

可以接入。现有 VERA image Jacobian 输出每个像素的 `J ∈ R^(2×A)`，
训练的是 `J(I_t) @ du ≈ flow`。本仓库 Omega-Warp 输出的是二维对应关系，
其 `warp` 是绝对目标坐标，`flow_px` 才是源像素到目标像素的位移。
因此新路径用冻结的 Omega-Warp 提供运动伪标签，并把 student 的 VGGT
backbone 换为 Omega aggregator，保留 VERA 的 Jacobian DPT decoder 和控制接口。

对同一 camera 的 `(I_t, I_(t+k))`：

```
teacher: Omega-Warp(I_t, I_(t+k)) → flow_px, certainty, valid
student: Omega(I_t 的各 camera) → DPT → J
loss: Σ valid·certainty·ρ(J @ normalized_du − normalized_flow_px)
      / Σ valid·certainty
```

这里监督的是 Jacobian 沿真实动作方向的结果，不是完整 Jacobian 真值。
VERA 的 Jacobian 是对机器人动作求导，不是对图像像素坐标求导；不能直接用
warp 的空间梯度替代它。
单个动作只能约束一个方向；仍需足够多样的机器人动作、局部小位移，以及与
动作相对应的图像变化。Warp 本身无法从无动作视频中确定机器人各动作维度的
Jacobian。相机运动、外物自主运动和匹配错误都可能污染标签。
`valid` 只检查坐标在图像内且数值有限，`certainty` 也不保证匹配正确。
实际收益需用机器人视频和下游控制实验验证，目前不能宣称优于原有 MegaFlow。
本次接入针对现有二维 image Jacobian；若要三维 scene Jacobian，还需要深度、
相机几何和坐标系一致的三维运动监督，二维 warp 本身不够。

## 已实现

- `models/vggt_omega_jacobian_field.py`：Omega patch16 aggregator，兼容
  `[B,3,H,W]` 和当前时刻的 `[B,V,3,H,W]`。可冻结或微调，复用原有 J/flow 输出。
- `warp_teacher.py`：读取对应 TOML、Omega backbone 权重和训练好的 warp head；
  验证 checkpoint 配置和 backbone 来源，冻结推理并按 pair 分批，不保存进 IDM 权重。
- `omega_warp_jacobian.py`：独立算法 `omega_warp_jacobian`；使用有效区域与置信度
  加权监督。可选 flow gradient、uncertainty、inverse-action 项同样遵守 mask。
  Jacobian TV 是独立先验，开启后仍覆盖整个图像。
- 数据集 `load_rgb_next=true`：加载 `t_src + linearize` 对应的目标帧；student
  从不接收 `rgb_next`。`load_flow=false` 时不读取 packed MegaFlow。

默认入口使用 MimicGen 的七维动作与两路 camera。DROID 的 time-aware 动作会根据
时间戳选择端点，当前新增路径明确拒绝此模式，避免把不匹配的帧和动作配在一起。
不能直接把这个配置用于不同动作定义的机器人。

## 权重到位后的运行方式

先在已安装 VERA 训练依赖（包括 `lightning` 和 `vggt`）的环境中安装本地包：

```bash
cd /path/to/vera
pip install -e ./vggt-omega-warp

export VERA_DATA_PREFIX=/path/to/data
export VERA_OMEGA_CHECKPOINT=/path/to/vggt_omega_1b_512.pt
export VERA_WARP_CHECKPOINT=/path/to/trained_warp_head.pt
export VERA_WARP_CONFIG=/path/to/matching_training_config.toml

python -m vera.main --config-name=config_jacobian_mimicgen_omega_warp \
  wandb.mode=disabled
```

三个权重/配置路径必须真实存在。没有 warp head 时会明确报错，不会使用随机 head
生成伪标签。原有 VERA checkpoint 不能直接作为新 Omega 模型的严格恢复权重；
这里从 Omega backbone 初始化，重新训练 Jacobian decoder。
Teacher 检查源代码指纹、模型配置和 backbone 权重身份，允许权重迁移到不同 GPU、
运行环境或 Git checkout 路径；代码版本不匹配仍会明确报错。

当前默认配置训练 student 的 Omega aggregator 和 Jacobian decoder；
`algorithm.model.freeze_aggregator=false`。历史冻结版实验显式使用 `true`。
每 GPU 每批一条序列，teacher 每次处理一个 pair，teacher backbone 和 warp head
均保持冻结。每个 DDP rank 持有独立的 teacher、student 和优化器状态。
多卡启动和新机器的数据准备见 [README](../README.md#omega-warp-training)。全无效的样本不会提供运动/逆动作梯度。
这些指标是对 teacher 的拟合程度，不能替代真值精度或控制成功率。

新配置把 resized RGB 像素位移乘 `0.1`；归一化保持零位移对应零。
原 MimicGen MegaFlow 的统计不自动视为 warp 的统计；可在拿到权重后重新测量，
或保持固定像素缩放。修改分辨率后也应重新检查尺度和 teacher 表现。

建议依次做：真实视频对应关系可视化；单批过拟合；固定数据和种子的
VGGT+原flow、Omega+原flow、Omega+warp 三组对照；最后测控制成功率。
这样可以分开判断 backbone 替换和监督替换各自的影响。

## 使用公开 MimicGen 原始图像

官方 VERA Hub 当前没有发布 `mimicgen_packed_v3_megaflow`。本地替代路径从
`amandlek/mimicgen_datasets` 下载原配置对应的九个 core 任务（共约 15.9 GB），
固定版本并验证 SHA-256，再转换为 RGB/轨迹 NPZ：

```bash
python scripts/data/download_mimicgen.py
python -m scripts.data.pack_mimicgen --workers 3
```

转换保留官方记录的 84×84 双相机图像，加载时缩放为 128×128；不重新渲染，
不生成 MegaFlow，也不声称与作者的高分辨率预处理包等价。默认输出为
`data/datasets/jacobian/mimicgen_official_rgb_warp`，共 9,000 条轨迹，任务顺序
与原配置一致，因此保留九个验证集的 episode pin。完整 index 仅在全部转换
成功后写出；转换支持在同一目录续跑。

从训练好的 Warp checkpoint 恢复配置和推理权重：

```bash
python scripts/prepare_omega_warp_teacher.py vera/checkpoints/best.pt \
  --backbone /path/to/vggt_omega_1b_512.pt \
  --output-dir outputs/omega_warp_preflight
```

工具使用受限的 weights-only 加载和明确列出的 NumPy 类型，验证 backbone
SHA-256，移除 optimizer/RNG 状态，并逐 tensor 检查导出的 head 与原文件一致。
原 checkpoint 保持不变，teacher 的源码、特征和模型配置一致性检查仍正常执行。

```bash
# 先转换两条独立测试轨迹，运行两步训练检查。
python -m scripts.data.pack_mimicgen --tasks stack_d0 --limit-per-task 2 \
  --workers 1 --output-root outputs/omega_warp_preflight/pack_smoke
bash scripts/train_mimicgen_omega_warp.sh smoke

# 完整数据转换结束后启动正式训练。
bash scripts/train_mimicgen_omega_warp.sh train
```

启动脚本默认使用仓库内 `.venv/bin/python` 和 GPU 1，输出位于
`outputs/mimicgen_omega_warp/`。可用 `VERA_PYTHON`、`CUDA_VISIBLE_DEVICES`、
`VERA_OMEGA_CHECKPOINT`、`VERA_WARP_CHECKPOINT`、`VERA_WARP_CONFIG` 覆盖。
所有 checkpoint 加载均设置 `TORCH_FORCE_WEIGHTS_ONLY_LOAD=1`。
入口为 `config_jacobian_mimicgen_omega_warp_official_rgb`，与原包配置分开。

## 组件测试范围

`python -m pytest -q tests`：49 项测试在本机训练环境中通过。覆盖 pair 顺序、
归一化、mask 梯度、已知 Jacobian 逆解、单/多视图、冻结/解冻、严格 checkpoint
加载、BF16/FP32 可视化，以及关闭 logger 后保留验证 loss 的行为。
Backbone 单元测试使用真实 24 层、宽度 64 的小 Omega；真实 1B 权重的训练检查
见下一节。单元测试不能替代机器人控制成功率验证。

## 本机真实权重检查（2026-09-29）

在 r34 的 RTX A5000 上，已使用提供的 `best.pt`（Warp epoch 119 / step
56,280）完成两步真实训练并保存 Lightning checkpoint。Warp 源码指纹和
Omega backbone SHA-256 均与 checkpoint 中的记录一致；导出推理 head 与
原 head 逐 tensor 一致。训练 checkpoint 的 global_step 为 2、所有 student
浮点权重均有限，保存的训练 loss 为 0.11966337。这只是训练链路检查，
不代表模型已经收敛或控制成功率已经验证。

证据保存在 `outputs/omega_warp_preflight/`：`data_validation.json`、
`train_smoke_validation.json`、`backbone_verified.json`、`runtime.json`，以及
下载、转换和训练日志。本机复用已有 PyTorch 2.12.0+cu126 runtime；新增依赖
安装在仓库 `.venv` 中，原 conda 环境未更改。

完整启动验证还暴露了 BF16 可视化的 dtype 问题，现已修复：单位动作矩阵
保持输入 dtype，转换为 NumPy 前将光流转为 float32。新增的 BF16/FP32
可视化回归测试均通过。

另外修复了 `wandb.mode=disabled` 时验证阶段仍调用空 logger 的问题：
loss 计算和标量记录继续执行，仅跳过媒体日志。对应回归测试保存在
`tests/test_validation_without_logger.py`。

九个官方原始任务已全部下载并通过 SHA-256 校验，9,000 条转换结果以及九个
固定验证 episode 均通过真实数据加载器检查。正式单 GPU 训练已通过全部
10 个启动验证 dataloader（九任务加 all），并进入连续训练。
运行配置、PID 和输出目录记录在 `outputs/omega_warp_preflight/run_summary.json`，
实时日志为 `outputs/omega_warp_preflight/train_full.log`。配置保留 600,000 步上限；
该记录表示训练已启动，不表示训练已经完成或收敛。


## 按官方 MimicGen 配置运行独立验证

```bash
CUDA_VISIBLE_DEVICES=2 bash scripts/validate_mimicgen_omega_warp.sh \
  /absolute/path/to/checkpoint.ckpt outputs/my_validation
```

这个入口复用原版 `BaseLightningExperiment.validation`、数据 loader 和
`ImageJacobian.validation_step`，继承原配置的 9 个 task pin（100、1100、…、8100）、
每 loader 一个 batch、batch size 1、8 帧、两路 128×128 图像和 `16-mixed` 精度。
它将 `validation_results.json` 与 W&B **离线**指标/视频写到输出目录，不上传 W&B。
Omega 验证还会计算归一化动作反解 MSE，即使该项训练权重为零；不会改变训练 loss。

原版 `all` 是拼接后的第一个 batch，因此在这个配置下重复 coffee_d0，并非九任务平均。
这些 pin 也包含在训练 episode pool 中，不是独立留出验证集。Warp teacher 伪标签与
原版 MegaFlow 不同，loss 不可直接横向比较；这些指标也不是仿真任务成功率。

2026-10-05 的验证输出位于 `outputs/omega_warp_validation_20261005/`，其中
`protocol.json` 记录实际数据、checkpoint 和协议差异；两个 checkpoint 的独立结果
保存在 `latest/` 和 `best_train/`。运行以下命令生成排除重复 `all` 的九任务平均和逐任务 CSV：

```bash
python scripts/summarize_mimicgen_validation.py outputs/omega_warp_validation_20261005
```


## 2026-10-05 官方 checkpoint 直接对照与机器人仿真

对照使用官方发布的 `idm-mimicgen-285ouq1q/model.ckpt`；新版固定为第 397,000 步。
模型文件固定版本并通过 SHA-256 校验，来源见
`outputs/official_comparison/assets_manifest.json`。

在相同九个 task pin、RGB、动作和 Warp teacher 标签上，统一光流单位后：
官方 / 新版的平均光流 EPE 为 **1.24998 / 0.56665 px**，归一化动作反解 MSE 为
**2.16832 / 0.30886**。这不是原版 MegaFlow loss 的复现；Warp 参考标签可能偏向新版。

机器人实验使用官方 `MimicgenRunner`、`RemotePolicy` 和 `VeraPolicyAdapter`，
仅以本地调用替代 websocket 传输。两组共用发布的 MimicGen WAN 1.3B、CoTracker3、
40 步去噪、TeaCache 0.10、21 帧上下文、每次执行 10 个动作及相同 gripper 控制器。
`stack_d0` 的相同 10 个初始状态和 seed 0–9，每回合上限 200 步；官方成功 **7/10**，
新版成功 **0/10**。这是小规模仿真实验，不是实机或独立留出集的成功率。

新版推理保持训练时的同时双相机注意力，并将 Jacobian 转成官方控制器使用的光流单位；
动作归一化统计相同。没有针对任一模型调控制器参数。原始训练 RGB 是白桌面，官方
runner 自动应用深木纹桌面，两组 rollout 使用相同深木纹场景；该输入域差异可能影响结果，
尚未做因果消融。发布的 planner 配置为 `skip_text_encoder=true`，两组均按原配置使用空文本嵌入。

运行中修复了两个共用接口问题：`MotionTrackConfig.backend` 被忽略而错误选择 AllTracker；
CoTracker 可视化的 visibility 类型及坐标设备不一致。完整 54 项测试通过。MimicGen
所需 `env_class` 接口来自 robomimic 0.5.0（具体 commit 记录在协议中）。

结果、逐回合 CSV、协议和并排视频：
`outputs/official_comparison/comparison_summary.json`、`robot_per_episode.csv`、
`robot_protocol.json`、`official_vs_omega_demo1.mp4`。
每个回合另有原始视频及动作/状态轨迹。

以下为原实验机器上的命令（在 repo 根目录）。`outputs/` 下的模型快照、配置、
下载清单和结果不提交到 Git，迁移这些对照实验需要另行复制对应文件；
新机器的数据准备与多卡训练使用 [README](../README.md#omega-warp-training) 中的步骤：

```bash
source outputs/official_comparison/env.sh
CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/compare_mimicgen_idm.py \
  --output outputs/official_comparison/official_val
CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/run_mimicgen_comparison.py \
  --variant official --output outputs/official_comparison/robot_official
CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/run_mimicgen_comparison.py \
  --variant omega --output outputs/official_comparison/robot_omega
python scripts/summarize_mimicgen_comparison.py outputs/official_comparison
```
