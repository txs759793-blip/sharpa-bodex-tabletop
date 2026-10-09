# Sharpa Wave × BODex：桌面抓取数据生成

把 Sharpa Wave 右手接入 [BODex](https://github.com/JYChen18/BODex) 生成桌面抓取，用 [DexGraspBench](https://github.com/JYChen18/DexGraspBench) 在 MuJoCo 里做物理评测，并通过调整生成参数把成功率从 34.9% 提高到 74.5%。

本仓库只包含：接入配置、对上游仓库的小补丁、运行脚本、固定的测试场景清单和结果。上游代码、物体数据集、Sharpa 官方手模型需要按下面的说明自行获取。

## 结果

评测条件固定：桌面抓取（手抓住物体后抬起 10 cm），物体 100 g，摩擦系数 0.6，**alpha=1，手的 kp 统一为 5**。固定随机抽取的 1000 个场景，每个场景 32 个候选，共 32000 个。成功率 = 成功的候选数 ÷ 全部候选数，不做任何筛选。

| 生成参数 | alpha=1、kp=5 | 官方 kp |
|---|---|---|
| 原始参数（`tabletop_baseline`） | 34.86% | 43.57% |
| 只改优化器三项 | 62.64% | 67.97% |
| **优化器三项 + 预抓取距离 1.5 cm（`tabletop_best`）** | **74.49%** | 73.10% |

`tabletop_best` 相对原始参数只改了四处：

| 参数 | 位置 | 原值 | 新值 |
|---|---|---|---|
| 迭代次数 `n_iters` | BODex `gradient_grasp_fc.yml` | 500 | 1500 |
| 步长 `base_scale` | 同上 | [0.01, 0.1, 0.1] | [0.02, 0.2, 0.2] |
| 第一阶段占比 `opt_progress` | 任务配置 | [0, 0.6, 0.8] | [0, 0.8, 0.9] |
| 预抓取距离 `distance` | 任务配置 | [0.01, 0.01, 0] | [0.015, 0.015, 0] |

更多数字、试过但无效的改法、已知局限见 [results/RESULTS.md](results/RESULTS.md)。

名词解释：
- **alpha**：挤压幅度系数。挤压姿态关节角 = 抓取 + alpha ×（抓取 − 预抓取）。BODex 保存的就是 alpha=1。
- **kp**：仿真中手指关节位置控制的比例增益。DexGraspBench 标准为所有关节 5；Sharpa 官方 XML 每个关节不同（0.9 到 13.2）。

## 环境

本结果在以下环境得到：Ubuntu 20.04，NVIDIA A800 80GB，驱动 560.35.03，Python 3.10，torch 2.2.2+cu121，mujoco 3.3.2，numpy 1.26.4，warp 1.17.0。

### 1. 获取上游代码（版本要一致）

```bash
git clone https://github.com/JYChen18/BODex.git          && git -C BODex checkout 06b9a3c90870d33bde9d6c665d4ed2819471407e
git clone https://github.com/JYChen18/DexGraspBench.git  && git -C DexGraspBench checkout d9ea6cf282de1f463c20fa54b4f68d7025bad40e
git clone https://github.com/sharpa-robotics/sharpa-urdf-usd-xml.git && git -C sharpa-urdf-usd-xml checkout 0d19cac602f46456b819e4b6a2c09a74982c9a3e
```

### 2. 打补丁

```bash
git -C BODex apply /path/to/this_repo/patches/bodex.patch
git -C DexGraspBench apply /path/to/this_repo/patches/dexgraspbench.patch
```

- `bodex.patch`：(a) cuRobo 慢速自碰撞分支每次调用前把输出缓冲清零。Sharpa 的碰撞球有 601 个，会走这个分支；不清零时上一次的结果会残留，生成成功率接近 0。(b) 两处与新版 warp / 编译环境的兼容性修改。
- `dexgraspbench.patch`：(a) 读取场景路径时兼容字符串和列表两种格式。(b) 与 mujoco 3.3 的接口兼容。**评测判定逻辑和参数没有改动。**

### 3. 按 BODex 的说明安装环境并准备物体数据

按 BODex 的 README 创建环境、编译安装，并下载 DGN_2k 物体数据，放到 `BODex/src/curobo/content/assets/object/DGN_2k/`（需要其中的 `processed_data/` 和 `scene_cfg/`）。同一个环境里还需要装好 DexGraspBench 的依赖（见它的 README，至少包括 `mujoco` 和 `hydra-core`），因为评测也用这个环境运行。

### 4. 填写路径

```bash
cp paths.example.yaml paths.yaml   # 然后把里面五个路径改成你机器上的绝对路径
```

脚本首次运行时会在 `<work_dir>/resolved/` 下生成带绝对路径的配置副本，并把手的评测配置写到 `DexGraspBench/config/hand/sharpa_wave.yaml`。除此之外不会改动上游仓库，也不会改动 Sharpa 官方模型文件（kp=5 用的是副本）。

## 复现

以下命令都在本仓库根目录运行，使用 `paths.yaml` 里那个环境的 Python。

```bash
# 1) 原始参数：生成 + 评测（官方 kp）。单张 A800 约 1.5 小时
python scripts/run_pipeline.py --config tabletop_baseline --name baseline --start 0 --end 1000 --gpu 0
# 2) 最好的参数。约 4.5 小时；可以分段在多张卡上同时跑，见下
python scripts/run_pipeline.py --config tabletop_best --name best --start 0 --end 1000 --gpu 1
# 3) 在标准条件 alpha=1、kp=5 下重新评测（只用 CPU，各约 10 分钟）
python scripts/reeval_alpha_kp.py --name baseline --alpha 1 --kp 5
python scripts/reeval_alpha_kp.py --name best --alpha 1 --kp 5
# 4) 汇总
python scripts/summarize.py baseline best
```

预期：`baseline` 在 alpha=1、kp=5 下约 34.9%，`best` 约 74.5%（标准误约 ±0.8 个百分点）。

想先快速检查流程是否通，把 `--end 1000` 改成 `--end 40`，十几分钟就能跑完。40 个场景上的参考值见 `results/RESULTS.md`。

**分段并行**：结果按 `--name` 存放，所以不同段要用不同的名字，例如 `--name best_0 --start 0 --end 250 --gpu 0`、`--name best_1 --start 250 --end 500 --gpu 1`，以此类推，最后分别评测再把成功数相加。本仓库报告的 1000 场景结果就是按 250 一段、四张卡跑出来的。

**注意分段大小**：BODex 每次同时解 8 个场景。分段或分批时不要让某一批只剩 1 个场景（即每个 chunk 的场景数除以 8 不要余 1），否则那一批可能输出无效位姿，脚本会报错停下。默认每个 chunk 40 个场景，1000、250、160、40 这些段长都没有问题。

**中断续跑**：同样的命令再跑一遍，会跳过已经完成的 chunk。

## 可视化

```bash
pip install viser trimesh mujoco numpy          # 建议装在单独的虚拟环境里
python scripts/viewer.py --left baseline --right best --port 8768
# 本机：ssh -L 8768:127.0.0.1:8768 <服务器>，然后浏览器打开 http://127.0.0.1:8768
```

左右并排显示同一场景下两组参数的抓取，可切换场景、候选编号、预抓取 / 抓取 / 挤压三个姿态，并显示成功或失败。

## 目录

```
configs/robot/           Sharpa 手接入 BODex 的配置：机器人配置、601 个碰撞球、手掌位姿对齐、URDF
configs/generation/      生成任务配置：tabletop_baseline.yml（原始）、tabletop_best.yml（最好）
configs/dexgraspbench/   评测用的手配置
patches/                 对 BODex 和 DexGraspBench 的补丁
scenes/                  固定的 1000 个桌面场景清单（随机种子 20261009，从 1918 个物体的 43256 个桌面场景中抽取）
scripts/                 run_pipeline.py（生成 + 转换 + 评测）、reeval_alpha_kp.py、summarize.py、viewer.py
results/                 结果汇总
```

## 许可与来源

- `configs/robot/right_sharpa_wave_bodex.urdf` 是为配合 BODex 使用而准备的 Sharpa 官方 URDF 的改写副本（Apache License 2.0，Copyright 2025 Sharpa Group），网格文件不在本仓库，运行时指向你本地的官方仓库。见 [NOTICE](NOTICE)。
- `configs/generation/*.yml` 以 BODex 自带的 Shadow 手配置为模板。
- BODex、DexGraspBench、cuRobo、DGN_2k 数据各自遵循其原始许可。
