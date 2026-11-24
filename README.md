# R2UHD

R2UHD 是 `relitev17`（revision 171）模型的整理版：架构注册名 **`R2UHDNet`**，
训练/验证模型 **`R2UHDModel`**，全部代码收敛在 `r2uhd/` 这一个文件夹内。

- 单模型、单尺度（`scale: 1`）、RGB 3 通道 uint8 输入输出。
- 338,147 个参数，整帧推理（4K 输入约 2.2 GiB 显存，RTX 5060 Ti 上 warm 约 0.5 s/帧）。
- H/4 引导 H/8、H/8 反馈 H/4 的三段双向结构；H/8 上用 16 个原型的互补聚类注意力；
  整帧聚类注意力，不支持分块推理。
- 训练：L1 + SSIM + FFT（FP32，AdamW，可选梯度累积）。
- 测试指标：原 `comput_psnr_ssim.py` 原样使用，一行未改。

## 目录结构

```
r2uhd/
├── basicsr/                    # 精简后的 BasicSR（只保留 R2UHD 需要的部分）
│   ├── archs/r2uhd_arch.py     # R2UHDNet
│   ├── models/r2uhd_model.py   # R2UHDModel
│   ├── models/base_model.py    # 训练/验证/断点等基础设施
│   ├── models/lr_scheduler.py  # CosineAnnealingRestartCyclicLR
│   ├── losses/                 # L1 / MSE / Charbonnier / WeightedTV / FFT
│   ├── data/                   # PairedImageDataset + 裁剪/增强 + prefetch
│   ├── utils/                  # registry / options / logger / img / matlab 工具
│   ├── train.py                # 训练入口
│   └── test.py                 # 可选的 BasicSR 测试入口（需要 test 数据集段）
├── options/                    # 六个任务配置
│   ├── train_r2uhd_haze.yml
│   ├── train_r2uhd_rain13k.yml
│   ├── train_r2uhd_uhdrain.yml
│   ├── train_r2uhd_snow.yml
│   ├── train_r2uhd_blur.yml
│   └── train_r2uhd_ll.yml
├── inference_r2uhd.py          # 测试脚本（由 inference_relitev17.py 改名，逻辑仅改必要处）
├── comput_psnr_ssim.py         # 原指标实现，原样复制，未修改
├── train.sh                    # 训练启动脚本（支持多卡、--auto_resume）
├── test.sh                     # 测试启动脚本（参数直传）
└── requirements.txt
```

## 环境

```bash
# PyTorch 按自己的 CUDA 版本安装（https://pytorch.org）
pip install -r requirements.txt
```

不需要 `pip install` 本仓库：在 `r2uhd/` 目录里运行脚本时，`basicsr` 解析为同目录下的
本地包。请勿同时把另一份 `basicsr` 装进同一环境，否则可能导入到别的版本。
`pyiqa` 只用于训练时的验证指标；测试的 PSNR/SSIM 走 `comput_psnr_ssim.py`
（它的 `import skimage.metrics` 需要 `scikit-image`）。LPIPS 需要可选依赖
`torchmetrics`，没装时会告警并跳过。

## 数据准备

训练使用成对目录，GT 和输入文件名一致：

```
datasets/UHD-Haze/train/gt/*.png
datasets/UHD-Haze/train/input/*.png
```

把 `options/train_r2uhd_*.yml` 里的 `/path/to/train/gt`、`/path/to/val/gt` 等占位路径
改成自己的目录即可。

## 训练

```bash
# 单卡，默认 options/train_r2uhd_haze.yml
bash train.sh

# 指定任务
bash train.sh options/train_r2uhd_snow.yml

# 断点续训（自动找最新一对 <iter>.state + net_g_<iter>.pth）
bash train.sh options/train_r2uhd_snow.yml --auto_resume

# 多卡
NPROC_PER_NODE=4 bash train.sh options/train_r2uhd_blur.yml

# 等价的 Python 命令（Windows 也可以直接这样跑）
python -m basicsr.train -opt options/train_r2uhd_haze.yml
```

产物在 `experiments/<name>/`：`models/`（`net_g_<iter>.pth`、`net_g_best_<iter>.pth`、
`net_g_latest.pth`）、`training_states/`、日志与可视化目录。`--auto_resume` 需要与
`<iter>.state` 同迭代的 `net_g_<iter>.pth` 同时存在。

配置中的几个约定：

| 键 | 行为 |
|---|---|
| `train.accumulation_steps` | 梯度累积。loss 按累积步数取平均，每 N 次迭代才 `optimizer.step()`；学习率调度仍按迭代步进。 |
| `train.optim_d` | R2UHD 没有判别器，此段被忽略（保留是为了旧配置可直接复用）。 |
| `train.scheduler.type` | 支持 `CosineAnnealingLR` 与 `CosineAnnealingRestartCyclicLR`；六个配置用后者，`periods` 之和小于 `total_iter` 时超出部分保持最后一个 `eta_min`。 |
| `val.metrics` | 训练验证用 pyiqa 计算；测试脚本另用 `comput_psnr_ssim`（见“指标口径”）。 |

## 测试（推理）

```bash
# 只推理，不算指标
bash test.sh -w /path/to/r2uhd_haze.pth -i /path/to/test/input -o results/haze

# 有 GT 时同时算 PSNR/SSIM/LPIPS，并写 inference_summary.json
bash test.sh -w /path/to/r2uhd_haze.pth -i /path/to/test/input -g /path/to/test/gt \
    -c options/train_r2uhd_haze.yml -o results/haze

# 也可以用环境变量传参
WEIGHT=/path/to/r2uhd_haze.pth INPUT=/path/to/test/input OUTPUT=results/haze bash test.sh

# 等价 Python 命令
python inference_r2uhd.py -i /path/to/test/input -g /path/to/test/gt \
    -w /path/to/r2uhd_haze.pth -c options/train_r2uhd_haze.yml -o results/haze
```

- 脚本由 `inference_relitev17.py` 改名而来，只改了模型/配置名与默认输出目录；数值路径
  与原脚本完全一致（同一权重、同一批图，逐图 PSNR/SSIM 差为 0）。
- `-c` 省略时，要求权重所在目录（若权重在 `models/` 下则为其上一级目录）里恰好只有
  一个 yml，脚本自动读取；旧 yml 里写 `ReLiteV17Net` 也能识别（会提示后按 `R2UHDNet`
  加载，参数布局相同）。
- PSNR/SSIM 用 `comput_psnr_ssim.calculate_psnr/calculate_ssim`，**固定 Y 通道**
  （与原脚本一致），`crop_border` 从 yml 的 `val.metrics` 读，可用 `--crop-border` 覆盖。
- LPIPS 可选（torchmetrics Alex，CPU），`--no-lpips` 关闭；没装 torchmetrics 时会告警跳过。
- 默认 FP32、padding 到 8 的整数倍（原生几何）；`--amp` 打开 CUDA FP16。
- 输出：恢复图（保持原扩展名）+ `inference_summary.json`（逐图指标、均值、权重
  sha256、网络配置、协议说明）。指标和原脚本一样在 **编码前** 对 uint8 数组计算；
  jpg 输出再解码会带压缩损失，想复现报告里的数字请让输入/输出用 png。

`test.sh` 只是启动器：所有参数原样转给 `inference_r2uhd.py`，另外支持
`WEIGHT`、`INPUT`、`GT`、`CONFIG`、`OUTPUT`、`DEVICE`、`PYTHON_BIN` 环境变量。

## 指标口径

测试脚本（`comput_psnr_ssim`，固定 Y 通道）和训练验证（pyiqa，按 yml 的
`test_y_channel`，六个配置是 RGB）**不是同一口径**，数值不能直接对比。
同一批 4K 恢复图上的实测：

| 指标 | comput_psnr_ssim（测试用） | pyiqa（训练验证用） |
|---|---|---|
| Y 通道 PSNR（crop 1） | 30.1648 | 28.8428 |
| Y 通道 SSIM（crop 1） | 0.9684 | 0.9596 |
| RGB PSNR（crop 1） | 28.0484 | 28.0484 |
| RGB SSIM（crop 1） | 0.9471 | 0.9544 |

原因：原实现的 Y 是 MATLAB BT.601 YCbCr（亮度 16–235），比 pyiqa 的 YIQ（0–255）
正好高 `10·log10((255/219)²) = +1.32 dB`；RGB SSIM 原实现是 11×11×11 的 3D 高斯
（跨通道卷积），pyiqa 是逐通道 SSIM 后平均。表里的数是在保存的 jpg 输出上重新解码后
算的，所以比 `inference_summary.json` 里的编码前数值低约 0.01 dB，只用于对比两套实现。

所以论文/表格数字以测试脚本为准；训练日志只用于看趋势和挑最优 checkpoint
（`key_metric: psnr` 是 pyiqa 口径）。

## 六个任务的配置与权重

| 配置 | 参考权重文件 | 数据集 |
|---|---|---|
| `options/train_r2uhd_haze.yml` | `r2uhd_haze.pth` | UHD-Haze |
| `options/train_r2uhd_rain13k.yml` | `r2uhd_rain13k.pth` | 4K-Rain13k |
| `options/train_r2uhd_uhdrain.yml` | `r2uhd_uhdrain.pth` | UHD-Rain |
| `options/train_r2uhd_snow.yml` | `r2uhd_uhdsnow.pth` | UHD-Snow |
| `options/train_r2uhd_blur.yml` | `r2uhd_blur.pth` | UHD-Blur |
| `options/train_r2uhd_ll.yml` | `r2uhd_uhdll.pth` | UHD-LL |

权重不随代码目录分发，用 `-w` 指定即可。

## 权重兼容性

- 每个检查点都带 `_architecture_revision = 171`。加载时严格校验 revision 与全部
  参数名/形状（`strict=True`），不匹配会直接报错，不会静默加载错误的权重。
- 旧 `relitev17` 训练代码产出的 `.pth` 可以直接加载。
- 旧的 yml 如果还写着 `network_g.type: ReLiteV17Net`，测试脚本会给出警告并按
  `R2UHDNet` 处理（参数布局完全相同）；训练脚本要求写 `R2UHDNet`。

## 相对 relitev17 的改动

- 命名统一为 R2UHD（arch、model、配置、脚本、默认输出目录）。
- `basicsr` 只保留 R2UHD 需要的 arch/model/loss/data/utils，删除了 V1–V16、Z1/Z2
  系列以及 SwinIR/FeMaSR/GAN/感知损失等无关代码。
- 推理脚本只改名 + 改必要逻辑（模型名、配置校验、默认输出目录），指标计算与协议不变。
- `train.accumulation_steps` 现在真正生效（旧模型会静默忽略该键）。
- 修复 `--auto_resume` 只恢复优化器/调度器、不恢复网络权重的问题。
- 修复 `CosineAnnealingRestartCyclicLR` 在超过 `sum(periods)` 时崩溃的问题。

## 已知限制

- 整帧聚类注意力不支持分块推理，显存占用随像素数增长。
- LPIPS 需要 `torchmetrics`（首次运行还要联网下载 Alex 权重）；缺失时只算 PSNR/SSIM。
- `comput_psnr_ssim.py` 原样保留，其中 RGB SSIM 分支固定调用 `.cuda()`；测试脚本固定走
  Y 通道，不会触发该分支。纯 CPU 机器上不要用 `test_y_channel=False` 调用它。
- `train.sh` / `test.sh` 需要 bash；Windows 上可直接按上面文档里的 Python 命令运行。

## 许可

代码源自 UHDformer / BasicSR，沿用上级目录 `LICENSE` 的非商用许可。
