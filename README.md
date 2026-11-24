# R2UHD

R2UHD 的训练、推理与评测代码，支持低光增强、去雾、去模糊、去雨和去雪。

**arXiv:**

## 环境安装

在独立 Python 环境中安装与本机 CUDA 匹配的 PyTorch 和 torchvision，再安装项目依赖：

```bash
python -m pip install -r requirements.txt
```

依赖要求为 PyTorch >= 2.0、torchvision >= 0.15，其余依赖见 `requirements.txt`。
所有命令均在 `r2uhd/` 目录下运行，直接使用目录中的本地 `basicsr` 包，无需额外安装 BasicSR。
Python 命令可用于 Windows；`train.sh` 和 `test.sh` 需要 Bash。

## 任务配置

| 任务 | 数据集 | 配置文件 |
| --- | --- | --- |
| 去雾 | UHD-Haze | `options/train_r2uhd_haze.yml` |
| 去雨 | 4K-Rain13k | `options/train_r2uhd_rain13k.yml` |
| 去雨 | UHD-Rain | `options/train_r2uhd_uhdrain.yml` |
| 去雪 | UHD-Snow | `options/train_r2uhd_snow.yml` |
| 去模糊 | UHD-Blur | `options/train_r2uhd_blur.yml` |
| 低光增强 | UHD-LL | `options/train_r2uhd_ll.yml` |

## 数据准备

按输入图像与清晰图像成对组织数据；配对图像的文件名和尺寸应一致。例如：

```text
datasets/UHD-Haze/
├── train/
│   ├── input/
│   └── gt/
├── val/
│   ├── input/
│   └── gt/
└── test/
    ├── input/
    └── gt/
```

在对应任务的 YAML 中修改下列四项路径，其他配置保持原文件内容。
目录可以使用数据集原有的 `low/high`、`input/target` 等名称，只需正确设置路径。
Windows 路径可写为 `D:/datasets/...`。

```yaml
datasets:
  train:
    dataroot_gt: datasets/UHD-Haze/train/gt
    dataroot_lq: datasets/UHD-Haze/train/input
  val:
    dataroot_gt: datasets/UHD-Haze/val/gt
    dataroot_lq: datasets/UHD-Haze/val/input
```

训练会从完整图像中随机裁剪；验证使用完整图像。独立测试通过推理命令中的 `-i` 和 `-g` 指定数据，不需要在 YAML 中添加测试集。

## 训练

以 UHD-Haze 为例：

```bash
python -m basicsr.train -opt options/train_r2uhd_haze.yml
```

其他任务使用上表对应的 YAML。也可以使用启动脚本：

```bash
bash train.sh options/train_r2uhd_haze.yml
```

六个配置文件提供的默认训练设置为：

| 项目 | 设置 |
| --- | --- |
| GPU 数量 | 1 |
| 随机种子 | 42 |
| 裁剪尺寸 | 768 × 768 |
| 每张 GPU 的 batch size | 6 |
| 梯度累积步数 | 2 |
| 总迭代数 | 602,000 |
| 优化器 | AdamW |
| 初始学习率 | 5e-4 |
| 权重衰减 | 1e-3 |

显存不足时可调整 `datasets.train.batch_size_per_gpu` 和 `train.accumulation_steps`；数据加载进程数通过 `datasets.train.num_worker_per_gpu` 设置。

### 多卡训练

在支持配置中 NCCL 后端的 CUDA 环境运行，例如使用两张 GPU：

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash train.sh options/train_r2uhd_haze.yml
```

### 训练输出与断点恢复

训练输出保存到 `experiments/<name>/`，其中 `<name>` 对应 YAML 的 `name`：

```text
experiments/<name>/
├── models/
├── training_states/
├── visualization/
└── <训练配置>.yml
```

自动续训命令：

```bash
python -m basicsr.train -opt options/train_r2uhd_haze.yml --auto_resume
```

续训需要同时保留 `training_states/<iter>.state` 和同迭代的 `models/net_g_<iter>.pth`。
只有模型权重时，可通过 YAML 中的 `path.pretrain_network_g` 初始化训练；恢复优化器和学习率调度状态需要训练状态文件。

## 推理与评测

权重需自行准备。以下示例假设权重位于 `pretrained_models/r2uhd_haze.pth`，请替换为实际路径。
`-c` 应指定与权重配套的训练 YAML；训练产物中的已保存配置可用于复现。

仅恢复图像：

```bash
python inference_r2uhd.py -i datasets/UHD-Haze/test/input -w pretrained_models/r2uhd_haze.pth -c options/train_r2uhd_haze.yml -o results/haze
```

提供 GT，同时计算 PSNR、SSIM 和可用的 LPIPS：

```bash
python inference_r2uhd.py -i datasets/UHD-Haze/test/input -g datasets/UHD-Haze/test/gt -w pretrained_models/r2uhd_haze.pth -c options/train_r2uhd_haze.yml -o results/haze
```

也可将相同参数传给启动脚本：

```bash
bash test.sh -i datasets/UHD-Haze/test/input -g datasets/UHD-Haze/test/gt -w pretrained_models/r2uhd_haze.pth -c options/train_r2uhd_haze.yml -o results/haze
```

恢复图像及 `inference_summary.json` 保存到 `-o` 指定的目录。JSON 包含逐图指标、平均指标、权重 SHA256、配置与评测参数。

| 参数 | 用途 |
| --- | --- |
| `-i` | 单张图像或图像目录；目录中的图像不递归读取 |
| `-g` | 可选 GT 目录，文件名与输入一致；单图输入也可指定 GT 文件 |
| `-w` | 模型权重路径 |
| `-c` | 与权重配套的训练 YAML |
| `-o` | 输出目录 |
| `--device cpu` | 使用 CPU；默认在 CUDA 可用时使用 GPU |
| `--amp` | 启用 CUDA FP16 推理；默认 FP32 |
| `--no-lpips` | 跳过 LPIPS |
| `--crop-border` | 覆盖 PSNR/SSIM 的边界裁剪像素数 |

输入应为 8 位 RGB 或灰度图像，不支持带 alpha 通道的输入。输出保留输入文件名和扩展名。
省略 `-c` 时，脚本要求权重目录中恰好有一个 YAML；若权重位于 `models/`，则从其上一级目录查找。

## 复现设置

- 使用与权重配套的 YAML；加载时会严格检查模型权重兼容性。
- 默认使用 FP32、整图推理，将尺寸补齐到 8 的倍数后裁回原始尺寸。复现时应保持相同的精度和补齐设置；脚本不提供分块推理。
- 测试 PSNR/SSIM 由 `comput_psnr_ssim.py` 在 Y 通道计算，边界裁剪默认读取 YAML 的 `val.metrics`，六个配置均为 1 像素。
- 训练验证使用 pyiqa，现有配置的 PSNR/SSIM 为 RGB 口径。报告测试指标时使用独立推理脚本，并保留 `inference_summary.json`。
- LPIPS 使用 torchmetrics 的 Alex 模型，在 CPU 上计算；首次使用需要下载相关权重。不可用时会跳过 LPIPS，仍计算 PSNR/SSIM。
- 指标在图像编码前的 uint8 数组上计算。需要从保存图像重新评测时，使用 PNG 等无损格式以避免有损编码影响。

## 致谢

代码基于 [UHDformer](https://github.com/supersupercong/UHDformer) 和 [BasicSR](https://github.com/XPixelGroup/BasicSR)。感谢原作者的开源工作。

## 许可

许可信息见上级目录的 [LICENSE](../LICENSE)。
