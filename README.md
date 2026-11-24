# R2UHD

Training, inference, and evaluation code for UHD image restoration, including low-light enhancement, dehazing, deblurring, deraining, and desnowing.

**arXiv:**

## Installation

```bash
git clone https://github.com/Sujiaxu999/R2UHD.git
cd R2UHD
```

Use a dedicated Python environment. Install PyTorch and torchvision for your CUDA environment, then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

The requirements specify PyTorch >= 2.0 and torchvision >= 0.15. See `requirements.txt` for the remaining dependencies.
Run all commands from the repository root to use the bundled `basicsr` package; a separate BasicSR installation is unnecessary.
Python commands can be used on Windows. The `train.sh` and `test.sh` scripts require Bash.

## Task Configurations

| Task | Dataset | Configuration |
| --- | --- | --- |
| Dehazing | UHD-Haze | `options/train_r2uhd_haze.yml` |
| Deraining | 4K-Rain13k | `options/train_r2uhd_rain13k.yml` |
| Deraining | UHD-Rain | `options/train_r2uhd_uhdrain.yml` |
| Desnowing | UHD-Snow | `options/train_r2uhd_snow.yml` |
| Deblurring | UHD-Blur | `options/train_r2uhd_blur.yml` |
| Low-light enhancement | UHD-LL | `options/train_r2uhd_ll.yml` |

## Dataset Preparation

Organize degraded inputs and ground-truth images in paired directories. Each pair should have matching filenames and dimensions. For example:

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

Update the following four paths in the task YAML, keeping its other settings. The excerpt below shows only the dataset paths:

```yaml
datasets:
  train:
    dataroot_gt: datasets/UHD-Haze/train/gt
    dataroot_lq: datasets/UHD-Haze/train/input
  val:
    dataroot_gt: datasets/UHD-Haze/val/gt
    dataroot_lq: datasets/UHD-Haze/val/input
```

Dataset directories may retain names such as `low/high` or `input/target`; point `dataroot_lq` to degraded inputs and `dataroot_gt` to ground truth. Windows paths can use forward slashes, such as `D:/datasets/...`.
Training uses random crops; validation uses full images. The inference script takes test data through `-i` and `-g`, so no test dataset section is required in the YAML.

## Training

Train on UHD-Haze:

```bash
python -m basicsr.train -opt options/train_r2uhd_haze.yml
```

For other tasks, select the corresponding YAML from the table above. Alternatively, use the launcher:

```bash
bash train.sh options/train_r2uhd_haze.yml
```

The six supplied configurations use these default settings:

| Setting | Value |
| --- | --- |
| GPUs | 1 |
| Random seed | 42 |
| Crop size | 768 × 768 |
| Batch size per GPU | 6 |
| Gradient accumulation steps | 2 |
| Total iterations | 602,000 |
| Optimizer | AdamW |
| Initial learning rate | 5e-4 |
| Weight decay | 1e-3 |

Adjust `datasets.train.batch_size_per_gpu` and `train.accumulation_steps` for available GPU memory. Set data-loading workers through `datasets.train.num_worker_per_gpu`.

### Multi-GPU Training

Use a CUDA environment that supports the configured NCCL backend. For example, train on two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash train.sh options/train_r2uhd_haze.yml
```

### Checkpoints and Resume

Training outputs are stored in `experiments/<name>/`, where `<name>` is the YAML's `name`:

```text
experiments/<name>/
├── models/
├── training_states/
├── visualization/
└── <training_config>.yml
```

Resume training automatically:

```bash
python -m basicsr.train -opt options/train_r2uhd_haze.yml --auto_resume
```

Keep both `training_states/<iter>.state` and the matching `models/net_g_<iter>.pth`.
To initialize from model weights alone, set `path.pretrain_network_g` in the YAML. Restoring optimizer and scheduler state also requires the training state file.

## Inference and Evaluation

Prepare a model checkpoint separately. The examples below assume `pretrained_models/r2uhd_haze.pth`; replace this with your actual checkpoint path.
Use `-c` to specify the matching training YAML. The configuration saved with the training outputs can also be used.

Restore images without ground truth:

```bash
python inference_r2uhd.py -i datasets/UHD-Haze/test/input -w pretrained_models/r2uhd_haze.pth -c options/train_r2uhd_haze.yml -o results/haze
```

Provide ground truth to calculate PSNR, SSIM, and LPIPS when available:

```bash
python inference_r2uhd.py -i datasets/UHD-Haze/test/input -g datasets/UHD-Haze/test/gt -w pretrained_models/r2uhd_haze.pth -c options/train_r2uhd_haze.yml -o results/haze
```

The launcher accepts the same arguments:

```bash
bash test.sh -i datasets/UHD-Haze/test/input -g datasets/UHD-Haze/test/gt -w pretrained_models/r2uhd_haze.pth -c options/train_r2uhd_haze.yml -o results/haze
```

Restored images and `inference_summary.json` are saved to the output directory. The JSON report includes per-image and average metrics, the checkpoint SHA256, configuration, and evaluation settings.

| Argument | Description |
| --- | --- |
| `-i` | Input image or directory; directory scanning is non-recursive |
| `-g` | Optional ground-truth directory with matching filenames, or a ground-truth file for a single input |
| `-w` | Model checkpoint path |
| `-c` | Matching training YAML |
| `-o` | Output directory |
| `--device cpu` | Run on CPU; CUDA is used by default when available |
| `--amp` | Enable CUDA FP16 inference; FP32 is the default |
| `--no-lpips` | Skip LPIPS |
| `--crop-border` | Override the border crop used for PSNR and SSIM |

Inputs must be 8-bit RGB or grayscale images without an alpha channel. Outputs retain the input filenames and extensions.
If `-c` is omitted, the checkpoint directory must contain exactly one YAML file. For a checkpoint inside `models/`, the parent directory is searched instead.

## Acknowledgments

This codebase is built upon [UHDformer](https://github.com/supersupercong/UHDformer) and [BasicSR](https://github.com/XPixelGroup/BasicSR). We thank the authors for their excellent work.
