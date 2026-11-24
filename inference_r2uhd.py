import argparse
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
import yaml

from basicsr.archs import build_network
from basicsr.utils import img2tensor, tensor2img
from comput_psnr_ssim import calculate_psnr, calculate_ssim

ROOT = Path(__file__).resolve().parent
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def resolve_config(weight, config=None):
    if config:
        path = Path(config)
    else:
        directory = weight.parent.parent if weight.parent.name == 'models' else weight.parent
        candidates = sorted(set(directory.glob('*.yml')) | set(directory.glob('*.yaml')))
        if len(candidates) != 1:
            raise ValueError('Cannot select one saved training YAML; specify -c/--config. '
                             'Use the checkpoint\'s saved YAML for independent stage depth, '
                             'clustering, and evaluation settings.')
        path = candidates[0]
    options = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    if not isinstance(options, Mapping):
        raise ValueError('The saved YAML must contain an options mapping')
    config_g = options.get('network_g', {})
    if not isinstance(config_g, Mapping):
        raise ValueError('The YAML must contain a network_g mapping')
    config_g = dict(config_g)
    if config_g.get('type') == 'ReLiteV17Net':
        # Old relitev17 YAMLs name the same architecture differently.
        print('[warn] legacy network_g.type ReLiteV17Net; loading it as R2UHDNet')
        config_g['type'] = 'R2UHDNet'
    if config_g.get('type') != 'R2UHDNet':
        raise ValueError('The YAML must contain network_g.type: R2UHDNet')
    if 'num_stages' not in config_g:
        raise ValueError('The saved YAML must explicitly contain network_g.num_stages')
    return path.resolve(), config_g


def load_weights(net, path, param_key='auto'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError('Expected a state_dict or a checkpoint mapping')
    if param_key == 'auto':
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            state, key = checkpoint, 'raw_state_dict'
        else:
            key = next((k for k in ('params', 'params_ema', 'state_dict')
                        if isinstance(checkpoint.get(k), Mapping)), None)
            if key is None:
                raise ValueError('No params/params_ema/state_dict found in checkpoint')
            state = checkpoint[key]
    else:
        if not isinstance(checkpoint.get(param_key), Mapping):
            raise ValueError(f'Checkpoint has no state dictionary named {param_key}')
        state, key = checkpoint[param_key], param_key
    if state and all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    revision = state.get('_architecture_revision')
    if not torch.is_tensor(revision) or revision.numel() != 1 or revision.item() != net._architecture_revision.item():
        raise ValueError('Missing or incompatible R2UHD architecture revision')
    net.load_state_dict(state, strict=True)
    return key


def check_image_size(x, multiple=8):
    h, w = x.shape[-2:]
    ph, pw = (-h) % multiple, (-w) % multiple
    if not (ph or pw):
        return x
    return F.pad(x, (0, pw, 0, ph), mode='reflect' if ph < h and pw < w else 'replicate')


def read_image(path):
    # imdecode supports Unicode filenames on Windows.
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f'Cannot decode image: {path}')
    if image.dtype != np.uint8:
        raise ValueError(f'Expected 8-bit image, got {image.dtype}: {path}')
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f'Expected RGB or grayscale image (no alpha): {path}')
    return image


def make_lpips(enabled):
    if not enabled:
        return None, 'disabled or no GT'
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        return LearnedPerceptualImagePatchSimilarity(net_type='alex').eval(), 'alex, CPU, full image [-1,1]'
    except Exception as exc:
        message = f'LPIPS unavailable; PSNR/SSIM still enabled: {exc}'
        print(f'[warn] {message}')
        return None, message


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input', default='D:/datasets/UHD-Rain/testing_set/input', help='Image or directory')
    parser.add_argument('-g', '--gt', help='Optional GT directory (same filenames), or GT file for a single input')
    parser.add_argument('-w', '--weight', required=True)
    parser.add_argument('-c', '--config', help='Saved training YAML (required directly or by unique discovery beside models/)')
    parser.add_argument('-o', '--output', default=str(ROOT/'results/R2UHD'))
    parser.add_argument('--param-key', default='auto', choices=['auto', 'params', 'params_ema', 'state_dict'])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--amp', action='store_true', help='CUDA FP16 autocast; default FP32')
    parser.add_argument('--no-lpips', action='store_true', help='Skip optional Alex LPIPS')
    parser.add_argument('--pad-multiple', type=int, default=8, help='8: native geometry; 128: legacy padded protocol')
    parser.add_argument('--crop-border', type=int, default=None, help='PSNR/SSIM crop border; defaults to saved YAML')
    args = parser.parse_args(argv)
    if args.pad_multiple < 8 or args.pad_multiple % 8:
        parser.error('--pad-multiple must be a positive multiple of 8')
    if args.crop_border is not None and args.crop_border < 0:
        parser.error('--crop-border must be nonnegative')
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda') or (args.amp and device.type != 'cuda'):
        parser.error('Use CPU/ CUDA; --amp requires CUDA')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    source, weight, destination = Path(args.input), Path(args.weight), Path(args.output)
    if not weight.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {weight}; specify -w')
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        paths = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    else:
        raise FileNotFoundError(f'Input not found: {source}')
    if not paths:
        raise ValueError(f'No supported images in {source}')
    gt = Path(args.gt) if args.gt else None
    if gt is not None and not (gt.is_dir() or (gt.is_file() and len(paths) == 1)):
        raise ValueError('GT must be a directory, or a file for a single input')
    pairs = [(p, (gt / p.name if gt.is_dir() else gt) if gt else None) for p in paths]
    for path, target in pairs:
        if target is not None and not target.is_file():
            raise FileNotFoundError(f'Missing paired GT: {target}')
        output_path = (destination/path.name).resolve()
        if output_path == path.resolve() or (target and output_path == target.resolve()):
            raise ValueError('Output must not overwrite the input or GT')
    config_path, config_g = resolve_config(weight, args.config)
    if args.crop_border is None and gt is not None:
        options = yaml.safe_load(config_path.read_text(encoding='utf-8-sig'))
        metrics = options.get('val', {}).get('metrics', {})
        crops = {int(metrics[name].get('crop_border', 0))
                 for name in ('psnr', 'ssim') if name in metrics}
        if len(crops) != 1 or next(iter(crops)) < 0:
            raise ValueError('Saved YAML needs matching PSNR/SSIM crop_border, '
                             'or specify --crop-border explicitly')
        args.crop_border = crops.pop()
    elif args.crop_border is None:
        args.crop_border = 0  # No metrics are calculated without paired GT.
    net = build_network(config_g)
    key = load_weights(net, weight, args.param_key)
    net = net.to(device).eval()
    print(f'Config: {config_path}\nNetwork: {config_g}\nWeights: {weight} [{key}]')
    print(f'Parameters: {net.parameter_count:,}; device: {device}; AMP: {args.amp}; pad: {args.pad_multiple}; '
          f'num_stages: {net.num_stages}; global_depth: {net.global_depth}; prototypes: {net.prototypes}')
    metric, lpips_status = make_lpips(gt is not None and not args.no_lpips)
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.inference_mode():
        for path, target in tqdm(pairs, unit='image', desc='R2UHD'):
            image = read_image(path)
            h, w = image.shape[:2]
            truth = read_image(target) if target else None
            if truth is not None:
                if truth.shape != image.shape:
                    raise ValueError(f'LQ/GT dimensions differ: {path} / {target}')
                if min(h,w) - 2*args.crop_border < 11:
                    raise ValueError('Image too small for SSIM after border cropping; omit GT for tiny-image inference')
            x = check_image_size(img2tensor(image).unsqueeze(0).to(device)/255., args.pad_multiple)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp):
                y = net(x)[..., :h, :w]
            if not torch.isfinite(y).all():
                raise RuntimeError(f'Nonfinite model output: {path}')
            restored = tensor2img(y.float())
            del x,y
            row = {'image':path.name, 'height':h, 'width':w}
            if truth is not None:
                row['psnr'] = float(calculate_psnr(restored,truth,crop_border=args.crop_border,test_y_channel=True))
                row['ssim'] = float(calculate_ssim(restored,truth,crop_border=args.crop_border,test_y_channel=True))
                if metric is not None:
                    pred_t = img2tensor(restored).unsqueeze(0)/127.5-1
                    gt_t = img2tensor(truth).unsqueeze(0)/127.5-1
                    metric.reset()  # Each row is this image's value, not a running metric.
                    row['lpips'] = float(metric(pred_t,gt_t))
                    del pred_t,gt_t
                tqdm.write(' | '.join([path.name]+[f'{k}: {row[k]:.6f}' for k in ('psnr','ssim','lpips') if k in row]))
            ok, encoded = cv2.imencode(path.suffix,restored)
            if not ok:
                raise OSError(f'Cannot encode output: {path.name}')
            encoded.tofile(destination/path.name)
            rows.append(row)
    averages = {k:sum(r[k] for r in rows)/len(rows) for k in ('psnr','ssim','lpips') if k in rows[0]}
    report = dict(weight=str(weight.resolve()),weight_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),
                  config=str(config_path),network_g=config_g,parameters=net.parameter_count, param_key=key,
                  num_stages=net.num_stages, global_depth=net.global_depth, prototypes=net.prototypes,
                  device=str(device),amp=args.amp,pad_multiple=args.pad_multiple,crop_border=args.crop_border,
                  protocol='Y PSNR/SSIM on clamped uint8 BGR arrays, before file encoding; comput_psnr_ssim',
                  lpips=lpips_status,count=len(rows),averages=averages,images=rows)
    report_path = destination/'inference_summary.json'
    report_path.write_text(json.dumps(json_safe(report),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(f'Saved {len(rows)} images to {destination}')
    for key,value in averages.items():
        print(f'avg_{key}: {value:.6f}')
    if gt is not None and metric is None:
        print(f'avg_lpips: N/A ({lpips_status})')
    print(f'Report: {report_path}')
    return report


if __name__ == '__main__':
    main()

