"""R2UHD training / evaluation adapter for BasicSR.

The recipe is L1 + SSIM + FFT supervision in FP32, AdamW, optional gradient
accumulation, and either a native cosine schedule or the BasicSR cyclic cosine
schedule. Both are supported because the released task recipes use the cyclic
one. Validation is lazy: pyiqa metrics are created only when ``val.metrics`` is
configured, and best weights are saved only from the key metric.

``train.optim_d`` and zero-weight unimplemented losses are tolerated so that
legacy option files stay loadable; any nonzero unimplemented loss raises. There
is no discriminator in R2UHD, so only ``optimizer_g`` is built.
"""
from collections import OrderedDict
from copy import deepcopy
import math
from pathlib import Path

import torch
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger, img2tensor, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
from .cal_ssim import SSIM
from .lr_scheduler import CosineAnnealingRestartCyclicLR

# Losses that R2UHD implements. ``pixel_ssim_opt`` is applied through the
# differentiable SSIM module rather than a registry loss.
NATIVE_LOSSES = ('pixel_opt', 'pixel_ssim_opt', 'fft_opt')


@MODEL_REGISTRY.register()
class R2UHDModel(BaseModel):
    def __init__(self, opt):
        opt = deepcopy(opt)
        BaseModel.__init__(self, opt)
        if opt.get('network_g', {}).get('type') != 'R2UHDNet':
            raise ValueError('R2UHDModel requires R2UHDNet')
        self.net_g = self.model_to_device(build_network(opt['network_g']))
        self.metric_funcs = None
        path = opt.get('path', {})
        if path.get('pretrain_network_g'):
            self.load_network(self.net_g, path['pretrain_network_g'],
                              path.get('strict_load', True), path.get('param_key_g', 'params'))
        if self.is_train:
            self.init_training_settings()
        else:
            self.net_g.eval()

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device, non_blocking=True)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device, non_blocking=True)
        elif hasattr(self, 'gt'):
            del self.gt

    def init_training_settings(self):
        self.net_g.train()
        train = self.opt['train']

        # Validate the supported loss entries; keep ``train`` unchanged so the
        # resolved YAML copied into the experiment directory stays faithful.
        specs = {}
        for key, expected in (('pixel_opt', 'L1Loss'), ('pixel_ssim_opt', 'SSIMLoss'),
                              ('fft_opt', 'FFTLoss')):
            spec = train.get(key, {})
            if spec.get('type', expected) != expected or spec.get('reduction', 'mean') != 'mean':
                raise ValueError(f'Unsupported {key}: {spec}')
            weight = float(spec.get('loss_weight', 0.0))
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f'Invalid loss weight in {key}: {weight}')
            specs[key] = weight
        if specs['pixel_opt'] <= 0:
            raise ValueError('A positive L1 loss is required')
        for key in [k for k in train if k.endswith('_opt') and k not in NATIVE_LOSSES]:
            if float((train[key] or {}).get('loss_weight', 0)) != 0:
                raise ValueError(f'{key} is not implemented by R2UHDModel but carries a nonzero weight')
        if train.get('grad_clip', 0) != 0:
            raise ValueError('The R2UHD recipe disables gradient clipping')
        if train.get('amp', False):
            raise ValueError('Training is FP32; AMP is available for inference through val.use_amp')
        if train['optim_g'].get('type') != 'AdamW':
            raise ValueError('R2UHDModel uses AdamW')

        self.accumulation_steps = int(train.get('accumulation_steps', 1))
        if self.accumulation_steps < 1:
            raise ValueError('accumulation_steps must be a positive integer')
        self.cri_pix = build_loss(deepcopy(train['pixel_opt'])).to(self.device)
        self.cri_fft = (build_loss(deepcopy(train['fft_opt'])).to(self.device)
                        if train.get('fft_opt') else None)
        self.ssim_loss_weight = specs['pixel_ssim_opt']
        self.ssim = SSIM().to(self.device)

        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        # Only the generator is optimized; a legacy ``optim_d`` entry is ignored.
        options = deepcopy(self.opt['train']['optim_g'])
        optim_type = options.pop('type')
        self.optimizer_g = getattr(torch.optim, optim_type)(self.net_g.parameters(), **options)
        self.optimizers.append(self.optimizer_g)

    def setup_schedulers(self):
        config = dict(self.opt['train']['scheduler'])
        kind = config.pop('type')
        if kind == 'CosineAnnealingLR':
            self.schedulers.extend(
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **config)
                for optimizer in self.optimizers)
        elif kind == 'CosineAnnealingRestartCyclicLR':
            self.schedulers.extend(
                CosineAnnealingRestartCyclicLR(optimizer, **config)
                for optimizer in self.optimizers)
        else:
            raise ValueError(f'Unsupported scheduler for R2UHDModel: {kind}')

    def optimize_parameters(self, current_iter):
        accumulation = self.accumulation_steps
        if (current_iter - 1) % accumulation == 0:
            self.optimizer_g.zero_grad(set_to_none=True)
        self.output = self.net_g(self.lq)
        with torch.autocast(device_type=self.device.type, enabled=False):
            pred, target = self.output.float(), self.gt.float()
            losses = OrderedDict(l_pix=self.cri_pix(pred, target).mean())
            if self.ssim_loss_weight > 0:
                losses['l_ssim'] = self.ssim_loss_weight * (1 - self.ssim(pred, target))
            if self.cri_fft is not None:
                losses['l_freq'] = self.cri_fft(pred, target).mean()
        (sum(losses.values()) / accumulation).backward()
        if current_iter % accumulation == 0:
            self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(losses)

    @torch.no_grad()
    def test(self):
        was_training = self.net_g.training
        self.net_g.eval()
        try:
            amp = self.device.type == 'cuda' and self.opt.get('val', {}).get('use_amp', False)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=amp):
                self.output = self.get_bare_model(self.net_g).test(self.lq)
        finally:
            self.net_g.train(was_training)

    def dist_validation(self, dataloader, current_iter, epoch, tb_logger, save_img=False, save_as_dir=None):
        if get_dist_info()[0] == 0:
            self.nondist_validation(dataloader, current_iter, epoch, tb_logger, save_img, save_as_dir)

    @torch.no_grad()
    def nondist_validation(self, dataloader, current_iter, epoch, tb_logger, save_img=False, save_as_dir=None):
        """Full-resolution validation on clamped uint8 outputs, no GT cropping."""
        val = self.opt.get('val', {})
        if self.metric_funcs is None:
            self.metric_funcs = {}
            if val.get('metrics'):
                import pyiqa
                for name, spec in val['metrics'].items():
                    spec = deepcopy(spec)
                    metric_type = spec.pop('type')
                    spec.pop('better', None)
                    self.metric_funcs[name] = pyiqa.create_metric(metric_type, device=self.device, **spec)
        dataset = dataloader.dataset.opt['name']
        totals, count = {k: 0.0 for k in self.metric_funcs}, 0
        with tqdm(dataloader, desc=f'Validation {dataset}', unit='batch',
                  dynamic_ncols=True, disable=not val.get('pbar', True)) as bar:
            for data in bar:
                self.feed_data(data)
                self.test()
                if self.metric_funcs and not hasattr(self, 'gt'):
                    raise ValueError('Reference metrics require GT')
                for i in range(self.output.shape[0]):
                    name = Path(data['lq_path'][i]).stem if 'lq_path' in data else str(count)
                    image = tensor2img(self.output[i].float())
                    if self.metric_funcs:
                        restored = img2tensor(image).unsqueeze(0).to(self.device) / 255.0
                        for key, metric in self.metric_funcs.items():
                            totals[key] += metric(restored, self.gt[i:i+1]).item()
                    if save_img:
                        folder = (Path(save_as_dir) if save_as_dir
                                  else Path(self.opt['path']['visualization']) / dataset / str(current_iter))
                        imwrite(image, str(folder / f'{name}.png'))
                    count += 1
                bar.set_postfix(images=count, image=name, refresh=False)
                del self.lq, self.output
                if hasattr(self, 'gt'):
                    del self.gt
        self.metric_results = {k: v / count for k, v in totals.items()} if count else {}
        if not self.metric_results:
            return
        self._initialize_best_metric_results(dataset)
        improved = {k: self._update_best_metric_result(dataset, k, v, current_iter)
                    for k, v in self.metric_results.items()}
        key = val.get('key_metric', 'psnr')
        if key not in improved:
            raise ValueError(f'Unknown key_metric: {key}')
        if self.is_train and improved[key]:
            self.save_network(self.net_g, 'net_g_best', current_iter, epoch)
        self._log_validation_metric_values(current_iter, dataset, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out = OrderedDict(lq=self.lq.detach().cpu()[:16], result=self.output.detach().cpu()[:16])
        if hasattr(self, 'gt'):
            out['gt'] = self.gt.detach().cpu()[:16]
        return out

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter, epoch)
        self.save_training_state(epoch, current_iter)
