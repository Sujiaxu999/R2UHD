"""BasicSR test entry point for R2UHD (optional; ``inference_r2uhd.py`` is the
primary evaluation script and needs no test dataset section).

The option file must contain one or more ``test`` datasets::

    datasets:
      test:
        name: UHD_Haze_Test
        type: PairedImageDataset
        dataroot_gt: /path/to/test/gt
        dataroot_lq: /path/to/test/input
        io_backend: {type: disk}

and the generator weights are taken from ``path.pretrain_network_g``::

    python -m basicsr.test -opt options/train_r2uhd_haze.yml \
        --force_yml path:pretrain_network_g=/path/to/r2uhd_haze.pth
"""
import logging
import sys
from os import path as osp

if __package__ in (None, ''):
    # allow `python basicsr/test.py` as well as `python -m basicsr.test`
    sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import torch

from basicsr.data import create_dataloader, create_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


def test_pipeline(root_path):
    opt, _ = parse_options(root_path, is_train=False)

    weight = opt.get('path', {}).get('pretrain_network_g')
    if not weight or not osp.isfile(weight):
        raise SystemExit('Set path.pretrain_network_g to an existing checkpoint, e.g. '
                         '--force_yml path:pretrain_network_g=/path/to/r2uhd_haze.pth')
    if not any(name.split('_')[0] == 'test' for name in opt['datasets']):
        raise SystemExit('The option file needs at least one "test" dataset section')

    torch.backends.cudnn.benchmark = True

    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    test_loaders = []
    for name, dataset_opt in opt['datasets'].items():
        if name.split('_')[0] != 'test':
            continue
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'],
            sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    model = build_model(opt)
    for test_loader in test_loaders:
        logger.info(f"Testing {test_loader.dataset.opt['name']}...")
        model.validation(test_loader, current_iter=opt['name'], epoch=0, tb_logger=None,
                         save_img=opt['val']['save_img'])


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
