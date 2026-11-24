"""BasicSR training entry point for R2UHD.

Run from the R2UHD directory::

    python -m basicsr.train -opt options/train_r2uhd_haze.yml
    python -m basicsr.train -opt options/train_r2uhd_haze.yml --auto_resume

Multi-GPU::

    python -m torch.distributed.run --nproc_per_node=2 --master_port=4398 \
        --module basicsr.train -opt options/train_r2uhd_haze.yml --launcher pytorch

Outputs go to ``experiments/<name>/`` (models, training_states, log,
visualization). ``--auto_resume`` restores the newest iteration that has both a
``.state`` and its ``net_g_<iter>.pth``.
"""
import datetime
import logging
import math
import os
import shutil
import sys
import time
import warnings
from os import path as osp

if __package__ in (None, ''):
    # allow `python basicsr/train.py` as well as `python -m basicsr.train`
    sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import torch

from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, scandir)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options

# pyiqa imports timm, whose legacy import path warns on every run.
warnings.filterwarnings('ignore', message='Importing from timm.models.layers is deprecated.*')


def archive_tb_logger(path):
    """Move an existing tensorboard run aside, keeping ``tb_logger`` tidy."""
    if osp.exists(path):
        name = osp.basename(osp.normpath(path))
        target = osp.join(osp.dirname(osp.normpath(path)) + '_archived',
                          f'{name}_{get_time_str()}')
        print(f'Path already exists. Rename it to {target}', flush=True)
        os.makedirs(osp.dirname(target), exist_ok=True)
        shutil.move(path, target)
    os.makedirs(path, exist_ok=True)


def init_tb_loggers(opt):
    # initialise wandb before tensorboard so both stay in sync
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger


def create_train_val_dataloader(opt, logger):
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'],
                                            dataset_enlarge_ratio)
            train_loader = create_dataloader(
                train_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'],
                sampler=train_sampler, seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / num_iter_per_epoch)
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'],
                sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')
    return train_loader, train_sampler, val_loader, total_epochs, total_iters


def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        return None
    map_location = (torch.device('cuda', torch.cuda.current_device())
                    if torch.cuda.is_available() else 'cpu')
    resume_state = torch.load(resume_state_path, map_location=map_location)
    check_resume(opt, resume_state['iter'])
    return resume_state


def train_pipeline(root_path):
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    torch.backends.cudnn.benchmark = True

    resume_state = load_resume_state(opt)
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            archive_tb_logger(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    # keep the resolved YAML next to the checkpoints for exact reproduction
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    tb_logger = init_tb_loggers(opt)

    (train_loader, train_sampler, val_loader,
     total_epochs, total_iters) = create_train_val_dataloader(opt, logger)

    model = build_model(opt)
    if resume_state:
        model.resume_training(resume_state)
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        current_iter = resume_state['iter']
        # Checkpoints are written at epoch boundaries; continue with the next
        # sampler epoch instead of replaying the finished one.
        epoch_finished = current_iter % len(train_loader) == 0
        start_epoch = resume_state['epoch'] + int(epoch_finished)
    else:
        start_epoch = 0
        current_iter = 0

    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, "cuda", "cpu".')

    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    epoch = start_epoch
    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_timer.record()
            if current_iter >= total_iters:
                break
            current_iter += 1
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            iter_timer.record()

            if current_iter == 1:
                # reset start time in msg_logger for a more accurate eta
                msg_logger.reset_start_time()
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            if current_iter % opt['logger']['show_tf_imgs_freq'] == 0:
                visual_imgs = model.get_current_visuals()
                if tb_logger:
                    for k, v in visual_imgs.items():
                        tb_logger.add_images(f'ckpt_imgs/{k}', v.clamp(0, 1), current_iter)

            if (current_iter % opt['logger']['save_checkpoint_freq'] == 0
                    or current_iter % opt['logger']['save_latest_freq'] == 0):
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            if (val_loader is not None and opt.get('val') is not None
                    and current_iter % opt['val']['val_freq'] == 0):
                model.validation(val_loader, current_iter, epoch, tb_logger, opt['val']['save_img'])

            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()

        if current_iter >= total_iters:
            break

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for 'latest'
    if val_loader is not None and opt.get('val') is not None:
        model.validation(val_loader, current_iter, epoch, tb_logger, opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
