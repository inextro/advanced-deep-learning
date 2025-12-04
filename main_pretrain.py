import os
import json
import time
import datetime
import argparse

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm
assert timm.__version__ == '0.3.2'
import timm.optim.optim_factory as optim_factory

import numpy as np
import utils
from utils import str2bool
from utils import NativeScalerWithGradNormCount as NativeScaler
from pathlib import Path
from models import fcmae
from engine_pretrain import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('FCMAE pre-training', add_help=False)
    parser.add_argument(
        '--batch_size', default=64, type=int, 
        help='Per GPU batch size'
    )
    parser.add_argument(
        '--epochs', default=800, type=int
    )
    parser.add_argument(
        '--warmup_epochs', default=40, type=int, metavar='N', 
        help='Epochs to warmup LR'
    )
    parser.add_argument(
        '--update_freq', default=1, type=int, 
        help='Gradient accumulation step'
    )

    # Model parameters
    parser.add_argument(
        '--model', default='fcmae_convnextv2_atto', type=str, choices=['fcmae_convnext_v1_atto', 'fcmae_convnext_v2_atto'], 
        help='Name of model to train (convnext_v1_atto | convnext_v2_atto)'
    )
    parser.add_argument(
        '--input_size', default=224, type=int, 
        help='Image input size'
    )
    parser.add_argument(
        '--mask_ratio', default=0.6, type=float, 
        help='Masking ratio (precentage of removed patches)'
    )
    parser.add_argument(
        '--norm_pix_loss', action='store_true', 
        help='Use (per-patch) normalized pixels as targtes for computing loss'
    )
    parser.set_defaults(norm_pix_loss=True)
    
    parser.add_argument(
        '--decoder_depth', type=int, default=1
    )
    parser.add_argument(
        '--decoder_embed_dim', type=int, default=320
    )

    # Optimizer paramters
    parser.add_argument(
        '--weight_decay', default=0.05, type=float, 
        help='Weight decay'
    )
    parser.add_argument(
        '--lr', default=None, type=float, metavar='LR', 
        help='Learning rate (absolute lr)'
    )
    parser.add_argument(
        '--blr', default=1.5e-4, type=float, metavar='LR', 
        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256'
    )
    parser.add_argument(
        '--min_lr', default=0.0, type=float, metavar='LR', 
        help='lower lr bound for cyclic schedulers that hit 0'
    )

    # Dataset parameters
    parser.add_argument(
        '--data_path', default='./data/', type=str, 
        help='Dataset path'
    )
    parser.add_argument(
        '--output_dir', default='./ckpt/', 
        help='Path where to save, empty for no saving'
    )
    parser.add_argument(
        '--log_dir', default='./log/', 
        help='Path where to tensorboard log'
    )
    parser.add_argument(
        '--device', default='cuda', 
        help='Device to use for training / testing'
    )
    parser.add_argument(
        '--seed', default=0, type=int
    )
    parser.add_argument(
        '--resume', default='', 
        help='Resume from checkpoint'
    )
    parser.add_argument(
        '--auto_resume', default=True, type=str2bool
    )
    parser.add_argument(
        '--save_ckpt', default=True, type=str2bool
    )
    parser.add_argument(
        '--save_ckpt_freq', default=1, type=int
    )
    parser.add_argument(
        '--save_ckpt_num', default=3, type=int
    )
    parser.add_argument(
        '--start_epoch', default=0, type=int, metavar='N', 
        help='Start epoch'
    )
    parser.add_argument(
        '--num_workers', default=10, type=int
    )
    parser.add_argument(
        '--pin_mem', default=True, type=str2bool, 
        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU'
    )

    # Evaluation parameters
    parser.add_argument(
        '--crop_pct', default=None, type=float
    )
    
    # Distributed training parameters
    parser.add_argument(
        '--world_size', default=1, type=int, 
        help='Number of distributed processes'
    )
    parser.add_argument(
        '--local_rank', default=-1, type=int
    )
    parser.add_argument(
        '--dist_on_itp', default=False, type=str2bool
    )
    parser.add_argument(
        '--dist_url', default='env://', 
        help='Url used to set up distributed training'
    )

    return parser


def main(args):
    utils.init_distributed_mode(args)

    print(args)
    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Simple augmentation
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3), # bicubic
        # transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed
    )
    print('Sampler_train = %s' % str(sampler_train))

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)

    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        pin_memory=args.pin_mem, 
        drop_last=True
    )

    model = fcmae.__dict__[args.model](
        mask_ratio=args.mask_ratio, 
        decoder_depth=args.decoder_depth, 
        decoder_embed_dim = args.decoder_embed_dim, 
        norm_pix_loss=args.norm_pix_loss
    )
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('Model = %s' % str(model_without_ddp))
    print('Number of params:', n_parameters)

    eff_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // eff_batch_size

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print('Base lr: %.2e' % (args.lr * 256 / eff_batch_size))
    print('Actual lr: %.2e' % args.lr)

    print('Accumulate grad iterations: %d' % args.update_freq)
    print('Effective batch size: %d' % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    
    loss_scaler = NativeScaler()

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, 
        optimizer=optimizer, loss_scaler=loss_scaler
    )

    print(f'Start training for {args.epochs} epochs')
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        
        train_stats = train_one_epoch(
            model, data_loader_train, 
            optimizer, device, epoch, loss_scaler, 
            log_writer=log_writer, 
            args=args
        )
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, 
                    loss_scaler=loss_scaler, epoch=epoch
                )
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()}, 
            'epoch': epoch, 
            'n_parameters': n_parameters
        }

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, 'log.txt'), mode='a', encoding='utf-8') as f:
                f.write(json.dumps(log_stats) + '\n')

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'Training time {total_time_str}')


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)