import sys
import math
import torch
import utils
from typing import Iterable

def train_one_epoch(
    model: torch.nn.Module, 
    data_loader: Iterable, optimizer: torch.optim.Optimizer, 
    device: torch.device, epoch: int, loss_scaler, 
    log_writer=None, 
    args=None
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter=' ')
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 20
    update_freq = args.update_freq

    optimizer.zero_grad()
    for data_iter_step, (samples, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % update_freq == 0:
            utils.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        if not isinstance(samples, list):
            samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        loss, _, _ = model(samples, labels, mask_ratio=args.mask_ratio)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f'Loss is {loss_value}, stopping training')
            sys.exit(1)

        loss /= update_freq
        loss_scaler(
            loss, optimizer, parameters=model.parameters(), 
            update_grad=(data_iter_step + 1) % update_freq == 0
        )

        if (data_iter_step + 1) % update_freq == 0:
            optimizer.zero_grad()
            torch.cuda.empty_cache() # clear the GPU cache at a regular interval for training ME network

        metric_logger.update(loss=loss_value)
        
        lr = optimizer.param_groups[0]['lr']
        metric_logger.update(lr=lr)

        loss_value_reduce = utils.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % update_freq == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.update(train_loss=loss_value_reduce, head='loss', step=epoch_1000x)
            log_writer.update(lr=lr, head='opt', step=epoch_1000x)

    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}