import argparse
import datetime
import os
import os.path as osp
import shutil
import time
import torch
import yaml
import torch.nn.parallel
import munch
import tqdm


from softgroup.data import (TreeDataset, build_dataloader)

from softgroup.model import SoftGroup

from softgroup.evaluation import (ScanNetEval, evaluate_offset_mae, evaluate_semantic_acc,
                                  evaluate_semantic_miou)
from softgroup.util import (AverageMeter, SummaryWriter, build_optimizer, checkpoint_save,
                            collect_results_gpu, cosine_lr_after_step, get_dist_info,
                            get_max_memory, get_root_logger, init_dist, is_main_process,
                            is_multiple, is_power2, load_checkpoint, get_data_paths)


def get_args():
    parser = argparse.ArgumentParser('SoftGroup')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('--dist', action='store_true', help='run with distributed parallel')
    parser.add_argument('--resume', type=str, help='path to resume from')
    parser.add_argument('--work_dir', type=str, help='working directory')
    parser.add_argument('--skip_validate', action='store_true', help='skip validation')
    args = parser.parse_args()
    return args


def train(epoch, model, optimizer, scaler, train_loader, cfg, logger, writer):
    model.train()
    iter_time = AverageMeter(True)
    data_time = AverageMeter(True)
    meter_dict = {}
    end = time.time()

    if train_loader.sampler is not None and cfg.dist:
        train_loader.sampler.set_epoch(epoch)

    for i, batch in enumerate(train_loader, start=1):
        data_time.update(time.time() - end)
        cosine_lr_after_step(optimizer, cfg.optimizer.lr, epoch - 1, cfg.step_epoch, cfg.epochs)
        with torch.cuda.amp.autocast(enabled=cfg.fp16):
            loss, log_vars = model(batch, return_loss=True)

        # meter_dict
        for k, v in log_vars.items():
            if k not in meter_dict.keys():
                meter_dict[k] = AverageMeter()
            meter_dict[k].update(v)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # time and print
        remain_iter = len(train_loader) * (cfg.epochs - epoch + 1) - i
        iter_time.update(time.time() - end)
        end = time.time()
        remain_time = remain_iter * iter_time.avg
        remain_time = str(datetime.timedelta(seconds=int(remain_time)))
        lr = optimizer.param_groups[0]['lr']

        if is_multiple(i, 10):
            log_str = f'Epoch [{epoch}/{cfg.epochs}][{i}/{len(train_loader)}]  '
            log_str += f'lr: {lr:.2g}, eta: {remain_time}, mem: {get_max_memory()}, '\
                f'data_time: {data_time.val:.2f}, iter_time: {iter_time.val:.2f}'
            for k, v in meter_dict.items():
                log_str += f', {k}: {v.val:.4f}'
            logger.info(log_str)
    writer.add_scalar('train/learning_rate', lr, epoch)
    for k, v in meter_dict.items():
        writer.add_scalar(f'train/{k}', v.avg, epoch)
    checkpoint_save(epoch, model, optimizer, cfg.work_dir, cfg.save_freq)


def validate(epoch, model, val_loader, cfg, logger, writer):
    logger.info('Validation')
    results = []
    all_sem_preds, all_sem_labels, all_offset_preds, all_offset_labels = [], [], [], []
    all_inst_labels, all_pred_insts, all_gt_insts = [], [], []
    _, world_size = get_dist_info()
    progress_bar = tqdm.tqdm(total=len(val_loader) * world_size, disable=not is_main_process())
    val_set = val_loader.dataset
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(val_loader):
            result = model(batch)
            results.append(result)
            progress_bar.update(world_size)
        progress_bar.close()
        results = collect_results_gpu(results, len(val_set))
    if is_main_process():
        for res in results:
            all_sem_preds.append(res['semantic_preds'])
            all_sem_labels.append(res['semantic_labels'])
            all_offset_preds.append(res['offset_preds'])
            all_offset_labels.append(res['offset_labels'])
            all_inst_labels.append(res['instance_labels'])
            if not cfg.model.semantic_only:
                all_pred_insts.append(res['pred_instances'])
                all_gt_insts.append(res['gt_instances'])
        if not cfg.model.semantic_only:
            logger.info('Evaluate instance segmentation')
            scannet_eval = ScanNetEval(val_set.CLASSES)
            eval_res = scannet_eval.evaluate(all_pred_insts, all_gt_insts)
            writer.add_scalar('val/AP', eval_res['all_ap'], epoch)
            writer.add_scalar('val/AP_50', eval_res['all_ap_50%'], epoch)
            writer.add_scalar('val/AP_25', eval_res['all_ap_25%'], epoch)
            logger.info('AP: {:.3f}. AP_50: {:.3f}. AP_25: {:.3f}'.format(
                eval_res['all_ap'], eval_res['all_ap_50%'], eval_res['all_ap_25%']))
        logger.info('Evaluate semantic segmentation and offset MAE')
        miou = evaluate_semantic_miou(all_sem_preds, all_sem_labels, cfg.model.ignore_label, logger)
        acc = evaluate_semantic_acc(all_sem_preds, all_sem_labels, cfg.model.ignore_label, logger)
        mae = evaluate_offset_mae(all_offset_preds, all_offset_labels, all_inst_labels, # look what this is
                                  cfg.model.ignore_label, logger)
        writer.add_scalar('val/mIoU_semantic', miou, epoch)
        writer.add_scalar('val/Acc_semantic', acc, epoch)
        writer.add_scalar('val/Offset_MAE', mae, epoch)


# EXECUTION STARTS HERE

def main():
    args = get_args()

    ########################################
    # print(f"path to config file: {args.config}")
    # print(f"run with distributed parallel: {args.dist}")
    # print(f"path to resume from: {args.resume}")
    # print(f"working directory: {args.work_dir}")
    # print(f"skip validation: {args.skip_validate}")

    # print(args)

    # import sys
    # sys.exit(0)
    ########################################

    # LOAD CONFIG FILE AS MUNCH DICT AND ADD SOME ENTRIES
    cfg_txt = open(args.config, 'r').read()
    cfg = munch.Munch.fromDict(yaml.safe_load(cfg_txt))

    if args.dist:
        init_dist()
    cfg.dist = args.dist

    # work_dir & logger
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])


    # MAKE DIRECTORY TO SAVE LOGFILES AND TENSORBOARDFILES TO AND CREATE LOGGER AND TENSORBOARDLOGGER WITH CERTAIN CHARACTERISTICS
    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file)
    logger.info(f'Config:\n{cfg_txt}')
    logger.info(f'Distributed: {args.dist}')
    logger.info(f'Mix precision training: {cfg.fp16}')
    shutil.copy(args.config, osp.join(cfg.work_dir, osp.basename(args.config)))
    writer = SummaryWriter(cfg.work_dir)

    # model
    model = SoftGroup(**cfg.model).cuda() # TAKES MODEL SPECIFICATION FROM YAML FILE AS INPUT, **dictionary IS STANDARD WAY TO PASS ARGUMENTS FROM DICT TO FUNCTION
    if args.dist:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)

    # optim
    optimizer = build_optimizer(model, cfg.optimizer)

    # pretrain, resume
    start_epoch = 1
    if args.resume:
        logger.info(f'Resume from {args.resume}')
        start_epoch = load_checkpoint(args.resume, logger, model, optimizer=optimizer)
    elif cfg.pretrain:
        logger.info(f'Load pretrain from {cfg.pretrain}')
        load_checkpoint(cfg.pretrain, logger, model)

    # data
    data_paths_train = get_data_paths(cfg.data.train, cfg.epochs, training=True)
    data_paths_val = get_data_paths(cfg.data.test, cfg.epochs, training=False)

    # train and val
    logger.info('Training')
    for epoch in range(start_epoch, cfg.epochs + 1):

        # load different dataset for every epoch for train
        train_set = TreeDataset(**cfg.data.train, data_paths=data_paths_train[epoch-1], logger=logger)
        val_set = TreeDataset(**cfg.data.test, data_paths=data_paths_val, logger=logger)
        train_loader = build_dataloader(train_set, training=True, dist=args.dist, **cfg.dataloader.train)
        val_loader = build_dataloader(val_set, training=False, dist=args.dist, **cfg.dataloader.test)

        train(epoch, model, optimizer, scaler, train_loader, cfg, logger, writer)
        if not args.skip_validate and (is_multiple(epoch, cfg.save_freq) or is_power2(epoch)):
            validate(epoch, model, val_loader, cfg, logger, writer)
        writer.flush()


if __name__ == '__main__':
    main()
