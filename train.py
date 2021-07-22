'''
Description: yolov5 train for sign
Author: Hejun Jiang
Date: 2020-12-24 15:15:34
LastEditTime: 2021-01-11 08:58:24
LastEditors: Hejun Jiang
Version: v0.0.1
Contact: jianghejun@hccl.ioa.ac.cn
Corporation: hccl
'''
import os
import time
import math
import yaml
import test
import torch
import random
import shutil
import logging
import argparse
import numpy as np
from tqdm import tqdm
from models import yolo
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from utils import loss, plots, general, metrics, datasets, autoanchor, torch_utils

logger = logging.getLogger(__name__)


def train(opt, device, tb_writer=None):
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'  # 保存模型的路径
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'  # 最近一次的模型
    best = wdir / 'best.pt'  # 最优的模型
    results_file = save_dir / 'results.txt'  # 保存结果的模型
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)  # 写入配置 for resume

    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
        logger.info(f'Hyperparameters {hyp}')  # 打印超参
        with open(save_dir / 'hyp.yaml', 'w') as f:
            yaml.dump(hyp, f, sort_keys=False)  # 写入超参数

    cuda = device.type != 'cpu'  # cuda true
    general.init_seeds(1)
    with open(opt.data) as f:  # 打开数据配置文件
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
    train_path = data_dict['train']  # 训练路径
    test_path = data_dict['val']  # 验证路径
    nc = int(data_dict['nc'])
    names = data_dict['names']
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check 是否相等

    weight = opt.resume + opt.pretrained if opt.resume or opt.pretrained else ''
    pretrained = weight.endswith('.pt')  # 是否有init 模型，有就为true
    if pretrained:  # 如果有预先训练的模型
        ckpt = torch.load(weight, map_location=device)  # load checkpoint
        if hyp.get('anchors'):
            ckpt['model'].yaml['anchors'] = round(hyp['anchors'])  # force autoanchor
        model = yolo.Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc).to(device)  # create
        exclude = ['anchor'] if opt.cfg or hyp.get('anchors') else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = torch_utils.intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weight))  # report
    else:  # 没有预先训练的模型
        model = yolo.Model(opt.cfg, ch=3, nc=nc).to(device)  # create 导入模型，通道为3

    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / opt.batch_size), 1)  # accumulate loss before optimizing 最小为1，64/16=4
    hyp['weight_decay'] *= opt.batch_size * accumulate / nbs  # scale weight_decay 0.0005*16*4/64 具体到每一批的weight_decay

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, torch.nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter):
            pg1.append(v.weight)  # apply decay

    if opt.adam:
        optimizer = torch.optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = torch.optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)  # 一般用这个 SGD

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    def lf(x): return ((1 + math.cos(x * math.pi / opt.epochs)) / 2) * (1 - hyp['lrf']) + hyp['lrf']  # cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # Results
        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weight, opt.epochs)
        if opt.epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weight, ckpt['epoch'], opt.epochs))
            opt.epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict

    gs = int(max(model.stride))  # grid size (max stride) 各自大小 32
    imgsz, imgsz_test = [general.check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples train validation

    if cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    ema = torch_utils.ModelEMA(model)

    dataloader, dataset = datasets.create_dataloader(train_path, imgsz, opt.batch_size, gs, opt, hyp=hyp,
                                                     augment=True, cache=opt.cache_images, rect=opt.rect, workers=opt.workers)

    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    ema.updates = start_epoch * nb // accumulate  # set EMA updates
    testloader = datasets.create_dataloader(test_path, imgsz_test, opt.batch_size, gs, opt, hyp=hyp,
                                            cache=opt.cache_images and not opt.finaltest, rect=True, workers=opt.workers)[0]  # testloader

    if not opt.resume:  # 如果是不用resume
        labels = np.concatenate(dataset.labels, 0)
        c = torch.tensor(labels[:, 0])  # classes
        plots.plot_labels(labels, save_dir=save_dir)
        if tb_writer:
            tb_writer.add_histogram('classes', c, 0)
        if not opt.noautoanchor:
            autoanchor.check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

    hyp['cls'] *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = general.labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    model.names = names

    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.cuda.amp.GradScaler(enabled=cuda)
    logger.info('Image sizes %g train, %g test\n'
                'Using %g dataloader workers\nLogging results to %s\n'
                'Starting training for %g epochs...' % (imgsz, imgsz_test, dataloader.num_workers, save_dir, opt.epochs))
    for epoch in range(start_epoch, opt.epochs):  # epoch ------------------------------------------------------------------
        model.train()

        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            iw = general.labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        mloss = torch.zeros(4, device=device)  # mean losses
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'targets', 'img_size'))
        pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                accumulate = max(1, np.interp(ni, xi, [1, nbs / opt.batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = torch.nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with torch.cuda.amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                losses, loss_items = loss.compute_loss(pred, targets.to(device), model)  # loss scaled by batch_size

            # Backward
            scaler.scale(losses).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 6) % (
                '%g/%g' % (epoch, opt.epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
            pbar.set_description(s)

            if ni < 3:
                f = save_dir / f'train_batch{ni}.jpg'  # filename
                plots.plot_images(images=imgs, targets=targets, paths=paths, fname=f)
        # end epoch ----------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        if ema:
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride'])
        final_epoch = epoch + 1 == opt.epochs
        if not opt.finaltest or final_epoch:  # Calculate mAP
            results, maps, times = test.test(opt.data,
                                             batch_size=opt.batch_size,
                                             imgsz=imgsz_test,
                                             model=ema.ema,
                                             dataloader=testloader,
                                             save_dir=save_dir,
                                             plot=final_epoch)

        # Write
        with open(results_file, 'a') as f:
            f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)

        # Log
        tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                'x/lr0', 'x/lr1', 'x/lr2']  # params
        for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
            if tb_writer:
                tb_writer.add_scalar(tag, x, epoch)  # tensorboard

        # Update best mAP
        fi = metrics.fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        if fi > best_fitness:
            best_fitness = fi

        # Save model
        save = (not opt.finalsave) or (final_epoch)  # true or 最后一次
        if save:
            with open(results_file, 'r') as f:  # create checkpoint
                ckpt = {'epoch': epoch,  # eopch 0开始
                        'best_fitness': best_fitness,  # 最高分
                        'training_results': f.read(),  # 训练结果
                        'model': ema.ema,  # 模型
                        'optimizer': None if final_epoch else optimizer.state_dict(),  # 最后一个epoch得时候不保存optimizer
                        'wandb_id': None}

            # Save last, best and delete
            torch.save(ckpt, last)
            if best_fitness == fi:
                torch.save(ckpt, best)
            del ckpt
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training ----------------------------------------------------------------------------------------------------

    # Strip optimizers
    n = opt.name if opt.name.isnumeric() else ''
    fresults, flast, fbest = save_dir / f'results{n}.txt', wdir / f'last{n}.pt', wdir / f'best{n}.pt'
    for f1, f2 in zip([wdir / 'last.pt', wdir / 'best.pt', results_file], [flast, fbest, fresults]):
        if f1.exists():
            os.rename(f1, f2)  # rename
            if str(f2).endswith('.pt'):  # is *.pt
                general.strip_optimizer(f2)  # strip optimizer
    # Finish
    plots.plot_results(save_dir=save_dir)  # save as results.png
    logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))

    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=str, default='', help='initial weight path, pretrained weight')  # 初始化weight, ''则表示无初始化权重
    parser.add_argument('--cfg', type=str, default='models/yolov5ss_sign.yaml', help='model.yaml path')  # 网络模型的配置文件
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')  # 训练数据的配置文件
    parser.add_argument('--hyp', type=str, default='models/hyp.scratch.yaml', help='hyperparameters path')  # 超参配置文件
    parser.add_argument('--epochs', type=int, default=300)  # 迭代轮询次数
    parser.add_argument('--batch-size', type=int, default=64, help='total batch size for all GPUs')  # batchsize
    parser.add_argument('--img-size', type=list, default=[640, 640], help='[train, test] image sizes')  # 训练的图像尺寸，外图像送入之后，自动resize成这个尺寸
    parser.add_argument('--rect', action='store_true', help='rectangular training')  # 是否矩形进行训练，而不是resize
    parser.add_argument('--resume', type=str, default='', help='resume most recent training, last.pt')  # 恢复最近的训练，后面不需要跟参数，就是const的值
    parser.add_argument('--finalsave', action='store_true', help='only save final checkpoint')  # 只保存最后的模型
    parser.add_argument('--finaltest', action='store_true', help='only test final epoch')  # 旨在最后一次迭代进行测试
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')  # 禁用自动定位检查
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')  # 缓存图片以加速训练
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')  # 使用加权图像选择进行训练
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')  # 训练设备选择
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')  # 多尺度训练
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')  # 是否使用adam优化器
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')  # 导入数据的进程数
    parser.add_argument('--project', default='runs/train', help='save to project/name')  # 存放训练结果的路径
    parser.add_argument('--name', default='exp', help='save to project/name')  # 训练模型的存放目录名
    opt = parser.parse_args()

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    if opt.resume:  # resume an interrupted run 重新开始之前被打断训练
        assert os.path.isfile(opt.resume), 'ERROR: --resume checkpoint does not exist'  # 检查上次训练的模型是否为真的文件
        resume = opt.resume
        with open(Path(opt.resume).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))  # replace
        opt.resume, opt.pretrained = resume, ''
        logger.info('Resuming training from %s' % opt.resume)
    else:
        assert os.path.isfile(opt.data) and os.path.isfile(opt.cfg) and os.path.isfile(opt.hyp), 'ERROR: opt.data, opt.cfg, opt.hyp path error'
        opt.save_dir = general.increment_path(Path(opt.project) / opt.name, exist_ok=False)  # increment run

    if opt.device != '' and opt.device != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.device
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print('using gpu', opt.device)
        else:
            device = torch.device("cpu")
            print('using cpu')
    else:
        device = torch.device("cpu")
        print('using cpu')

    logger.info(opt)  # 打印设置
    logger.info(f'Start Tensorboard with "tensorboard --logdir {opt.project}", view at http://localhost:6006/')  # 设置tensorboard
    tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
    train(opt, device, tb_writer)  # 开始训练，超参 参数 设备 tensorboard
