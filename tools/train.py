from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import sys
import time
from glob import glob

import _init_paths
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import yaml
from torch.utils.tensorboard import SummaryWriter

from lib.core.function import train_one_epoch, validate
from lib.core.loss import build_criterion
from lib.dataset import build_dataloader
from lib.models import build_model
from lib.optim import build_optimizer
from lib.scheduler import build_scheduler
from lib.utils.args import parse_args
from lib.utils.misc import Timer, build_expname
from lib.utils.utils import *
from lib.utils.similarity import compute_feat_similarity
from lib.utils.upload import cos_upload_file


def main():
    cudnn.benchmark = True
    cudnn.deterministic = True

    args = parse_args()

    # process argparse & yaml
    if not args.config:
        opt = vars(args)
        args = yaml.load(open(args.config), Loader=yaml.FullLoader)
        opt.update(args)
        args = opt
    else:  # yaml priority is higher than args
        opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
        opt.update(vars(args))
        args = argparse.Namespace(**opt)

    args.name = build_expname(args)

    writer = SummaryWriter(
        'exps/%s/runs/%s-%05d' %
        (args.name, time.strftime('%m-%d',
                                  time.localtime()), random.randint(0, 100)))

    if not os.path.exists('exps/%s' % args.name):
        os.makedirs('exps/%s' % args.name)

    print('--------Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    print('--------------------')

    with open('exps/%s/args.txt' % args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' % (arg, getattr(args, arg)), file=f)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=log_format,
        datefmt='%m/%d %I:%M:%S %p',
    )

    fh = logging.FileHandler(os.path.join('exps', args.name, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info(args)

    #######################################################################

    train_loader = build_dataloader(args.dataset, type='train', args=args)
    test_loader = build_dataloader(args.dataset, type='val', args=args)

    # create model
    num_classes = 10 if args.dataset =="cifar10" else 100
    model = build_model(args.model, num_classes=num_classes)
    logging.info(f'param of model {args.model} is {count_params(model)}')

    # stat(model, (3, 32, 32))
    # from torchsummary import summary
    # summary(model, input_size=(3, 32, 32), batch_size=-1)

    model = model.cuda()

    criterion = build_criterion(args).cuda()
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(args, optimizer)

    log = pd.DataFrame(
        index=[],
        columns=['epoch', 'lr', 'loss', 'acc', 'val_loss', 'val_acc'])

    logging.info('Training Start...')

    timer = Timer()

    best_acc = 0
    
    for epoch in range(args.epochs):
        # logging.info("Epoch [%03d/%03d]" % (epoch, args.epochs))
        # train for one epoch
        train_log, behaviour_dict = train_one_epoch(
                args,
                train_loader,
                model,
                criterion,
                optimizer,
                epoch,
                scheduler=scheduler,
                writer=writer,
        )

        train_time = timer()

        # evaluate on validation set
        val_log = validate(args,
                           test_loader,
                           model,
                           criterion,
                           epoch,
                           writer=writer)

        scheduler.step()

        logging.info(
            f"Epoch:[{epoch:03d}/{args.epochs:03d}] lr:{scheduler.get_last_lr()[0]:.4f} train_acc:{train_log['acc']/100.:.3f} train_loss:{train_log['loss']:.4f} train_time:{train_time:03.2f} valid_acc:{val_log['acc']/100.:.3f} valid_loss:{val_log['loss']:.4f} valid_time:{timer():03.2f} total_time: {timer()+train_time:.2f}"
        )

        tmp = pd.Series(
            [
                epoch,
                scheduler.get_last_lr()[0],
                train_log['loss'],
                train_log['acc'],
                val_log['loss'],
                val_log['acc'],
            ],
            index=['epoch', 'lr', 'loss', 'acc', 'val_loss', 'val_acc'],
        )

        log = log.append(tmp, ignore_index=True)
        log.to_csv('exps/%s/log.csv' % args.name, index=False)
        
        numpy_logit = torch.cat(behaviour_dict["logit"]).cpu().detach().numpy()
        numpy_feat = torch.cat(behaviour_dict["feat"]).cpu().detach().numpy()
        numpy_index = torch.cat(behaviour_dict["index"]).cpu().detach().numpy()
        numpy_feat_ = np.concatenate([numpy_index[:,np.newaxis],numpy_feat],axis=1)
        numpy_logit_ = np.concatenate([numpy_index[:,np.newaxis],numpy_logit],axis=1)
        np.save(f'exps/{args.name}/runs/feat_{epoch}.npy',numpy_feat_)
        np.save(f'exps/{args.name}/runs/logit_{epoch}.npy',numpy_logit_)
        
        
        if val_log['acc'] > best_acc:
            

            useless_files = glob('exps/%s/*.pth' % args.name)
            for file in useless_files:
                os.remove(file)
            torch.save(
                model.state_dict(),
                'exps/%s/model_%d.pth' % (args.name, (val_log['acc'] * 100)),
            )
            best_acc = val_log['acc']


    # 计算相似度
    feat_root = f'exps/{args.name}/runs'
    feat_paths = glob(os.path.join(feat_root,"feat_*.npy"))
    compute_feat_similarity(feat_paths)

    
    # 上传cos[邻居id、logit、log.csv][path = args.name]
    for p_logit in glob(os.path.join(feat_root,"logit_*.npy")):
        cos_upload_file(p_logit)
    for p_topk in glob(os.path.join(feat_root,"topk_*.npy")):
        cos_upload_file(p_topk)
    cos_upload_file(f'exps/{args.name}/log.csv')
    cos_upload_file(f'exps/{args.name}/args.txt')
    
    
    
if __name__ == '__main__':
    main()
