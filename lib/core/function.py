from __future__ import absolute_import, division, print_function

import logging
import time
from collections import OrderedDict

import numpy as np
import torch
from core.evaluate import accuracy
from torch.cuda.amp import autocast

from lib.utils.utils import *



def train_one_epoch(
    args,
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    scheduler=None,
    writer=None,
):
    losses = AverageMeter()
    scores = AverageMeter()
    behaviour_dict = {"logit":[],"feat":[],"index":[]}

    model.train()
    if args.half:
        model = model.half()

    for i, (index,input,target) in enumerate(train_loader):
        optimizer.zero_grad()
        target_a = target[:,:1].view(-1).long()
        target_b = target[:,1:2].view(-1).long()
        target_l = target[:,2:3].view(-1).float()
        if args.ricap:
            I_x, I_y = input.size()[2:]

            w = int(
                np.round(I_x *
                         np.random.beta(args.ricap_beta, args.ricap_beta)))
            h = int(
                np.round(I_y *
                         np.random.beta(args.ricap_beta, args.ricap_beta)))
            w_ = [w, I_x - w, w, I_x - w]
            h_ = [h, h, I_y - h, I_y - h]

            cropped_images = {}
            c_ = {}
            W_ = {}
            for k in range(4):
                idx = torch.randperm(input.size(0))
                x_k = np.random.randint(0, I_x - w_[k] + 1)
                y_k = np.random.randint(0, I_y - h_[k] + 1)
                cropped_images[k] = input[idx][:, :, x_k:x_k + w_[k],
                                               y_k:y_k + h_[k]]
                c_[k] = target[idx].cuda()
                W_[k] = w_[k] * h_[k] / (I_x * I_y)

            patched_images = torch.cat(
                (
                    torch.cat((cropped_images[0], cropped_images[1]), 2),
                    torch.cat((cropped_images[2], cropped_images[3]), 2),
                ),
                3,
            )
            patched_images = patched_images.cuda()

            if args.amp:
                with autocast():
                    output = model(patched_images)
                    loss = sum(
                        [W_[k] * criterion(output, c_[k]) for k in range(4)])
            else:
                output = model(patched_images)
                loss = sum(
                    [W_[k] * criterion(output, c_[k]) for k in range(4)])

            acc = sum([W_[k] * accuracy(output, c_[k])[0] for k in range(4)])
        elif args.mixup:
            l = np.random.beta(args.mixup_alpha, args.mixup_alpha)

            idx = torch.randperm(input.size(0))
            input_a, input_b = input, input[idx]
            target_a, target_b = target, target[idx]

            mixed_input = l * input_a + (1 - l) * input_b

            target_a = target_a.cuda()
            target_b = target_b.cuda()
            mixed_input = mixed_input.cuda()

            output = model(mixed_input)
            loss = l * criterion(output, target_a) + (1 - l) * criterion(
                output, target_b)

            acc = (l * accuracy(output, target_a)[0] +
                   (1 - l) * accuracy(output, target_b)[0])
        elif args.cutmix:
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2,
                                                      bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
                       (input.size()[-1] * input.size()[-2]))
            # compute output
            output = model(input)
            loss = criterion(output, target_a) * lam + criterion(
                output, target_b) * (1.0 - lam)

            acc = accuracy(output, target)[0]
        elif args.optims in ['sam', 'asam']:
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.ascent_step()
            acc = accuracy(output, target)[0]
        else:
            # logging.info("train.py line 134")
            if args.half:
                input = input.cuda().half()
            else:
                input = input.cuda()
            
            target_a = target_a.cuda()
            target_b = target_b.cuda()
            target_l = target_l.cuda()
            input = input.cuda()
            
            output,feat = model(input)
            logit = torch.nn.functional.softmax(output,dim=1)
            loss = criterion(output, target_a, target_b, target_l)
            acc = accuracy(output, target_a)[0]
            
            behaviour_dict["logit"].append(logit)
            behaviour_dict["feat"].append(feat)
            behaviour_dict["index"].append(index)

        # compute gradient and do optimizing step
        if args.amp:
            # optimizer.zero_grad()
            args.scaler.scale(loss).backward()
            args.scaler.step(optimizer)
            args.scaler.update()
        elif args.optims in ['sam', 'asam']:
            loss = criterion(model(input), target)
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.gradient_clip)
            optimizer.descent_step()
        else:
            # optimizer.zero_grad()
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.gradient_clip)
            optimizer.step()

        losses.update(loss.item(), input.size(0))
        scores.update(acc.item(), input.size(0))

    log = OrderedDict([('loss', losses.avg), ('acc', scores.avg)])
    if writer is not None:
        writer.add_scalar('Train/Loss', losses.avg, epoch)
        writer.add_scalar('Train/Acc', scores.avg, epoch)

    return log,behaviour_dict



@torch.no_grad()
def validate(args, val_loader, model, criterion, epoch, writer):
    losses = AverageMeter()
    scores = AverageMeter()
    criterion = nn.CrossEntropyLoss()
    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        if args.half:
            input = input.cuda().half()
        else:
            input = input.cuda()
        target = target.cuda()

        output,_ = model(input)  
        loss = criterion(output, target)

        acc1, _ = accuracy(output, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        scores.update(acc1.item(), input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('acc', scores.avg),
    ])

    if writer is not None:
        writer.add_scalar('Val/Loss', losses.avg, epoch)
        writer.add_scalar('Val/Acc', scores.avg, epoch)

    return log


@torch.no_grad()
def uncertain_data_epoch(model,loader1,loader2,loader3,loader4,loader5):
    model.eval()
    uncertain_epoch_1 = []
    uncertain_epoch_2 = []
    uncertain_epoch_3 = []
    uncertain_epoch_4 = []
    uncertain_epoch_5 = []
    
    for i, (index,input,target) in enumerate(loader1):
        input = input.cuda()
        output,feat = model(input)
        logit = torch.nn.functional.softmax(output,dim=1)
        uncertains = torch.sum(logit*torch.log(logit),dim=1)
        uncertain_epoch_1.append(uncertains)
            
    for i, (index,input,target) in enumerate(loader2):
        input = input.cuda()
        output,feat = model(input)
        logit = torch.nn.functional.softmax(output,dim=1)
        uncertains = torch.sum(logit*torch.log(logit),dim=1)
        uncertain_epoch_2.append(uncertains)
        
    for i, (index,input,target) in enumerate(loader3):
        input = input.cuda()
        output,feat = model(input)
        logit = torch.nn.functional.softmax(output,dim=1)
        uncertains = torch.sum(logit*torch.log(logit),dim=1)
        uncertain_epoch_3.append(uncertains)
    
    for i, (index,input,target) in enumerate(loader4):
        input = input.cuda()
        output,feat = model(input)
        logit = torch.nn.functional.softmax(output,dim=1)
        uncertains = torch.sum(logit*torch.log(logit),dim=1)
        uncertain_epoch_4.append(uncertains)
        
    for i, (index,input,target) in enumerate(loader5):
        input = input.cuda()
        output,feat = model(input)
        logit = torch.nn.functional.softmax(output,dim=1)
        uncertains = torch.sum(logit*torch.log(logit),dim=1)
        uncertain_epoch_5.append(uncertains)
        
    uncertain_total1= torch.cat(uncertain_epoch_1).cpu().detach().numpy()[:, np.newaxis]
    uncertain_total2= torch.cat(uncertain_epoch_2).cpu().detach().numpy()[:, np.newaxis]
    uncertain_total3= torch.cat(uncertain_epoch_3).cpu().detach().numpy()[:, np.newaxis]
    uncertain_total4= torch.cat(uncertain_epoch_4).cpu().detach().numpy()[:, np.newaxis]
    uncertain_total5= torch.cat(uncertain_epoch_5).cpu().detach().numpy()[:, np.newaxis]
    total = np.concatenate([uncertain_total1, uncertain_total2,uncertain_total3,uncertain_total4,uncertain_total5], axis=1)
    
    return total

@torch.no_grad()
def uncertain_model_epoch(model,loader):
    uncertain_epoch_1 = []
    uncertain_epoch_2 = []
    uncertain_epoch_3 = []
    uncertain_epoch_4 = []
    uncertain_epoch_5 = []
    
    for i, (index,input,target) in enumerate(loader):
        input = input.cuda()
        output,feat = model(input)
        logit = torch.nn.functional.softmax(output,dim=1)
        uncertains = torch.sum(logit*torch.log(logit),dim=1)
        uncertain_epoch_1.append(uncertains)
        
    for i, (index,input,target) in enumerate(loader):
        input = input.cuda()
        output,feat = model(input)
        logit = torch.nn.functional.softmax(output,dim=1)
        uncertains = torch.sum(logit*torch.log(logit),dim=1)
        uncertain_epoch_2.append(uncertains)
        
    for i, (index,input,target) in enumerate(loader):
        input = input.cuda()
        output,feat = model(input)
        logit = torch.nn.functional.softmax(output,dim=1)
        uncertains = torch.sum(logit*torch.log(logit),dim=1)
        uncertain_epoch_3.append(uncertains)

    for i, (index,input,target) in enumerate(loader):
        input = input.cuda()
        output,feat = model(input)
        logit = torch.nn.functional.softmax(output,dim=1)
        uncertains = torch.sum(logit*torch.log(logit),dim=1)
        uncertain_epoch_4.append(uncertains)
        
    for i, (index,input,target) in enumerate(loader):
        input = input.cuda()
        output,feat = model(input)
        logit = torch.nn.functional.softmax(output,dim=1)
        uncertains = torch.sum(logit*torch.log(logit),dim=1)
        uncertain_epoch_5.append(uncertains)
               
    uncertain_total1= torch.cat(uncertain_epoch_1).cpu().detach().numpy()[:, np.newaxis]
    uncertain_total2= torch.cat(uncertain_epoch_2).cpu().detach().numpy()[:, np.newaxis]
    uncertain_total3= torch.cat(uncertain_epoch_3).cpu().detach().numpy()[:, np.newaxis]
    uncertain_total4= torch.cat(uncertain_epoch_4).cpu().detach().numpy()[:, np.newaxis]
    uncertain_total5= torch.cat(uncertain_epoch_5).cpu().detach().numpy()[:, np.newaxis]
    total = np.concatenate([uncertain_total1, uncertain_total2,uncertain_total3,uncertain_total4,uncertain_total5], axis=1)
    
    return total