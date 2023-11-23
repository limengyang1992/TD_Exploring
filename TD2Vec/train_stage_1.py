
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from glob import glob
import logging
from tqdm import tqdm
from lion_pytorch import Lion

from datautils_init import dataloader_finetune
from td2vec import TD2VEC
from sklearn.metrics import classification_report
import numpy as np


torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(
        description="lstm_attention", add_help=False)

    # train path
    parser.add_argument('--output_dim', type=int, default=25)
    parser.add_argument('--logs_path', type=str, default="train.txt")

    parser.add_argument('--save_path',  type=str, default="best_s1.pth")
    # train
    parser.add_argument('--epoch', type=int, default=100, help="Epoch")
    parser.add_argument('--batch_size', type=int,
                        default=32, help="Batch size")
    parser.add_argument('--dropout', type=float,
                        default=0.7, help="Use dropout")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")

    # model
    parser.add_argument('--output_dims', type=int, default=25)
    parser.add_argument('--embed_dim', type=int, default=300)
    

    parser_args, _ = parser.parse_known_args()
    target_parser = argparse.ArgumentParser(parents=[parser])
    args = target_parser.parse_args()
    return args


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    pbar = tqdm(iterator,total=len(iterator))
    for batch in pbar:
    # for batch in iterator:
        optimizer.zero_grad()
        x,y = batch
        x,y = x.cuda(),y.cuda()
        pred,score,attn = model(x)
        loss = criterion(pred, y)
        acc = np.mean((torch.argmax(pred, 1) == y).cpu().numpy())

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
        # logger.info(f"loss {loss},acc {acc}")

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    with torch.no_grad():
        pbar = tqdm(iterator,total=len(iterator))
        labels = []
        predicts = []
        vector = []
        attention = []
        feature = []
        inputs = []
        for batch in pbar:
            x,y = batch
            x,y = x.cuda(),y.cuda()
            pred,attn,feat = model(x)
            inputs.extend(list(x.detach().cpu().numpy()))
            vector.extend(list(pred.detach().cpu().numpy()))
            attention.extend(list(attn.detach().cpu().numpy()))
            feature.extend(list(feat.detach().cpu().numpy()))
            loss = criterion(pred, y.cuda())
            
            acc = np.mean((torch.argmax(pred, 1) == y.cuda()).cpu().numpy())
            predicts.extend(list(torch.argmax(pred, 1).cpu().numpy()))
            labels.extend(list(y.cpu().numpy()))
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    target_names = ['clean', 'noise', 'head', 'tail', 'adver']
    print(classification_report(labels, predicts, target_names=target_names,digits=4))

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
def main(args):

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(args.logs_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    

    logger.info(args)
    
    train_loader,valid_loader = dataloader_finetune(128)
    model = TD2VEC(ts2vec_pt="ckpt/92.pt",output_dims=100,class_num=5)
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = Lion(model.parameters(), lr=args.lr)

    logger.info("start traning...")

    best_acc = 0
    for epoch in range(args.epoch):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion)

        if best_acc <= valid_acc:
            best_acc = valid_acc
            pth = model.state_dict()
            torch.save(pth, args.save_path)

        logger.info(
            f'Epoch: {epoch+1:02}, Train Acc: {train_acc * 100:.2f}%, valid Acc: {valid_acc * 100:.2f}% , Best Acc: {best_acc * 100:.2f}%')
        # scheduler.step(2)

    # # load model
    # test_model = net.BILSTM(embed_matrix, args, device)
    # test_model.to(device)
    # test_model.load_state_dict(torch.load("vector_cache/best.pth"))
    # # test acc
    # test_loss, test_acc = evaluate(test_model, test_iter, criterion)
    # logger.info(f'Test Acc: {test_acc * 100:.2f}%')
    
    
# CUDA_VISIBLE_DEVICES=0 python train2.py --output_dim=25 --logs_path log_gpu0.txt --save_path best_25.pth
# CUDA_VISIBLE_DEVICES=1 python train2.py --output_dim=50 --logs_path log_gpu1.txt --save_path best_50.pth
# CUDA_VISIBLE_DEVICES=2 python train2.py --output_dim=75 --logs_path log_gpu2.txt --save_path best_75.pth
# CUDA_VISIBLE_DEVICES=3 python train2.py --output_dim=150 --logs_path log_gpu3.txt --save_path best_150.pth


if __name__ == '__main__':
    args = parse_args()
    main(args)