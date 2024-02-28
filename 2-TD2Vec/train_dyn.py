
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

from datautils_dyn import dataloader_finetune
from td2vec import TD2VEC

import numpy as np
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("log2.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(
        description="lstm_attention", add_help=False)

    # train path
    parser.add_argument('--output_dim', type=int, default=2)
    parser.add_argument('--train_path', type=str,
                        default="dataset/stsa_seg/train_clean.csv")
    parser.add_argument('--valid_path', type=str,
                        default="dataset/stsa_seg/dev_clean.csv")
    parser.add_argument('--test_path', type=str,
                        default="dataset/stsa_seg/test_clean.csv")

    # train
    parser.add_argument('--epoch', type=int, default=100, help="Epoch")
    parser.add_argument('--batch_size', type=int,
                        default=32, help="Batch size")
    parser.add_argument('--dropout', type=float,
                        default=0.7, help="Use dropout")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")

    # model
    parser.add_argument('--max_length', type=int, default=96)
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--hidden_dim', type=int, default=512)

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
        pred,attn = model(x)
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
        for batch in pbar:
            x,y,path = batch
            x,y = x.cuda(),y.cuda()
            pred,attn = model(x)
            loss = criterion(pred, y)
            acc = np.mean((torch.argmax(pred, 1) == y).cpu().numpy())
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def main(args):


    logger.info("Model: ts2vec with Attension")
    
    train_loader,valid_loader = dataloader_finetune(128)
    model = TD2VEC(ts2vec_pt="ckpt/92.pt")
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
            torch.save(pth, "ckpt/best2.pth")

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


if __name__ == '__main__':
    args = parse_args()
    main(args)