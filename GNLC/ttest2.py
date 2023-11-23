
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

from datautils import dataloader_finetune
from td2vec import TS2VEC
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report


import numpy as np
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("log4.txt")
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

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(outputs, labels):
    """Computes accuracy for given outputs and ground truths"""

    _, predicted = torch.max(outputs, 1)
    n_samples = labels.size(0)
    n_correct = (predicted == labels).sum().item()
    acc = 100.0 * n_correct / n_samples
    return acc

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    accuracies = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm(iterator,total=len(iterator))
        labels = []
        predicts = []
        vector = []
        attention = []
        feature = []
        for batch in pbar:
            x,y = batch
            x,y = x.cuda(),y.cuda()
            feat = model.encode(x)
            feature.extend(list(feat.detach().cpu().numpy()))

    return feature


def main(args):
    from seq2vec import TS2Vec
    logger.info("Model: ts2vec with Attension")
    train_loader,valid_loader = dataloader_finetune(128)
    test_model = TS2Vec()
    test_model.load("ckpt/92.pt")
    # test acc
    criterion = nn.CrossEntropyLoss().to(device)
    feature = test_model.encode(valid_loader,encoding_window = 'full_series')
    # feature = evaluate(test_model, valid_loader, criterion)
    np.save("feat_conv.npy",np.array(feature))
    
    

if __name__ == '__main__':
    args = parse_args()
    main(args)