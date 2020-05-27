import json
import os

from config import get_opt

from dataset import MedicalExtractionDataset

import torch
from torch.utils.data import DataLoader


def train(opt):
    train_ds = MedicalExtractionDataset(opt.train_data)
    dev_ds = MedicalExtractionDataset(opt.dev_data)
    test_ds = MedicalExtractionDataset(opt.test_data)

    train_dl = DataLoader(train_ds,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          num_workers=8,
                          )



if __name__ == '__main__':
    print('processId:', os.getpid())
    print('prarent processId:', os.getppid())
    opt = get_opt()
    print(json.dumps(opt.__dict__, indent=4))
    train(opt)
