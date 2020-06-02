import os
import random
from datetime import datetime

import numpy as np
import torch


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def get_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor


def logging(s):
    print(datetime.now(), s)


def print_params(model):
    print('total parameters:', sum([np.prod(list(p.size())) for p in model.parameters() if p.requires_grad]))


class EarlyStopping:
    def __init__(self, patience=7, mode="max", criterion='val_loss', delta=0.001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf
        self.criterion = criterion

    def __call__(self, epoch_score, model, model_path, epoch=None, learning_rate=None):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path, epoch, learning_rate)
        elif score < self.best_score + self.delta:
            if score > self.best_score:
                self.best_score = score
                self.save_checkpoint(epoch_score, model, model_path, epoch, learning_rate)
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path, epoch, learning_rate)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path, epoch=None, learning_rate=None):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print('{} improved ({} --> {}). Saving model!'.format(
                self.criterion, self.val_score, epoch_score))
            torch.save({
                'epoch': epoch,
                'learning_rate': learning_rate,
                'checkpoint': model.state_dict()
            }, model_path)
        self.val_score = epoch_score
