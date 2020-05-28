import torch

from datetime import datetime


def get_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor


def logging(s):
    print(datetime.now(), s)


def print_params(model):
    print('total parameters:', sum([np.prod(list(p.size())) for p in model.parameters() if p.requires_grad]))

