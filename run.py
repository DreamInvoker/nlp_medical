import json
import os

from config import get_opt

from dataset import MedicalExtractionDataset
from model import MedicalExtractionModel
import torch
from torch import optim
from torch.utils.data import DataLoader

from utils import get_cuda, logging, print_params
import time


def train(opt):
    train_ds = MedicalExtractionDataset(opt.train_data)
    dev_ds = MedicalExtractionDataset(opt.dev_data)

    dev_dl = DataLoader(dev_ds,
                        batch_size=opt.dev_batch_size,
                        shuffle=False,
                        num_workers=1
                        )

    model = MedicalExtractionModel(opt)
    print(model.parameters)
    print_params(model)

    start_epoch = 1
    learning_rate = opt.lr
    total_epochs = opt.epochs
    log_step = opt.log_step
    pretrain_model = opt.pretrain_model
    model_name = opt.model_name  # 要保存的模型名字

    # load pretrained model
    if pretrain_model != '':
        chkpt = torch.load(pretrain_model, map_location=torch.device('cpu'))
        model.load_state_dict(chkpt['checkpoints'])
        logging('load model from {}'.format(pretrain_model))
        start_epoch = chkpt['epoch'] + 1
        learning_rate = chkpt['learning_rate']
        logging('resume from epoch {} with learning_rate {}'.format(start_epoch, learning_rate))
    else:
        logging('training from scratch with learning_rate {}'.format(learning_rate))

    model = get_cuda(model)

    # TODO 如果用Bert可以改成AdamW
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # TODO loss function
    # criterion =

    checkpoint_dir = opt.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # training process
    # 1.
    global_step = 0
    total_loss = 0
    for epoch in range(1, total_epochs + 1):
        start_time = time.time()
        train_dl = DataLoader(train_ds,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              num_workers=8
                              )
        model.train()
        for batch in train_dl:
            optimizer.zero_grad()
            # TODO 喂数据

            # TODO loss计算
            loss = None

            loss.backward()
            optimizer.step()

            global_step += 1
            total_loss += loss.item()
            if global_step % log_step == 0:
                cur_loss = total_loss / log_step
                elapsed = time.time() - start_time
                logging(
                    '| epoch {:2d} | step {:4d} |  ms/b {:5.2f} | train loss {:5.3f} '.format(
                        epoch, global_step, elapsed * 1000 / log_step, cur_loss*1000))
                total_loss = 0
                start_time = time.time()

        if epoch % opt.test_epoch == 0:
            model.eval()
            with torch.no_grad():
                for batch in dev_dl:
                    # TODO 在验证集上测试
                    pass

        # save model
        # TODO 可以改成只save在dev上最佳的模型
        if epoch % opt.save_model_freq == 0:
            path = os.path.join(checkpoint_dir, model_name + '_{}.pt'.format(epoch))
            torch.save({
                'epoch': epoch,
                'learning_rate': learning_rate,
                'checkpoint': model.state_dict()
            }, path)


if __name__ == '__main__':
    print('processId:', os.getpid())
    print('prarent processId:', os.getppid())
    opt = get_opt()
    print(json.dumps(opt.__dict__, indent=4))
    train(opt)
