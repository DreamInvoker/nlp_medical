import json

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from dataset_zs import MedicalExtractionDatasetForSubjectAndBody
from model import MedicalExtractionModel
from utils import get_cuda, logging, print_params, EarlyStopping, seed_everything


# TODO 监控验证集上的评价指标比如F1或者jaccard score
# TODO 加入warm up

def train(opt):
    train_ds = MedicalExtractionDatasetForSubjectAndBody(opt.train_data)
    dev_ds = MedicalExtractionDatasetForSubjectAndBody(opt.dev_data)
    # test_ds = MedicalExtractionDatasetForSubjectAndBody(opt.test_data)

    dev_dl = DataLoader(dev_ds,
                        batch_size=opt.dev_batch_size,
                        shuffle=False,
                        num_workers=16
                        )

    model = MedicalExtractionModel(opt)
    print(model.parameters)
    print_params(model)

    start_epoch = 1
    learning_rate = opt.lr
    total_epochs = opt.epochs
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

    # TODO 可以改成AdamW
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.BCEWithLogitsLoss()

    checkpoint_dir = opt.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    es = EarlyStopping(patience=5, mode="min", criterion='val loss')
    for epoch in range(start_epoch, total_epochs + 1):
        train_loss = 0.0
        val_loss = 0.0

        train_dl = DataLoader(train_ds,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              num_workers=1
                              )
        model.train()
        tk_train = tqdm(train_dl, total=len(train_dl))
        for batch in tk_train:
            optimizer.zero_grad()
            subject_target_ids = batch['subject_target_ids']
            body_target_ids = batch['body_target_ids']

            subject_logits, body_logits = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['mask'],
                token_type_ids=batch['token_type_ids']
            )
            loss = criterion(subject_logits, subject_target_ids) + criterion(body_logits, body_target_ids)
            loss.backward()
            optimizer.step()
            tk_train.set_postfix(train_loss='{:5.3f} / 1000'.format(1000 * loss.item()),
                                 epoch='{:2d}'.format(epoch))
            train_loss += loss.item() * subject_target_ids.shape[0]

        # TODO 如果eval时间太久，可能不太需要每个epoch都验证一次
        # if epoch % opt.test_epoch == 0:
        model.eval()

        with torch.no_grad():
            tk_val = tqdm(dev_dl, total=len(dev_dl))
            for batch in tk_val:
                subject_target_ids = batch['subject_target_ids']
                body_target_ids = batch['body_target_ids']

                subject_logits, body_logits = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['mask'],
                    token_type_ids=batch['token_type_ids']
                )
                loss = criterion(subject_logits, subject_target_ids) + criterion(body_logits, body_target_ids)
                tk_val.set_postfix(val_loss='{:5.3f} / 1000'.format(1000 * loss.item()),
                                   epoch='{:2d}'.format(epoch))
                val_loss += loss.item() * subject_target_ids.shape[0]

        avg_train_loss = train_loss * 1000 / len(train_ds)
        avg_val_loss = val_loss * 1000 / len(dev_ds)

        logging('train loss per example: {:5.3f} / 1000'.format(avg_train_loss))
        logging('val loss per example: {:5.3f} / 1000'.format(avg_val_loss))

        # 保留最佳模型方便evaluation
        save_model_path = os.path.join(checkpoint_dir, model_name + '_best.pt'.format(epoch))
        es(avg_val_loss, model, model_path=save_model_path, epoch=epoch, learning_rate=learning_rate)
        if es.early_stop:
            print("Early stopping")
            break

        # 保存epoch的模型方便断点续训
        if epoch % opt.save_model_freq == 0:
            save_model_path = os.path.join(checkpoint_dir, model_name + '_{}.pt'.format(epoch))
            torch.save({
                'epoch': epoch,
                'learning_rate': learning_rate,
                'checkpoint': model.state_dict()
            }, save_model_path)


if __name__ == '__main__':
    from multiprocessing import set_start_method

    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    print('processId:', os.getpid())
    print('prarent processId:', os.getppid())
    opt = get_opt()
    print(json.dumps(opt.__dict__, indent=4))

    seed_everything(19960122)
    train(opt)
