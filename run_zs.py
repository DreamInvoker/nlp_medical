import json

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from config import *
from dataset_zs import MedicalExtractionDatasetForSubjectAndBody
from evaluate import Metrics
from model import MedicalExtractionModel
from utils import get_cuda, logging, print_params, EarlyStopping, seed_everything


def print_eval(metrics_list, batch_size_list, attr_type):
    avg_p, avg_r, avg_f1, avg_jac = 0.0, 0.0, 0.0, 0.0
    count = 0
    for (p, r, f1, jac), bsz in zip(metrics_list, batch_size_list):
        avg_p += p * bsz
        avg_r += r * bsz
        avg_f1 += f1 * bsz
        avg_jac += jac * bsz
        count += bsz
    avg_p, avg_r, avg_f1, avg_jac = avg_p / count, avg_r / count, avg_f1 / count, avg_jac / count

    print('{}\tP: {:.3f}\tR: {:.3f}\tF1: {:.3f}\tJaccard: {:.3f}'.format(attr_type, avg_p, avg_r, avg_f1, avg_jac))


def test(model, ds, loader, criterion, threshold=0.5, name='val'):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        subject_metrics_list = []
        body_metrics_list = []
        batch_size_list = []
        tk_val = tqdm(loader, total=len(loader))
        for batch in tk_val:
            subject_target_ids = batch['subject_target_ids']
            body_target_ids = batch['body_target_ids']
            batch_size = subject_target_ids.shape[0]

            subject_logits, body_logits = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['mask'],
                token_type_ids=batch['token_type_ids']
            )
            loss = criterion(subject_logits, subject_target_ids) + criterion(body_logits, body_target_ids)
            test_loss += loss.item() * batch_size

            # TODO 暂时对于body和subject用统一的thresh
            subject_metrics_list.append(Metrics(subject_logits, subject_target_ids, threshold=threshold))
            body_metrics_list.append(Metrics(body_logits, body_target_ids, threshold=threshold))
            batch_size_list.append(batch_size)

        avg_test_loss = test_loss * 1000 / len(ds)
        print('{} loss per example: {:5.3f} / 1000'.format(name, avg_test_loss))

        print_eval(subject_metrics_list, batch_size_list, 'subject')
        print_eval(body_metrics_list, batch_size_list, 'body')

    return test_loss


def train(opt):
    train_ds = MedicalExtractionDatasetForSubjectAndBody(opt.train_data)
    dev_ds = MedicalExtractionDatasetForSubjectAndBody(opt.dev_data)
    test_ds = MedicalExtractionDatasetForSubjectAndBody(opt.test_data)

    train_dl = DataLoader(train_ds,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          num_workers=opt.num_worker
                          )
    dev_dl = DataLoader(dev_ds,
                        batch_size=opt.dev_batch_size,
                        shuffle=False,
                        num_workers=opt.num_worker
                        )

    test_dl = DataLoader(test_ds,
                         batch_size=opt.dev_batch_size,
                         shuffle=False,
                         num_workers=opt.num_worker
                         )

    model = MedicalExtractionModel(opt)
    # print(model.parameters)
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

    num_train_steps = int(len(train_ds) / opt.batch_size * opt.epochs)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.AdamW(optimizer_parameters, lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=opt.num_warmup_steps,
        num_training_steps=num_train_steps
    )
    threshold = opt.threshold
    criterion = nn.BCEWithLogitsLoss()

    checkpoint_dir = opt.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    es = EarlyStopping(patience=opt.patience, mode="min", criterion='val loss')
    for epoch in range(start_epoch, total_epochs + 1):
        train_loss = 0.0
        val_loss = 0.0

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
            scheduler.step()

            tk_train.set_postfix(train_loss='{:5.3f} / 1000'.format(1000 * loss.item()),
                                 epoch='{:2d}'.format(epoch))
            train_loss += loss.item() * subject_target_ids.shape[0]

        avg_train_loss = train_loss * 1000 / len(train_ds)
        print('train loss per example: {:5.3f} / 1000'.format(avg_train_loss))

        avg_val_loss = test(model, dev_ds, dev_dl, criterion, threshold, 'val')

        # 保留最佳模型方便evaluation
        save_model_path = os.path.join(checkpoint_dir, model_name + '_best.pt')
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

    # load best model and test
    best_model_path = os.path.join(checkpoint_dir, model_name + '_best.pt')
    chkpt = torch.load(best_model_path, map_location=torch.device('cpu'))
    model.load_state_dict(chkpt['checkpoints'])
    logging('load best model from {} and test ...'.format(best_model_path))
    avg_test_loss = test(model, test_ds, test_dl, criterion, threshold, 'test')
    print('test loss: {}'.format(avg_test_loss))


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
