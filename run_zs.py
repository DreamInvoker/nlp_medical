import json

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from config import *
from dataset_zs import MedicalExtractionDataset
from evaluate import test
from model import MedicalExtractionModel, MedicalExtractionModelForBody
from utils import get_cuda, logging, print_params, EarlyStopping, seed_everything


def train(opt, isbody=False):
    train_ds = MedicalExtractionDataset(opt.train_data)
    dev_ds = MedicalExtractionDataset(opt.dev_data)
    test_ds = MedicalExtractionDataset(opt.test_data)

    dev_dl = DataLoader(dev_ds, batch_size=opt.dev_batch_size, shuffle=False, num_workers=opt.num_worker)
    test_dl = DataLoader(test_ds, batch_size=opt.dev_batch_size, shuffle=False, num_workers=opt.num_worker)

    if isbody:
        logging('training for body')
        model = MedicalExtractionModelForBody(opt)
    else:
        logging('training for subject, decorate and body')
        model = MedicalExtractionModel(opt)
    # print(model.parameters)
    print_params(model)

    start_epoch = 1
    learning_rate = opt.lr
    total_epochs = opt.epochs
    pretrain_model = opt.pretrain_model
    model_name = opt.model_name  # 要保存的模型名字

    # load pretrained model
    if pretrain_model != '' and not isbody:
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
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    checkpoint_dir = opt.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    es = EarlyStopping(patience=opt.patience, mode="min", criterion='val loss')
    for epoch in range(start_epoch, total_epochs + 1):
        train_loss = 0.0
        model.train()
        train_dl = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_worker)
        tk_train = tqdm(train_dl, total=len(train_dl))
        for batch in tk_train:
            optimizer.zero_grad()
            subject_target_ids = batch['subject_target_ids']
            decorate_target_ids = batch['decorate_target_ids']
            freq_target_ids = batch['freq_target_ids']
            body_target_ids = batch['body_target_ids']
            mask = batch['mask'].float()
            body_mask = batch['body_mask']
            loss = None
            if isbody:
                body_logits = model(
                    input_ids=batch['body_input_ids'],
                    attention_mask=batch['body_mask'],
                    token_type_ids=batch['body_token_type_ids']
                )
                loss = torch.sum(criterion(body_logits, body_target_ids)
                                 * body_mask.unsqueeze(-1)) / torch.sum(body_mask)
            else:
                subject_logits, decorate_logits, freq_logits = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['mask'],
                    token_type_ids=batch['token_type_ids']
                )
                loss = torch.sum((criterion(subject_logits, subject_target_ids) +
                                  criterion(decorate_logits, decorate_target_ids) +
                                  criterion(freq_logits, freq_target_ids))
                                 * mask.unsqueeze(-1)) / torch.sum(mask)

            loss.backward()
            optimizer.step()
            scheduler.step()

            tk_train.set_postfix(train_loss='{:5.3f} / 1000'.format(1000 * loss.item()),
                                 epoch='{:2d}'.format(epoch))
            train_loss += loss.item() * subject_target_ids.shape[0]

        avg_train_loss = train_loss * 1000 / len(train_ds)
        print('train loss per example: {:5.3f} / 1000'.format(avg_train_loss))

        avg_val_loss = test(model, dev_ds, dev_dl, criterion, threshold, 'val', isbody=isbody)

        # 保留最佳模型方便evaluation
        if isbody:
            save_model_path = os.path.join(checkpoint_dir, model_name + '_body_best.pt')
        else:
            save_model_path = os.path.join(checkpoint_dir, model_name + '_best.pt')

        es(avg_val_loss, model, model_path=save_model_path, epoch=epoch, learning_rate=learning_rate)
        if es.early_stop:
            print("Early stopping")
            break

        # 保存epoch的模型方便断点续训
        if epoch % opt.save_model_freq == 0:
            if isbody:
                save_model_path = os.path.join(checkpoint_dir, model_name + '_body_{}.pt'.format(epoch))
            else:
                save_model_path = os.path.join(checkpoint_dir, model_name + '_{}.pt'.format(epoch))
            torch.save({
                'epoch': epoch,
                'learning_rate': learning_rate,
                'checkpoints': model.state_dict()
            }, save_model_path)

    # load best model and test
    if isbody:
        best_model_path = os.path.join(checkpoint_dir, model_name + '_body_best.pt')
    else:
        best_model_path = os.path.join(checkpoint_dir, model_name + '_best.pt')
    chkpt = torch.load(best_model_path, map_location=torch.device('cpu'))
    model.load_state_dict(chkpt['checkpoints'])
    if isbody:
        logging('load best body model from {} and test ...'.format(best_model_path))
    else:
        logging('load best model from {} and test ...'.format(best_model_path))
    test(model, test_ds, test_dl, criterion, threshold, 'test', isbody)
    model.cpu()


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
    train(opt, isbody=True)
