import json
import os

import torch
from sklearn import metrics
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_opt
from dataset_zs import MedicalExtractionDataset
from model import MedicalExtractionModel, MedicalExtractionModelForBody
from utils import get_cuda


def token_metrics(pred_logits, gold_labels, threshold, mask=None):
    pred_label = pred_logits.cpu().numpy()
    gold_label = gold_labels.cpu().numpy()

    batch_size = gold_label.shape[0]
    pred_label[pred_label < threshold] = 0
    pred_label[pred_label >= threshold] = 1
    mask = mask.unsqueeze(-1).cpu().numpy()
    if mask is not None:
        pred_label = pred_label * mask
        gold_label = gold_label * mask

    prec = 0
    rec = 0
    f1 = 0
    jac = 0
    for gold, pred in zip(gold_label, pred_label):
        prec += metrics.precision_score(gold, pred, labels=[1], average='micro', zero_division=1)
        rec += metrics.recall_score(gold, pred, labels=[1], average='micro', zero_division=1)
        f1 += metrics.f1_score(gold, pred, labels=[1], average='micro', zero_division=1)
        jac += metrics.jaccard_score(gold, pred, labels=[1], average='micro')

    prec /= batch_size
    rec /= batch_size
    f1 /= batch_size
    jac /= batch_size
    return prec, rec, f1, jac


def span_metrics(pred_logits, gold_labels, threshold, mask=None):
    pred_label = pred_logits.cpu().numpy()
    gold_label = gold_labels.cpu().numpy()

    batch_size = gold_label.shape[0]
    pred_label[pred_label < threshold] = 0
    pred_label[pred_label >= threshold] = 1
    mask = mask.cpu().numpy()
    if mask:
        pred_label = pred_label * mask
        gold_label = gold_label * mask

    prec = 0
    rec = 0
    f1 = 0

    gold_span = get_pos_list(gold_label)
    pred_span = get_pos_list(pred_label)
    for batch_index in range(batch_size):
        print(batch_index)
        p, r, f = span_metrics_eval(gold_span[batch_index], pred_span[batch_index])
        prec += p
        rec += r
        f1 += f

    prec /= batch_size
    rec /= batch_size
    f1 /= batch_size

    return prec, rec, f1


class Span_eval():
    def __init__(self):
        self.TP = 0
        self.TP_FP = 0
        self.TP_FN = 0

    def span_metrics(self, pred_logits, gold_labels, threshold=0.5, mask=None):
        pred_label = pred_logits.cpu().numpy()
        gold_label = gold_labels.cpu().numpy()

        batch_size = gold_label.shape[0]
        pred_label[pred_label < threshold] = 0
        pred_label[pred_label >= threshold] = 1
        mask = mask.unsqueeze(-1).cpu().numpy()
        if mask is not None:
            pred_label = pred_label * mask
            gold_label = gold_label * mask

        gold_span = get_pos_list(gold_label)
        pred_span = get_pos_list(pred_label)
        for batch_index in range(batch_size):
            self.metrics_eval(gold_span[batch_index], pred_span[batch_index])

    def metrics_eval(self, gold_span_pos, pred_span_pos):
        cnt = 0
        for span in pred_span_pos:
            if span in gold_span_pos:
                cnt += 1

        self.TP += cnt
        self.TP_FP += len(pred_span_pos)
        self.TP_FN += len(gold_span_pos)

    def cal_metrics(self):
        prec = self.TP / self.TP_FP
        recall = self.TP / self.TP_FN
        if prec == 0 and recall == 0:
            f1 = 1
        else:
            f1 = 2 * (prec * recall) / (prec + recall)

        return prec, recall, f1


def span_metrics_eval(gold_span_pos, pred_span_pos):
    cnt = 0

    for span in pred_span_pos:
        if span in gold_span_pos:
            cnt += 1
    if pred_span_pos != []:
        prec = cnt / len(pred_span_pos)
    else:
        prec = 1

    cnt = 0

    for span in gold_span_pos:
        if span in pred_span_pos:
            cnt += 1

    if gold_span_pos != []:
        recall = cnt / len(gold_span_pos)
    else:
        recall = 1

    if pred_span_pos == [] and gold_span_pos == []:
        f1 = 1
    elif prec == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2 * (prec * recall) / (prec + recall)

    return prec, recall, f1


def get_pos_list(label):
    batch_size = label.shape[0]
    length = label.shape[1]

    batch_pos_list = []
    for batch_index in range(batch_size):
        span_list = []
        pos_list = []
        for idx in range(length - 1):

            if idx == 0 and label[batch_index][idx] == 1:
                pos_list.append(idx)

            if label[batch_index][idx] == 0 and label[batch_index][idx + 1] == 1:
                pos_list.append(idx + 1)

            elif label[batch_index][idx] == 1 and label[batch_index][idx + 1] == 0:
                pos_list.append(idx)
                span_list.append(pos_list)
                pos_list = []

            else:
                pass
            if idx == length - 2 and label[batch_index][idx + 1] == 1:
                pos_list.append(idx + 1)
                span_list.append(pos_list)
                pos_list = []
                continue

        batch_pos_list.append(span_list)
        # print(batch_pos_list)

    return batch_pos_list


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

    print(
        '{} token\tP: {:.3f}\tR: {:.3f}\tF1: {:.3f}\tJaccard: {:.3f}'.format(attr_type, avg_p, avg_r, avg_f1, avg_jac))


def print_eval_all(evaluation, attr_type):
    p, r, f1 = evaluation.cal_metrics()

    print('{} span\tP: {:.3f}\tR: {:.3f}\tF1: {:.3f}'.format(attr_type, p, r, f1))


def test(model, ds, loader, criterion, threshold=0.5, name='val', isbody=False):
    model.eval()
    test_loss = 0.0
    subject_evaluation = Span_eval()
    decorate_evaluation = Span_eval()
    freq_evaluation = Span_eval()
    body_evaluation = Span_eval()
    subject_metrics_list = []
    decorate_metrics_list = []
    freq_metrics_list = []
    body_metrics_list = []
    batch_size_list = []
    with torch.no_grad():

        tk_val = tqdm(loader, total=len(loader))
        for batch in tk_val:
            subject_target_ids = batch['subject_target_ids']
            decorate_target_ids = batch['decorate_target_ids']
            freq_target_ids = batch['freq_target_ids']
            body_target_ids = batch['body_target_ids']
            batch_size = subject_target_ids.shape[0]
            mask = batch['mask'].float()
            body_mask = batch['body_mask'].float()
            loss = None
            if isbody:
                body_logits = model(
                    input_ids=batch['body_input_ids'],
                    attention_mask=batch['body_mask'],
                    token_type_ids=batch['body_token_type_ids']
                )
                loss = torch.sum(criterion(body_logits, body_target_ids)
                                 * body_mask.unsqueeze(-1)) / torch.sum(body_mask)
                body_evaluation.span_metrics(body_logits, body_target_ids, threshold, body_mask)
                body_metrics_list.append(
                    token_metrics(body_logits, body_target_ids, threshold=threshold, mask=body_mask))
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
                subject_evaluation.span_metrics(subject_logits, subject_target_ids, threshold, mask)
                decorate_evaluation.span_metrics(decorate_logits, decorate_target_ids, threshold, mask)
                freq_evaluation.span_metrics(freq_logits, freq_target_ids, threshold, mask)

                subject_metrics_list.append(
                    token_metrics(subject_logits, subject_target_ids, threshold=threshold, mask=mask))
                decorate_metrics_list.append(
                    token_metrics(decorate_logits, decorate_target_ids, threshold=threshold, mask=mask))
                freq_metrics_list.append(token_metrics(freq_logits, freq_target_ids, threshold=threshold, mask=mask))

                batch_size_list.append(batch_size)

            test_loss += loss.item() * batch_size

        avg_test_loss = test_loss * 1000 / len(ds)
        print('{} loss per example: {:5.3f} / 1000'.format(name, avg_test_loss))

        if isbody:
            print_eval_all(body_evaluation, 'body')
            print_eval(body_metrics_list, batch_size_list, 'body')
        else:
            print_eval(subject_metrics_list, batch_size_list, 'subject')
            print_eval_all(subject_evaluation, 'subject')
            print_eval(decorate_metrics_list, batch_size_list, 'decorate')
            print_eval_all(decorate_evaluation, 'decorate')
            print_eval(freq_metrics_list, batch_size_list, 'freq')
            print_eval_all(freq_evaluation, 'freq')

    return avg_test_loss


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
    test_ds = MedicalExtractionDataset(opt.test_data)
    test_dl = DataLoader(test_ds, batch_size=opt.dev_batch_size, shuffle=False, num_workers=opt.num_worker)

    body_model = MedicalExtractionModelForBody(opt)
    model = MedicalExtractionModel(opt)

    # load best model and test
    checkpoint_dir = opt.checkpoint_dir
    model_name = opt.model_name
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    threshold = opt

    body_best_model_path = os.path.join(checkpoint_dir, model_name + '_body_best.pt')
    chkpt = torch.load(body_best_model_path, map_location=torch.device('cpu'))
    body_model.load_state_dict(chkpt['checkpoints'])

    best_model_path = os.path.join(checkpoint_dir, model_name + '_best.pt')
    chkpt = torch.load(best_model_path, map_location=torch.device('cpu'))
    model.load_state_dict(chkpt['checkpoints'])

    body_model = get_cuda(body_model)
    test(body_model, test_ds, test_dl, criterion, threshold, 'test', True)
    body_model.cpu()

    model = get_cuda(model)
    test(model, test_ds, test_dl, criterion, threshold, 'test', False)
    model.cpu()
