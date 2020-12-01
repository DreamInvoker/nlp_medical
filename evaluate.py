import json
import os

import numpy as np
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
    mask = mask.cpu().numpy()
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
        mask = mask.cpu().numpy()
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


def get_spans(label_list, offsets, raw_text):
    span_list = set([])
    text = ""
    for label, offset in zip(label_list, offsets):
        if label == 0:
            span_list.add(text)
            text = ""
        else:
            text += ''.join(raw_text[offset[0]:offset[1]])
    span_list = list(span_list)
    if "" in span_list:
        span_list.remove("")
    return span_list

def test(model, ds, loader, criterion, threshold=0.5, name='val', isbody=False, output=False):
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
        output_list = []
        for batch in tk_val:
            subject_target_ids = batch['subject_target_ids']
            decorate_target_ids = batch['decorate_target_ids']
            freq_target_ids = batch['freq_target_ids']
            body_target_ids = batch['body_target_ids']
            batch_size = subject_target_ids.shape[0]
            mask = batch['mask'].float().unsqueeze(-1)
            body_mask = batch['body_mask'].float().unsqueeze(-1)
            raw_text = batch['raw_text']
            symptom_name = batch['symptom_name']
            body_text_offsets = batch['body_text_offsets'].numpy()
            offsets = batch['offsets'].numpy()
            loss = None
            if isbody:
                body_logits = model(
                    input_ids=batch['body_input_ids'],
                    attention_mask=batch['body_mask'],
                    token_type_ids=batch['body_token_type_ids']
                )
                loss = torch.sum(criterion(body_logits, body_target_ids)
                                 * body_mask) / torch.sum(body_mask)

                body_pred = ((torch.sigmoid(body_logits) >= threshold) * body_mask).long()

                body_evaluation.span_metrics(torch.sigmoid(body_logits), body_target_ids, threshold, body_mask)
                body_metrics_list.append(
                    token_metrics(torch.sigmoid(body_logits), body_target_ids, threshold=threshold, mask=body_mask))

                if output:
                    for i in range(batch_size):
                        length = int(torch.sum(body_mask[i]).item())
                        body_pred_i = body_pred[i, :length].squeeze().cpu().numpy().tolist()
                        raw_text_i = raw_text[i]
                        body_text_offsets_i = body_text_offsets[i, :length]
                        symptom_name_i = symptom_name[i]
                        target_i = body_target_ids[i, :length].long().squeeze().cpu().numpy().tolist()
                        span_list = get_spans(body_pred_i, body_text_offsets_i, raw_text_i)
                        target_list = get_spans(target_i, body_text_offsets_i, raw_text_i)

                        output_list.append({
                            'text': raw_text_i,
                            'symptom_name': symptom_name_i,
                            'predict_body': span_list,
                            'gold_body': target_list
                        })

            else:
                subject_logits, decorate_logits, freq_logits = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['mask'],
                    token_type_ids=batch['token_type_ids']
                )
                loss = torch.sum((criterion(subject_logits, subject_target_ids) +
                                  criterion(decorate_logits, decorate_target_ids) +
                                  criterion(freq_logits, freq_target_ids))
                                 * mask) / torch.sum(mask)
                subject_pred = (torch.sigmoid(subject_logits) >= threshold) * mask
                decorate_pred = (torch.sigmoid(decorate_logits) >= threshold) * mask
                freq_pred = (torch.sigmoid(freq_logits) >= threshold) * mask

                if output:
                    for i in range(batch_size):
                        length = int(torch.sum(mask[i]).item())
                        subject_pred_i = subject_pred[i, :length].long().squeeze().cpu().numpy().tolist()
                        decorate_pred_i = decorate_pred[i, :length].long().squeeze().cpu().numpy().tolist()
                        freq_pred_i = freq_pred[i, :length].long().squeeze().cpu().numpy().tolist()
                        raw_text_i = raw_text[i]
                        offsets_i = offsets[i, :length]
                        symptom_name_i = symptom_name[i]
                        subject_target_i = subject_target_ids[i, :length].long().squeeze().cpu().numpy().tolist()
                        decorate_target_i = decorate_target_ids[i, :length].long().squeeze().cpu().numpy().tolist()
                        freq_target_i = freq_target_ids[i, :length].long().squeeze().cpu().numpy().tolist()
                        subject_span_list = get_spans(subject_pred_i, offsets_i, symptom_name_i)
                        decorate_span_list = get_spans(decorate_pred_i, offsets_i, symptom_name_i)
                        freq_span_list = get_spans(freq_pred_i, offsets_i, symptom_name_i)
                        subject_target_list = get_spans(subject_target_i, offsets_i, symptom_name_i)
                        decorate_target_list = get_spans(decorate_target_i, offsets_i, symptom_name_i)
                        freq_target_list = get_spans(freq_target_i, offsets_i, symptom_name_i)

                        output_list.append({
                            'text': raw_text_i,
                            'symptom_name': symptom_name_i,
                            'predict_subject': subject_span_list,
                            'gold_subject': subject_target_list,
                            'predict_decorate': decorate_span_list,
                            'gold_decorate': decorate_target_list,
                            'predict_freq': freq_span_list,
                            'gold_freq': freq_target_list,
                        })

                subject_evaluation.span_metrics(torch.sigmoid(subject_logits), subject_target_ids, threshold, mask)
                decorate_evaluation.span_metrics(torch.sigmoid(decorate_logits), decorate_target_ids, threshold, mask)
                freq_evaluation.span_metrics(torch.sigmoid(freq_logits), freq_target_ids, threshold, mask)

                subject_metrics_list.append(
                    token_metrics(torch.sigmoid(subject_logits), subject_target_ids, threshold=threshold, mask=mask))
                decorate_metrics_list.append(
                    token_metrics(torch.sigmoid(decorate_logits), decorate_target_ids, threshold=threshold, mask=mask))
                freq_metrics_list.append(token_metrics(torch.sigmoid(freq_logits), freq_target_ids, threshold=threshold, mask=mask))

            batch_size_list.append(batch_size)

            test_loss += loss.item() * batch_size

        avg_test_loss = test_loss * 1000 / len(ds)
        print('{} loss per example: {:5.3f} / 1000'.format(name, avg_test_loss))

        if isbody:
            print_eval_all(body_evaluation, 'body')
            print_eval(body_metrics_list, batch_size_list, 'body')
            if output:
                with open('body_output.json', 'w') as fw:
                    json.dump(output_list, fw)
        else:
            print_eval(subject_metrics_list, batch_size_list, 'subject')
            print_eval_all(subject_evaluation, 'subject')
            print_eval(decorate_metrics_list, batch_size_list, 'decorate')
            print_eval_all(decorate_evaluation, 'decorate')
            print_eval(freq_metrics_list, batch_size_list, 'freq')
            print_eval_all(freq_evaluation, 'freq')

            if output:
                with open('output.json', 'w') as fw:
                    json.dump(output_list, fw)

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
    threshold = opt.threshold

    body_best_model_path = os.path.join(checkpoint_dir, model_name + '_body_best.pt')
    chkpt = torch.load(body_best_model_path, map_location=torch.device('cpu'))
    body_model.load_state_dict(chkpt['checkpoints'])

    best_model_path = os.path.join(checkpoint_dir, model_name + '_best.pt')
    chkpt = torch.load(best_model_path, map_location=torch.device('cpu'))
    model.load_state_dict(chkpt['checkpoints'])

    body_model = get_cuda(body_model)
    print('test body ...')
    test(body_model, test_ds, test_dl, criterion, threshold, 'test', True, True)
    body_model.cpu()

    model = get_cuda(model)
    print('test others ...')
    test(model, test_ds, test_dl, criterion, threshold, 'test', False, True)
    model.cpu()
