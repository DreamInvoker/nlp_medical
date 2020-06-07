
import torch
from sklearn import metrics
import numpy as np

def token_metrics(pred_logits,gold_labels,threshold,mask=None):
    pred_label = pred_logits.cpu().numpy()
    gold_label = gold_labels.cpu().numpy()

    batch_size = gold_label.shape[0]
    pred_label[pred_label < threshold] = 0
    pred_label[pred_label >= threshold] = 1

    if mask:
        pred_label = pred_label*mask
        gold_label = gold_label*mask


    prec = 0
    rec = 0
    f1 = 0
    jac = 0
    for gold, pred in zip(gold_label, pred_label):
        prec += metrics.precision_score(gold, pred, labels=[1], average='micro',zero_division=1)
        rec += metrics.recall_score(gold, pred, labels=[1], average='micro',zero_division=1)
        f1 += metrics.f1_score(gold, pred, labels=[1], average='micro',zero_division=1)
        jac += metrics.jaccard_score(gold, pred, labels=[1], average='micro',zero_division=1)

    prec /= batch_size
    rec /= batch_size
    f1 /= batch_size
    jac /= batch_size
    return prec, rec, f1, jac

def span_metrics(pred_logits,gold_labels,threshold,mask=None):
    pred_label = pred_logits.cpu().numpy()
    gold_label = gold_labels.cpu().numpy()

    batch_size = gold_label.shape[0]
    pred_label[pred_label < threshold] = 0
    pred_label[pred_label >= threshold] = 1

    if mask:
        pred_label = pred_label*mask
        gold_label = gold_label*mask

    prec = 0
    rec = 0
    f1 = 0

    gold_span = get_pos_list(gold_label)
    pred_span = get_pos_list(pred_label)
    for batch_index in range(batch_size):
        print(batch_index)
        p,r,f =span_metrics_eval(gold_span[batch_index],pred_span[batch_index])
        prec+=p
        rec+=r
        f1+=f

    prec/=batch_size
    rec /=batch_size
    f1  /=batch_size

    return prec , rec, f1


class Span_eval():
    def __init__(self):
        self.TP = 0
        self.TP_FP = 0
        self.TP_FN = 0

    def span_metrics(self,pred_logits,gold_labels,threshold=0.5,mask=None):
        pred_label = pred_logits.cpu().numpy()
        gold_label = gold_labels.cpu().numpy()

        batch_size = gold_label.shape[0]
        pred_label[pred_label < threshold] = 0
        pred_label[pred_label >= threshold] = 1

        if mask:
            pred_label = pred_label * mask
            gold_label = gold_label * mask

        gold_span = get_pos_list(gold_label)
        pred_span = get_pos_list(pred_label)
        for batch_index in range(batch_size):

            self.metrics_eval(gold_span[batch_index], pred_span[batch_index])


    def metrics_eval(self,gold_span_pos,pred_span_pos):
        cnt = 0
        for span in pred_span_pos:
            if span in gold_span_pos:
                cnt += 1

        self.TP +=cnt
        self.TP_FP += len(pred_span_pos)
        self.TP_FN +=len(gold_span_pos)

    def cal_metrcs(self):
        prec = self.TP/self.TP_FP
        recall = self.TP/self.TP_FN
        if prec ==0 and recall == 0:
            f1 = 1
        else:
            f1   = 2*(prec*recall)/(prec+recall)

        return prec,recall,f1

def span_metrics_eval(gold_span_pos,pred_span_pos):
    cnt = 0

    for span in pred_span_pos:
        if span in gold_span_pos:
            cnt+=1
    if pred_span_pos!=[]:
        prec = cnt/len(pred_span_pos)
    else:
        prec = 1

    cnt = 0

    for span in gold_span_pos:
        if span in pred_span_pos:
            cnt+=1

    if gold_span_pos!=[]:
        recall = cnt / len(gold_span_pos)
    else:
        recall = 1

    if pred_span_pos==[] and gold_span_pos==[]:
        f1 = 1
    elif prec == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2*(prec*recall)/(prec+recall)

    return prec,recall,f1


def get_pos_list(label):

    batch_size = label.shape[0]
    length = label.shape[1]

    batch_pos_list =[]
    for batch_index in range(batch_size):
        span_list = []
        pos_list = []
        for idx in range(length-1):

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
                pos_list.append(idx+1)
                span_list.append(pos_list)
                pos_list = []
                continue

        batch_pos_list.append(span_list)
        #print(batch_pos_list)

    return batch_pos_list


if __name__ == '__main__':
    gold = torch.Tensor([[0,1,1,0,1,1,1],[0,1,0,0,1,1,1]])
    #mask =torch.Tensor([[1,1,1,1,0,0,0],[1,1,1,1,1,1,1]])
    pred = torch.Tensor([[0,1,1,0,0,1,1],[0,1,0,1,0,1,1]])

    #gold =gold*mask
    print(gold)
    print(pred)
    eval = Span_eval()
    eval.span_metrics(pred,gold,0.5)
    print(eval.cal_metrcs())



