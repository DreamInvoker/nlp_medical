from sklearn import metrics


def Metrics(pred_logits, gold_labels, threshold):
    pred_label = pred_logits.cpu().numpy()
    gold_label = gold_labels.cpu().numpy()
    batch_size = gold_label.shape[0]
    pred_label[pred_label < threshold] = 0
    pred_label[pred_label >= threshold] = 1
    acc = 0
    rec = 0
    f1 = 0
    jac = 0
    for gold, pred in zip(gold_label, pred_label):
        acc += metrics.precision_score(gold, pred, labels=[1], average='micro', zero_division=1)
        rec += metrics.recall_score(gold, pred, labels=[1], average='micro', zero_division=1)
        f1 += metrics.f1_score(gold, pred, labels=[1], average='micro', zero_division=1)
        jac += metrics.jaccard_score(gold, pred, labels=[1], average='micro')

    acc /= batch_size
    rec /= batch_size
    f1 /= batch_size
    jac /= batch_size
    return acc, rec, f1, jac


if __name__ == '__main__':
    print(metrics.jaccard_score([0, 0, 0, 0, 1, 1, 1], [1, 1, 0, 0, 1, 0, 1], labels=[1], average='micro'))
    print(metrics.precision_score([0, 0, 0, 0, 1, 1, 1], [1, 1, 0, 0, 1, 0, 1], labels=[1], average='micro'))
    print(metrics.recall_score([0, 0, 0, 0, 1, 1, 1], [1, 1, 0, 0, 1, 0, 1], labels=[1], average='micro'))
    print(metrics.f1_score([0, 0, 0, 0, 1, 1, 1], [1, 1, 0, 0, 1, 0, 1], labels=[1], average='micro'))
