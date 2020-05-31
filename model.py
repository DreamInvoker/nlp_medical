# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertTokenizer
import tokenizers

# TODO 减少hard code，将模型配置放在一起
# class ModelConfig:
#     MAX_LEN = 128
#     TRAIN_BATCH_SIZE = 64
#     VALID_BATCH_SIZE = 16
#     EPOCHS = 5
#     BERT_PATH = "../input/bert-base-uncased/"
#     MODEL_PATH = "model.bin"
#     TRAINING_FILE = "../input/tweet-train-folds/train_folds.csv"
#     TOKENIZER = tokenizers.BertWordPieceTokenizer(
#         f"{BERT_PATH}/vocab.txt",
#         lowercase=True
#     )

class MedicalExtractionModel(nn.Module):

    def __init__(self, opt):
        super(MedicalExtractionModel, self).__init__()

        config = BertConfig('./PLM/bert-base-chinese')
        self.encoder = BertModel.from_pretrained('./PLM/bert-base-chinese')
        self.body_fc = nn.Linear(768, 1)
        self.dropout = nn.Dropout(0.5)
        self.subject_fc = nn.Linear(768, 1)

    def reset_param(self, layer):
        # nn.init.uniform(layer['weight'], 0.02)
        # nn.init.xavier_normal(layer['bias'], 0)
        pass

    def forward(self, **params):
        hidden_state, cls = self.encoder(input_ids=params['input_ids'],
                                         attention_mask=params['attention_mask'],
                                         token_type_ids=params['token_type_ids'])
        x = self.dropout(input=hidden_state)
        subject_logits = self.subject_fc(input=x)
        body_logits = self.body_fc(input=x)

        # subject_start_logits, subject_end_logits = subject_span.split(split_size=1, dim=-1)
        # subject_start_logits = subject_start_logits.squeeze(-1)
        # subject_end_logits = subject_start_logits.squeeze(-1)
        #
        # body_start_logits, body_end_logits = body_span.split(split_size=1, dim=-1)
        # body_start_logits = body_start_logits.squeeze(-1)
        # body_end_logits = body_start_logits.squeeze(-1)

        return subject_logits, body_logits#, subject_start_logits, subject_end_logits, body_start_logits, body_end_logits


if __name__ == '__main__':

    text = '[0幽门肥厚性狭窄6]dis （ [8hypertrophicpyloricstenosis34]dis ， [36HPS38]dis ）是因 [42[幽门环状括约肌]bod 增厚50]sym ' \
           '[51[幽门管]bod 延长55]sym [56正常结构消失61]sym ，导致 [65[胃出口]bod 部梗阻70]sym [71[胃]bod 代偿性扩张 、 ' \
           '肥厚和蠕动加快84]sym [85[幽门平滑肌细胞]bod 肥大 ， 而非增生98]sym [99幽门肥厚性狭窄105]dis 的临床及病理表现，' \
           '认为该病是一种先天性疾病。 胃空肠吻合术是当时经典的治疗方法，死亡率高达60%。 黏膜外幽门成形术是另一种术式，但因为缝合易撕裂水肿的 [180肌肉181]bod ，导致 [185大出血187]dis ，疗效也不理想。自从1911年Ramstedt首次放弃 缝合肌肉后， 幽门环肌切开术成为标准术式。'
    print(text)
    print('-'*20)
    text.encode('utf-8')
    filtered_text = ''
    idx = 0
    true_idx = 0
    while idx < len(text):
        word = text[idx]
        temp = true_idx - 1
        digits = 1
        if temp == -1:
            digits = 1
        else:
            while temp // 10 != 0:
                digits += 1
                temp //= 10
        temp = true_idx
        digits_left = 1
        if temp == -1:
            digits_left = 1
        else:
            while temp // 10 != 0:
                digits_left += 1
                temp //= 10

        if word == ' ' and idx < len(text) - 1 and text[idx + 1] == '[':
            idx += 2
            idx += digits
            continue
        if word == '[' and idx < len(text) - digits_left and text[idx + 1:idx+1+digits_left] == str(true_idx):
            idx += 1
            idx += digits_left
            continue
        if word == '[':
            idx += 1
            continue
        if word == ' ':
            idx += 1
            continue
        if word == ']' and idx < len(text) - 5 and text[idx:idx+5] in [']bod ', ']dis ', ']sym ', ']item ']:
            idx += 5
            continue
        if idx < len(text) - digits and text[idx:idx+digits] == str(true_idx - 1) and text[idx + digits] == ']':
            idx += digits + 5
            continue

        filtered_text += word
        true_idx += 1
        idx += 1
    # filtered_text = filtered_text.replace(' ', '')
    print(filtered_text)

    symptom = 'No.0: 幽门环状括约肌增厚'
    symptom_name = symptom.split(': ')[1]
    print('-'*20)
    print('symptom name:', symptom_name)
    attr = {
                "has_problem": False,
                "subject": {
                     "val": "增厚",
                     "pos": [
                          49,
                          50
                     ]
                },
                "self": {
                     "val": "幽门环状括约肌增厚",
                     "pos": [
                          42,
                          50
                     ]
                },
                "body": {
                     "val": "幽门环状括约肌",
                     "pos": [
                          42,
                          48
                     ]
                },
                "decorate": {
                     "val": "",
                     "pos": []
                },
                "frequency": {
                     "val": "",
                     "pos": []
                },
                "item": {
                     "val": "",
                     "pos": []
                },
                "disease": {
                     "val": "幽门肥厚性狭窄 hypertrophicpyloricstenosis HPS",
                     "pos": [
                          0,
                          6,
                          8,
                          34,
                          36,
                          38
                     ]
                }
            }
    attr_dict = {}
    if not attr['has_problem']:

        for k in attr:
            success = True
            if k == 'has_problem':
                continue
            k_list = [0] * len(filtered_text)
            pos = attr[k]['pos']
            val = str(attr[k]['val']).split(' ')
            for idx, v in enumerate(val):
                start, end = None, None
                if v == '' and len(val) == 1:
                    pass
                elif v == filtered_text[pos[idx * 2]:pos[idx * 2 + 1] + 1]:
                    start, end = pos[idx * 2], pos[idx * 2 + 1] + 1
                elif val == filtered_text[pos[idx * 2]:pos[idx * 2] + len(val)]:
                    start, end = pos[idx * 2], pos[idx * 2] + len(val)
                elif val == filtered_text[pos[idx * 2 + 1] - len(val) + 1:pos[idx * 2 + 1] + 1]:
                    start, end = pos[idx * 2 + 1] - len(val) + 1, pos[idx * 2 + 1] + 1
                else:
                    success = False

                if start is not None and end is not None:
                    for index in range(start, end):
                        # if index == start:
                        #     k_list[index] = 'B'
                        # elif index == end - 1:
                        #     k_list[index] = 'E'
                        # else:
                        #     k_list[index] = 'I'
                        k_list[index] = 1
                    # if end - start <= 1:
                    #     k_list[start] = 'S'
            attr_dict[k] = k_list
            print('{}:{}'.format(k, success))

    # label = {
    #     'B': 0,
    #     'I': 1,
    #     'E': 2,
    #     'O': 3,
    #     'S': 4
    # }
    tokenizer = tokenizers.BertWordPieceTokenizer(
        vocab_file='./PLM/bert-base-chinese/vocab.txt'
    )

    subject = attr_dict['subject']
    body = attr_dict['body']
    # subject = [label[sub] for sub in subject]
    # body = [label[bod] for bod in body]

    symptom_pos = attr['self']['pos']
    span_bound = max(0, symptom_pos[0] - 20), min(symptom_pos[1] + 20, len(filtered_text))
    text_span = filtered_text[span_bound[0]:span_bound[1] + 1]
    text_token = tokenizer.encode(text_span)
    text_ids = text_token.ids[1:-1]
    text_offsets = text_token.offsets[1:-1]
    subject_target_span = subject[span_bound[0]:span_bound[1] + 1]
    body_target_span = body[span_bound[0]:span_bound[1] + 1]

    subject_target_ids = []
    body_target_ids = []
    for idx, offset in enumerate(text_offsets):
        subject_total = sum(subject_target_span[offset[0]:offset[1]])
        if subject_total > 0:
            subject_target_ids.append(1)
        else:
            subject_target_ids.append(0)

        body_total = sum(body_target_span[offset[0]:offset[1]])
        if body_total > 0:
            body_target_ids.append(1)
        else:
            body_target_ids.append(0)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    # sym_tokens = tokenizer.encode(symptom_name)

    symptom_tokens = tokenizer.encode(symptom_name)
    symptom_ids = symptom_tokens.ids[1:-1]

    input_ids = [101] + symptom_ids + [102] + text_ids + [102]
    token_type_ids = [0] * (len(symptom_ids) + 2) + [1] * (len(text_ids) + 1)
    mask = [1] * len(token_type_ids)
    tweet_offsets = [(0, 0)] * (len(symptom_ids) + 2) + text_offsets + [(0, 0)]
    subject_target_ids = [0] * (len(symptom_ids) + 2) + subject_target_ids + [0]
    body_target_ids = [0] * (len(symptom_ids) + 2) + body_target_ids + [0]

    padding_length = 60 - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
        subject_target_ids = subject_target_ids + ([0] * padding_length)
        body_target_ids = body_target_ids + ([0] * padding_length)
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)
    subject_target_ids = torch.tensor(subject_target_ids, dtype=torch.float).unsqueeze(0).unsqueeze(-1)
    body_target_ids = torch.tensor(body_target_ids, dtype=torch.float).unsqueeze(0).unsqueeze(-1)
    # input_ids = torch.tensor(input_ids, dtype=torch.long)
    criterion = nn.BCEWithLogitsLoss()
    pass
    model = MedicalExtractionModel(None)

    subject_logits, body_logits = model(
        input_ids=input_ids,
        attention_mask=mask,
        token_type_ids=token_type_ids
    )

    loss = criterion(subject_logits, subject_target_ids) + criterion(body_logits, body_target_ids)
    print(loss.item())

