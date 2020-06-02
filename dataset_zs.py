import json

import torch
from torch.utils.data import Dataset

from config import PLMConfig
from preprocess import text_process, process_sym_attr, get_entity
from utils import get_cuda


class MedicalExtractionDatasetForSubjectAndBody(Dataset):

    def __init__(self, data_path, max_len=300, left_side_span=30, right_side_span=5):
        super(MedicalExtractionDatasetForSubjectAndBody, self).__init__()
        self.data_path = data_path
        self.raw_data = None
        with open(file=data_path, mode='r', encoding='utf-8') as fr:
            self.raw_data = json.load(fr)
        self.data = []
        for idx in range(len(self.raw_data)):
            self.pre_process(idx)
        print(f'{data_path} has {len(self.data)} symptoms.')
        # TODO TypeError: can't pickle Tokenizer objects
        # self.tokenizer = PLMConfig.tokenizer
        self.max_len = max_len
        self.left_side_span = left_side_span
        self.right_side_span = right_side_span

    def __getitem__(self, item):
        example = self.data[item]
        raw_text = example['raw_text']
        symptom_name = example['symptom_name']
        attr_dict = example['attr_dict']
        symptom_pos = example['symptom_pos']

        subject = attr_dict['subject']
        body = attr_dict['body']
        span_bound = max(0, symptom_pos[0] - self.left_side_span), min(symptom_pos[1] + self.right_side_span,
                                                                       len(raw_text))
        text_span = raw_text[span_bound[0]:span_bound[1] + 1]
        text_token = PLMConfig.tokenizer.encode(text_span)
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

        symptom_tokens = PLMConfig.tokenizer.encode(symptom_name)
        symptom_ids = symptom_tokens.ids[1:-1]

        input_ids = [101] + symptom_ids + [102] + text_ids + [102]
        token_type_ids = [0] * (len(symptom_ids) + 2) + [1] * (len(text_ids) + 1)
        mask = [1] * len(token_type_ids)
        text_offsets = [(0, 0)] * (len(symptom_ids) + 2) + text_offsets + [(0, 0)]
        subject_target_ids = [0] * (len(symptom_ids) + 2) + subject_target_ids + [0]
        body_target_ids = [0] * (len(symptom_ids) + 2) + body_target_ids + [0]

        padding_length = self.max_len - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([0] * padding_length)
            mask = mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
            text_offsets = text_offsets + ([(0, 0)] * padding_length)
            subject_target_ids = subject_target_ids + ([0] * padding_length)
            body_target_ids = body_target_ids + ([0] * padding_length)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        mask = torch.tensor(mask, dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        subject_target_ids = torch.tensor(subject_target_ids, dtype=torch.float).unsqueeze(-1)
        body_target_ids = torch.tensor(body_target_ids, dtype=torch.float).unsqueeze(-1)

        return {
            'input_ids': get_cuda(input_ids),
            'mask': get_cuda(mask),
            'token_type_ids': get_cuda(token_type_ids),
            'subject_target_ids': get_cuda(subject_target_ids),
            'body_target_ids': get_cuda(body_target_ids),
            'text_offsets': text_offsets,
            'raw_text': raw_text,
            'symptom_name': symptom_name
        }

    def __len__(self):
        return len(self.data)

    def pre_process(self, index):
        item = self.raw_data[index]
        text = item['text']
        symptoms = item['symptom']
        assert isinstance(symptoms, dict), 'dataset error: {}, index: {}'.format(self.data_path, index)
        raw_text = text_process(text)
        # entity_dict = get_entity(text_contains_entities=text)
        for symtom_name, attribute_dict in symptoms.items():
            if attribute_dict['has_problem']:
                continue
            symptom_name = None
            symptom_pos = attribute_dict['self']['pos']
            if len(symptom_pos) == 0:
                continue
            try:
                a = symptom_pos[0]
                b = symptom_pos[1]
            except:
                continue
            try:
                symptom_name = symtom_name.split(': ')[1]
            except:
                symptom_name = symtom_name.split(':')[1]

            attr_processed_dict = process_sym_attr(raw_text, attribute_dict)
            self.data.append({
                'raw_text': raw_text,
                'symptom_name': symptom_name,
                'attr_dict': attr_processed_dict,
                'symptom_pos': symptom_pos
            })
