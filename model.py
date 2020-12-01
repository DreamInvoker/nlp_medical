# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

from config import PLMConfig


def reset_linear_param(layer):
    nn.init.normal_(layer.weight, std=0.02)
    nn.init.normal_(layer.bias, 0)


class MedicalExtractionModel(nn.Module):

    def __init__(self, opt):
        super(MedicalExtractionModel, self).__init__()

        config = BertConfig.from_pretrained(PLMConfig.MODEL_PATH + '/config.json', output_hidden_states=True)
        self.encoder = BertModel.from_pretrained(PLMConfig.MODEL_PATH, config=config)
        self.dropout = nn.Dropout(opt.dropout)
        self.subject_fc = nn.Linear(768, 1)
        self.decorate_fc = nn.Linear(768, 1)
        self.freq_fc = nn.Linear(768, 1)

        reset_linear_param(self.subject_fc)
        reset_linear_param(self.decorate_fc)
        reset_linear_param(self.freq_fc)

    def forward(self, **params):
        hidden_state, cls, hs = self.encoder(input_ids=params['input_ids'],
                                             attention_mask=params['attention_mask'],
                                             token_type_ids=params['token_type_ids'])
        x = torch.stack([hs[-1], hs[-2], hs[-3], hs[-4]])
        x = torch.mean(x, 0)
        x = self.dropout(x)
        subject_logits = self.subject_fc(input=x)
        decorate_logits = self.decorate_fc(input=x)
        freq_logits = self.freq_fc(input=x)

        return subject_logits, decorate_logits, freq_logits


class MedicalExtractionModelForBody(nn.Module):

    def __init__(self, opt):
        super(MedicalExtractionModelForBody, self).__init__()

        config = BertConfig.from_pretrained(PLMConfig.MODEL_PATH + '/config.json', output_hidden_states=True)
        self.encoder = BertModel.from_pretrained(PLMConfig.MODEL_PATH, config=config)
        self.dropout = nn.Dropout(opt.dropout)
        self.body_fc = nn.Linear(768, 1)

        reset_linear_param(self.body_fc)

    def forward(self, **params):
        hidden_state, cls, hs = self.encoder(input_ids=params['input_ids'],
                                             attention_mask=params['attention_mask'],
                                             token_type_ids=params['token_type_ids'])
        x = torch.stack([hs[-1], hs[-2], hs[-3], hs[-4]])
        x = torch.mean(x, 0)
        x = self.dropout(x)
        body_logits = self.body_fc(input=x)

        return body_logits
