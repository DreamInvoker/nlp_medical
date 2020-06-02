# -*- coding: utf-8 -*-
import torch.nn as nn
from transformers import BertModel, BertConfig

from config import PLMConfig


class MedicalExtractionModel(nn.Module):

    def __init__(self, opt):
        super(MedicalExtractionModel, self).__init__()

        # TODO 手动导入不成功，咋回事
        config = BertConfig(PLMConfig.MODEL_PATH + "/config.json")
        self.encoder = BertModel.from_pretrained(PLMConfig.MODEL_PATH)
        self.dropout = nn.Dropout(opt.dropout)
        self.subject_fc = nn.Linear(768, 1)
        self.body_fc = nn.Linear(768, 1)

        self.reset_linear_param(self.subject_fc)
        self.reset_linear_param(self.body_fc)

    def reset_linear_param(self, layer):
        nn.init.normal_(layer.weight, std=0.02)
        nn.init.normal_(layer.bias, 0)
        pass

    def forward(self, **params):
        hidden_state, cls = self.encoder(input_ids=params['input_ids'],
                                         attention_mask=params['attention_mask'],
                                         token_type_ids=params['token_type_ids'])
        # TODO 可以使用最后多层来取平均
        x = self.dropout(input=hidden_state)
        subject_logits = self.subject_fc(input=x)
        body_logits = self.body_fc(input=x)

        return subject_logits, body_logits
