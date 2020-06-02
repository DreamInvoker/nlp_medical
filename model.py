# -*- coding: utf-8 -*-
import torch.nn as nn
from transformers import BertModel, BertConfig

from config import PLMConfig


# TODO 减少hard code，将模型配置放在一起


class MedicalExtractionModel(nn.Module):

    def __init__(self, opt):
        super(MedicalExtractionModel, self).__init__()

        # TODO 手动导入不成功，咋回事
        config = BertConfig(PLMConfig.MODEL_PATH + "/config.json")
        self.encoder = BertModel.from_pretrained(PLMConfig.MODEL_PATH)
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

        return subject_logits, body_logits  # , subject_start_logits, subject_end_logits, body_start_logits, body_end_logits
