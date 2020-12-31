#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
Bert + CRF

Authors: PanXu
Date:    2020/09/09 14:33:00
"""

import logging
from typing import Union, Dict

import torch
from torch import Tensor
from torch.nn import Dropout, Linear

from transformers import BertModel, BertConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling

from easytext.model import Model, ModelOutputs
from easytext.data import Vocabulary, PretrainedVocabulary, LabelVocabulary
from easytext.modules import ConditionalRandomField
from easytext.utils import bio as BIO
from easytext.component.register import ComponentRegister

from ner.data.vocabulary_builder import VocabularyBuilder
from ner.models.ner_model_outputs import NerModelOutputs

import itertools

def _find_tensors(obj):
    """
    Recursively find all tensors contained in the specified object.
    """
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return itertools.chain(*map(_find_tensors, obj))
    if isinstance(obj, dict):
        return itertools.chain(*map(_find_tensors, obj.values()))
    return []


@ComponentRegister.register(name_space="ner")
class BertWithCrf(Model):
    """
    Bert With Crf
    """

    def __init__(self,
                 bert_dir: str,
                 vocabulary_builder: VocabularyBuilder,
                 dropout: float,
                 is_used_crf: bool):
        """
        初始化
        :param bert_dir: 预训练好的 bert 模型所在 dir
        :param vocabulary_builder: vocabulary builder
        :param dropout: bert 最后一层输出的 dropout
        :param is_used_crf: 是否使用 crf, True: 使用 crf; False: 不使用 crf
        """

        super().__init__()

        self.label_vocabulary = vocabulary_builder.label_vocabulary
        self.dropout = Dropout(dropout)
        self.is_used_crf = is_used_crf
        self.bert = BertModel.from_pretrained(bert_dir)
        bert_config: BertConfig = self.bert.config

        self.classifier = Linear(bert_config.hidden_size, self.label_vocabulary.label_size)

        if self.is_used_crf:
            constraints = BIO.allowed_transitions(label_vocabulary=self.label_vocabulary)
            self.crf = ConditionalRandomField(num_tags=self.label_vocabulary.label_size,
                                              constraints=constraints)
        else:
            self.crf = None

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self,
                input_ids: Tensor,
                attention_mask: Tensor,
                token_type_ids: Tensor,
                sequence_mask: Tensor,
                metadata: Dict) -> NerModelOutputs:
        """
        运行模型
        :param input_ids: bert input ids
        :param attention_mask: bert attention mask
        :param token_type_ids: bert token type ids
        :param sequence_mask: sequence mask, 对于 CLS, SEP, PADDING 的 mask = 0, 其他实际的 token 的 mask = 1
        :param metadata: meta data
        :return: NerModelOutputs
        """

        assert input_ids.dim() == 2, f"input_ids shape: {input_ids.dim()} 与 (batch_size, seq_len) 不匹配"

        bert_output: BaseModelOutputWithPooling = self.bert(input_ids=input_ids,
                                                            attention_mask=attention_mask,
                                                            token_type_ids=token_type_ids,
                                                            position_ids=None,
                                                            return_dict=True)

        sequence_output = self.dropout(bert_output.last_hidden_state)

        logits = self.classifier(sequence_output)

        model_outputs = NerModelOutputs(logits=logits,
                                        mask=sequence_mask,
                                        crf=self.crf)

        return model_outputs
