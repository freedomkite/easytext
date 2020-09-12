#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
ner loss

Authors: PanXu
Date:    2020/06/27 19:49:00
"""
import torch

from torch.nn import CrossEntropyLoss

from easytext.loss import Loss
from easytext.model import ModelOutputs
from easytext.modules import ConditionalRandomField
from easytext.data import LabelVocabulary

from ner.models import NerModelOutputs


class NerCRFLoss(Loss):
    """
    Ner CRF Loss
    """

    def __init__(self, is_used_crf: bool, label_vocabulary: LabelVocabulary):
        """
        loss 初始化
        :param is_used_crf: 是否使用了crf, True: 使用了; False: 没有使用
        :param label_vocabulary: label vocabulary
        """
        super().__init__()
        self.label_vocabulary = label_vocabulary
        self.is_used_crf = is_used_crf

        if not self.is_used_crf:
            self.loss = CrossEntropyLoss(ignore_index=self.label_vocabulary.padding_index)

    def __call__(self, model_outputs: ModelOutputs, golden_label: torch.Tensor) -> torch.Tensor:
        model_outputs: NerModelOutputs = model_outputs

        # shape: (batch_size, seq_len, label_size)
        logits = model_outputs.logits
        assert model_outputs.logits.dim() == 3, \
            f"model_outputs.logits.dim() != 3, 应该是 (batch_size, seq_len, label_size)"

        # shape: (batch_size, seq_len)
        mask = model_outputs.mask
        assert mask.dim() == 2, f"mask.dim() != 2, 应该是 (batch_size, seq_len)"

        if self.is_used_crf:
            crf: ConditionalRandomField = model_outputs.crf
            assert crf is not None, f"is_used_crf: {self.is_used_crf}, 但是 model_outputs.crf is None"
            return -crf(inputs=logits,
                        tags=golden_label,
                        mask=mask)

        else:
            # 将 logits 转换成二维
            logits = logits.view(-1, self.label_vocabulary.label_size)
            golden_label = golden_label.view(-1)
            return self.loss(logits, golden_label)
