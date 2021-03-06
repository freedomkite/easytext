#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
测试 vocabulary collate

Authors: PanXu
Date:    2020/06/27 00:06:00
"""
import torch
from torch.utils.data import DataLoader

from easytext.data import Vocabulary, LabelVocabulary, PretrainedVocabulary

from ner.tests import ASSERT

from ner.data import VocabularyCollate


def test_vocabuary_collate(vocabulary):
    """
    测试 vocabualry collate
    :param conll2003_dataset:
    :return: None
    """

    word_vocabulary = vocabulary["token_vocabulary"]
    ASSERT.assertEqual(13 + 2, word_vocabulary.size)

    label_vocabulary = vocabulary["label_vocabulary"]
    ASSERT.assertEqual(3, label_vocabulary.label_size)

