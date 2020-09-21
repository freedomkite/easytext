#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
ner 初始化

Authors: PanXu
Date:    2020/06/09 00:41:00
"""
import os
import json
from typing import List, Dict

import torch

from transformers import BertTokenizer

from easytext.data import LabelVocabulary
from easytext.data.tokenizer import ZhTokenizer
from easytext.data import Instance
from easytext.model import ModelInputs
from easytext.utils.nn import cuda_util

from ner import ROOT_PATH
from ner.models import NerV4, NerModelOutputs
from ner.data import BertModelCollate
from ner.data.dataset.military_dataset import MilitaryDataset
from ner.label_decoder import NerMaxModelLabelDecoder
from ner.config import NerConfigFactory


class NerV4Predictor:

    def __init__(self, config: Dict, cuda_device: str):
        self.config = config
        self.cuda_device = cuda_device
        
        self._model = None
        self._model_collate = None
        self._tokenizer = None
    
    def load(self):

        if self._model is None:
            model_config = self.config["model"]

            bert_dir = model_config["bert_dir"]
            dropout = model_config["dropout"]
            is_used_crf = model_config["is_used_crf"]
            label_vocabulary_dir = "data/models/military_ner_v4/vocabulary/label_vocabulary"
            label_vocabulary_dir = os.path.join(ROOT_PATH, label_vocabulary_dir)
            label_vocabulary = LabelVocabulary.from_file(label_vocabulary_dir)
            self._model = NerV4(bert_dir=bert_dir, 
                           label_vocabulary=label_vocabulary, 
                           dropout=dropout, 
                           is_used_crf=is_used_crf)
            model_file_path = "data/models/military_ner_v4/model.pt"
            model_file_path = os.path.join(ROOT_PATH, model_file_path)
            self._model.load_state_dict(torch.load(model_file_path))

            if self.cuda_device is not None and self.cuda_device != "":
                self._model = self._model.cuda(torch.device(self.cuda_device))

            self._model.eval()

            bert_tokenizer = BertTokenizer.from_pretrained(bert_dir)
            self._model_collate = BertModelCollate(tokenizer=bert_tokenizer,
                                             sequence_label_vocab=label_vocabulary,
                                             sequence_max_len=300)
            self._tokenizer = ZhTokenizer()
            self._label_decoder = NerMaxModelLabelDecoder(label_vocabulary=label_vocabulary)
        
    def __call__(self, texts: List[str]):
        
        instances = list()

        for text in texts:
            instance = Instance()
            instance["tokens"] = self._tokenizer.tokenize(text)
            instance["metadata"] = {"text": text}

            instances.append(instance)

        with torch.no_grad():
            model_inputs = self._model_collate(instances)
            model_inputs: ModelInputs = model_inputs

            batch_size, batch_inputs, _ = model_inputs.batch_size, \
                                                model_inputs.model_inputs, \
                                                model_inputs.labels
            if self.cuda_device is not None and self.cuda_device != "":
                batch_inputs = cuda_util.cuda(batch_inputs, 
                                              cuda_device=torch.device(self.cuda_device))

            model_output: NerModelOutputs = self._model(**batch_inputs)

            spans = self._label_decoder(model_outputs=model_output)

            return spans

if __name__ == "__main__":
    texts = ["2010年06月11日09:53,四川在线-华西都市报美国总统奥巴马9日在白宫与来访的巴勒斯坦民族权力机构主席阿巴斯举行会谈。", 
             "据透露，上星期沙龙访问罗马时，曾与美国特使艾布拉姆斯举行了秘密会谈。",
             "首先，日中在政治上一改以往的半敌对状态，安倍在与李克强会谈后举行的联合记者会中表示：日中关系已从竞争进入了协调的新阶段。"]
    
    config = NerConfigFactory(model_name=NerV4.NAME, 
                              dataset_name=MilitaryDataset.NAME, 
                              debug=False).create()

    predictor = NerV4Predictor(config=config, cuda_device="cuda:1")
    predictor.load()

    spans = predictor(texts)

    for text, span in zip(texts, spans):

        for sp in span:
            sp["text"] = text[sp["begin"]: sp["end"]]
        print(f"{text}:\n {json.dumps(spans, ensure_ascii=False)}")