from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import pickle
import time
import math

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

import torch.autograd as autograd
from scipy import stats

class MyBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(MyBertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
        
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, note=""):
        """Constructs a InputExample.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.note = note

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, guid):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.guid = guid
        

class MAProcessor(object):
    
    def get_direct_control_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "micro_train.pkl")), "missed_micro") + self._create_examples(self._read_tsv(os.path.join(data_dir, "nonmicro_large_train.pkl")), "clean") + self._create_examples(self._read_tsv(os.path.join(data_dir, "hs_train.pkl")), "hateful")
    
    def get_dirctr_corrected_train_examples(self, data_dir, correction_dir, correction_size):
        """See base class."""
        sorted_ex_for_correction = self._read_tsv(os.path.join(correction_dir, "sorted_ex_for_correction.pkl"))
        correct_idx_list = sorted_ex_for_correction[:correction_size]
        examples = self._create_examples(self._read_tsv(os.path.join(data_dir, "micro_train.pkl")), "missed_micro") + self._create_examples(self._read_tsv(os.path.join(data_dir, "nonmicro_large_train.pkl")), "clean") + self._create_examples(self._read_tsv(os.path.join(data_dir, "hs_train.pkl")), "hateful")
        new_examples = []
        for i, example in enumerate(examples):
            if i in correct_idx_list:
                new_examples.append(InputExample(guid=i, text_a=example.text_a, text_b=None, label="1")) # flip the label
            else:
                new_examples.append(InputExample(guid=i, text_a=example.text_a, text_b=None, label=example.label))
        return new_examples
    
    def get_dirctr_checked_train_examples(self, data_dir, correction_dir, correction_size):
        """See base class."""
        sorted_ex_for_correction = self._read_tsv(os.path.join(correction_dir, "sorted_ex_for_correction.pkl"))
        correct_idx_list = sorted_ex_for_correction[:correction_size]
        len_micro_train = len(self._read_tsv(os.path.join(data_dir, "micro_train.pkl")))
        examples = self._create_examples(self._read_tsv(os.path.join(data_dir, "micro_train.pkl")), "missed_micro") + self._create_examples(self._read_tsv(os.path.join(data_dir, "nonmicro_large_train.pkl")), "clean") + self._create_examples(self._read_tsv(os.path.join(data_dir, "hs_train.pkl")), "hateful")
        new_examples = []
        for i, example in enumerate(examples):
            if i in correct_idx_list and i < len_micro_train: # only flip the true microaggressions
                new_examples.append(InputExample(guid=i, text_a=example.text_a, text_b=None, label="1")) # flip the label
            else:
                new_examples.append(InputExample(guid=i, text_a=example.text_a, text_b=None, label=example.label))
        return new_examples
    
    def get_dirctr_gold_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "micro_train.pkl")), "gold_micro") + self._create_examples(self._read_tsv(os.path.join(data_dir, "nonmicro_large_train.pkl")), "clean") + self._create_examples(self._read_tsv(os.path.join(data_dir, "hs_train.pkl")), "hateful")
    
    def get_direct_control_test_examples(self, data_dir):
        """See base class."""
        micro_test_ex = self._create_examples(self._read_tsv(os.path.join(data_dir, "micro_test.pkl")), "gold_micro")
        clean_test_ex = self._create_examples(self._read_tsv(os.path.join(data_dir, "nonmicro_large_test.pkl")), "clean")
        hs_test_ex = self._create_examples(self._read_tsv(os.path.join(data_dir, "hs_test.pkl")), "hateful")
        return (micro_test_ex, clean_test_ex, hs_test_ex)
    
    def get_adv_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "micro_adv.pkl")), "missed_micro")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = i
            text_a = line[0]
            if set_type == "hateful":
                label = "1"
            elif set_type == "missed_micro":
                label = "0"
            elif set_type == "gold_micro":
                label = "1"
            elif set_type == "clean":
                label = "0"
            else:
                raise ValueError("Check your set type")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
    
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "rb") as f:
            pairs = pickle.load(f)
            return pairs

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              guid=example.guid))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, label_ids):
    # axis-0: seqs in batch; axis-1: potential labels of seq
    outputs = np.argmax(out, axis=1)
    matched = outputs == label_ids
    num_correct = np.sum(matched)
    num_total = len(label_ids)
    return num_correct, num_total
