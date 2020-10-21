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

from model.bert_util import *

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def gather_flat_grad(grads):
    views = []
    for p in grads:
        if p.data.is_sparse:
            view = p.data.to_dense().view(-1)
        else:
            view = p.data.view(-1)
        views.append(view)
    return torch.cat(views, 0)

def hv(loss, model_params, v):
    grad = autograd.grad(loss, model_params, create_graph=True, retain_graph=True)
    Hv = autograd.grad(grad, model_params, grad_outputs=v)
    return Hv

def get_inverse_hvp_lissa(v, model, param_influence, train_lissa_loader, args):
    ihvp = None
    for i in range(args.lissa_repeat):
        cur_estimate = v
        lissa_data_iterator = iter(train_lissa_loader)
        for j in range(args.lissa_depth):
            try:
                tmp_elem = next(lissa_data_iterator)
                input_ids, input_mask, segment_ids, label_ids, guids = tmp_elem
            except StopIteration:
                lissa_data_iterator = iter(train_lissa_loader)
                tmp_elem = next(lissa_data_iterator)
                input_ids, input_mask, segment_ids, label_ids, guids = tmp_elem
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            segment_ids = segment_ids.to(args.device)
            label_ids = label_ids.to(args.device)
            
            model.zero_grad()
            train_loss = model(input_ids, segment_ids, input_mask, label_ids)
            hvp = hv(train_loss, param_influence, cur_estimate)
            cur_estimate = [_a + (1 - args.damping) * _b - _c / args.scale
                            for _a, _b, _c in zip(v, cur_estimate, hvp)]
            
            if (j % args.logging_steps == 0) or (j == args.lissa_depth - 1):
                logger.info(" Recursion at depth %d: norm is %f", j,
                            np.linalg.norm(gather_flat_grad(cur_estimate).cpu().numpy()))
        if ihvp == None:
            ihvp = cur_estimate
        else:
            ihvp = [_a + _b for _a, _b in zip(ihvp, cur_estimate)]
    
    return_ihvp = gather_flat_grad(ihvp)
    return_ihvp /= args.lissa_repeat
    return return_ihvp

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--trained_model_dir",
                        default="",
                        type=str,
                        help="Where is the fine-tuned (with the cloze-style LM objective) BERT model?")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--freeze_bert',
                        action='store_true',
                        help="Whether to freeze BERT")
    parser.add_argument('--full_bert',
                        action='store_true',
                        help="Whether to use full BERT")
    parser.add_argument('--num_train_samples',
                        type=int,
                        default=-1,
                        help="-1 for full train set, otherwise please specify")
    parser.add_argument('--damping',
                        type=float,
                        default=0.0,
                        help="probably need damping for deep models")
    parser.add_argument('--test_idx',
                        type=int,
                        default=1,
                        help="test index we want to examine")
    parser.add_argument('--start_test_idx',
                        type=int,
                        default=-1,
                        help="when not -1, --test_idx will be disabled")
    parser.add_argument('--end_test_idx',
                        type=int,
                        default=-1,
                        help="when not -1, --test_idx will be disabled")
    parser.add_argument("--lissa_repeat",
                        default=1,
                        type=int)
    parser.add_argument("--lissa_depth_pct",
                        default=1.0,
                        type=float)
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument('--scale',
                        type=float,
                        default=1e4,
                        help="probably need scaling for deep models")
    parser.add_argument("--alt_mode",
                        default="",
                        type=str,
                        help="whether to use extended data split (ext) or only control data split (ctr)")
    
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        #raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        logger.info("WARNING: Output directory already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    ma_processor = MAProcessor()
    label_list = ma_processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Prepare model
    model = MyBertForSequenceClassification.from_pretrained(args.trained_model_dir, num_labels=num_labels)
    model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    if args.freeze_bert:
        frozen = ['bert']
    elif args.full_bert:
        frozen = []
    else:
        frozen = ['bert.embeddings.',
                  'bert.encoder.layer.0.',
                  'bert.encoder.layer.1.',
                  'bert.encoder.layer.2.',
                  'bert.encoder.layer.3.',
                  'bert.encoder.layer.4.',
                  'bert.encoder.layer.5.',
                  'bert.encoder.layer.6.',
                  'bert.encoder.layer.7.',
                 ] # *** change here to filter out params we don't want to track ***

    param_influence = []
    for n, p in param_optimizer:
        if (not any(fr in n for fr in frozen)):
            param_influence.append(p)
        elif 'bert.embeddings.word_embeddings.' in n:
            pass # need gradients through embedding layer for computing saliency map
        else:
            p.requires_grad = False
            
    param_shape_tensor = []
    param_size = 0
    for p in param_influence:
        tmp_p = p.clone().detach()
        param_shape_tensor.append(tmp_p)
        param_size += torch.numel(tmp_p)
    logger.info("  Parameter size = %d", param_size)
    
    if args.alt_mode == "dirctr":
        train_examples = ma_processor.get_direct_control_train_examples(args.data_dir)
    else:
        raise ValueError("Check your data alt mode")

    train_features = convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, tokenizer)
    logger.info("***** Train set *****")
    logger.info("  Num examples = %d", len(train_examples))
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_id = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    all_guids = torch.tensor([f.guid for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_id, all_guids)
    train_dataloader = DataLoader(train_data, sampler=SequentialSampler(train_data), batch_size=1)
    
    test_examples = ma_processor.get_adv_examples(args.data_dir)
    
    test_features = convert_examples_to_features(
        test_examples, label_list, args.max_seq_length, tokenizer)
    logger.info("***** Test set *****")
    logger.info("  Num examples = %d", len(test_examples))
    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    all_label_id = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    all_guids = torch.tensor([f.guid for f in test_features], dtype=torch.long)
    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_id, all_guids)
    test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=1)
    
    args.lissa_depth = int(args.lissa_depth_pct * len(train_examples))
    
    test_idx = args.test_idx
    start_test_idx = args.start_test_idx
    end_test_idx = args.end_test_idx
    
    influence_dict = dict()
    ihvp_dict = dict()
    
    for tmp_idx, (input_ids, input_mask, segment_ids, label_ids, guids) in enumerate(test_dataloader):
        if args.start_test_idx != -1 and args.end_test_idx != -1:
            if tmp_idx < args.start_test_idx:
                continue
            if tmp_idx > args.end_test_idx:
                break
        else:
            if tmp_idx < args.test_idx:
                continue
            if tmp_idx > args.test_idx:
                break
                
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        
        influence_dict[tmp_idx] = np.zeros(len(train_examples))
        
        ######## L_TEST GRADIENT ########
        model.eval()
        model.zero_grad()
        test_loss = model(input_ids, segment_ids, input_mask, label_ids)
        test_grads = autograd.grad(test_loss, param_influence)
        ################
        
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        ######## IHVP ########
        train_dataloader_lissa = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
        model.eval() #model.train()
        logger.info("######## START COMPUTING IHVP ########")
        inverse_hvp = get_inverse_hvp_lissa(test_grads, model, param_influence, train_dataloader_lissa, args)
        logger.info("######## FINISHED COMPUTING IHVP ########")
        ################
        
        ihvp_dict[tmp_idx] = inverse_hvp.detach().cpu() # put to CPU to save GPU memory
        
    for tmp_idx in ihvp_dict.keys():
        ihvp_dict[tmp_idx] = ihvp_dict[tmp_idx].to(args.device)
        
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
        
    for train_idx, (_input_ids, _input_mask, _segment_ids, _label_ids, _) in enumerate(tqdm(train_dataloader, desc="Train set index")):
        model.eval() #model.train()
        _input_ids = _input_ids.to(device)
        _input_mask = _input_mask.to(device)
        _segment_ids = _segment_ids.to(device)
        _label_ids = _label_ids.to(device)

        ######## L_TRAIN GRADIENT ########
        model.zero_grad()
        train_loss = model(_input_ids, _segment_ids, _input_mask, _label_ids)
        train_grads = autograd.grad(train_loss, param_influence)
        ################
    
        with torch.no_grad():
            for tmp_idx in ihvp_dict.keys():
                influence_dict[tmp_idx][train_idx] = torch.dot(ihvp_dict[tmp_idx], gather_flat_grad(train_grads)).item()
                
    for k, v in influence_dict.items():
        influence_filename = f"influence_test_idx_{k}.pkl"
        pickle.dump(v, open(os.path.join(args.output_dir, influence_filename), "wb"))

if __name__ == "__main__":
    main()
