"""
Baseline of no DA.
1. train encoder + classifier on src data
2. inference on target data
"""

import torch
import torch.nn as nn
from params.param import *
from core import eval_src, eval_tgt, train_no_da
from models import BERTEncoder, BERTClassifier
from utils import read_data, get_data_loader, init_model, init_random_seed
from pytorch_pretrained_bert import BertTokenizer
import argparse


def arguments():
    # argument parsing
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting")
    parser.add_argument('--src', type=str, default="books", choices=["books", "dvd", "electronics", "kitchen"],
                        help="Specify src dataset")
    parser.add_argument('--tgt', type=str, default="books", choices=["books", "dvd", "electronics", "kitchen"],
                        help="Specify tgt dataset")
    parser.add_argument('--enc_train', default=False, action='store_true',
                        help='Train source encoder')
    parser.add_argument('--seqlen', type=int, default=200,
                        help="Specify maximum sequence length")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="batch size")
    parser.add_argument('--lr', type=float, default=1e-5,
                        help="learning_rate")
    parser.add_argument('--patience', type=int, default=5,
                        help="Specify patience of early stopping for pretrain")
    parser.add_argument('--num_epochs', type=int, default=200,
                        help="Specify the number of epochs for train")
    parser.add_argument('--log_step_pre', type=int, default=1,
                        help="Specify log step size for pretrain")
    parser.add_argument('--eval_step_pre', type=int, default=5,
                        help="Specify eval step size for pretrain")
    parser.add_argument('--save_step_pre', type=int, default=100,
                        help="Specify save step size for pretrain")
    parser.add_argument('--log_step', type=int, default=1,
                        help="Specify log step size for adaptation")
    parser.add_argument('--save_step', type=int, default=100,
                        help="Specify save step size for adaptation")
    parser.add_argument('--model_root', type=str, default='snapshots',
                        help="model_root")
    parser.add_argument('--num_gpus', type=int, default=2,
                        help="num_gpus")
    args = parser.parse_args()
    return args


def get_dataset(args):
    # preprocess data
    print("=== Processing datasets ===")
    src_train = read_data('./data/processed/' + args.src + '/train.txt')
    src_test = read_data('./data/processed/' + args.src + '/test.txt')
    tgt_train = read_data('./data/processed/' + args.tgt + '/train.txt')
    tgt_test = read_data('./data/processed/' + args.tgt + '/test.txt')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    src_train_sequences = []
    src_test_sequences = []
    tgt_train_sequences = []
    tgt_test_sequences = []
    for i in range(len(src_train.review)):  # 1587
        tokenized_text = tokenizer.tokenize(src_train.review[i])
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        src_train_sequences.append(indexed_tokens)
    for i in range(len(src_test.review)):
        tokenized_text = tokenizer.tokenize(src_test.review[i])
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        src_test_sequences.append(indexed_tokens)
    for i in range(len(tgt_train.review)):
        tokenized_text = tokenizer.tokenize(tgt_train.review[i])
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tgt_train_sequences.append(indexed_tokens)
    for i in range(len(tgt_test.review)):
        tokenized_text = tokenizer.tokenize(tgt_test.review[i])
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tgt_test_sequences.append(indexed_tokens)
    # load dataset
    src_train_loader = get_data_loader(src_train_sequences, src_train.label, args)
    src_test_loader = get_data_loader(src_test_sequences, src_test.label, args)
    tgt_train_loader = get_data_loader(tgt_train_sequences, tgt_train.label, args)
    tgt_test_loader = get_data_loader(tgt_test_sequences, tgt_test.label, args)
    return src_train_loader, src_test_loader, tgt_train_loader, tgt_test_loader


def main():
    args = arguments()

    # init random seed
    init_random_seed(manual_seed)

    src_train_loader, src_test_loader, tgt_train_loader, tgt_test_loader = get_dataset(args)

    print("=== Datasets successfully loaded ===")

    # load models
    src_encoder = init_model(BERTEncoder(),
                             restore=src_encoder_restore)
    src_classifier = init_model(BERTClassifier(),
                                restore=src_classifier_restore)

    # if torch.cuda.device_count() > 1:
    #     print('Let\'s use {} GPUs!'.format(torch.cuda.device_count()))
    #     src_encoder = nn.DataParallel(src_encoder)
    #     src_classifier = nn.DataParallel(src_classifier)

    # enable encoder params: we should train encoder
    if not args.enc_train:
        for param in src_encoder.parameters():
            param.requires_grad = True

    # argument setting
    print("=== Argument Setting ===")
    print("src: " + args.src)
    print("tgt: " + args.tgt)
    print("enc_train: " + str(args.enc_train))
    print("seqlen: " + str(args.seqlen))
    print("num_epochs: " + str(args.num_epochs))
    print("batch_size: " + str(args.batch_size))
    print("learning_rate: " + str(args.lr))

    # train source model
    print("=== Training classifier for source domain ===")
    src_encoder, src_classifier = train_no_da(
        args, src_encoder, src_classifier, src_train_loader, src_test_loader)

    # eval source model
    print("=== Evaluating classifier for source domain ===")
    eval_src(src_encoder, src_classifier, src_test_loader)

    # eval target encoder on test set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> domain adaption <<<")
    eval_tgt(src_encoder, src_classifier, tgt_test_loader)


if __name__ == '__main__':
    main()
