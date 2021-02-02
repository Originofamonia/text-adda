"""
Inference trained encoder (classifier);
save encoded features from encoder for visualization.
"""

import torch
from params.param import *
from core import eval_src, eval_tgt, train_src, train_tgt, eval_tgt_save_features
from models import BERTEncoder, BERTClassifier, Discriminator
from utils import read_data, get_data_loader, init_model, init_random_seed
from pytorch_pretrained_bert import BertTokenizer
import argparse


def main():
    # argument parsing
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting")
    parser.add_argument('--src', type=str, default="books", choices=["books", "dvd", "electronics", "kitchen"],
                        help="Specify src dataset")
    parser.add_argument('--tgt', type=str, default="dvd", choices=["books", "dvd", "electronics", "kitchen"],
                        help="Specify tgt dataset")
    parser.add_argument('--enc_train', default=False, action='store_true',
                        help='Train source encoder')
    parser.add_argument('--seqlen', type=int, default=200,
                        help="Specify maximum sequence length")
    parser.add_argument('--patience', type=int, default=5,
                        help="Specify patience of early stopping for pretrain")
    parser.add_argument('--num_epochs_pre', type=int, default=200,
                        help="Specify the number of epochs for pretrain")
    parser.add_argument('--log_step_pre', type=int, default=1,
                        help="Specify log step size for pretrain")
    parser.add_argument('--eval_step_pre', type=int, default=10,
                        help="Specify eval step size for pretrain")
    parser.add_argument('--save_step_pre', type=int, default=100,
                        help="Specify save step size for pretrain")
    parser.add_argument('--num_epochs', type=int, default=100,
                        help="Specify the number of epochs for adaptation")
    parser.add_argument('--log_step', type=int, default=1,
                        help="Specify log step size for adaptation")
    parser.add_argument('--save_step', type=int, default=100,
                        help="Specify save step size for adaptation")
    parser.add_argument('--model_root', type=str, default='snapshots',
                        help="model_root")
    args = parser.parse_args()

    # argument setting
    print("=== Argument Setting ===")
    print("src: " + args.src)
    print("tgt: " + args.tgt)
    print("enc_train: " + str(args.enc_train))
    print("seqlen: " + str(args.seqlen))
    print("patience: " + str(args.patience))
    print("num_epochs_pre: " + str(args.num_epochs_pre))
    print("log_step_pre: " + str(args.log_step_pre))
    print("eval_step_pre: " + str(args.eval_step_pre))
    print("save_step_pre: " + str(args.save_step_pre))
    print("num_epochs: " + str(args.num_epochs))
    print("log_step: " + str(args.log_step))
    print("save_step: " + str(args.save_step))

    # init random seed
    init_random_seed(manual_seed)

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
    src_data_loader = get_data_loader(src_train_sequences, src_train.label, args.seqlen)
    src_data_loader_eval = get_data_loader(src_test_sequences, src_test.label, args.seqlen)
    tgt_data_loader = get_data_loader(tgt_train_sequences, tgt_train.label, args.seqlen)
    tgt_data_loader_eval = get_data_loader(tgt_test_sequences, tgt_test.label, args.seqlen)

    print("=== Datasets successfully loaded ===")

    # load models
    src_encoder_file = "snapshots/src-encoder.pt"
    src_classifier_file = "snapshots/src-classifier.pt"
    src_encoder = init_model(BERTEncoder(),
                             restore=src_encoder_file)
    # src_classifier = init_model(BERTClassifier(),
    #                             restore=src_classifier_file)
    # tgt_encoder = init_model(BERTEncoder(),
    #                          restore=tgt_encoder_restore)
    # critic = init_model(Discriminator(),
    #                     restore=d_model_restore)

    # freeze encoder params
    # if not args.enc_train:
    #     for param in src_encoder.parameters():
    #         param.requires_grad = False

    # eval target encoder on test set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    feature_name = 'snapshots/tgt_dvd_features'
    eval_tgt_save_features(src_encoder, tgt_data_loader_eval, feature_name)
    # print(">>> domain adaption <<<")
    # eval_tgt_save_features(tgt_encoder, src_classifier, tgt_data_loader_eval)


if __name__ == '__main__':
    main()
