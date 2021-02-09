"""Main script for ADDA."""

import torch
import torch.nn as nn
from params.param import *
from core import eval_src, eval_tgt, train_src, train_tgt
from models import BERTEncoder, BERTClassifier, Discriminator
from utils import read_data, get_data_loader, init_model, init_random_seed
from pytorch_pretrained_bert import BertTokenizer
import argparse


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
    src_data_loader = get_data_loader(src_train_sequences, src_train.label, args)
    src_data_loader_eval = get_data_loader(src_test_sequences, src_test.label, args)
    tgt_data_loader = get_data_loader(tgt_train_sequences, tgt_train.label, args)
    tgt_data_loader_eval = get_data_loader(tgt_test_sequences, tgt_test.label, args)
    return src_data_loader, src_data_loader_eval, tgt_data_loader, tgt_data_loader_eval


def get_arguments():
    # argument parsing
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting")
    parser.add_argument('--src', type=str, default="books", choices=["books", "dvd", "electronics", "kitchen"],
                        help="Specify src dataset")
    parser.add_argument('--tgt', type=str, default="dvd", choices=["books", "dvd", "electronics", "kitchen"],
                        help="Specify tgt dataset")
    parser.add_argument('--enc_train', default=False, action='store_true', help='Train source encoder')
    parser.add_argument('--seqlen', type=int, default=200, help="Specify maximum sequence length")
    parser.add_argument('--patience', type=int, default=5,
                        help="Specify patience of early stopping for pretrain")
    parser.add_argument('--num_epochs_pre', type=int, default=7,
                        help="Specify the number of epochs for pretrain")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size")
    parser.add_argument('--lr', type=float, default=1e-5,help="src encoder lr")
    parser.add_argument('--t_lr', type=float, default=1e-5, help="tgt encoder lr")
    parser.add_argument('--c_lr', type=float, default=1e-5, help="critic lr")
    parser.add_argument('--beta1', type=float, default=0.5, help="beta1")
    parser.add_argument('--beta2', type=float, default=0.99, help="beta2")
    parser.add_argument('--log_step_pre', type=int, default=1,
                        help="Specify log step size for pretrain")
    parser.add_argument('--eval_step_pre', type=int, default=10,
                        help="Specify eval step size for pretrain")
    parser.add_argument('--save_step_pre', type=int, default=100,
                        help="Specify save step size for pretrain")
    parser.add_argument('--num_epochs', type=int, default=11,
                        help="Specify the number of epochs for adaptation")
    parser.add_argument('--log_step', type=int, default=1,
                        help="Specify log step size for adaptation")
    parser.add_argument('--save_step', type=int, default=100,
                        help="Specify save step size for adaptation")
    parser.add_argument('--model_root', type=str, default='snapshots', help="model_root")
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()

    # init random seed
    init_random_seed(manual_seed)

    src_data_loader, src_data_loader_eval, tgt_data_loader, tgt_data_loader_eval = get_dataset(args)

    # argument setting
    print("=== Argument Setting ===")
    print("src: " + args.src)
    print("tgt: " + args.tgt)
    print("patience: " + str(args.patience))
    print("num_epochs_pre: " + str(args.num_epochs_pre))
    print("eval_step_pre: " + str(args.eval_step_pre))
    print("save_step_pre: " + str(args.save_step_pre))
    print("num_epochs: " + str(args.num_epochs))
    print("src encoder lr: " + str(args.lr))
    print("tgt encoder lr: " + str(args.t_lr))
    print("critic lr: " + str(args.c_lr))
    print("batch_size: " + str(args.batch_size))

    # load models
    src_encoder_restore = "snapshots/src-encoder-adda-{}.pt".format(args.src)
    src_classifier_restore = "snapshots/src-classifier-adda-{}.pt".format(args.src)
    tgt_encoder_restore = "snapshots/tgt-encoder-adda-{}.pt".format(args.src)
    d_model_restore = "snapshots/critic-adda-{}.pt".format(args.src)
    src_encoder = init_model(BERTEncoder(),
                             restore=src_encoder_restore)
    src_classifier = init_model(BERTClassifier(),
                                restore=src_classifier_restore)
    tgt_encoder = init_model(BERTEncoder(),
                             restore=tgt_encoder_restore)
    critic = init_model(Discriminator(),
                        restore=d_model_restore)

    # no, fine-tune BERT
    # if not args.enc_train:
    #     for param in src_encoder.parameters():
    #         param.requires_grad = False

    if torch.cuda.device_count() > 1:
        print('Let\'s use {} GPUs!'.format(torch.cuda.device_count()))
        src_encoder = nn.DataParallel(src_encoder)
        src_classifier = nn.DataParallel(src_classifier)
        tgt_encoder = nn.DataParallel(tgt_encoder)
        critic = nn.DataParallel(critic)

    # train source model
    print("=== Training classifier for source domain ===")
    src_encoder, src_classifier = train_src(
        args, src_encoder, src_classifier, src_data_loader, src_data_loader_eval)

    # eval source model
    print("=== Evaluating classifier for source domain ===")
    eval_src(src_encoder, src_classifier, src_data_loader_eval)

    # train target encoder by GAN
    print("=== Training encoder for target domain ===")
    if not (tgt_encoder.module.restored and critic.module.restored and
            tgt_model_trained):
        tgt_encoder = train_tgt(args, src_encoder, tgt_encoder, critic,
                                src_data_loader, tgt_data_loader)

    # eval target encoder on test set of target dataset
    print("Evaluate tgt test data on src encoder: {}".format(args.tgt))
    eval_tgt(src_encoder, src_classifier, tgt_data_loader_eval)
    print("Evaluate tgt test data on tgt encoder: {}".format(args.tgt))
    eval_tgt(tgt_encoder, src_classifier, tgt_data_loader_eval)


if __name__ == '__main__':
    main()
