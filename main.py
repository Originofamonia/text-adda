"""Main script for ADDA."""

import torch
from params import param
from core import eval_src, eval_tgt, train_src, train_tgt
from models import BERTEncoder, BERTClassifier, Discriminator
from utils import read_data, get_data_loader, init_model, init_random_seed
from pytorch_pretrained_bert import BertTokenizer

if __name__ == '__main__':
    # init random seed
    init_random_seed(param.manual_seed)

    # preprocess data
    src_train = read_data('./data/processed/electronics/train.txt')
    src_test = read_data('./data/processed/electronics/test.txt')
    tgt_train = read_data('./data/processed/kitchen/train.txt')
    tgt_test = read_data('./data/processed/kitchen/test.txt')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    src_train_sequences = []
    src_test_sequences = []
    tgt_train_sequences = []
    tgt_test_sequences = []

    for i in range(len(src_train.review)):
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
    src_data_loader = get_data_loader(src_train_sequences, src_train.label, param.maxlen)
    src_data_loader_eval = get_data_loader(src_test_sequences, src_test.label, param.maxlen)
    tgt_data_loader = get_data_loader(tgt_train_sequences, tgt_train.label, param.maxlen)
    tgt_data_loader_eval = get_data_loader(tgt_test_sequences, tgt_test.label, param.maxlen)

    # load models
    src_encoder = init_model(BERTEncoder(),
                             restore=param.src_encoder_restore)
    src_classifier = init_model(BERTClassifier(),
                                restore=param.src_classifier_restore)
    tgt_encoder = init_model(BERTEncoder(),
                             restore=param.tgt_encoder_restore)
    critic = init_model(Discriminator(),
                        restore=param.d_model_restore)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # freeze encoder params
    for param in src_encoder.parameters():
        param.requires_grad = False

    # train source model
    print("=== Training classifier for source domain ===")
    # if not (src_encoder.restored and src_classifier.restored and
    #         param.src_model_trained):
    src_encoder, src_classifier = train_src(
        src_encoder, src_classifier, src_data_loader, src_data_loader_eval)

    # eval source model
    print("=== Evaluating classifier for source domain ===")
    eval_src(src_encoder, src_classifier, src_data_loader_eval)

    # train target encoder by GAN
    print("=== Training encoder for target domain ===")
    if not (tgt_encoder.restored and critic.restored and
            param.tgt_model_trained):
        tgt_encoder = train_tgt(src_encoder, tgt_encoder, critic,
                                src_data_loader, tgt_data_loader)

    # eval target encoder on test set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> source only <<<")
    eval_tgt(src_encoder, src_classifier, tgt_data_loader_eval)
    print(">>> domain adaption <<<")
    eval_tgt(tgt_encoder, src_classifier, tgt_data_loader_eval)
