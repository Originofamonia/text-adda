"""Test script to classify target data."""

import torch.nn as nn
import numpy as np


def eval_tgt(encoder, classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (reviews, labels) in data_loader:
        preds = classifier(encoder(reviews))
        loss += criterion(preds, labels).item()
        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum().item()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = %.4f, Avg Accuracy = %.4f" % (loss, acc))


def eval_tgt_save_features(encoder, classifier, data_loader):
    """
    Evaluation for target encoder by source classifier on target dataset.
    Save features after encoder for visualization.
    """
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    feat_list = []
    label_list = []
    for (reviews, labels) in data_loader:
        features = encoder(reviews)
        feat_list.append(features.cpu().detach().numpy())
        label_list.append(labels.cpu().detach().numpy())

    features_npy = np.vstack(feat_list)
    labels_npy = np.hstack(label_list)
    np.savez('snapshots/features', features_npy, labels_npy)
