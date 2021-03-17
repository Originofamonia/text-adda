"""Pre-train encoder and classifier for source dataset."""

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from params import param
from utils import save_model


def train_src(args, encoder, classifier, train_loader, test_loader):
    """Train classifier for source domain."""
    # instantiate EarlyStop
    earlystop = EarlyStop(args.patience)

    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=args.lr,
        # betas=(param.beta1, param.beta2)
    )
    criterion = nn.CrossEntropyLoss()

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()

    for epoch in range(args.num_epochs_pre):
        pbar = tqdm(train_loader)
        for step, (reviews, labels) in enumerate(pbar):

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            preds = classifier(encoder(reviews))
            loss = criterion(preds, labels)

            # optimize source classifier
            loss.backward()
            optimizer.step()

            # print step info
            if step % args.log_step_pre == 0:
                desc = "Epoch [{}/{}] Step [{}/{}]: loss={:.4f}".format(epoch,
                                                                        args.num_epochs_pre,
                                                                        step,
                                                                        len(train_loader),
                                                                        loss.item())
                pbar.set_description(desc=desc)

        # eval model on test set
        # if epoch % args.eval_step_pre == 0:
        #     print('Epoch [{}/{}]'.format(epoch, args.eval_step_pre))
        #     eval_src(encoder, classifier, train_loader)
        #     earlystop.update(eval_src(encoder, classifier, test_loader))

        if earlystop.stop:
            break

    # # save final model
    # save_model(encoder, "ADDA-src-encoder-{}.pt".format(args.src))
    # save_model(classifier, "ADDA-src-classifier-{}.pt".format(args.src))

    return encoder, classifier


def train_no_da(args, encoder, classifier, train_loader, test_loader):
    """Train without DA"""
    # instantiate EarlyStop
    earlystop = EarlyStop(args.patience)

    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=args.lr,
        # betas=(param.beta1, param.beta2)
    )
    criterion = nn.CrossEntropyLoss()

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()

    for epoch in range(args.num_epochs):
        pbar = tqdm(train_loader)
        for step, (reviews, labels) in enumerate(pbar):

            optimizer.zero_grad()

            # compute loss for critic
            preds = classifier(encoder(reviews))
            loss = criterion(preds, labels)

            # optimize source classifier
            loss.backward()
            optimizer.step()

            # print step info
            if step % args.log_step_pre == 0:
                desc = "Epoch {}/{} Step {}/{}: loss={:.4f}".format(epoch,
                                                                    args.num_epochs,
                                                                    step,
                                                                    len(train_loader),
                                                                    loss.item())
                pbar.set_description(desc=desc)

        # eval model on test set
        if epoch % args.eval_step_pre == 0:
            print('Epoch [{}/{}]'.format(epoch, args.eval_step_pre))
            eval_src(encoder, classifier, train_loader)
            earlystop.update(eval_src(encoder, classifier, test_loader))

        if earlystop.stop:
            break

    # save final model
    save_model(encoder, "src-encoder-{}.pt".format(args.src))
    save_model(classifier, "src-classifier-{}.pt".format(args.src))

    return encoder, classifier


def eval_src(encoder, classifier, data_loader):
    """Evaluate classifier for source domain."""
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

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()

    return loss


class EarlyStop:
    def __init__(self, patience):
        self.count = 0
        self.maxAcc = 0
        self.patience = patience
        self.stop = False

    def update(self, acc):
        if acc < self.maxAcc:
            self.count += 1
        else:
            self.count = 0
            self.maxAcc = acc

        if self.count > self.patience:
            self.stop = True
