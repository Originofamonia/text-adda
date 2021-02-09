"""Adversarial adaptation to train target encoder."""

import os
import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm
from params import param
from utils import make_cuda


def train_tgt(args, src_encoder, tgt_encoder, critic,
              src_data_loader, tgt_data_loader):
    """Train encoder for target domain."""

    # set train state for Dropout and BN layers
    tgt_encoder.train()
    critic.train()

    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
                               lr=args.c_lr,)
                               # betas=(args.beta1, args.beta2))
    optimizer_critic = optim.Adam(critic.parameters(),
                                  lr=args.d_lr,)
                                  # betas=(args.beta1, args.beta2))
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    for epoch in range(args.num_epochs):
        # zip source and target data pair
        pbar = tqdm(zip(src_data_loader, tgt_data_loader))
        for step, ((reviews_src, _), (reviews_tgt, _)) in enumerate(pbar):

            # zero gradients for optimizer
            optimizer_critic.zero_grad()

            # extract and concat features
            feat_src = src_encoder(reviews_src)
            feat_tgt = tgt_encoder(reviews_tgt)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)

            # predict on discriminator
            pred_concat = critic(feat_concat.detach())

            # prepare real and fake label
            label_src = make_cuda(torch.ones(feat_src.size(0)).long())
            label_tgt = make_cuda(torch.zeros(feat_tgt.size(0)).long())
            label_concat = torch.cat((label_src, label_tgt), 0)

            # compute loss for critic
            loss_critic = criterion(pred_concat, label_concat)
            loss_critic.backward()

            # optimize critic
            optimizer_critic.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            ############################
            # 2.2 train target encoder #
            ############################

            # zero gradients for optimizer
            optimizer_tgt.zero_grad()

            # extract and target features
            feat_tgt = tgt_encoder(reviews_tgt)

            # predict on discriminator
            pred_tgt = critic(feat_tgt)

            # prepare fake labels
            label_tgt = make_cuda(torch.ones(feat_tgt.size(0)).long())

            # compute loss for target encoder
            loss_tgt = criterion(pred_tgt, label_tgt)
            loss_tgt.backward()

            # optimize target encoder
            optimizer_tgt.step()

            #######################
            # 2.3 print step info #
            #######################
            if (step + 1) % args.log_step == 0:
                desc = "Epoch [{}/{}] Step [{}/{}]: " \
                       "d_loss={:.4f} g_loss={:.4f} acc={:.4f}".format(epoch + 1,
                                                                       args.num_epochs,
                                                                       step + 1,
                                                                       len_data_loader,
                                                                       loss_critic.item(),
                                                                       loss_tgt.item(),
                                                                       acc.item())
                pbar.set_description(desc=desc)

        #############################
        # 2.4 save model argseters #
        #############################
        if (epoch + 1) % args.save_step == 0:
            torch.save(critic.state_dict(), os.path.join(
                args.model_root,
                "ADDA-critic-{}.pt".format(epoch + 1)))
            torch.save(tgt_encoder.state_dict(), os.path.join(
                args.model_root,
                "ADDA-target-encoder-{}.pt".format(epoch + 1)))

    torch.save(critic.state_dict(), os.path.join(
        args.model_root,
        "ADDA-critic-final.pt"))
    torch.save(tgt_encoder.state_dict(), os.path.join(
        args.model_root,
        "ADDA-target-encoder-final.pt"))
    return tgt_encoder
