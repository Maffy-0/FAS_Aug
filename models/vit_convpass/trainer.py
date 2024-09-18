import os

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

from utils.utils import AverageMeter
from .network import build_net

pd.set_option('display.max_columns', None)

import logging

from models.base import BaseTrainer
from utils.criterion import CrossEntropyLabelSmooth, SupConLoss

class Trainer(BaseTrainer):
    """
    Trainer encapsulates all the logic necessary for
    training the Recurrent Attention Model.

    All hyperparameters are provided by the user in the
    config file.
    """

    """
    Trainer encapsulates all the logic necessary for
    training the Recurrent Attention Model.

    All hyperparameters are provided by the user in the
    config file.
    """

    def __init__(self, config):
        """
        Construct a new Trainer instance.
        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data iterator
        """
        super(Trainer, self).__init__(config)

    def set_model(self):
        self.network = build_net(arch_name=self.config.MODEL.ARCH, pretrained=self.config.MODEL.IMAGENET_PRETRAIN)
        if self.config.CUDA:
            self.network.cuda()

        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion_smooth = CrossEntropyLabelSmooth(epsilon = self.config.DATA.LABEL_SMOOTHING)
        self.criterion_contrastive = SupConLoss()
        if self.config.MODEL.FIX_BACKBONE:
            for name, p in self.network.named_parameters():
                if 'adapter' in name or 'head' in name:
                    p.requires_grad = True
                    # import pdb; pdb.set_trace()
                else:
                    p.requires_grad = False
        # Set up optimizer
        if self.config.TRAIN.OPTIM.TYPE == 'SGD':
            logging.info('Setting: Using SGD Optimizer')
            self.optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.network.parameters()),
                lr=self.init_lr,
            )

        elif self.config.TRAIN.OPTIM.TYPE == 'Adam':
            logging.info('Setting: Using Adam Optimizer')
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.network.parameters()),
                lr=self.init_lr,
            )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.TRAIN.EPOCHS)        
