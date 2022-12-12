import torch
import torch.nn.functional as F

import torch.optim as optim
from utils.utils import AverageMeter
import os
import time

from tqdm import tqdm
from models.bc.network import build_net
from models.base import BaseTrainer
import logging
import pdb

from torch.nn import DataParallel
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.criterion import CrossEntropyLabelSmooth, SupConLoss

class Trainer(BaseTrainer):
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
        self.config = config

    def set_model(self):
        self.network = build_net(self.config)
        if self.config.CUDA:
            self.network.cuda()
            if self.train_mode:
                self.network = DataParallel(self.network)
        self.optimizer = SGD(filter(lambda p: p.requires_grad, self.network.parameters()), 
                             lr=self.init_lr, momentum = self.momentum, nesterov = False, weight_decay = 0)
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        # self.criterion_smooth = CrossEntropyLabelSmooth(epsilon = self.config.DATA.LABEL_SMOOTHING).cuda()
        self.criterion_contrastive = SupConLoss().cuda()











