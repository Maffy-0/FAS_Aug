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
        self.config = config
        self.global_step = 1
        self.start_epoch = 1

        # Training control config
        self.epochs = self.config.TRAIN.EPOCHS
        self.batch_size = self.config.DATA.BATCH_SIZE

        self.counter = 0

        self.epochs = self.config.TRAIN.EPOCHS


        self.test_metrcis = {
            'HTER@0.5': 1.0,
            'EER': 1.0,
            'MIN_HTER': 1.0,
            'AUC': 0
        }

        # Optimizer config
        self.momentum = self.config.TRAIN.MOMENTUM
        self.init_lr = self.config.TRAIN.INIT_LR
        self.lr_patience = self.config.TRAIN.LR_PATIENCE
        self.train_patience = self.config.TRAIN.PATIENCE

    def set_model(self):
        self.network = build_net(arch_name=self.config.MODEL.ARCH, pretrained=self.config.MODEL.IMAGENET_PRETRAIN)
        if self.config.CUDA:
            self.network.cuda()

        self.criterion = torch.nn.CrossEntropyLoss()
        # self.criterion_smooth = CrossEntropyLabelSmooth(epsilon = self.config.DATA.LABEL_SMOOTHING)
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

    def train(self, ):

        if self.config.TRAIN.RESUME and os.path.exists(self.config.TRAIN.RESUME):
            logging.info("Resume=True.")
            self.load_checkpoint(self.config.TRAIN.RESUME)

        if self.config.CUDA:
            logging.info("Number of GPUs: {}".format(torch.cuda.device_count()))
            self.network = torch.nn.DataParallel(self.network)

        for epoch in range(self.start_epoch, self.epochs + 1):
            # if self.tensorboard:
            #     self.tensorboard.add_scalar('lr', self.init_lr, self.global_step)
            logging.info('\nEpoch: {}/{} - LR: {:.6f}'.format(
                epoch, self.epochs, self.init_lr))

            train_loss_avg, loss_base ,loss_rex, loss_con = self._train_one_epoch(epoch)
            # train for 1 epoch
            logging.info("Avg Training loss = {}  Avg loss_base = {}  Avg loss_rex = {}  Avg loss_con = {}".format(train_loss_avg, loss_base ,loss_rex, loss_con))
            # evaluate on validation set'
            
            with torch.no_grad():
                if self.valid_loader:
                    val_loss_avg = self.validate(epoch, self.valid_loader, test_mode=False)
                    logging.info("\nAvg Validation loss = {}".format(val_loss_avg))
                    self.toPKL.add_dict({'valid_loss' : val_loss_avg})                
                test_output, test_loss_avg = self.validate(epoch, self.test_loader)
                logging.info("\nAvg Testing loss = {}".format(test_loss_avg))

            if test_output['MIN_HTER'] < self.test_metrcis['MIN_HTER']:
                self.counter = 0
                self.test_metrcis['EER'] = test_output['EER']
                self.test_metrcis['MIN_HTER'] = test_output['MIN_HTER']
                self.test_metrcis['AUC'] = test_output['AUC']
                self.save_checkpoint(
                    {'epoch': epoch,
                        'val_metrics': self.test_metrcis,
                        'global_step': self.global_step,
                        'model_state': self.network.module.state_dict(),
                        'optim_state': self.optimizer.state_dict(),
                        }
                )

            else:
                self.counter += 1

            logging.info('Current Best MIN_HTER={}%, AUC={}%'.format(100*self.test_metrcis['MIN_HTER'],
                                                                    100*self.test_metrcis['AUC']))

            self.toPKL.add_dict(
            {
                'loss_rex' : loss_rex,
                'loss_con' : loss_con,
                'loss_base' : loss_base,
                'train_loss' : train_loss_avg,
                'test_loss' : test_loss_avg,
                'EER' : test_output['EER'],
                'HTER' : test_output['MIN_HTER'],
                'AUC' : test_output['AUC']
            }
            )
            self.toPKL.save_logs()    


            if self.counter > self.train_patience:
                logging.info("[!] No improvement in a while, stopping training.")
                break
        self.toCSV(self.test_metrcis)                                                                        

    def load_batch_data(self):
        pass
