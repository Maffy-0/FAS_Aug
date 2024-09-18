import os
import torch
import logging
import numpy as np

from utils.metrics import metric_report_from_dict
from utils.utils import AverageMeter, to_Pickle
# from utils.criterion import CrossEntropyLabelSmooth
from tqdm import tqdm

import pandas as pd
pd.set_option('display.max_columns', None)

import logging

from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
# from torch.cuda.amp import GradScaler, autocast

from data.dataloader import get_dataloader
from data.transform import get_basetransform

from data.transform import random_parse_policies, MultiAugmentation
import importlib
from torch.nn import DataParallel
import copy

class BaseTrainer(object):
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
        self.config = config
        self.global_step = 1
        self.start_epoch = 1

        # Training control config
        self.epochs = self.config.TRAIN.EPOCHS
        self.batch_size = self.config.DATA.BATCH_SIZE

        self.counter = 0

        #  # Meanless at this version
        self.epochs = self.config.TRAIN.EPOCHS
        # self.val_freq = 1
        #
        # # Network config

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

        self.num_classes = self.config.MODEL.NUM_CLASSES
        self.M = self.config.TRAIN.AUG.NUM_POLICIES
        self.Q = self.config.TRAIN.AUG.NUM_SUBPOLICIES
        self.num_ops = self.config.TRAIN.AUG.NUM_OPS
        self.num_mag = self.config.TRAIN.AUG.NUM_MAG
        self.toPKL = to_Pickle(self.config.OUTPUT_DIR)

    def set_trainMode(self, train_mode):
        self.train_mode = train_mode

    def set_model(self):
        return None

    def __get_dataset___(self):
        logging.info("zip_dataset module used: {}".format(self.config.DATA.DATASET))
        self.Dataset = importlib.import_module('data.zip_dataset').__dict__[self.config.DATA.DATASET]
        return self.Dataset

    def set_dataloader(self):
        config = self.config
        Dataset = self.__get_dataset___()
        self.transform_aug_list = []
        transform_basic = get_basetransform(config)

        # create augmented domain
        if self.train_mode:
            for i in range(config.DATA.EXTRA_DOMAIN):
                tran = copy.deepcopy(transform_basic)
                tran.transforms.insert(1, None)
                self.transform_aug_list.append(tran)

        self.train_loader_list, self.valid_loader, self.test_loader = get_dataloader(config, Dataset, 
                    transform = transform_basic, transform_aug_list=self.transform_aug_list, 
                    train_mode = self.train_mode)
        self.train_loader_len = len(self.train_loader_list)

        self.train_loader_ListnIter = []
        for i in range(len(self.train_loader_list)):
            self.train_loader_ListnIter.append({
                "data_loader": self.train_loader_list[i],
                "iteration": iter(self.train_loader_list[i])
            })                


    def set_iteration(self):
        lenInfo = "length of iteration of domain = "
        for i in self.train_loader_list:
            lenInfo += str(len(i))+' '
        logging.info(lenInfo)
        if self.config.TRAIN.ITERATION > 0:
            # self define
            self.iteration = self.config.TRAIN.ITERATION
        elif len(self.train_loader_list)>len(self.config.DATA.TRAIN_LIST):
            # get iteration of augmented dataloader
            self.iteration = int(len(self.train_loader_list[-1]))
        else:
            # mean
            self.iteration = 0
            for loader in self.train_loader_list:
                self.iteration += len(loader)
            self.iteration = int(self.iteration/len(self.train_loader_list)) 
        logging.info("Number of training iteration: {}".format(self.iteration))
                

    def set_augment(self):
        augment_lib = importlib.import_module("data."+self.config.TRAIN.AUG.BAG).__dict__[self.config.TRAIN.AUG.BAG]
        self.augment = augment_lib(self.config)

    def train(self, ):
        
        if self.config.TRAIN.RESUME and os.path.exists(self.config.TRAIN.RESUME):
            logging.info("Resume=True.")
            self.load_checkpoint(self.config.TRAIN.RESUME)  
        if self.config.CUDA:
            logging.info("Number of GPUs: {}".format(torch.cuda.device_count()))
            self.network = torch.nn.DataParallel(self.network)

        for epoch in range(self.start_epoch, self.epochs + 1):
            logging.info('\nEpoch: {}/{} - LR: {:.6f}'.format(
                epoch, self.epochs, self.init_lr))

            # train for 1 epoch
            train_loss_avg, loss_base ,loss_rex, loss_con = self._train_one_epoch(epoch)
            logging.info("Avg Training loss = {}  Avg loss_base = {}  Avg loss_rex = {}  Avg loss_con = {}".format(train_loss_avg, loss_base ,loss_rex, loss_con))

            # evaluate on validation set
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

    def _train_one_epoch(self, epoch):
        losses_ave = AverageMeter()
        lossesB_ave = AverageMeter()
        lossesR_ave = AverageMeter()
        lossesC_ave = AverageMeter()

        self.network.train()

        # get data augmentation policies randomly
        parsed_policies_list = []
        for i in range(self.config.DATA.EXTRA_DOMAIN):
            parsed_policies = random_parse_policies(self.augment, self.M, self.Q, self.num_mag)
            parsed_policies_list.append(parsed_policies)     

        # apply the policies to the dataloader's transform
        for i in range(self.config.DATA.EXTRA_DOMAIN):
            parsed_policies = parsed_policies_list[i]
            logging.info(parsed_policies)
            trfs = self.transform_aug_list[i].transforms
            trfs[1] = MultiAugmentation(self.augment, parsed_policies, epoch)

        self.features = []
        def hook_function(modele,input,output):
            self.features.append(torch.nn.functional.normalize(input[0], dim=1))

        if 'vit' in self.config.MODEL.ARCH:
            hook_handle = self.network.module.head.register_forward_hook(hook_function)
        else:
            hook_handle = self.network.module.fc.register_forward_hook(hook_function)

        with tqdm(total=self.iteration) as pbar:
            for iter_num in range(self.iteration): 

                loss, lossBase, lossRex, lossCon = self._train_one_batch(optimizer=self.optimizer)

                pbar.set_description(
                    (
                        " total loss={:.3f} ".format(loss.item(),)
                    )
                )
                pbar.update(1)
                losses_ave.update(loss.item())
                lossesB_ave.update(lossBase.item())
                lossesR_ave.update(lossRex.item())
                lossesC_ave.update(lossCon.item())

                self.global_step += 1
            
        hook_handle.remove()

        self.lr_scheduler.step()
        for i in range(self.config.DATA.EXTRA_DOMAIN):
            dict_key = 'policies'+str(i+1)                   
            self.toPKL.add_dict({dict_key : parsed_policies_list[i]})   

        return losses_ave.avg, lossesB_ave.avg, lossesR_ave.avg, lossesC_ave.avg

    def _train_one_batch(self, optimizer):
        batch_data = self.get_data_from_loader_list(self.train_loader_ListnIter)
        network_input, label = batch_data[0].cuda(), batch_data[1].cuda()
        pred = self.network(network_input)

        lossSARE = torch.tensor(0.)
        lossCon = torch.tensor(0.)
        lossBase = self._total_loss_calculation(pred, label)
        feature = self.features.pop()
        lossSARE = self.sare_loss_varOnSpoof(pred,label)
        lossCon = self.supConLoss_real(feature,label)
        loss = lossBase + self.config.TRAIN.ALPHA*lossCon + self.config.TRAIN.BETA*lossSARE            

        # compute gradients and update SGD
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        return loss, lossBase, lossRex, lossCon

    def validate(self, epoch, val_data_loader, test_mode=True):
        
        val_results = self.test(val_data_loader)

        val_loss = val_results['avg_loss']
        scores_gt_dict = val_results['scores_gt']
        scores_pred_dict = val_results['scores_pred']
        if test_mode:
            frame_metric_dict, video_metric_dict = metric_report_from_dict(scores_gt_dict, scores_pred_dict, 0.5)
            df_frame = pd.DataFrame(frame_metric_dict, index=[0])
            df_video = pd.DataFrame(video_metric_dict, index=[0])

            logging.info("Frame level metrics: \n" + str(df_frame))
            logging.info("Video level metrics: \n" + str(df_video))

            return frame_metric_dict, val_loss
        else:
            return val_loss

    def test(self, test_data_loader):
        avg_test_loss = AverageMeter()
        pred_dict = {}
        gt_dict = {}
        self.network.eval()
        with torch.no_grad():
            for data in tqdm(test_data_loader):
                network_input, label, vid_path = data[1], data[2], data[3]
                label = label['face_label']
                network_input = network_input.cuda()

                pred = self.network(network_input)
                prob = self._get_prob_from_pred(pred)
                gt_dict, pred_dict = self._collect_scores_from_loader(gt_dict, pred_dict,
                                                                      label.numpy(), prob,
                                                                      vid_path)

                test_loss = self._total_loss_calculation(pred, label)
                avg_test_loss.update(test_loss.item(), network_input.size()[0])

        test_results = {
            'scores_gt': gt_dict,
            'scores_pred': pred_dict,
            'avg_loss': avg_test_loss.avg,
        }
        return test_results

    def get_data_from_loader_list(self, train_loader_list):
        img_list = []
        label_list = []
        # print(train_loader_list)
        for loader in train_loader_list:
            # print(loader)
            try:
                _, img, label, _ = loader["iteration"].next()
            except:
                loader["iteration"] = iter(loader["data_loader"])
                _, img, label, _ = loader["iteration"].next()
            img_list.append(img)
            label_list.append(label['face_label'])
        
        imgs = torch.cat(img_list, 0)
        labels = torch.cat(label_list, 0)
        return imgs, labels

    def _collect_scores_from_loader(self, gt_dict, pred_dict, ground_truths, pred_scores, video_ids):
        batch_size = ground_truths.shape[0]

        for i in range(batch_size):
            video_name = video_ids[i]
            if video_name not in pred_dict.keys():
                pred_dict[video_name] = list()
            if video_name not in gt_dict.keys():
                gt_dict[video_name] = list()

            pred_dict[video_name] = np.append(pred_dict[video_name], pred_scores[i])
            gt_dict[video_name] = np.append(gt_dict[video_name], ground_truths[i])

        return gt_dict, pred_dict

    def save_checkpoint(self, state):

        ckpt_dir = os.path.join(self.config.OUTPUT_DIR, 'ckpt')
        os.makedirs(ckpt_dir, exist_ok=True)

        epoch = state['epoch']
        if self.config.TRAIN.SAVE_BEST:
            filename = 'best.ckpt'.format(epoch)
        else:
            filename = 'epoch_{}.ckpt'.format(epoch)
        ckpt_path = os.path.join(ckpt_dir, filename)
        logging.info("[*] Saving model to {}".format(ckpt_path))
        torch.save(state, ckpt_path)

    def load_checkpoint(self, ckpt_path):

        logging.info("[*] Loading model from {}".format(ckpt_path))

        ckpt = torch.load(ckpt_path)
        # load variables from checkpoint
        self.start_epoch = ckpt['epoch']
        self.global_step = ckpt['global_step']
        self.test_metric = ckpt['val_metrics']
        # self.best_valid_acc = ckpt['best_valid_acc']
        self.network.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optim_state'])

        logging.info(
            "[*] Loaded {} checkpoint @ epoch {}".format(
                ckpt_path, ckpt['epoch'])
        )

    def inference(self, *args, **kargs):
        """
            Input images
            Output prob and scores
        """
        output_prob = self.network(*args, **kargs)  # By default: a binary classifier network
        return output_prob


    def _get_prob_from_pred(self, pred):
        output_scores = torch.softmax(pred, 1)
        output_scores = output_scores.cpu().numpy()[:, 1]
        return output_scores

    def _total_loss_calculation(self, output_prob, target):
        face_label = target.cuda()
        return self.criterion(output_prob, face_label)

    def toCSV(self, metrics):
        path = self.config.RESULT_FILE
        metrics['output_path'] = self.config.OUTPUT_DIR
        df = pd.DataFrame(metrics, columns=['EER', 'MIN_HTER', 'AUC'], index=[metrics['output_path']])
        df.to_csv(path, mode='a+', header=False)        

    def sare_loss_varOnSpoof(self,pred,label):
        loss_per_domain = torch.zeros(self.train_loader_len).cuda()
        for domain_id, (y_per_domain, labels_per_domain) in enumerate(zip(pred.chunk(self.train_loader_len, dim=0), label.chunk(self.train_loader_len, dim=0))):
            fakeLabel_idx = (labels_per_domain == 0).nonzero(as_tuple=True)
            fake_y= y_per_domain[fakeLabel_idx]
            fake_labels = labels_per_domain[fakeLabel_idx]
            fake_loss = self._total_loss_calculation(fake_y, fake_labels)

            if not torch.isnan(fake_loss):
                loss_per_domain[domain_id] += fake_loss

        # risk extrapolation         
        loss_spoofDomain_mean = loss_per_domain.mean()
        loss_spoofDomain_variance = ((loss_per_domain - loss_spoofDomain_mean) ** 2).mean()

        return loss_spoofDomain_variance

    def supConLoss_real(self,feature,label):
        lossCon = torch.tensor(0.)
        realLabel_idx = (label == 1).nonzero(as_tuple=True)
        real_label = label[realLabel_idx]
        real_feature = feature[realLabel_idx]
        if real_feature.shape[0] > 1:
            lossCon = self.criterion_contrastive(real_feature.unsqueeze(1), labels=real_label)
        return lossCon

    def supConLoss(self,feature,label):
        lossCon = torch.tensor(0.)
        realLabel_idx = (label == 1).nonzero(as_tuple=True)
        fakeLabel_idx = (label == 0).nonzero(as_tuple=True)
        real_label = label[realLabel_idx]
        fake_label = label[fakeLabel_idx]
        real_feature = feature[realLabel_idx]
        fake_feature = feature[fakeLabel_idx]
        if real_feature.shape[0] <= 1:
            lossCon = self.criterion_contrastive(fake_feature.unsqueeze(1), labels=fake_label)
        elif fake_feature.shape[0] <= 1:
            lossCon = self.criterion_contrastive(real_feature.unsqueeze(1), labels=real_label)  
        else:
            lossCon = self.criterion_contrastive(feature.unsqueeze(1), labels=label)                    

        return lossCon    

    def sare_loss(self,pred,label):
        loss_per_domain = torch.zeros(self.train_loader_len).cuda()
        for domain_id, (y_per_domain, labels_per_domain) in enumerate(zip(pred.chunk(self.train_loader_len, dim=0), label.chunk(self.train_loader_len, dim=0))):
            loss_per_domain[domain_id] = self._total_loss_calculation(y_per_domain, labels_per_domain)

        # risk extrapolation         
        loss_spoofDomain_mean = loss_per_domain.mean()
        loss_spoofDomain_variance = ((loss_per_domain - loss_spoofDomain_mean) ** 2).mean()

        return loss_spoofDomain_variance