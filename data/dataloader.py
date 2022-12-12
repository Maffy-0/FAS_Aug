import os

import torch
import torch.nn as nn
# import torchvision
import torch.distributed as dist

# from torch.utils.data import Subset, ConcatDataset
# from sklearn.model_selection import StratifiedShuffleSplit
# from .transform import test_collate_fn

import pandas as pd
import copy
# import importlib

def parse_multiple_data_list(data_list_path):
    csv_list = []
    for csv_path in data_list_path:
        csv_list.append(pd.read_csv(csv_path, header=None))
    csv = pd.concat(csv_list, ignore_index=True)
    data_list = csv.get(0)
    face_labels = csv.get(1)
    return data_list, face_labels

def parse_data_list(data_list_path):
    csv = pd.read_csv(data_list_path, header=None)
    data_list = csv.get(0)
    face_labels = csv.get(1)

    return data_list, face_labels

def get_dataset_from_list(data_list_path, dataset_cls, transform, num_frames=1000, root_dir='', single_dataList = True):
    if single_dataList:
        # print("single", data_list_path)
        data_file_list, face_labels = parse_data_list(data_list_path)
    else:
        # print("multiple", data_list_path)
        data_file_list, face_labels = parse_multiple_data_list(data_list_path)
    # print(transform)
    num_file = data_file_list.size
    dataset_list = []

    for i in range(num_file):
        face_label = int(face_labels.get(i)==0) # 0 means real face and non-zero represents spoof
        file_path = data_file_list.get(i)

        zip_path = root_dir + file_path
        if not os.path.exists(zip_path):
            print("Skip {} (not exists)".format(zip_path))
            continue
        else:
            dataset = dataset_cls(zip_path, face_label, transform, num_frames=num_frames)
            if len(dataset) == 0:
                print("Skip {} (zero elements)".format(zip_path))
                continue
            else:
                dataset_list.append(dataset)

    final_dataset = torch.utils.data.ConcatDataset(dataset_list)

    return final_dataset

def get_dataloader(config, dataset, transform = None, transform_aug_list=None, train_mode=True):

    batch_size = config.DATA.BATCH_SIZE
    num_workers = 0 if config.DEBUG else config.DATA.NUM_WORKERS
    dataset_root_dir = config.DATA.ROOT_DIR 
    dataset_subdir = config.DATA.SUB_DIR  
    dataset_dir = os.path.join(dataset_root_dir, dataset_subdir)

    train_loader_list = []
    train_aug_loader_list = []
    valid_loader = None
    test_loader = None

    if train_mode:
        assert config.DATA.TRAIN_LIST, "CONFIG.DATA.TRAIN should be provided"
        for i in range(len(config.DATA.TRAIN_LIST)):
            trainset = get_dataset_from_list(config.DATA.TRAIN_LIST[i], dataset, 
                                            transform=transform, num_frames=config.DATA.NUM_FRAMES, root_dir=dataset_dir,
                                            single_dataList = True)
            train_loader_list.append(torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                                                                num_workers=num_workers, pin_memory=True,
                                                                sampler=None, drop_last=True,))


        for i in range(config.DATA.EXTRA_DOMAIN):
            trainset_aug = get_dataset_from_list(config.DATA.TRAIN_LIST, dataset, 
                                                transform=transform_aug_list[i], num_frames=int(config.DATA.NUM_FRAMES/3+0.5), root_dir=dataset_dir,
                                                single_dataList = False)         
                
            mergeDataset = torch.utils.data.DataLoader(trainset_aug, batch_size=batch_size, shuffle=True, 
                                                                num_workers=num_workers, pin_memory=True,
                                                                sampler=None, drop_last=True, )
            train_aug_loader_list.append(mergeDataset)                                                                

        train_loader_list.extend(train_aug_loader_list)
        if config.DATA.VAL:
            validset = get_dataset_from_list(config.DATA.VAL, dataset, 
                                            transform=transform, num_frames=config.DATA.NUM_FRAMES, root_dir=dataset_dir)
            valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,sampler=None, drop_last=False)                
        assert config.DATA.TEST, "CONFIG.DATA.TEST should be provided"
        testset = get_dataset_from_list(config.DATA.TEST, dataset, 
                                        transform=transform, num_frames=config.DATA.NUM_FRAMES, root_dir=dataset_dir)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,sampler=None, drop_last=False)
        
        
    else:
        assert config.DATA.TEST, "CONFIG.DATA.TEST should be provided"
        testset = get_dataset_from_list(config.DATA.TEST, dataset, transform=transform, num_frames=config.DATA.NUM_FRAMES, root_dir=dataset_dir)
        
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
            sampler=None, drop_last=False
        )        

    return train_loader_list, valid_loader, test_loader