from torchvision.transforms import transforms
# from .fs_augmentations_m import *
from PIL import Image
import torch
import random
import logging

def get_basetransform(config):

    image_size = config.DATA.IN_SIZE
        
    transform_list = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor()
        ])

    if config.DATA.NORMALIZE.ENABLE:
        normalize = transforms.Normalize(config.DATA.NORMALIZE.MEAN, config.DATA.NORMALIZE.STD)
        transform_list.transforms.insert(2, normalize)
        logging.info("Mean STD Normalize is ENABLED")
    return transform_list

class Augmentation(object):
    def __init__(self, augment, policy):
        """
        For example, policy is [[(op, mag), (op,mag)]]*Q
        """
        self.augment = augment
        self.policy = policy
        self.changeLabel = False

    def __call__(self, img,epoch):
        sub_policy = random.choice(self.policy)
        self.changeLabel = False
        for op,mag in sub_policy:
            img, cl= self.augment.apply_augment(img, op, mag,epoch)
            self.changeLabel = cl or self.changeLabel
        return img
    
class MultiAugmentation(object):
    def __init__(self, augment, policies, epoch):
        self.policies = [Augmentation(augment, policy) for policy in policies] # len = M
        self.epoch = epoch
        self.count = 0
    
    def get_changeLabel(self):
        changeLabel = [policy.changeLabel for policy in self.policies]
        return changeLabel

    def __call__(self,img):
        imgs = [policy(img,self.epoch) for policy in self.policies]
        return imgs[0]


def random_parse_policies(augment, M, S, num_mags):
    # policies : (M,4(op,mag,op,mag)*5(sub_policy))
    # parsed_policies : [[[(op, mag), (op,mag)]]*5] * M
    
    al = augment.al
    
    # M, S = policies_shape
    # S = S//4
    parsed_policies = []
    for i in range(M):
        parsed_policy = []
        for j in range(S):
            op1, op2 = random.choices(al, k=2)
            mag1 = random.choice(range(num_mags))
            mag2 = random.choice(range(num_mags))
            # print(op1[0], mag1, op2[0], mag2)
            parsed_policy.append([(op1[0],mag1/(num_mags-1)),(op2[0],mag2/(num_mags-1))])
        parsed_policies.append(parsed_policy)
    
    return parsed_policies
