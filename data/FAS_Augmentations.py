import random

# import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
from PIL import Image, ImageCms, ImageFilter
import numpy as np
import torch
from torchvision.transforms.transforms import Compose
import os
from .fas_aug_helper import *
import cv2
import os

class FAS_Augmentations(object):
    def __init__(self, config):
        self.save = config.TRAIN.AUG.SAVE
        if self.save:
            self.config = config
        self.al = [
        # (operation, min, max, changeLabel)
        ("Color_Diversity", 0.1, 10, False),  # 0
        ("Color_Distortion", 0.1, 10, True),  # 1
        ("Reflection", 0.01, 0.2, True),  # 2
        ("BN_Halftone", 0.01, 0.4, True),  # 3
        ("Moire_Pattern", 0.01, 0.3, True),  # 4
        ("SFC_Halftone", 0.01, 0.2, True),  # 5
        ("Hand_Trembling", 1, 16, False),  # 6
        ("Low_Resolution", 0.01, 0.9, False),  # 7
        # ("Original", 0, 0.1, False) # 8
    ]
        self.name_dict = {
            "Color_Diversity": self.Color_Diversity,
            "Color_Distortion": self.Color_Distortion,
            "Reflection": self.Reflection,
            "BN_Halftone": self.BN_Halftone,
            "Moire_Pattern": self.Moire_Pattern,
            "SFC_Halftone": self.SFC_Halftone,
            "Hand_Trembling": self.Hand_Trembling,
            "Low_Resolution": self.Low_Resolution,
            "Original": self.Original
        }
        self.augment_dict = {}
        for i in self.al:
            self.augment_dict[i[0]] = (self.name_dict[i[0]], i[1],i[2],i[3])   
        return

    def Reflection(self, img, a):
        assert 0.0 < a <= 0.5
        img = img.convert('RGBA')
        texture = getTexture('R', img.size)
        # further texture process
        comb_rgba = Image.blend(img, texture, a)
        res = comb_rgba.convert('RGB')
        return res

    def BN_Halftone(self, img, a):
        assert 0.0 < a <= 0.5
        img = img.convert('RGBA')
        texture = getTexture('BN', img.size)
        # further texture process
        comb_rgba = Image.blend(img, texture, a)
        res = comb_rgba.convert('RGB')
        return res

    def Moire_Pattern(self, img, a):
        assert 0.0 < a <= 0.5
        img = img.convert('RGBA')
        texture = getTexture('MP', img.size)
        # further texture process
        comb_rgba = Image.blend(img, texture, a)
        res = comb_rgba.convert('RGB')
        return res

    def Color_Diversity(self, img, nil):  # [-0.3, 0.3]
        assert 0 < nil
        fas_aug_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        rgb_profile_path = os.path.join(fas_aug_base, 'data/profile/RGB Profiles/')
        rgb_profile_dict = {
                "AdobeRGB1998.icc" : "A98",
                "AppleRGB.icc" : "A",
                "ColorMatchRGB.icc" : "CM",
                "sRGB2014.icc" : "S",
                "CIERGB.icc" : "C",
                "ProPhoto.icc" : "P",
                "ProPhotoD65.icc" : "P65",
                "sRGB Gamma22.icc" : "S22",
                "MaxRGB.icc" : "Max",
                "DonRGB4.icm" : "D",
                "BestRGB.icm" : "B"
            }
        rp1 = random.choice(list(rgb_profile_dict)) 
        rp2 = random.choice(list(rgb_profile_dict))   
        r1 = rgb_profile_dict[rp1] 
        r2 = rgb_profile_dict[rp2] 
        rgb_p1 = rgb_profile_path+rp1
        rgb_p2 = rgb_profile_path+rp2
        res = ImageCms.profileToProfile(img, rgb_p1, rgb_p2)
        return res

    def Color_Distortion(self, img, nil):
        assert 0 < nil
        fas_aug_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        rgb_profile_path = os.path.join(fas_aug_base, 'data/profile/RGB Profiles/')
        rgb_profile_dict = {
                "AdobeRGB1998.icc" : "A98",
                "AppleRGB.icc" : "A",
                "ColorMatchRGB.icc" : "CM",
                "sRGB2014.icc" : "S",
                "CIERGB.icc" : "C",
                "ProPhoto.icc" : "P",
                "ProPhotoD65.icc" : "P65",
                "sRGB Gamma22.icc" : "S22",
                "MaxRGB.icc" : "Max",
                "DonRGB4.icm" : "D",
                "BestRGB.icm" : "B"
            }
        cmyk_profile_path = os.path.join(fas_aug_base, 'data/profile/CMYK Profiles/')
        cmyk_profile_dict = {
                "EuroscaleCoated.icc" : "EC",
                "EuroscaleUncoated.icc" : "EU",
                "JapanStandard.icc" : "J",
                "USSheetfedCoated.icc" : "USC",
                "USSheetfedUncoated.icc" : "USU",
                "USWebCoated.icc" : "UWC",
                "USWebUncoated.icc" : "UWU"
            }
        rp2 = random.choice(list(rgb_profile_dict))
        cp2 = random.choice(list(cmyk_profile_dict))
        r2 = rgb_profile_dict[rp2]
        c2 = cmyk_profile_dict[cp2]    
        rgb_p2 = rgb_profile_path+rp2
        cmyk_p2  = cmyk_profile_path+cp2
        cmyk2rgb_trans = ImageCms.buildTransformFromOpenProfiles(cmyk_p2, rgb_p2, "CMYK", "RGB")
        img = img.convert('CMYK')
        res = ImageCms.applyTransform(img, cmyk2rgb_trans)
        return res

    def SFC_Halftone(self, img, a):
        assert 0.0 < a <= 0.5
        h = img.size[0]
        w = img.size[1]
        shrink = cv2.resize(np.asarray(img), (int(h/3),int(w/3)))
        ht = sfc_halftone(shrink)
        ht = Image.fromarray(np.uint8(ht)).convert('RGBA')
        ht = ht.resize((h,w))
        img = img.convert('RGBA')
        comb_rgba = Image.blend(img, ht, a) # 0.05-0.4
        res = comb_rgba.convert('RGB') 
        return res

    def Hand_Trembling(self, img, ks):
        ks = int(ks)
        assert 0 < ks <= 20
        dict_direction = {'H': 'Horizontal',
                            'V': 'Vertical',
                            'D': 'Top-left to Bottom-right',
                            'U': 'Bottom-left to Top-right'}
        direction = random.choice(list(dict_direction))
        kernel = np.zeros((ks, ks))
        if direction == 'H':
            kernel[int((ks - 1)/2), :] = np.ones(ks)
        elif direction == 'V':
            kernel[:, int((ks - 1)/2)] = np.ones(ks)
        elif direction == 'D':
            for i in range (ks):
                kernel[i][i] = 1
        elif direction == 'U':
            for i in range (ks):
                kernel[i][ks - 1 - i] = 1
        # print(kernel)
        kernel = kernel / kernel.sum()   
        img = np.asarray(img)
        img_motionBlur = cv2.filter2D(img,-1,kernel)
        res = Image.fromarray(np.uint8(img_motionBlur))
        return res

    def Low_Resolution(self, img, sr):
        assert 0 < sr <= 1
        hw = img.size
        shrink_size = int( (((sr-0.0001)*(1/6-1)/(1-0.0001))+1) * hw[0] )
        # print(shrink_size)
        shrink_interpolation = {"Area":cv2.INTER_AREA, 
                                "Cubic":cv2.INTER_CUBIC, 
                                "Linear":cv2.INTER_LINEAR, 
                                "Lanczos4":cv2.INTER_LANCZOS4, 
                                "Nearest":cv2.INTER_NEAREST}
        enlarge_interpolation = {"Area":cv2.INTER_AREA, 
                                "Nearest":cv2.INTER_NEAREST}
        s_i = random.choice(list(shrink_interpolation))
        e_i = random.choice(list(enlarge_interpolation))
        shrink_i = shrink_interpolation[s_i]
        enlarge_i = enlarge_interpolation[e_i]
        img = np.asarray(img)
        shrinked_img = cv2.resize(img, (shrink_size,shrink_size), interpolation=shrink_i)
        enlarged_img = cv2.resize(shrinked_img, (hw[0],hw[1]), interpolation=enlarge_i)
        res = Image.fromarray(np.uint8(enlarged_img))    
        return res

    def Original (self, img, _):
        res = img
        return res

    def get_augment(self, name):
        return self.augment_dict[name]

    def apply_augment(self, img, name, level, epoch):
        augment_fn, low, high, changeLabel = self.get_augment(name)
        # print(img,level)
        if level == 0:
            return img.copy(), False
        mag = level * (high - low) + low
        res = augment_fn(img.copy(), mag)
        
        if self.save:
            # print("save: ",epoch)
            folder_path = os.path.join(self.config.OUTPUT_DIR,"Augmented_img/"+"epoch"+str(epoch))
            os.makedirs(folder_path, exist_ok=True)
            output_name = name+'_'+str(level)+'.png'
            output_path = os.path.join(folder_path, output_name)
            res.save(output_path)
        return res, changeLabel

# if __name__ == '__main__':
#     input_pic_path = '/home/rizhao/projects/Cecelia/a-transform/input/6.png'
#     input_pic = Image.open(input_pic_path)
#     res = Low_Resolution(input_pic, 0.5)
#     res.save('./test/LQ.png')