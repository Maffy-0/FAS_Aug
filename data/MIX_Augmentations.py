# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
from cgitb import reset
import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
from PIL import Image, ImageCms, ImageFilter
import numpy as np
import torch
from torchvision.transforms.transforms import Compose
import os
from .fas_aug_helper import *
import cv2
import os

class MIX_Augmentations(object):
    def __init__(self, config):
        self.save = config.TRAIN.AUG.SAVE
        if self.save:
            self.config = config
        self.al = [
        # (operation, min, max, changeLabel)
        ("changeColorGamut", 0.1, 10, False),  # 0
        ("changeRGB2CMYK", 0.1, 10, True),  # 1
        ("addReflection", 0.01, 0.2, True),  # 2
        ("addBlueNoise", 0.01, 0.4, True),  # 3
        ("addMoirePattern", 0.01, 0.3, True),  # 4
        ("addHalftone", 0.01, 0.2, True),  # 5
        ("addMotionBlur", 1, 16, False),  # 6
        ("addQualityLose", 0.01, 0.9, False),  # 7
        ("ShearX", 0, 0.3, False),  # 0
        ("ShearY", 0, 0.3, False),  # 1
        ("TranslateX", 0, 0.45, False),  # 2
        ("TranslateY", 0, 0.45, False),  # 3
        ("Rotate", 0, 30, False),  # 4
        ("AutoContrast", 0, 1, False),  # 5
        ("Invert", 0, 1, False),  # 6
        ("Equalize", 0, 1, False),  # 7
        ("Solarize", 0, 256, False),  # 8
        ("Posterize", 4, 8, False),  # 9
        ("Contrast", 0.1, 1.9, False),  # 10
        ("Color", 0.1, 1.9, False),  # 11
        ("Brightness", 0.1, 1.9, False),  # 12
        ("Sharpness", 0.1, 1.9, False),  # 13
        ("Cutout", 0, 0.2, False),  # 14        
    ]
        self.name_dict = {
            "changeColorGamut": self.changeColorGamut,
            "changeRGB2CMYK": self.changeRGB2CMYK,
            "addReflection": self.addReflection,
            "addBlueNoise": self.addBlueNoise,
            "addMoirePattern": self.addMoirePattern,
            "addHalftone": self.addHalftone,
            "addMotionBlur": self.addMotionBlur,
            "addQualityLose": self.addQualityLose,
            "original": self.original,
            "ShearX": self.ShearX,
            "ShearY": self.ShearY,
            "TranslateX": self.TranslateX,
            "TranslateY": self.TranslateY,
            "Rotate": self.Rotate,
            "AutoContrast": self.AutoContrast,
            "Invert": self.Invert,
            "Equalize": self.Equalize,
            "Solarize": self.Solarize,
            "Posterize": self.Posterize,
            "Contrast": self.Contrast,
            "Color": self.Color,
            "Brightness": self.Brightness,
            "Sharpness": self.Sharpness,
            "Cutout": self.Cutout            
        }
        self.random_mirror = True
        self.augment_dict = {}
        for i in self.al:
            self.augment_dict[i[0]] = (self.name_dict[i[0]], i[1],i[2],i[3])   

        return

    def addReflection(self, img, a):
        assert 0.0 < a <= 0.5
        img = img.convert('RGBA')
        texture = getTexture('R', img.size)
        # further texture process
        comb_rgba = Image.blend(img, texture, a)
        res = comb_rgba.convert('RGB')
        # res.save('./output/test/R.png')
        return res

    def addBlueNoise(self, img, a):
        assert 0.0 < a <= 0.5
        img = img.convert('RGBA')
        texture = getTexture('BN', img.size)
        # further texture process
        comb_rgba = Image.blend(img, texture, a)
        res = comb_rgba.convert('RGB')
        # res.save('./output/test/BN.png')
        return res

    def addMoirePattern(self, img, a):
        assert 0.0 < a <= 0.5
        img = img.convert('RGBA')
        texture = getTexture('MP', img.size)
        # further texture process
        comb_rgba = Image.blend(img, texture, a)
        res = comb_rgba.convert('RGB')
        # res.save('./output/test/MP.png')
        return res

    def changeColorGamut(self, img, nil):  # [-0.3, 0.3]
        assert 0 < nil
        rgb_profile_path = 'data/profile/RGB Profiles/'
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
        # res.save('./output/test/RGB.png')
        return res

    def changeRGB2CMYK(self, img, nil):
        assert 0 < nil
        rgb_profile_path = 'data/profile/RGB Profiles/'
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
        cmyk_profile_path = 'data/profile/CMYK Profiles/'
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
        # print(rgb_p2, cmyk_p2)
        cmyk2rgb_trans = ImageCms.buildTransformFromOpenProfiles(cmyk_p2, rgb_p2, "CMYK", "RGB")
        img = img.convert('CMYK')
        res = ImageCms.applyTransform(img, cmyk2rgb_trans)
        # res.save('./output/test/CMYK.png')
        return res

    def addHalftone(self, img, a):
        assert 0.0 < a <= 0.5
        h = img.size[0]
        w = img.size[1]
        shrink = cv2.resize(np.asarray(img), (int(h/3),int(w/3)))
        ht = halftone(shrink)
        ht = Image.fromarray(np.uint8(ht)).convert('RGBA')
        ht = ht.resize((h,w))
        img = img.convert('RGBA')
        comb_rgba = Image.blend(img, ht, a) # 0.05-0.4
        res = comb_rgba.convert('RGB') 
        # res.save('./output/test/H.png')
        return res

    def addMotionBlur(self, img, ks):
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
        # print('MB: ',np.max(img_motionBlur))
        res = Image.fromarray(np.uint8(img_motionBlur))
        # res.save('./output/test/MB.png')
        return res

    def addQualityLose(self, img, sr):
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
        # res.save('./output/test/QL.png')
        return res

    def original (self, img, _):
        res = img
        # res.save('./output/test/original.png')
        return res
    
    def ShearX(self, img, v):  # [-0.3, 0.3]
        assert -0.3 <= v <= 0.3
        if self.random_mirror and random.random() > 0.5:
            v = -v
        return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


    def ShearY(self, img, v):  # [-0.3, 0.3]
        assert -0.3 <= v <= 0.3
        if self.random_mirror and random.random() > 0.5:
            v = -v
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


    def TranslateX(self, img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
        assert -0.45 <= v <= 0.45
        if self.random_mirror and random.random() > 0.5:
            v = -v
        v = v * img.size[0]
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


    def TranslateY(self, img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
        assert -0.45 <= v <= 0.45
        if self.random_mirror and random.random() > 0.5:
            v = -v
        v = v * img.size[1]
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


    def TranslateXAbs(self, img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
        assert 0 <= v <= 10
        if random.random() > 0.5:
            v = -v
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


    def TranslateYAbs(self, img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
        assert 0 <= v <= 10
        if random.random() > 0.5:
            v = -v
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


    def Rotate(self, img, v):  # [-30, 30]
        assert -30 <= v <= 30
        if self.random_mirror and random.random() > 0.5:
            v = -v
        return img.rotate(v)


    def AutoContrast(self, img, _):
        return PIL.ImageOps.autocontrast(img)


    def Invert(self, img, _):
        return PIL.ImageOps.invert(img)


    def Equalize(self, img, _):
        return PIL.ImageOps.equalize(img)


    def Flip(self, img, _):  # not from the paper
        return PIL.ImageOps.mirror(img)


    def Solarize(self, img, v):  # [0, 256]
        assert 0 <= v <= 256
        return PIL.ImageOps.solarize(img, v)


    def Posterize(self, img, v):  # [4, 8]
        assert 4 <= v <= 8
        v = int(v)
        return PIL.ImageOps.posterize(img, v)


    def Posterize2(self, img, v):  # [0, 4]
        assert 0 <= v <= 4
        v = int(v)
        return PIL.ImageOps.posterize(img, v)


    def Contrast(self, img, v):  # [0.1,1.9]
        assert 0.1 <= v <= 1.9
        return PIL.ImageEnhance.Contrast(img).enhance(v)


    def Color(self, img, v):  # [0.1,1.9]
        assert 0.1 <= v <= 1.9
        return PIL.ImageEnhance.Color(img).enhance(v)


    def Brightness(self, img, v):  # [0.1,1.9]
        assert 0.1 <= v <= 1.9
        return PIL.ImageEnhance.Brightness(img).enhance(v)


    def Sharpness(self, img, v):  # [0.1,1.9]
        assert 0.1 <= v <= 1.9
        return PIL.ImageEnhance.Sharpness(img).enhance(v)


    def Cutout(self, img, v):  # [0, 60] => percentage: [0, 0.2]
        assert 0.0 <= v <= 0.2
        if v <= 0.:
            return img

        v = v * img.size[0]
        return self.CutoutAbs(img, v)

    def CutoutAbs(self, img, v):  # [0, 60] => percentage: [0, 0.2]
        # assert 0 <= v <= 20
        if v < 0:
            return img
        w, h = img.size
        x0 = np.random.uniform(w)
        y0 = np.random.uniform(h)

        x0 = int(max(0, x0 - v / 2.))
        y0 = int(max(0, y0 - v / 2.))
        x1 = min(w, x0 + v)
        y1 = min(h, y0 + v)

        xy = (x0, y0, x1, y1)
        color = (125, 123, 114)
        # color = (0, 0, 0)
        img = img.copy()
        PIL.ImageDraw.Draw(img).rectangle(xy, color)
        return img        
        
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
#     res = addQualityLose(input_pic, 0.5)
#     res.save('./test/LQ.png')