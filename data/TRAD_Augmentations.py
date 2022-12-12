# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
import os
from torchvision.transforms.transforms import Compose

class TRAD_Augmentations(object):
    def __init__(self, config):
        self.al = [
        ("ShearX", 0, 0.3),  # 0
        ("ShearY", 0, 0.3),  # 1
        ("TranslateX", 0, 0.45),  # 2
        ("TranslateY", 0, 0.45),  # 3
        ("Rotate", 0, 30),  # 4
        ("AutoContrast", 0, 1),  # 5
        ("Invert", 0, 1),  # 6
        ("Equalize", 0, 1),  # 7
        ("Solarize", 0, 256),  # 8
        ("Posterize", 4, 8),  # 9
        ("Contrast", 0.1, 1.9),  # 10
        ("Color", 0.1, 1.9),  # 11
        ("Brightness", 0.1, 1.9),  # 12
        ("Sharpness", 0.1, 1.9),  # 13
        ("Cutout", 0, 0.2),  # 14
    ]
        self.name_dict = {
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
        self.augment_dict = {}
        for i in self.al:
            self.augment_dict[i[0]] = (self.name_dict[i[0]], i[1],i[2],False)   
        # print(self.augment_dict) 
        self.random_mirror = True
    
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

    def apply_augment(self, img, name, level,epoch):
        augment_fn, low, high = self.get_augment(name)
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
        return res

# if __name__ == '__main__':
#     input_pic_path = '/home/rizhao/projects/Cecelia/a-transform/input/6.png'
#     input_pic = Image.open(input_pic_path)
#     res = addQualityLose(input_pic, 0.5)
#     res.save('./test/LQ.png')