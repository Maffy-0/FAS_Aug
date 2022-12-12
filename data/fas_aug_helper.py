import os
import random
from PIL import Image
import numpy as np
import cv2

# for Reflection, Blue Noise, Moire Pattern
def get_background_dict(backDirPath, backNameList):
    backDict = {}
    for name in backNameList:
        backPath = os.path.join(backDirPath, name)
        backDict[name] = Image.open(backPath).convert('RGB')
    return backDict

R_DIRPATH = "data/background/"
R_NAME_LIST = os.listdir(R_DIRPATH)
R_DICT = get_background_dict(R_DIRPATH, R_NAME_LIST)

BN_DIRPATH = "data/noiseTexture/"
BN_NAME_LIST = os.listdir(BN_DIRPATH)
BN_DICT = get_background_dict(BN_DIRPATH, BN_NAME_LIST)

MP_DIRPATH = "data/MPTexture/"
MP_NAME_LIST = os.listdir(MP_DIRPATH)
MP_DICT = get_background_dict(MP_DIRPATH, MP_NAME_LIST)

def backgroundCrop(b_img,img_size,brmp):
    scale = 1
    if brmp == 'R': scale = 2
    if brmp == 'MP': scale = 3
    if b_img.size[0] > scale*img_size[0]:
        h = random.randint(0, b_img.size[0] - scale*img_size[0])
        b_img = b_img.crop((h, 0, scale*img_size[0]+h, b_img.size[1]))   
    if b_img.size[1] > scale*img_size[1]:
        w = random.randint(0, b_img.size[1] - scale*img_size[1])
        b_img = b_img.crop((0, w, b_img.size[0], scale*img_size[1]+w))
    return b_img.resize(img_size)

def getTexture(brmp, img_size):
    background_nameList_dir = {
            "R" : R_NAME_LIST,
            "BN" : BN_NAME_LIST,
            "MP" : MP_NAME_LIST
    }
    texture_dir = {
            "R" : R_DICT,
            "BN" : BN_DICT,
            "MP" : MP_DICT
    }
    b_name = random.choice(background_nameList_dir[brmp])
    b_img = texture_dir[brmp][b_name].convert('RGBA')
    b_img_crop = backgroundCrop(b_img,img_size,brmp)
    return b_img_crop
#

# Halftone
def get_image_range(image):
    return np.min(image), np.max(image)

def adjust_gray(image, new_min, new_max):
    image_min, image_max = get_image_range(image)
    h, w  = image.shape
    # print(image_min, image_max)
    adjusted = np.zeros((h, w))
    adjusted = (image - image_min)*((new_max - new_min)/(image_max - image_min)) + new_min
    return adjusted.astype(np.uint8)

def gen_halftone_masks():
    m = np.zeros((3, 3, 10))
    m[:, :, 1] = m[:, :, 0]
    m[0, 1, 1] = 1
    m[:, :, 2] = m[:, :, 1]
    m[2, 2, 2] = 1
    m[:, :, 3] = m[:, :, 2]
    m[0, 0, 3] = 1
    m[:, :, 4] = m[:, :, 3]
    m[2, 0, 4] = 1
    m[:, :, 5] = m[:, :, 4]
    m[0, 2, 5] = 1
    m[:, :, 6] = m[:, :, 5]
    m[1, 2, 6] = 1
    m[:, :, 7] = m[:, :, 6]
    m[2, 1, 7] = 1
    m[:, :, 8] = m[:, :, 7]
    m[1, 0, 8] = 1
    m[:, :, 9] = m[:, :, 8]
    m[1, 1, 9] = 1
    return m

def halftone(image):
    gray      = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adjusted  = adjust_gray(gray, 0, 9)
    m         = gen_halftone_masks()
    # print(image.shape)
    height, width, channel = image.shape


    halftoned = np.zeros((3*height, 3*width))
    for j in range(height):
        for i in range(width):
            index = adjusted[j, i]
            halftoned[3*j:3+3*j, 3*i:3+3*i] = m[:, :, index]

    halftoned = 255*halftoned
    return np.asarray(halftoned, dtype='uint8' )