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

# 現在のファイルの位置を基準として相対パスを設定
current_dir = os.path.dirname(os.path.abspath(__file__))

R_DIRPATH = os.path.join(current_dir, "background")
if os.path.exists(R_DIRPATH):
    R_NAME_LIST = os.listdir(R_DIRPATH)
    R_DICT = get_background_dict(R_DIRPATH, R_NAME_LIST)
else:
    R_NAME_LIST = []
    R_DICT = {}

BN_DIRPATH = os.path.join(current_dir, "noiseTexture")
if os.path.exists(BN_DIRPATH):
    BN_NAME_LIST = os.listdir(BN_DIRPATH)
    BN_DICT = get_background_dict(BN_DIRPATH, BN_NAME_LIST)
else:
    BN_NAME_LIST = []
    BN_DICT = {}

MP_DIRPATH = os.path.join(current_dir, "MPTexture")
if os.path.exists(MP_DIRPATH):
    MP_NAME_LIST = os.listdir(MP_DIRPATH)
    MP_DICT = get_background_dict(MP_DIRPATH, MP_NAME_LIST)
else:
    MP_NAME_LIST = []
    MP_DICT = {}

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
    
    # テクスチャファイルが存在しない場合の対処
    if not background_nameList_dir[brmp]:
        # 単色のテクスチャを生成
        print(f"警告: {brmp} テクスチャが見つかりません。デフォルトテクスチャを使用します。")
        if brmp == "R":  # Reflection - 薄いグレー
            color = (200, 200, 200, 128)
        elif brmp == "BN":  # Blue Noise - ランダムノイズ
            noise = np.random.randint(0, 256, (img_size[1], img_size[0], 3))
            return Image.fromarray(noise.astype(np.uint8)).convert('RGBA')
        else:  # MP - Moire Pattern - 薄い白
            color = (240, 240, 240, 64)
        
        return Image.new('RGBA', img_size, color)
    
    b_name = random.choice(background_nameList_dir[brmp])
    b_img = texture_dir[brmp][b_name].convert('RGBA')
    b_img_crop = backgroundCrop(b_img,img_size,brmp)
    return b_img_crop
#

# SFC-Halftone
def get_image_range(image):
    return np.min(image), np.max(image)

def adjust_gray(image, new_min, new_max):
    image_min, image_max = get_image_range(image)
    h, w  = image.shape
    # print(image_min, image_max)
    adjusted = np.zeros((h, w))
    
    # ゼロ除算を防ぐための対策
    if image_max - image_min == 0:
        # 全ピクセルが同じ値の場合、中間値で埋める
        adjusted.fill((new_min + new_max) / 2)
    else:
        adjusted = (image - image_min)*((new_max - new_min)/(image_max - image_min)) + new_min
    
    # NaN/Infチェックと修正
    adjusted = np.nan_to_num(adjusted, nan=(new_min + new_max) / 2, 
                           posinf=new_max, neginf=new_min)
    
    # 値の範囲をクリップ
    adjusted = np.clip(adjusted, new_min, new_max)
    
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

def sfc_halftone(image):
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