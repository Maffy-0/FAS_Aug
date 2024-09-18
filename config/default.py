from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------#
# Init
# -----------------------------------------------------------------------------#
_C = CN(new_allowed=True)
_C.OUTPUT_DIR = 'output/train'
_C.RESULT_FILE = './output/csvResult.csv'
_C.DEBUG = False
_C.SEED = 666
_C.CUDA = True
_C.NOTES = '' # Any comments. Can help remember what the experiment is about.


_C.MODEL = CN(new_allowed=True)
_C.DATA = CN(new_allowed=True)
_C.TRAIN = CN(new_allowed=True)
_C.TEST = CN(new_allowed=True)


# ---------------------------------------------------------------------------- #
# Data config
# ---------------------------------------------------------------------------- #

_C.DATA.VAL = ''

_C.DATA.TRAIN_LIST = ['data/data_list/datalist/CASIA-ALL.csv',
                'data/data_list/datalist/MSU-MFSD-ALL.csv',
                'data/data_list/datalist/REPLAY-TRAIN.csv',
                ]

_C.DATA.TEST = 'data/data_list/datalist/OULU-NPU-ALL.csv'       

_C.DATA.EXTRA_DOMAIN = 3

_C.DATA.IN_SIZE = 256 # Input image size
_C.DATA.DATASET = 'ZipDataset'
_C.DATA.ROOT_DIR = '/home/rizhao/data/FAS/all_public_datasets_zip/' 
_C.DATA.SUB_DIR = 'EXT0.0'
_C.DATA.NUM_FRAMES = 5 # number of frames extracted from a video
_C.DATA.BATCH_SIZE = 2
_C.DATA.NUM_WORKERS = 10
_C.DATA.LABEL_SMOOTHING = 0.0

_C.DATA.NORMALIZE = CN(new_allowed=True)
_C.DATA.NORMALIZE.ENABLE = False
_C.DATA.NORMALIZE.MEAN = [0.485, 0.456, 0.406]
_C.DATA.NORMALIZE.STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------- #
# MODEL/NETWORK  config
# ---------------------------------------------------------------------------- #
_C.MODEL.NUM_CLASSES = 2

# ---------------------------------------------------------------------------- #
# MODEL/NETWORK  config
# ---------------------------------------------------------------------------- #
_C.TRAIN.RESUME = '' # Path to the resume ckpt
_C.TRAIN.INIT_LR = 1e-4 # 1e-3 for --trainer bc , 1e-4 for --trainer vit_convpass/vit_adapter
_C.TRAIN.BETA = 10.0
_C.TRAIN.ALPHA = 0.02
_C.TRAIN.ITERATION = 0

_C.TRAIN.EPOCHS = 200
_C.TRAIN.MOMENTUM = 0.0
_C.TRAIN.LR_PATIENCE = 0
_C.TRAIN.PATIENCE = 20
_C.TRAIN.SAVE_BEST = True 

# ---------------------------------------------------------------------------- #
# Augmentation
# ---------------------------------------------------------------------------- #
_C.TRAIN.AUG = CN(new_allowed=True) # Augmentation
_C.TRAIN.AUG.SAVE = False
_C.TRAIN.AUG.NUM_OPS = 8
_C.TRAIN.AUG.NUM_MAG = 10
_C.TRAIN.AUG.NUM_POLICIES = 1 # M
_C.TRAIN.AUG.NUM_SUBPOLICIES = 5 # Q
_C.TRAIN.AUG.BAG = 'FAS_Augmentations' #'MIX_Augmentations', 'TRAD_Augmentations' use different algorithm for computing loss and data loadiung, so which is not runable at this code.

# TEST Config
_C.TEST.CKPT = '' # checkpoint to load
_C.TEST.TAG = 'test'
_C.TEST.NO_INFERENCE = False # Load metrics from TEST.OUTPUT_DIR and conduct testing
_C.TEST.THR = 0.5 # Threshold for calculating HTER




def get_cfg_defaults():
    '''Get a yacs CfgNode object with default values for my_project.'''
    # Return a clone so that the defaults will not be altered
    # This is for the 'local variable' use pattern
    return _C.clone()