import os

from yacs.config import CfgNode as CN


_C = CN()
_C.MODEL = CN()
_C.MODEL.NAME = "default"
_C.MODEL.DEVICE = "cuda"

_C.MODEL.NUM_LAYERS = 18 # resnet layers
# _C.MODEL.WEIGHTS_INIT = "pretrained" # pretrained or scratch

_C.MODEL.SCALES = [0,1,2,3]

# Options
# _C.MODEL.USE_STEREO = False
# _C.MODEL.USE_PREDICTIVE_MASK = False
# _C.MODEL.DISABLE_AUTOMASKING = False
# _C.MODEL.USE_SSIM = True


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.HEIGHT = 192
_C.INPUT.WIDTH = 640

_C.INPUT.FRAME_IDS = [0,-1,1] # from 'main'
_C.INPUT.CAM_IDS = [] # stereo, cam0, cam1, ...


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 16


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.SCHEDULER_STEP_SIZE = 15
_C.SOLVER.NUM_EPOCHS = 10

_C.SOLVER.DISPARITY_SMOOTHNESS = 1e-3
_C.SOLVER.MIN_DEPTH = 0.1
_C.SOLVER.MAX_DEPTH = 100


# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 16


# ---------------------------------------------------------------------------- #
# Test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 4


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "./output"