'''
@file mask.py
@brief segmentation wrapper to retrieve object mask(s), given image and input labels
'''
import os

class Mask:
    def __init__(self):
        self.dims = None
        self.H = None
        self.W = None
        self.boxes = None
        self.labels = None
        self.pred_dict = None

    def __init__(self, ckpt=None, config=None, labels=None, threshold=0.5, device='cuda'):
        pass