'''
@file mask_sam.py
@brief Segment Anything Model (SAM) implementaion of mask.py
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
sys.path.append("../../")
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import copy
import cv2
import numpy as np
from magpie_perception.mask import Mask
# from magpie_perception.mask import Mask

class MaskSAM2(Mask):
    def __init__(self, ckpt=None):
        super().__init__()
        # self.sam = build_sam()
        # self.predictor = SamPredictor(self.sam)
        if ckpt is None:
            ckpt = "facebook/sam2.1-hiera-large"
        self.predictor = SAM2ImagePredictor.from_pretrained(ckpt, device="cpu")
        self.dims = None
        self.H = None
        self.W = None
        self.boxes = None
        self.labels = None
        self.pred_dict = None
        self.image = None
        self.image_cv2 = None
        self.masks = None

    def set_image_and_labels(self, image, boxes, labels):
        self.image = np.array(image)
        self.W, self.H = self.image.shape[:2][::-1] # TODO: check if this is correct
        self.dims = [self.H, self.W]
        self.boxes = boxes
        self.labels = labels
        self.pred_dict = {
            "boxes": boxes,
            "size": self.dims, # H, W
            "labels": labels      
        }
        # don't know why colors aren't correct without this prior manual rotation
        cv2_array = image[:, :, ::-1].copy()
        self.image_cv2 = cv2.cvtColor(cv2_array, cv2.COLOR_RGB2BGR)
        self.predictor.set_image(self.image_cv2)

    def get_masks(self, labels):
        boxes_copy = copy.deepcopy(self.boxes)
        boxes = boxes_copy[:10]
        # transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes_copy, self.image.shape[:2][::-1])
        masks, _, _ = self.predictor.predict(
            point_coords = None,
            point_labels = None,
            box = boxes,
            multimask_output = False,
        )
        self.masks = masks
        return masks
    
    def show_mask(self, mask, ax, random_color=False):
        '''
        @param mask bool image mask
        @param ax matplotlib fig
        '''
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
    
    def show_all_masks(self, image):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for mask in self.masks:
            self.show_mask(mask, plt.gca(), random_color=True)
        plt.axis('off')