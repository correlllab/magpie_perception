'''
@file mask.py
@brief segmentation wrapper to retrieve object mask(s), given image and input labels
'''
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

    def plot_image(self, image, masks, boxes=None, scores=None, show_plot=True):
        def show_mask(mask, ax, random_color=False):
            if random_color:
                color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            else:
                color = np.array([30/255, 144/255, 255/255, 0.6])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)

        def show_box(box, ax, conf=None):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, f'{idx}: {conf:.2f}', color='red', verticalalignment='top', fontsize="small", 
                bbox={'facecolor': 'white', 'alpha': 0.35, 'pad': 1})

        plt.imshow(image)
        for mask in masks:
            show_mask(mask, plt.gca(), random_color=False)
        if boxes is not None:
            confidences = boxes.copy() if scores is None else scores
            idx = 0
            for box, conf in zip(boxes, confidences):
                show_box(box, plt.gca(), conf=conf if scores is not None else None)
                idx += 1
        plt.axis('off')
        # get figure as np array in RGB format
        fig = plt.gcf()
        fig.canvas.draw()
        predicted_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        predicted_image = predicted_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        self.pred_image = predicted_image
        if not show_plot:
            plt.close()
            plt.close(fig)
