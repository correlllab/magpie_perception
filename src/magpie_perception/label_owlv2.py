'''
@file label_owlv2.py
@brief OWLv2 implementation of label.py
'''
import sys
sys.path.append("../")
import torch
import numpy as np
from magpie_perception.label import Label
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

class LabelOWLv2(Label):
    def __init__(self, topk=3, score_threshold=0.005, pth="google/owlv2-base-patch16-ensemble", cpu_override=True):
        '''
        @param camera camera object, expects realsense_wrapper
        '''
        super().__init__(cpu_override=cpu_override)
        self.processor = Owlv2Processor.from_pretrained(pth)
        self.model = Owlv2ForObjectDetection.from_pretrained(pth, device_map=self.device)
        self.SCORE_THRESHOLD = score_threshold
        self.TOP_K = topk


    def scale_thresh_by_factor( self, factor ):
        """ Adjust the threshold by some factor """
        self.SCORE_THRESHOLD *= factor
        return self.SCORE_THRESHOLD


    def get_preds(self, outputs, target_sizes):
        logits = torch.max(outputs["logits"][0], dim=-1)
        scores = torch.sigmoid(logits.values).cpu().detach().numpy()
        # Get prediction labels and boundary boxes
        labels = logits.indices.cpu().detach().numpy()
        self.results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=self.SCORE_THRESHOLD)
        boxes  = self.results[0]['boxes'].cpu().detach().numpy()
        pboxes = self.results[0]['boxes'].cpu().detach().numpy()
        scores = self.results[0]['scores'].cpu().detach().numpy()
        labels = self.results[0]['labels'].cpu().detach().numpy()
        # sort labels by score, high to low
        sorted_indices = np.argsort(scores)[::-1]

        # store member variables
        # cut off score indices below threshold
        self.sorted_indices = sorted_indices[scores[sorted_indices] > self.SCORE_THRESHOLD]
        self.sorted_scores = scores[self.sorted_indices]
        self.sorted_labels = labels[self.sorted_indices]
        self.sorted_text_labels = np.array([self.queries[label] for label in labels[self.sorted_indices]])
        self.sorted_boxes = boxes[self.sorted_indices]
        self.sorted_boxes_coords = boxes[self.sorted_indices]
        self.sorted_labeled_boxes = list(zip(self.sorted_boxes, self.sorted_labels))
        self.sorted_labeled_boxes_coords = list(zip(self.sorted_boxes_coords, self.sorted_labels))
        self.sorted = list(zip(self.sorted_scores, self.sorted_labels, self.sorted_indices, self.sorted_boxes))
        
        return scores, labels, boxes, pboxes

    def label(self, input_image, input_labels, abbrev_labels, topk=False, plot=False):
        '''
        @param input_labels list of input labels
        @param input_image np.array image to label
        @return pboxes list of predicted boxes
        @return uboxes list of unnormalized boxes
        '''
        self.image = img = np.asarray(input_image)
        img_tensor = torch.tensor(img, dtype=torch.float32)
        inputs = self.processor(input_labels, images=img_tensor, padding=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        self.dims = img.shape[:2][::-1] # TODO: check if this is correct
        self.W = self.dims[0]
        self.H = self.dims[1]

        target_sizes = torch.Tensor([[max(self.W,self.H), max(self.W,self.H)]])
        
        self.queries = abbrev_labels
        scores, labels, boxes, pboxes = self.get_preds(outputs, target_sizes)
        self.plot_predictions(topk=topk, show_plot=plot)
        return self.results, self.sorted_boxes_coords, self.sorted_scores, self.sorted_labels

