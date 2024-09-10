'''
@file label.py
@brief VLM wrapper to retrieve object bounding box and label, given image and input labels
        Additionally, integrate with point cloud data to get 3D bounding boxes
'''
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class Label:
    def __init__(self):
        self.TOP_K = 3
        self.sorted = None
        self.dims = None
        self.H = None
        self.W = None
        self.SCORE_THRESHOLD = 0.01
        self.preds_plot = None
        self.queries = None
        self.results = None
        self.sorted_indices = None
        self.sorted_labels = None
        self.sorted_text_labels = None
        self.sorted_scores = None
        self.sorted_boxes = None
        self.sorted_boxes_coords = None
        self.sorted_labeled_boxes = None
        self.sorted_labeled_boxes_coords = None
        self.sorted = None
        self.boxes = None

    def get_boxes(input_image, text_queries, scores, boxes, labels):
        pass

    def get_top_boxes(self, topk=None):
        if topk is None:
            topk = self.TOP_K
        return self.sorted[:topk]

    def get_index(self, index):
        if index < len(self.sorted):
            return self.sorted[index]
        return None

    def label(self, image, labels):
        pass

    def plot_predictions(self, input_image, text_queries, scores, boxes, labels, topk=False, show_plot=True):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(input_image)
        ax.set_axis_off()

        idx = 0
        if topk:
            scores = self.sorted_scores[:self.TOP_K]
            boxes  = self.sorted_boxes_coords[:self.TOP_K]
            labels = self.sorted_labels[:self.TOP_K] # oops
        for score, box, label in zip(scores, boxes, labels):
            if score < self.SCORE_THRESHOLD and not topk:
                continue
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            label_text = label if type(label) in (str, np.str_) else text_queries[label]
            ax.text(x1, y1, f'{label_text} ({idx}): {score:.3f}', color='red', verticalalignment='top', 
                    bbox={'facecolor': 'white', 'alpha': 0.35, 'pad': 1})

            idx += 1

        fig.canvas.draw()
        predicted_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        predicted_image = predicted_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        self.preds_plot = predicted_image
        if not show_plot: plt.close(fig)

    def get_preds(self, outputs, target_sizes):
        pass
    
    def set_threshold(self, threshold):
        self.SCORE_THRESHOLD = threshold
