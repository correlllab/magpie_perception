import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from magpie_perception import pcd

def label_wrist_image(wrist_img):
    '''
    labels wrist image with overlaid x,y,z axes in line with F/T sensor
    '''
    h, w, c = wrist_img.shape

    x_color = (0, 255, 0) # green
    y_color = (0, 0, 255) # blue
    z_color = (255, 0, 0) # red

    # Define axis line lengths (adjust as needed)
    axis_length = min(w, h) // 3

    # Calculate starting point for axes (center of the image)
    center_x, center_y = w // 2, h // 2

    # Draw X-axis
    cv2.arrowedLine(wrist_img, (center_x, center_y), (center_x + axis_length, center_y), x_color, 6)
    cv2.putText(wrist_img, "+X", (center_x + axis_length - 15, center_y -20), cv2.FONT_HERSHEY_SIMPLEX, 2, x_color, 6)

    # Draw Y-axis
    cv2.arrowedLine(wrist_img, (center_x, center_y), (center_x, center_y - axis_length), y_color, 6)
    cv2.putText(wrist_img, "-Y", (center_x - 15, center_y - axis_length - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, y_color, 6)

    # Draw Z-axis (pointing towards the viewer, represented as a circle)
    cv2.circle(wrist_img, (center_x, center_y), 9, z_color, -1)  # Filled circle
    cv2.putText(wrist_img, "+Z", (center_x - 45, center_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 2, z_color, 6)

    return wrist_img

def find_object(rgbd_img, query, label_model, mask_model, seg_type, wrist_camera):
    img = np.array(rgbd_img.color)
    label_model.TOP_K = 1
    label_model.label(img, query, query, topk=True, plot=False)
    index = 0
    boxes = []
    if seg_type == "box-dbscan":
        boxes = label_model.sorted_labeled_boxes_coords
        pred_image = label_model.preds_plot
    elif seg_type == "mask":
        mask_sam2 = mask_model
        mask_sam2.set_image_and_labels(np.array(rgbd_img.color), 
                                        label_model.sorted_boxes_coords, 
                                        label_model.sorted_labels)
        masks = mask_sam2.get_masks(label_model.sorted_labels)
        mask_sam2.plot_image(rgbd_img.color, masks[:3], 
                                label_model.sorted_boxes_coords[:3],
                                label_model.sorted_scores, show_plot=False)
        pred_image = mask_sam2.preds_plot
        boxes = masks.astype(bool)
    print(boxes)
    print(type(rgbd_img))
    _, ptcld, GOAL_POSE, _ = pcd.get_segment(boxes, 
                                        index, 
                                        rgbd_img, 
                                        wrist_camera, 
                                        type=seg_type, 
                                    #  type="box", 
                                    #  method="quat", 
                                        method="iterative", 
                                        display=False)
    
    return GOAL_POSE, pred_image
