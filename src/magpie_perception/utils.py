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

    x_color = (255, 0, 0) # red
    y_color = (0, 255, 0) # green
    z_color = (0, 0, 255) # blue

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

def label_wrist_image_rotation_axes(wrist_img, rotation_matrix, camera_intrinsics, axis_length=0.1):
    '''
    Overlays 3D coordinate axes (X, Y, Z) onto an image using the provided rotation matrix
    and camera intrinsics.

    Parameters:
        wrist_img (np.ndarray): The input image.
        rotation_matrix (np.ndarray): A 3x3 rotation matrix.
        camera_intrinsics (np.ndarray): A 3x3 camera intrinsics matrix.
        axis_length (float): Length of the axis vectors in meters or arbitrary units.

    Returns:
        np.ndarray: The labeled image.
    '''
    h, w, _ = wrist_img.shape

    # Define axis colors: X (green), Y (blue), Z (red)
    colors = {
        'x': (0, 255, 0),
        'y': (0, 0, 255),
        'z': (255, 0, 0)
    }

    def compute_center_3d(intrinsics, image_shape, depth=0.1):
        h, w = image_shape[:2]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]

        # Image center pixel coordinates
        u, v = w // 2, h // 2

        # Back-project pixel (u, v) at given depth
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        return np.array([[x, y, z]], dtype=np.float32).T  # shape (3, 1)

    # Center point in 3D (e.g., origin in camera frame)
    origin_3d = compute_center_3d(camera_intrinsics, wrist_img.shape, depth=0.1)

    # Define axis endpoints in 3D using the rotation matrix
    axes_3d = {
        'x': origin_3d + rotation_matrix @ np.array([[axis_length, 0, 0]]).T,
        'y': origin_3d + rotation_matrix @ np.array([[0, axis_length, 0]]).T,
        'z': origin_3d + rotation_matrix @ np.array([[0, 0, axis_length]]).T,
    }

    # Project 3D points to 2D using the camera intrinsics
    def project(point_3d):
        point_cam = point_3d
        # if point_cam[2] <= 1e-6:
        #     return None  # or (0, 0), or raise an exception
        point_proj = camera_intrinsics @ point_cam
        point_proj /= point_proj[2]
        return int(point_proj[0]), int(point_proj[1])

    center_2d = project(origin_3d)

    for axis, end_3d in axes_3d.items():
        end_2d = project(end_3d)
        cv2.arrowedLine(wrist_img, center_2d, end_2d, colors[axis], 6, tipLength=0.1)
        label_pos = (end_2d[0] + 5, end_2d[1] + 5)
        cv2.putText(wrist_img, f"+{axis.upper()}", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors[axis], 4)

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
