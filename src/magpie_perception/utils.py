import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from magpie_perception import pcd
import itertools
from spatialmath import SO3
from scipy.spatial.transform import Rotation as R

def optimize_and_correct_frame(R_input, step_angle=90):
    """
    Optimize rotation to align frame to canonical and return corrected frame while maintaining orientation 'essence' of input.
    Canonical frame being the egocentric world frame in which +Z is up in the world, +X is right in the world, +Y is forward in the world.

    Args:
        R_input (np.ndarray): 3x3 rotation matrix representing arbitrary frame.
        step_angle (int): 90 (default) or 180 degrees allowed.

    Returns:
        np.ndarray: Corrected 3x3 rotation matrix (after applying best correction).
    """
    def frame_alignment_error(R_frame):
        R_err = R_frame.T
        return R.from_matrix(R_err).magnitude()

    angles = [np.pi] if step_angle == 180 else [np.pi/2, -np.pi/2, np.pi]
    moves = [(axis, angle) for axis in range(3) for angle in angles]

    best_error = np.inf
    best_rotation = np.eye(3)

    for n in [1,2,3]:
        for seq in itertools.product(moves, repeat=n):
            R_total = np.eye(3)
            for axis, angle in seq:
                if axis == 0: R_step = SO3.Rx(angle).A
                elif axis == 1: R_step = SO3.Ry(angle).A
                else: R_step = SO3.Rz(angle).A
                R_total = R_total @ R_step
            R_candidate = R_input @ R_total
            err = frame_alignment_error(R_candidate)
            if err < best_error:
                best_error = err
                best_rotation = R_total

    R_corrected = R_input @ best_rotation
    return R_corrected, best_rotation

def plot_frames_3d(R_input, R_corrected, R_best_rotation):
    """Plot canonical, input, corrected, and corrected-inverse frames."""
    def plot_frame(ax, R_frame, label, origin, scale=0.2):
        colors = ['r', 'g', 'b']
        for i in range(3):
            vec = R_frame[:,i] * scale
            ax.quiver(*origin, *vec, color=colors[i])
        ax.text(*origin, label, fontsize=10)

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    # Frames
    plot_frame(ax, np.eye(3), 'Canonical', origin=np.array([0,0,0]))
    plot_frame(ax, R_input, 'Original', origin=np.array([0.5,0,0]))
    plot_frame(ax, R_corrected, 'Corrected', origin=np.array([1.0,0,0]))

    R_recovered = R_corrected @ R_best_rotation.T
    plot_frame(ax, R_recovered, 'Recovered', origin=np.array([1.5,0,0]))

    ax.set_xlim([-0.5,2.0])
    ax.set_ylim([-1.5,1.5])
    ax.set_zlim([-1.5,1.5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Frame Alignment Visualization')
    plt.tight_layout()
    plt.show()

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

def label_wrist_image_rotation_axes(wrist_img, rotation_matrix, camera_intrinsics, axis_length=0.1, origin=None):
    '''
    Overlays 3D coordinate axes (X, Y, Z) onto an image using the provided rotation matrix
    and camera intrinsics.

    Parameters:
        wrist_img (np.ndarray): The input image.
        rotation_matrix (np.ndarray): A 3x3 rotation matrix.
        camera_intrinsics (np.ndarray): A 3x3 camera intrinsics matrix.
        axis_length (float): Length of the axis vectors in meters or arbitrary units.
        origin (tuple or None): Optional (x, y) pixel coordinates to plot the axes. If None, uses image center.

    Returns:
        np.ndarray: The labeled image.
    '''
    h, w, _ = wrist_img.shape

    # Define axis colors: X (red), Y (green), Z (blue)
    colors = {
        'x': (255, 0, 0),
        'y': (0, 255, 0),
        'z': (0, 0, 255),
    }

    def pixel_to_3d(u, v, intrinsics, depth=0.1):
        """Back-project 2D pixel (u,v) to 3D point assuming fixed depth."""
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]

        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        return np.array([[x, y, z]], dtype=np.float32).T  # shape (3, 1)

    # Choose center pixel
    if origin is not None:
        u, v = origin  # User-specified origin (already pixel coordinates)
    else:
        u, v = w // 2, h // 2  # Image center

    # Compute the 3D point corresponding to the pixel
    origin_3d = pixel_to_3d(u, v, camera_intrinsics, depth=0.1)

    # Define axis endpoints in 3D using the rotation matrix
    axes_3d = {
        'x': origin_3d + rotation_matrix @ np.array([[axis_length, 0, 0]]).T,
        'y': origin_3d + rotation_matrix @ np.array([[0, axis_length, 0]]).T,
        'z': origin_3d + rotation_matrix @ np.array([[0, 0, axis_length]]).T,
    }

    def project(point_3d):
        """Projects a 3D point to 2D pixel coordinates."""
        point_proj = camera_intrinsics @ point_3d
        point_proj /= point_proj[2]
        return int(point_proj[0]), int(point_proj[1])

    center_2d = (int(u), int(v))  # We already know pixel center

    # Precompute projected endpoints and their lengths
    axis_endpoints = []
    for axis, end_3d in axes_3d.items():
        end_2d = project(end_3d)
        dx = end_2d[0] - center_2d[0]
        dy = end_2d[1] - center_2d[1]
        length = np.hypot(dx, dy)  # Euclidean distance
        axis_endpoints.append((length, axis, end_2d))

    # Sort by length, longer first, shorter last
    axis_endpoints.sort(reverse=True)

    for _, axis, end_2d in axis_endpoints:
        cv2.arrowedLine(wrist_img, center_2d, end_2d, colors[axis], 6, tipLength=0.1)
        label_pos = (end_2d[0] + 5, end_2d[1] + 5)
        cv2.putText(wrist_img, f"+{axis.upper()}", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors[axis], 4)
    # for axis, end_3d in axes_3d.items():
    #     end_2d = project(end_3d)
    #     cv2.arrowedLine(wrist_img, center_2d, end_2d, colors[axis], 6, tipLength=0.1)
    #     label_pos = (end_2d[0] + 5, end_2d[1] + 5)
    #     cv2.putText(wrist_img, f"+{axis.upper()}", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors[axis], 4)

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
