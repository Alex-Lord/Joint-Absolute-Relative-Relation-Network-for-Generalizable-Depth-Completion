    
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import label
# 你需要定义的卷积核
# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_13 = np.ones((13, 13), np.uint8)
FULL_KERNEL_25 = np.ones((25, 25), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

# 3x3 cross kernel
CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)

# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 7x7 cross kernel
CROSS_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

# 7x7 diamond kernel
DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

DIAMOND_KERNEL_9 = np.asarray(
    [
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
    ], dtype=np.uint8)

DIAMOND_KERNEL_13 = np.asarray(
    [
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8)

def outlier_removal(lidar):
    # sparse_lidar = np.squeeze(lidar)
    threshold = 1.0
    sparse_lidar = lidar
    valid_pixels = (sparse_lidar > 0.1).astype(np.float)

    lidar_sum_7 = cv2.filter2D(sparse_lidar, -1, DIAMOND_KERNEL_7)
    lidar_count_7 = cv2.filter2D(valid_pixels, -1, DIAMOND_KERNEL_7)

    lidar_aveg_7 = lidar_sum_7 / (lidar_count_7 + 0.00001)
    potential_outliers_7 = ((sparse_lidar - lidar_aveg_7) > threshold).astype(np.float)

    lidar_sum_9 = cv2.filter2D(sparse_lidar, -1, DIAMOND_KERNEL_9)
    lidar_count_9 = cv2.filter2D(valid_pixels, -1, DIAMOND_KERNEL_9)

    lidar_aveg_9 = lidar_sum_9 / (lidar_count_9 + 0.00001)
    potential_outliers_9 = ((sparse_lidar - lidar_aveg_9) > threshold).astype(np.float)

    lidar_sum_13 = cv2.filter2D(sparse_lidar, -1, DIAMOND_KERNEL_13)
    lidar_count_13 = cv2.filter2D(valid_pixels, -1, DIAMOND_KERNEL_13)

    lidar_aveg_13 = lidar_sum_13 / (lidar_count_13 + 0.00001)
    potential_outliers_13 = ((sparse_lidar - lidar_aveg_13) > threshold).astype(np.float)

    potential_outliers = potential_outliers_7 + potential_outliers_9 + potential_outliers_13
    lidar_cleared = (sparse_lidar * (1 - potential_outliers)).astype(np.float32)
    # lidar_cleared = np.expand_dims(lidar_cleared, -1)

    # potential_outliers = np.expand_dims(potential_outliers, -1)

    return lidar_cleared, potential_outliers

def fill_in_fast_tensor(depth_tensor, max_depth=1.0, custom_kernel=DIAMOND_KERNEL_5,
                        extrapolate=False, blur_type='bilateral'):
    """Fast, in-place depth completion for tensor.

    Args:
        depth_tensor: projected depths tensor with shape [b, c=1, h, w]
        max_depth: max depth value for inversion
        custom_kernel: kernel to apply initial dilation
        extrapolate: whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'bilateral' - preserves local structure (recommended)
            'gaussian' - provides lower RMSE

    Returns:
        depth_tensor: dense depth tensor with same shape [b, c=1, h, w]
    """
    depth_tensor_np = depth_tensor.squeeze(1).cpu().numpy()  # Convert to numpy array with shape [b, h, w]

    for i in range(depth_tensor_np.shape[0]):  # Iterate over the batch
        depth_map = depth_tensor_np[i]

        valid_pixels = (depth_map > 0.1)
        depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

        depth_map = cv2.dilate(depth_map, custom_kernel)

        depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

        empty_pixels = (depth_map < 0.1)
        dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
        depth_map[empty_pixels] = dilated[empty_pixels]

        if extrapolate:
            top_row_pixels = np.argmax(depth_map > 0.1, axis=0)
            top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]

            for pixel_col_idx in range(depth_map.shape[1]):
                depth_map[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = \
                    top_pixel_values[pixel_col_idx]

        empty_pixels = depth_map < 0.1
        dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
        depth_map[empty_pixels] = dilated[empty_pixels]

        depth_map = depth_map.astype('float32')
        depth_map = cv2.medianBlur(depth_map, 5)
        depth_map = depth_map.astype('float64')

        if blur_type == 'bilateral':
            depth_map = depth_map.astype('float32')
            depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
            depth_map = depth_map.astype('float64')
        elif blur_type == 'gaussian':
            valid_pixels = (depth_map > 0.1)
            blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
            depth_map[valid_pixels] = blurred[valid_pixels]

        valid_pixels = (depth_map > 0.1)
        depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

        mask = (depth_map <= 0.1)
        if np.sum(mask) != 0:
            labeled_array, num_features = label(mask)
            for j in range(num_features):
                index = j + 1
                m = (labeled_array == index)
                m_dilate1 = cv2.dilate(1.0 * m, FULL_KERNEL_7)
                m_dilate2 = cv2.dilate(1.0 * m, FULL_KERNEL_13)
                m_diff = m_dilate2 - m_dilate1
                v = np.mean(depth_map[m_diff > 0])
                depth_map = np.ma.array(depth_map, mask=m_dilate1, fill_value=v)
                depth_map = depth_map.filled()
                depth_map = np.array(depth_map)

        depth_tensor_np[i] = depth_map

    depth_tensor_filled = torch.from_numpy(depth_tensor_np).unsqueeze(1).to(depth_tensor.device)

    return depth_tensor_filled
