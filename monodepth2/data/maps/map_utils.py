import numpy as np

def get_rotation_translation_mat(rotation, translation):
    """
    Get the integrated transformation matrix from the motion data.
    translate from the origin coordinate first,
    then rotate as Rz * Ry * Rx
    :param rotation: in-order of rx, ry, rz
    :param translation: in-order offset of x, y, z
    :return: affine transformation matrix
    """
    rx, ry, rz = rotation
    rot_mat_rx = np.array([[1, 0, 0, 0],
                           [0, np.cos(rx), -np.sin(rx), 0],
                           [0, np.sin(rx), np.cos(rx), 0],
                           [0, 0, 0, 1]])
    rot_mat_ry = np.array([[np.cos(ry), 0, np.sin(ry), 0],
                           [0, 1, 0, 0],
                           [-np.sin(ry), 0, np.cos(ry), 0],
                           [0, 0, 0, 1]])
    rot_mat_rz = np.array([[np.cos(rz), -np.sin(rz), 0., 0],
                           [np.sin(rz), np.cos(rz), 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

    tform = rot_mat_rz.dot(rot_mat_ry).dot(rot_mat_rx)
    tform[:3, 3] = translation
    return tform

def scale_cam_intrinsic(intrinsic, src_shape, dst_shape, override=False):
    """
    scale intrinsic due to image size changed
    :param intrinsic: source intrinsic
    :param src_shape: source image shape
    :param dst_shape: destination image shape
    :return:
    """
    src_w, src_h = src_shape
    dst_w, dst_h = dst_shape
    rx, ry = src_w * 1. / dst_w, src_h * 1. / dst_h
    mat_intr = intrinsic if override else intrinsic.copy()
    mat_intr[0, 0] /= rx
    mat_intr[0, 2] /= rx
    mat_intr[1, 1] /= ry
    mat_intr[1, 2] /= ry
    return mat_intr