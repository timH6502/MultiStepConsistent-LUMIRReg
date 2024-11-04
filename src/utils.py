import logging
import sys

from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.ndimage import map_coordinates
from skimage.morphology import ball

from surface_distance import compute_robust_hausdorff, compute_surface_distances


def init_logger(logger_name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Initialize a logger with the specified name.

    Parameters
    ----------
    logger_name : str
        The name of the logger.
    level : int, default=20 (INFO)
        The logging level.

    Returns
    -------
    logging.Logger
        Logger instance.
    """
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def generate_identity_grid(image_size: tuple[int, int], batch_size: int, device: torch.device = None) -> torch.Tensor:
    """
    Generates a grid that represents the identity deformation.

    Parameters
    ----------
    image_size : tuple[int, int]
        Size of the image to be transformed.
    batch_size : int
        Batch size.
    device : torch.device, default=None
        Device where the grid will be moved to. If None, cuda:0 will be selected if
        cuda is available otherwise defaults to cpu.

    Returns
    -------
    torch.Tensor
        A tensor representing the identity grid.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return F.affine_grid(torch.eye(3, 4, device=device).unsqueeze(0).repeat(batch_size, 1, 1),
                         (batch_size, 1, image_size[0], image_size[1], image_size[2]), align_corners=False)


def dice_coeff(fixed: torch.Tensor, transformed_moving: torch.Tensor) -> torch.Tensor:
    """
    Calculates the dice coefficient between the fixed and transformed moving label tensors.

    This code is from the notebook provided by Mattias.

    Parameters
    ----------
    fixed : torch.Tensor
        Tensor containing the labels of the fixed image.
    transformed_moving : torch.Tensor
        Tensor containing the labels of the transformed moving image.

    Returns
    -------
    torch.Tensor
        Tensor containing the dice coefficients.
    """
    B = transformed_moving.shape[0]
    dice = torch.zeros(1, len(torch.unique(
        transformed_moving)), device=fixed.device)
    for i, label_id in enumerate(torch.unique(transformed_moving).cpu().numpy()):
        label_id = int(label_id)
        fixed_flat = (fixed == label_id).view(B, -1).float()
        transformed_flat = (transformed_moving == label_id).view(B, -1).float()
        intersection = torch.mean(fixed_flat * transformed_flat)
        dice[:, i] = (2 * intersection) / \
            (torch.mean(fixed_flat) + torch.mean(transformed_flat) + 1e-8)
    return dice


def compute_hausdorff_95_distance(fixed_labels: torch.Tensor,
                                  moving_labels: torch.Tensor,
                                  transformed_moving_labels: torch.Tensor,
                                  original_image_size: tuple[int, int, int] = None) -> tuple[float, list[float]]:
    """
    Compute robust hausdorff distances.
    Modification of compute_hd95 from https://github.com/MDL-UzL/L2R/blob/main/evaluation/utils.py

    Parameters
    ----------
    fixed_labels: torch.Tensor
        Tensor containing the labels of the fixed image.
    moving_labels : torch.Tensor
        Tensor containing the labels of the moving image.
    transformed_moving_labels : torch.Tensor
        Tensor containing the transformed moving labels.
    original_image_size: tuple[int, int, int], default=None
        Size of the original images. Given images will be rescaled to this size.
        If None, no rescaling will be performed.

    Returns
    -------
    tuple[float, list[float]]
        Average hausdorff distance and label-wise hausdorff distances.
    """
    hd95 = []
    if original_image_size is not None:
        fixed_labels = torch.nn.Upsample(
            size=original_image_size,
            mode='nearest')(fixed_labels
                            .squeeze()
                            .unsqueeze(0)
                            .unsqueeze(0)).squeeze().numpy()
        moving_labels = torch.nn.Upsample(
            size=original_image_size,
            mode='nearest')(moving_labels
                            .squeeze()
                            .unsqueeze(0)
                            .unsqueeze(0)).squeeze().numpy()
        transformed_moving_labels = torch.nn.Upsample(
            size=original_image_size,
            mode='nearest')(transformed_moving_labels
                            .squeeze()
                            .unsqueeze(0)
                            .unsqueeze(0)).squeeze().numpy()
    else:
        fixed_labels = fixed_labels.squeeze().numpy()
        moving_labels = moving_labels.squeeze().numpy()
        transformed_moving_labels = transformed_moving_labels.squeeze().numpy()

    for i, label_id in enumerate(torch.unique(torch.from_numpy(transformed_moving_labels)).cpu().numpy()):
        label_id = int(label_id)
        if ((fixed_labels == label_id).sum() == 0) or ((moving_labels == label_id).sum() == 0):
            hd95.append(np.NAN)
        else:
            hd95.append(compute_robust_hausdorff(compute_surface_distances(
                (fixed_labels == label_id), (transformed_moving_labels == label_id), np.ones(3)), 95.))
    mean_hd95 = np.nanmean(hd95)
    return mean_hd95, hd95


def initialize_weights(m: nn.Module) -> None:
    """
    Initializes the weights of a given module using Kaiming initialization.

    Parameters
    ----------
    m : nn.Module
        The module whose weights will be initialized.
    """
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def csv_to_img(csv_path: Path, img_size: tuple[int, int, int]) -> torch.Tensor:
    """
    Reads a csv file containing coordinates and creates a tensor of the given size,
    where each coordinate in the csv file is assigned a unique value in the tensor.


    Parameters
    ----------
    csv_path : Path
        Path to the csv file that contains the coordinates.
    img_size : tuple[int, int, int]
        Size of the output tensor.

    Returns
    -------
        torch.Tensor
            A tensor with the values set.
    """
    df = pd.read_csv(csv_path, header=None)
    tensor = torch.zeros(img_size, dtype=torch.float32)
    for i, row in df.iterrows():
        tensor[int(row[0]), int(row[1]), int(row[2])] = i + 1
    return tensor


def index_img_to_df(img: torch.Tensor) -> pd.DataFrame:
    """
    Extracts indices of non-zero values in a tensor.

    Parameters
    ----------
    img : torch.Tensor
        Input tensor with non-zero values at specific coordinates.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the coordinates ordered by their value.
    """
    indices = img.nonzero(as_tuple=False)
    values = img[indices[:, 0], indices[:, 1], indices[:, 2]]
    sorted_indices = indices[values.argsort()]
    sorted_indices = [tuple(idx.tolist()) for idx in sorted_indices]
    df = pd.DataFrame(sorted_indices)
    return df


def transform_landmarks(landmarks: torch.Tensor,
                        displacement_field: 'DisplacementField',
                        original_image_size: tuple[int, int, int] = None,
                        radius: int = 3) -> torch.Tensor:
    """
    Transform landmarks based on the given displacement field.

    Parameters
    ----------
    landmarks : torch.Tensor
        Tensor containing the landmarks to be transformed.
    displacement_field : DisplacementField
        The displacement field used to transform the landmarks.
    original_image_size : tuple[int, int, int]
        The original size of the image that the displacement field corresponds to.
        If not None, the displacement field will be resized for this calculation.
    radius : int, default=3
        The radius of the ball used for the transformation.

    Returns
    -------
    torch.Tensor
        Tensor of transformed landmarks.
    """
    if original_image_size is not None:
        displacement_field = displacement_field.reshape(original_image_size)
    transformed_landmarks = []
    for _, landmark in enumerate(landmarks):
        pixel_coord = landmark.numpy().astype(np.uint8)
        pixel_coord = np.clip(pixel_coord, a_min=[0, 0, 0], a_max=np.array(list(
            displacement_field.displacement_field.shape[2:])) - 1)
        x = np.zeros(tuple(displacement_field.displacement_field.shape[2:]))
        x[tuple(pixel_coord)] = 1
        b = np.array(ball(radius))
        x = scipy.ndimage.binary_dilation(x, b).astype(np.uint8)
        x = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0).to(
            displacement_field.displacement_field.device)
        x = displacement_field(x, mode='nearest')
        x = x.squeeze().cpu().numpy()
        x_coord, y_coord, z_coord = np.nonzero(x)
        transformed_landmark = np.array(
            [x_coord.mean(), y_coord.mean(), z_coord.mean()])
        transformed_landmarks.append(transformed_landmark)

    return torch.from_numpy(np.array(transformed_landmarks))


def compute_tre(fix_lms: torch.Tensor,
                mov_lms: torch.Tensor,
                displacement_field: 'DisplacementField',
                original_image_size: tuple[int, int, int],
                spacing_mov: float = 1.5) -> float:
    """
    Compute target registration error.
    This is a modification of https://github.com/MDL-UzL/L2R/blob/main/evaluation/evaluation.py.

    Parameters
    ----------
    fix_lms : torch.Tensor
        Tensor containing the fixed landmarks.
    mov_lms : torch.Tensor
        Tensor containing the moving landmarks.
    displacement_field : DisplacementField
        Displacement field that defines how the fixed landmarks should be warped.
    original_image_size: tuple[int, int, int]
        The original size of the image that the displacement field corresponds to.
    spacing_mov : float, default=1.5
        Spacing between points.

    Returns
    -------
    float
        The average target registration error.
    """
    fix_lms = fix_lms.squeeze().cpu().numpy()
    mov_lms = mov_lms.squeeze().cpu().numpy()
    disp = displacement_field.reshape(
        original_image_size).displacement_field.permute(0, 2, 3, 4, 1).squeeze().cpu()

    disp[..., 0] *= (original_image_size[0] - 1) / 2
    disp[..., 1] *= (original_image_size[1] - 1) / 2
    disp[..., 2] *= (original_image_size[2] - 1) / 2
    disp = disp.numpy()

    fix_lms_disp_x = map_coordinates(disp[:, :, :, 0], fix_lms.transpose())
    fix_lms_disp_y = map_coordinates(disp[:, :, :, 1], fix_lms.transpose())
    fix_lms_disp_z = map_coordinates(disp[:, :, :, 2], fix_lms.transpose())
    fix_lms_disp = np.array(
        (fix_lms_disp_x, fix_lms_disp_y, fix_lms_disp_z)).transpose()

    fix_lms_warped = fix_lms + fix_lms_disp

    return np.linalg.norm((fix_lms_warped - mov_lms) * spacing_mov, axis=1).mean()


def jacobian_determinant(disp: np.ndarray) -> np.ndarray:
    """
    Computes the Jacobian determinants of a given displacement field.

    Parameters
    ----------
    disp : np.ndarray
        Numpy array representing a displacement field (without identity mapping).

    Returns
    -------
    np.ndarray
        Jacobian determinants at each voxel.

    Code adapted from https://github.com/MDL-UzL/L2R/blob/main/evaluation/utils.py.
    """
    _, _, H, W, D = disp.shape

    gradx = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(
                               disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)

    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(
                               disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)

    gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(
                               disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) -\
        jacobian[1, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]) +\
        jacobian[2, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[1,
                                   2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])

    return jacdet
