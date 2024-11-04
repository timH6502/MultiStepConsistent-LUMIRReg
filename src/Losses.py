import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class JacobianDeterminantLoss(nn.Module):
    """
    This class can be used to calculate the negative Jacobian regularization loss
    for a given deformation field.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, displacement_field: torch.Tensor) -> torch.Tensor:
        """
        Calculates the negative Jacobian determinant regularization
        loss for a given deformation field.

        Parameters
        ----------
        displacement_field : torch.Tensor
            Tensor representing the deformation field.

        Returns
        -------
        torch.Tensor
            Negative Jacobian determinant loss

        Notes
        -----
        Given a displacement field u and the identity transform x,
        the Jacobian J_\phi(p) of \phi(x) = x + u(x) is calculated for each
        position p. Since we are only interested in the values, where the
        Jacobian is positive, we apply ReLU to the resulting tensor and take
        its mean.
        """
        neg_jacobian_determinant = - \
            self.calculate_jacobian_determinant(displacement_field)
        positives = F.relu(neg_jacobian_determinant)
        return positives.mean()

    def calculate_jacobian_determinant(self, displacement_field: torch.Tensor) -> torch.Tensor:
        """
        Calculates the determinant for a given displacement field.

        Parameters
        ----------
        displacement_field : torch.Tensor
            Tensor representing the displacement field.

        Returns
        -------
        torch.Tensor
            Jacobian determinant.

        Notes
        -----
        The gradient is estimated as described in https://doi.org/10.2307/2008770
        """
        phi = displacement_field.clone()

        phi[:, 0, :, :, :] = phi[:, 0, :, :, :] * (phi.shape[2] - 1) / 2
        phi[:, 1, :, :, :] = phi[:, 1, :, :, :] * (phi.shape[3] - 1) / 2
        phi[:, 2, :, :, :] = phi[:, 2, :, :, :] * (phi.shape[4] - 1) / 2

        dphi_x_dx = torch.gradient(phi[:, 0, :, :, :], dim=-3)[0] + 1
        dphi_x_dy = torch.gradient(phi[:, 0, :, :, :], dim=-2)[0]
        dphi_x_dz = torch.gradient(phi[:, 0, :, :, :], dim=-1)[0]
        dphi_y_dx = torch.gradient(phi[:, 1, :, :, :], dim=-3)[0]
        dphi_y_dy = torch.gradient(phi[:, 1, :, :, :], dim=-2)[0] + 1
        dphi_y_dz = torch.gradient(phi[:, 1, :, :, :], dim=-1)[0]
        dphi_z_dx = torch.gradient(phi[:, 2, :, :, :], dim=-3)[0]
        dphi_z_dy = torch.gradient(phi[:, 2, :, :, :], dim=-2)[0]
        dphi_z_dz = torch.gradient(phi[:, 2, :, :, :], dim=-1)[0] + 1

        jacobian_det = (
            dphi_x_dx * (dphi_y_dy * dphi_z_dz - dphi_y_dz * dphi_z_dy) +
            dphi_x_dy * (dphi_y_dz * dphi_z_dx - dphi_y_dx * dphi_z_dz) +
            dphi_x_dz * (dphi_y_dx * dphi_z_dy - dphi_y_dy * dphi_z_dx)
        )

        return jacobian_det


class NCC(nn.Module):
    """
    This class can be used to calculate the local patchwise normalized cross correlation loss.
    The code is a modified version of the code from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, window_size: int = 7, eps: float = 1e-5) -> None:
        """
        Initializes the module.

        Parameters
        ----------
        window_size : int, default=7
            Windows size of the patches.
        eps : float, default=1e-5
            Value added to prevent division by zero.
        """
        super().__init__()
        self.win_raw = window_size
        self.eps = eps
        self.win = window_size

    def forward(self, I, J) -> torch.Tensor:
        """
        Calculate the NCC loss given two images.


        Parameters
        ----------
        I : torch.Tensor
            First image.
        J : torch.Tensor
            Second image.

        Returns
        -------
        torch.Tensor
            Tensor containing the loss.
        """
        ndims = 3
        win_size = self.win_raw
        self.win = [self.win_raw] * ndims

        weight_win_size = self.win_raw
        weight = torch.ones((1, 1, weight_win_size, weight_win_size,
                            weight_win_size), device=I.device, requires_grad=False)
        conv_fn = F.conv3d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size/2))
        J_sum = conv_fn(J, weight, padding=int(win_size/2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size/2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size/2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)


class SmoothLoss(nn.Module):
    """
    Class to compute smoothness loss of a tensor.
    """

    def __init__(self, lambda_weight: float = 1) -> None:
        """
        Initializes the SmoothLoss.

        Parameters
        ----------
        lambda_weight: float, default=1
            Value to be multiplied before squaring.
        """
        super().__init__()
        self.lambda_weight = lambda_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the smoothness loss of the given tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor (e.g. a displacement field).

        Returns
        -------
        torch.Tensor
            Smoothness loss.
        """
        dx = (x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
              ).mul(self.lambda_weight).square().mean()
        dy = (x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
              ).mul(self.lambda_weight).square().mean()
        dz = (x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
              ).mul(self.lambda_weight).square().mean()
        return dx + dy + dz
