from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from Losses import JacobianDeterminantLoss, SmoothLoss
from Transforms import CompositionTransform, Exponential, SpatialTransform
from utils import generate_identity_grid


class DisplacementField(nn.Module):
    """
    Representation of a displacement field.
    """

    def __init__(self, displacement_field: torch.Tensor, exponential_time_steps: int = 10) -> None:
        """
        Initialize the DisplacementField.

        Parameters
        ----------
        displacement_field : torch.Tensor
            The displacement field represented by this class.
        time_steps : int, default=10
            Number of times steps used for scaling and squaring.
        """
        super().__init__()
        self.displacement_field = displacement_field
        self.time_steps = exponential_time_steps
        self.composition = CompositionTransform()
        self.spatial_transform = SpatialTransform()
        self.exponential_transform = Exponential(
            time_steps=exponential_time_steps)
        self.jac_det_loss = JacobianDeterminantLoss()
        self.smooth_loss_fn = SmoothLoss()
        self.displacement_field_shape = displacement_field.shape[2:]
        self.identity = generate_identity_grid(
            self.displacement_field_shape, self.displacement_field.shape[0], self.displacement_field.device)

    def forward(self, x: Union['DisplacementField', torch.Tensor],
                mode: str = 'bilinear') -> Union['DisplacementField', torch.Tensor]:
        """
        Forward pass. If x is a displacement field, the composition of this displacement field
        and x is returned. If x is a tensor, it is assumed that this is an image and the
        image will be deformed according to the displacement field.

        Parameters
        ----------
        x : Union[DisplacementField, torch.Tensor]
            Another displacement field to compose with or an image to transform.
        mode : str, default='bilinear'
            Mode used for sampling.

        Returns
        -------
        Union[DisplacementField, torch.Tensor]
            The composed displacement field or the transformed image.
        """
        if isinstance(x, DisplacementField):
            # Compose fields
            out = self.composition(
                self.displacement_field, x.displacement_field, self.identity, mode)
            out = DisplacementField(out, self.time_steps)
        else:
            # Apply to image
            out = self.spatial_transform(
                x, self.displacement_field, self.identity, mode=mode)
        return out

    def exp(self) -> 'DisplacementField':
        """
        Compute the exponential of the displacement field.

        Returns
        -------
        DisplacementField
            The scaled and squared displacement field.
        """
        scaled_squared = self.exponential_transform(
            self.displacement_field, self.identity)
        return DisplacementField(scaled_squared, self.time_steps)

    def jacobian_determinant_loss(self) -> torch.Tensor:
        """
        Calculates the negative Jacobian determinant loss of the displacement field.

        Returns
        -------
        torch.Tensor
            Tensor containing the loss value.
        """
        return self.jac_det_loss(self.displacement_field)

    def percentage_negative_jacobian_determinant(self) -> float:
        """
        Calculate the percentage of negative values in the Jacobian of this displacement field.

        Returns
        -------
        float
            Percentage of negative values in the Jacobian determinant.
        """
        jacobian_determinant = self.jac_det_loss.calculate_jacobian_determinant(
            self.displacement_field)
        n_neg = (jacobian_determinant < 0).sum().float().item()
        return 100 * n_neg / jacobian_determinant.numel()

    def standard_deviation_pos_jac_determinant(self) -> float:
        """
        Calculates the standard deviation of the logarithm of the Jacobian determinant.

        This function computes the standard deviation of the logarithm of the Jacobian determinant
        after shifting its values by 3 and clipping them to avoid negative values.

        Returns
        -------
        float
            The standard deviation of the log-transformed, shifted and clipped Jacobian determinant.

        Notes
        -----
        The decision to shift by (the 'fairly arbitrary' value) 3 is based on empirical
        findings that this value is larger than the most negative determinant value encountered.
        """
        jacobian_determinant = self.jac_det_loss.calculate_jacobian_determinant(
            self.displacement_field)
        jac_det = (jacobian_determinant + 3).clip(0.000000001, 1000000000)
        log_jac_det = torch.log(jac_det)
        return log_jac_det.std().item()

    def smooth_loss(self) -> torch.Tensor:
        """
        Calculates the smooth loss of the displacement field.

        Returns
        -------
        torch.Tensor
            Tensor containing the loss value.
        """
        return self.smooth_loss_fn(self.displacement_field)

    def tanh(self) -> 'DisplacementField':
        """
        Applies tanh to the displacement field.

        Returns
        -------
        DisplacementField
            tanh of the displacement field.
        """
        return DisplacementField(torch.tanh(self.displacement_field))

    def mul(self, value: float) -> 'DisplacementField':
        """
        Multiplies the displacement field with the given value.

        Parameters
        ----------
        value : float
            Value to muliply.

        Returns
        -------
        DisplacementField
            The multiplied displacement field.
        """
        return DisplacementField(self.displacement_field.mul(value))

    def smooth(self, kernel: int = 3) -> 'DisplacementField':
        """
        Smoothens this displacement field by applying average pooling twice.

        Parameters
        ----------
        kernel : int, default=3
            Kernel size used for average pooling.

        Returns
        -------
        DisplacementField
            Smoothed displacement field.
        """
        half_width = (kernel - 1) // 2
        displacement_field = F.avg_pool3d(F.avg_pool3d(
            self.displacement_field, kernel, stride=1, padding=half_width), kernel, stride=1, padding=half_width)
        return DisplacementField(displacement_field)

    def transform_point(self, point: torch.Tensor) -> torch.Tensor:
        """
        This function transforms a point according to this displacement field.

        Parameters
        ----------
        point : torch.Tensor
            The point to be transformed.

        Returns
        -------
        torch.Tensor
            The transformed point.
        """
        point = point.clone()

        image_size = torch.Tensor(list(self.displacement_field.shape[2:])).to(
            self.displacement_field.device)
        ints = [max(0, min(int(image_size[i]) - 1, int(point[i])))
                for i in range(3)]
        fracs = [0 if ints[i] == 0 else image_size[i] if point[i] ==
                 image_size[i] else point[i] - int(point[i]) for i in range(3)]
        # Calculate weights
        w000 = (1 - fracs[0]) * (1 - fracs[1]) * (1 - fracs[2])
        w001 = (1 - fracs[0]) * (1 - fracs[1]) * fracs[2]
        w010 = (1 - fracs[0]) * fracs[1] * (1 - fracs[2])
        w011 = (1 - fracs[0]) * fracs[1] * fracs[2]
        w100 = fracs[0] * (1 - fracs[1]) * (1 - fracs[2])
        w101 = fracs[0] * (1 - fracs[1]) * fracs[2]
        w110 = fracs[0] * fracs[1] * (1 - fracs[2])
        w111 = fracs[0] * fracs[1] * fracs[2]

        # Calculate values
        v000 = self.displacement_field[0, :, ints[0], ints[1], ints[2]]
        v001 = self.displacement_field[0, :, ints[0], ints[1],
                                       (ints[2] + 1) if ints[2] + 1 < image_size[2] else ints[2]]
        v010 = self.displacement_field[0, :, ints[0], (
            ints[1] + 1) if ints[1] + 1 < image_size[1] else ints[1], ints[2]]
        v011 = self.displacement_field[0, :, ints[0], (ints[1] + 1) if ints[1] + 1 < image_size[1]
                                       else ints[1], (ints[2] + 1) if ints[2] + 1 < image_size[2] else ints[2]]
        v100 = self.displacement_field[0, :, (
            ints[0] + 1) if ints[0] + 1 < image_size[0] else ints[0], ints[1], ints[2]]
        v101 = self.displacement_field[0, :, (ints[0] + 1) if ints[0] + 1 < image_size[0]
                                       else ints[0], ints[1], (ints[2] + 1) if ints[2] + 1 < image_size[2] else ints[2]]
        v110 = self.displacement_field[0, :, (ints[0] + 1) if ints[0] + 1 < image_size[0]
                                       else ints[0], (ints[1] + 1) if ints[1] + 1 < image_size[1] else ints[1], ints[2]]
        v111 = self.displacement_field[0, :, (ints[0] + 1) if ints[0] + 1 < image_size[0] else ints[0], (ints[1] + 1)
                                       if ints[1] + 1 < image_size[1] else ints[1], (ints[2] + 1)
                                       if ints[2] + 1 < image_size[2] else ints[2]]

        displacement = w000 * v000 + w001 * v001 + w010 * v010 + \
            w011 * v011 + w100 * v100 + w101 * v101 + w110 * v110 + w111 * v111

        point = point.to(self.displacement_field.device)
        normalized_point = 2.0 * point / (image_size - 1) - 1.0
        point_transformed = (normalized_point -
                             displacement + 1.0) * (image_size - 1) / 2.0

        return point_transformed

    def reshape(self, size: tuple[int, int, int], mode: str = 'trilinear') -> 'DisplacementField':
        """
        Reshapes the displacement field.

        Parameters
        ----------
        size : tuple[int, int, int]
            Desiresd output size of the displacement field.
        mode : str, default='trilinear'
            Mode to be used for upsampling.

        Returns
        -------
        DisplacementField
            Object representation of the reshaped displacement field.
        """
        displacement_x = torch.nn.Upsample(
            size=size, mode=mode)(self.displacement_field[:, 0].unsqueeze(0)).squeeze()
        displacement_y = torch.nn.Upsample(
            size=size, mode=mode)(self.displacement_field[:, 1].unsqueeze(0)).squeeze()
        displacement_z = torch.nn.Upsample(
            size=size, mode=mode)(self.displacement_field[:, 2].unsqueeze(0)).squeeze()
        displacement_field = torch.stack(
            [displacement_x, displacement_y, displacement_z], dim=0).unsqueeze(0)
        return DisplacementField(displacement_field, exponential_time_steps=self.time_steps)
