import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositionTransform(nn.Module):
    """
    Class that can be used to compose displacement fields.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self,
                displacement_field_a: torch.Tensor,
                displacement_field_b: torch.Tensor,
                identity: torch.Tensor,
                mode: str = 'bilinear') -> torch.Tensor:
        """
        Computes the composition of two displacement fields.

        Parameters
        ----------
        displacement_a : torch.Tensor
            Tensor representing the first displacement field.
        displacement_b : torch.Tensor
            Tensor representing the second displacement field.
        identity : torch.Tensor
            Identity transformation.
        mode : str, default='bilinear'
            Interpolation mode used for grid sampling.

        Returns
        -------
        torch.Tensor
            Composed displacement fields, i.e. displacement_a \circ displacement_b.
        """
        grid = identity + displacement_field_a.permute(0, 2, 3, 4, 1)
        sampled_displacement_b = F.grid_sample(
            displacement_field_b, grid, mode=mode, align_corners=False)
        composition = displacement_field_a + \
            sampled_displacement_b
        return composition


class Exponential(nn.Module):
    """
    This class computes the exponential of a displacement field by using scaling and squaring.
    This procedure is described in https://doi.org/10.1007/11866565_113

    Attributes
    ----------
    time_steps : int, default=7
        Number of iterations for computing the exponential.
    """

    def __init__(self, time_steps: int = 10) -> None:
        super().__init__()
        self.time_steps = time_steps

    def forward(self, displacement_field: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        """
        Computes the exponential of the displacement field iteratively by composing small flow increments.

        Parameters
        ----------
        displacement_field : torch.Tensor
            Tensor representing the displacement field.
        identity : torch.Tensor
            Identity transformation.

        Returns
        -------
        torch.Tensor
            Exponential of the displacement field.
        """
        # Divide displacement field by 2^T.
        flow = displacement_field / (2 ** self.time_steps)

        for _ in range(self.time_steps):
            # Create deformation field.
            grid = identity + flow.permute(0, 2, 3, 4, 1)
            # Compose
            flow = flow + F.grid_sample(flow, grid,
                                        mode='bilinear', align_corners=False)
        return flow


class SpatialTransform(nn.Module):
    """
    Class that can be used to apply spatial transformations to an image using a displacement field.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self,
                image: torch.Tensor,
                displacement_field: torch.Tensor,
                identity: torch.Tensor,
                mode: str = 'bilinear') -> torch.Tensor:
        """
        Warps the image according to the given displacement field.

        Parameters
        ----------
        image : torch.Tensor
            Tensor representing the image.
        displacement_field : torch.Tensor
            Tensor representing the displacement_field
        identity : torch.Tensor
            Identity transformation.
        mode : str, default='bilinear'
        Interpolation mode used for grid sampling.

        Returns
        -------
        torch.Tensor
            Transformed image.
        """
        # Create deformation field
        deformation_field = identity + \
            displacement_field.permute(0, 2, 3, 4, 1)
        # Transform image
        transformed_img = F.grid_sample(
            image, deformation_field, mode=mode, align_corners=False)

        return transformed_img
