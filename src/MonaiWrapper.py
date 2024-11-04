"""
This module provides wrapper classes for some of Monai's models that were tested.
"""

import torch
import torch.nn as nn

from monai.networks.nets import (
    UNet, VNet, VoxelMorphUNet, VoxelMorph)


class ResidualUNet(nn.Module):
    """
    A wrapper around the Monai Unet model, implementing a residual U-Net architecture.
    See https://docs.monai.io/en/stable/networks.html#unet
    """

    def __init__(self, **unet_kwargs) -> None:
        """
        Initializes this class.

        Parameters
        ----------
        **unet_kwargs : dict, optional
            Additional keyword arguments to customize the UNet model. These can include:
            - spatial_dims: int, default=3
                Number of spatial dimensions (e.g., 2 for 2D, 3 for 3D).
            - in_channels: int, default=2
                Number of input channels.
            - out_channels: int, default=3
                Number of output channels.
            - channels: tuple of int, default=(16, 32, 32, 32, 32, 32)
                Number of channels at each layer.
            - strides: tuple of int, default=(2, 2, 2, 1, 1)
                Stride for each convolutional layer.
            - num_res_units: int, default=4
                Number of residual units per layer.
        """
        super().__init__()
        default_kwargs = dict(
            spatial_dims=3,
            in_channels=2,
            out_channels=3,
            channels=(16, 32, 32, 32, 32, 32),
            strides=(2, 2, 2, 1, 1),
            num_res_units=4
        )
        default_kwargs.update(unet_kwargs)
        self.model = UNet(**default_kwargs)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            The first input tensor.
        y : torch.Tensor
            The second input tensor.

        Returns
        -------
        torch.Tensor
            The output of the UNet model after processing the concatenated input tensors.
        """
        x_in = torch.cat((x, y), 1)
        return self.model(x_in)


class MonaiVNet(nn.Module):
    """
    A wrapper around the Monai Unet model, implementing a residual U-Net architecture.
    See: https://docs.monai.io/en/stable/networks.html#vnet
    """

    def __init__(self, **vnet_kwargs) -> None:
        """
        Initializes this class.

        Parameters
        ----------
        **vnet_kwargs : dict, optional
            Additional keyword arguments to customize the VNet model. These can include:
            - spatial_dims: int, default=3
                Number of spatial dimensions (e.g., 2 for 2D, 3 for 3D).
            - in_channels: int, default=2
                Number of input channels.
            - out_channels: int, default=3
                Number of output channels.
            - dropout_prob_down: float, default=0.5
                Dropout probability for the downward path.
            - dropout_prob_up: tuple of float, default=(0.5, 0.5)
                Dropout probabilities for the upward path.
        """
        super().__init__()
        default_kwargs = dict(
            spatial_dims=3,
            in_channels=2,
            out_channels=3,
            dropout_prob_down=0.5,
            dropout_prob_up=(0.5, 0.5)
        )
        default_kwargs.update(vnet_kwargs)
        self.model = VNet(**default_kwargs)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            The first input tensor.
        y : torch.Tensor
            The second input tensor.

        Returns
        -------
        torch.Tensor
            The output of the VNet model after processing the concatenated input tensors.
        """
        x_in = torch.cat((x, y), 1)
        return self.model(x_in)


class MonaiVoxelMorph(nn.Module):
    """
    A wrapper around the Monai VoxelMorph framework.
    See https://docs.monai.io/en/latest/networks.html#voxelmorph
    """

    def __init__(self, integration_steps: int = 10, half_res: bool = False, **unet_kwargs) -> None:
        """
        Initializes this class.

        Parameters
        ----------
        integration_steps : int, default=10
            Number of integration steps for scaling and squaring.
        half_res : bool, default=False
            Whether to perform integration on half resolution.
        **unet_kwargs : dict, optional
            Additional keyword arguments for the UNet backbone.
        """
        super().__init__()
        default_kwargs = dict(
            spatial_dims=3,
            in_channels=2,
            unet_out_channels=32,
            channels=(16, 32, 32, 32, 32, 32),
            final_conv_channels=(16, 16)
        )
        default_kwargs.update(unet_kwargs)

        backbone = VoxelMorphUNet(**default_kwargs)

        self.model = VoxelMorph(
            backbone=backbone,
            integration_steps=integration_steps,
            half_res=half_res
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            The first input tensor.
        y : torch.Tensor
            The secont input tensor.

        Returns
        -------
        torch.Tensor
            The output from VoxelMorph.
        """
        _, displacement_field = self.model(x, y)
        return displacement_field
