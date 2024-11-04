import torch
import torch.nn as nn

from DisplacementField import DisplacementField


class DiffNetwork(nn.Module):
    """
    This class is a represnetation of g(I^A, I^B) = N_\theta[I^A, I^B] - N_\theta[I^B, I^A]
    as described in https://doi.org/10.48550/arXiv.2305.00087 , where N_\theta denotes a neural network,
    I^A and I^B are images.
    """

    def __init__(self, backbone: nn.Module,
                 lambda_jacobian: float,
                 lambda_smooth: float) -> None:
        """
        Initializes the DiffNetwork.

        Parameters
        ----------
        backbone : nn.Module
            The neural network backbone used to compute representations.
        lambda_jacobian : float
            Weight parameter of the Jacobian determinant loss.
        lambda_smooth : float
            Weight parameter of the smoothness loss.
        """
        super().__init__()
        self.backbone = backbone
        self.lambda_jacobian = lambda_jacobian
        self.lambda_smooth = lambda_smooth

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor) -> tuple[DisplacementField, torch.Tensor]:
        """
        Calculates g(I^A, I^B) := N[A, B] - N[B, A], where A and B are batches of images
        and N is the backbone.

        Parameters
        ----------
        moving : torch.Tensor
            Tensor representing the moving image.
        fixed : torch.Tensor
            Tensor representing the fixed image.

        Returns
        -------
        tuple[DisplacementField, torch.Tensor]
            Tuple containing the displacement field and the total loss.
        """
        n_a_b = self.backbone(moving, fixed)
        n_b_a = self.backbone(fixed, moving)
        difference = n_a_b - n_b_a
        displacement_field = DisplacementField(difference)

        loss = displacement_field.jacobian_determinant_loss() * self.lambda_jacobian
        loss += displacement_field.smooth_loss() * self.lambda_smooth

        return displacement_field, loss
