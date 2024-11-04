import torch
import torch.nn as nn

from DisplacementField import DisplacementField


class RegistrationNetwork(nn.Module):
    """
    This is a representation of \Phi[I^A, I^B] = exp(g(I^A, I^B))
    """

    def __init__(self,
                 diff_network: nn.Module,
                 lambda_jacobian: float = 0,
                 lambda_smooth: float = 0,
                 multiplier: float = 1) -> None:
        """
        Initializes the RegistrationNetwork

        Parameters
        ----------
        diff_network : nn.Module
            The difference network g(I^A, I^B) := N[A, B] - N[B, A].
        lambda_jacobian : float, default=0
            Factor for the negative jacobian determinant loss.
        lambda_smooth : float, default=0
            Factor for the smooth loss.
        multiplier : float, default=1
            Displacement field will be multiplied with this value before scaling and squaring.
            This is more of a cosmetic that does not improve anything.
        """
        super().__init__()
        self.diff_network = diff_network
        self.lambda_jacobian = lambda_jacobian
        self.lambda_smooth = lambda_smooth
        self.multiplier = multiplier

    def forward(self,
                moving: torch.Tensor,
                fixed: torch.Tensor) -> tuple[torch.Tensor, DisplacementField, torch.Tensor]:
        """
        Forward pass of the registration network.

        Parameters
        ----------
        moving : torch.Tensor
            Tensor representing the moving image.
        fixed : torch.Tensor
            Tensor representing the fixed image.

        Returns
        -------
        tuple[torch.Tensor, DisplacementField, torch.Tensor]
            Tuple containing the transformed moving image, the displacement field,
            and the total loss. The loss includes the loss provided by the
            diff_network.
        """
        diff, loss = self.diff_network(moving, fixed)
        displacement_field = diff.smooth().mul(
            self.multiplier).exp()
        transformed = displacement_field(moving)
        loss += displacement_field.jacobian_determinant_loss() * self.lambda_jacobian
        loss += displacement_field.smooth_loss() * self.lambda_smooth
        return transformed, displacement_field, loss
