import torch
import torch.nn as nn

from DisplacementField import DisplacementField


class TwoStepConsistent(nn.Module):
    """
    Two-step inverse consistent operator, as described in https://doi.org/10.48550/arXiv.2305.00087
    """

    def __init__(self, phi: nn.Module, psi: nn.Module) -> None:
        """
        Initialize the TwoStepConsistent operator.

        Parameters
        ----------
        phi : nn.Module
            First registration step module. Must be inverse consistent.
        psi : nn.Module
            Second registration step module. Must be inverse consistent.

        Notes
        -----
        If phi and psi are inverse consistent, this module is inverse consistent as well and can be passed
        as a new psi to this class.
        """
        super().__init__()
        self.phi = phi
        self.psi = psi

    def forward(self,
                moving: torch.Tensor,
                fixed: torch.Tensor) -> tuple[torch.Tensor, DisplacementField, torch.Tensor]:
        """
        Performs the two-step consistent registration.

        Parameters
        ----------
        moving : torch.Tensor
            Tensor representing the moving image.
        fixed : torch.Tensor
            Tensor representing the fixed image.

        Returns
        -------
        tuple[torch.Tensor, DisplacementField, torch.Tensor]
            Tuple contatining the transformed moving image, the displacement field and a
            combined loss from phi and psi.
        """
        transformed_moving_phi, sqrt_phi, loss_phi = self.phi(moving, fixed)
        transformed_fixed_phi, _, _ = self.phi(fixed, moving)

        _, psi, loss_psi = self.psi(
            transformed_moving_phi, transformed_fixed_phi)

        displacement_field = sqrt_phi(psi(sqrt_phi))
        transformed_moving = displacement_field(moving)

        loss = loss_phi + loss_psi
        return transformed_moving, displacement_field, loss
