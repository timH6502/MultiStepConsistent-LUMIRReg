import random

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch

from tqdm import tqdm

from DiffNetwork import DiffNetwork
from LumirDataset import LumirValDataset
from MonaiWrapper import ResidualUNet
from RegistrationNetwork import RegistrationNetwork
from TwoStepConsistent import TwoStepConsistent
from utils import init_logger


def run_inference(config: dict[str, Any]) -> None:
    logger = config['logger']
    device = config['device']
    logger.info('Building model')
    backbone_phi = ResidualUNet().to(device)
    diff_network_phi = DiffNetwork(
        backbone_phi, lambda_jacobian=0, lambda_smooth=0)
    phi = RegistrationNetwork(
        diff_network_phi,
        lambda_jacobian=0,
        lambda_smooth=0,
        multiplier=0.5)

    backbone_psi = ResidualUNet().to(device)
    diff_network_psi = DiffNetwork(
        backbone_psi, lambda_jacobian=0, lambda_smooth=0)
    psi = RegistrationNetwork(
        diff_network_psi,
        lambda_jacobian=0,
        lambda_smooth=0)

    backbone_xi = ResidualUNet(num_res_units=2).to(device)
    diff_network_xi = DiffNetwork(
        backbone_xi, lambda_jacobian=0, lambda_smooth=0)
    xi = RegistrationNetwork(
        diff_network_xi,
        lambda_jacobian=0,
        lambda_smooth=0,
        multiplier=0.5)

    phi_psi = TwoStepConsistent(phi, psi)
    registration_network = TwoStepConsistent(xi, phi_psi)

    logger.info('Loading weights')
    registration_network.load_state_dict(
        torch.load(config['pretrained_weights_path'], weights_only=False, map_location=torch.device(device)))
    registration_network.eval()

    logger.info('Initializing data loader')
    val_data_loader = LumirValDataset(
        config['json_path'],
        img_size=config['image_size']).as_data_loader(batch_size=1)

    logger.info('Running inference.')
    for data in tqdm(val_data_loader):
        moving = data['img_data_moving'].to(device)
        fixed = data['img_data_fixed'].to(device)
        original_image_size = data['original_image_size']

        with torch.no_grad():
            _, displacement_field, _ = registration_network(moving, fixed)
            displacement_field = displacement_field.reshape(
                original_image_size)
            scaling_factors = torch.tensor([(original_image_size[0] - 1) / 2,
                                            (original_image_size[1] - 1) / 2,
                                            (original_image_size[2] - 1) / 2],
                                           device=displacement_field.displacement_field.device)
            displacement_field_tensor = displacement_field.displacement_field.squeeze() * \
                scaling_factors.view(3, 1, 1, 1)
            nifti_img = nib.Nifti1Image(
                displacement_field_tensor.squeeze().permute(1, 2, 3, 0).flip(-1)
                .detach().cpu().numpy().astype(np.float32), np.eye(4))
            nib.save(
                nifti_img, config['displacement_save_path'] / data['save_name'][0])


if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger = init_logger(__name__)
    config = dict(
        pretrained_weights_path=Path(
            './pretrained_weights/model_1725062136.pth'),
        time_steps=10,
        image_size=(128, 128, 128),
        device=torch.device('cpu'),
        json_path=Path('./LUMIR/LUMIR_dataset.json'),
        logger=logger,
        displacement_save_path=Path('./output/')
    )

    config['displacement_save_path'].mkdir(parents=True, exist_ok=True)
    if config['pretrained_weights_path'].exists() and config['json_path'].exists():
        run_inference(config)
    elif not config['pretrained_weights_path'].exists():
        logger.error('Could not find pretrained weights.')
    else:
        logger.error('Could not find json file.')
