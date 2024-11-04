import random
import time

from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from monai.losses import BendingEnergyLoss
from tqdm import tqdm

from DiffNetwork import DiffNetwork
from Losses import NCC
from LumirDataset import LumirTrainDataset, LumirValDataset
from MonaiWrapper import ResidualUNet
from RegistrationNetwork import RegistrationNetwork
from Trainer import Trainer
from TwoStepConsistent import TwoStepConsistent
from utils import init_logger, jacobian_determinant


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    logger = init_logger(__name__)
    batch_size = 1
    accumulation_steps = 1
    epochs = 1
    num_workers = 16
    lambda_jacobian = 100
    lambda_smooth = 0
    lambda_jacobian_deformation = 10
    lambda_smooth_deformation = 0
    use_amp = False
    gradient_clipping = 1
    save_path = Path('./checkpoints/')
    displacement_save_path = Path('./disp_pred/folder/')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    learning_rate = 1e-3
    image_size = (128, 128, 128)
    pretrained_path = Path('./pretrained_weights/model_1725062136.pth')

    data_loader = LumirTrainDataset(image_size=image_size).as_data_loader(
        batch_size=batch_size, num_workers=num_workers)

    # Network definition
    backbone_phi = ResidualUNet().to(device)
    diff_network_phi = DiffNetwork(
        backbone_phi, lambda_jacobian=lambda_jacobian, lambda_smooth=lambda_smooth)
    phi = RegistrationNetwork(
        diff_network_phi,
        lambda_jacobian=lambda_jacobian_deformation,
        lambda_smooth=lambda_smooth_deformation,
        multiplier=0.5)

    backbone_psi = ResidualUNet().to(device)
    diff_network_psi = DiffNetwork(
        backbone_psi, lambda_jacobian=lambda_jacobian, lambda_smooth=lambda_smooth)
    psi = RegistrationNetwork(
        diff_network_psi,
        lambda_jacobian=lambda_jacobian_deformation,
        lambda_smooth=lambda_smooth_deformation)

    backbone_xi = ResidualUNet(num_res_units=2).to(device)
    diff_network_xi = DiffNetwork(
        backbone_xi, lambda_jacobian=lambda_jacobian, lambda_smooth=lambda_smooth)
    xi = RegistrationNetwork(
        diff_network_xi,
        lambda_jacobian=lambda_jacobian_deformation,
        lambda_smooth=lambda_smooth_deformation,
        multiplier=0.5)

    phi_psi = TwoStepConsistent(phi, psi)
    registration_network = TwoStepConsistent(xi, phi_psi)

    if pretrained_path is not None:
        registration_network.load_state_dict(
            torch.load(pretrained_path, weights_only=False))
        logger.info('Model has been loaded.')
        registration_network.train(True)

    optimizer = torch.optim.AdamW(
        registration_network.parameters(), lr=learning_rate, weight_decay=0, amsgrad=True)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=2, cooldown=0)
    sim_loss = NCC(window_size=9)
    reg_loss = BendingEnergyLoss(normalize=True)

    # Train network
    trainer = Trainer(model=registration_network,
                      optimizer=optimizer,
                      data_loader=data_loader,
                      device=device,
                      sim_loss=sim_loss,
                      reg_loss=reg_loss,
                      save_path=save_path,
                      lr_scheduler=lr_scheduler,
                      use_amp=use_amp,
                      gradient_clipping=gradient_clipping,
                      accumulation_steps=accumulation_steps)
    logger.info('Starting training.')
    trainer.train(epochs=epochs)

    # Load saved model
    if epochs > 0:
        registration_network.load_state_dict(
            torch.load(trainer.save_path, weights_only=False))
    elif pretrained_path is not None:
        registration_network.load_state_dict(
            torch.load(pretrained_path, weights_only=False))
        registration_network.eval()
    else:
        logger.error('Model not defined.')
        quit()

    # Inference
    logger.info('Inference')
    val_data_loader = LumirValDataset(
        Path('./LUMIR/LUMIR_dataset.json'), img_size=image_size).as_data_loader(batch_size=1)

    displacement_save_path.mkdir(exist_ok=True, parents=True)

    der_id = []
    jac_det_perc = []
    sdlogj = []
    forward_pass_times = []

    for data in tqdm(val_data_loader):
        moving = data['img_data_moving'].to(device)
        fixed = data['img_data_fixed'].to(device)
        original_moving = data['img_data_moving_orig'].to(device)
        original_fixed = data['img_data_fixed_orig'].to(device)
        original_image_size = data['original_image_size']

        with torch.no_grad():
            start_time = time.time()
            _, displacement_field, _ = registration_network(
                moving, fixed)
            end_time = time.time()
            forward_pass_times.append(end_time - start_time)
            _, displacement_field_fixed_moving, _ = registration_network(
                fixed, moving)
            displacement_field = displacement_field.reshape(
                original_image_size)
            displacement_field_fixed_moving = displacement_field_fixed_moving.reshape(
                original_image_size)
            transformed_moving = displacement_field(original_moving)
            der_id.append(displacement_field(
                displacement_field_fixed_moving).displacement_field.abs().mean().item())
            jacobian_det = jacobian_determinant(
                displacement_field.displacement_field.cpu().numpy())
            jac_det_perc.append((jacobian_det < 0).mean())
            sdlogj.append(
                np.log((jacobian_det + 3).clip(0.000000001, 1000000000)).std())
            # Save displacement fields
            scaling_factors = torch.tensor([(original_image_size[0] - 1) / 2,
                                            (original_image_size[1] - 1) / 2,
                                            (original_image_size[2] - 1) / 2],
                                           device=displacement_field.displacement_field.device)

            displacement_field_tensor = displacement_field.displacement_field.squeeze() * \
                scaling_factors.view(3, 1, 1, 1)
            nifti_img = nib.Nifti1Image(
                displacement_field_tensor.squeeze().permute(1, 2, 3, 0).flip(-1)
                .detach().cpu().numpy().astype(np.float32), np.eye(4))
            nib.save(nifti_img, displacement_save_path / data['save_name'][0])
            nifti_img = nib.Nifti1Image(
                transformed_moving.squeeze().detach().cpu().numpy().astype(np.float32), np.eye(4))

    logger.info(
        f'Norm diff from Id: {np.array(der_id).mean()}, std: {np.array(der_id).std()}')
    logger.info(
        'Average percentage of non-positive Jacobians:'
        f' {np.array(jac_det_perc).mean()}, std: {np.array(jac_det_perc).std()}')
    logger.info(
        f'sdlogj: {np.array(sdlogj).mean()}, std: {np.array(sdlogj).std()}')
    logger.info(
        f'Average forward pass time: {np.array(forward_pass_times).mean()}, std: {np.array(forward_pass_times).std()}')
