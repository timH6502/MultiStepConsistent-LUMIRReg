import time

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm

from EarlyStopping import EarlyStopping
from utils import init_logger


class Trainer:
    """
    Class for training a registration model.

    Attributes
    ----------
    model : torch.nn.Module
        The model to train.
    optimizer : torch.optim.Optimizer
        The optimizer used for training.
    data_loader : torch.utils.data.DataLoader
        The data loader providing the training images.
    device : torch.device
        The device used for training.
    save_path : Path
        Directory where models will be saved.
    sim_loss : nn.Module
        Similarity loss function used for training.
    reg_loss : nn.Module
        Regularization loss function used for training.
    lr_scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        The learning rate scheduler to be used.
    use_amp : bool, default=False
        Whether to use automatic mixed precision (AMP) for training.
    gradient_clipping : float, default=None
        If not None, gradients will be clipped to this value.
    accumulation_steps : int, default=1
        Steps of gradient accumulation.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 data_loader: torch.utils.data.DataLoader,
                 device: torch.device,
                 save_path: Path,
                 sim_loss: nn.Module,
                 reg_loss: nn.Module = None,
                 lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                 use_amp: bool = False,
                 gradient_clipping: float = None,
                 accumulation_steps: int = 1) -> None:

        self.model = model.to(device)
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.device = device
        self.sim_loss = sim_loss
        self.reg_loss = reg_loss
        self.lr_scheduler = lr_scheduler
        self.use_amp = use_amp
        self.gradient_clipping = gradient_clipping
        self.accumulation_steps = accumulation_steps if accumulation_steps != 0 else 1
        save_path.mkdir(parents=True, exist_ok=True)
        self.save_path = str(save_path / f'model_{int(time.time())}.pth')
        self.early_stopping = EarlyStopping(
            min_delta=0, patience=10, save_path=self.save_path)
        self.logger = init_logger(__name__)
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            self.grad_scaler = torch.amp.GradScaler(
                'cuda', init_scale=2**16, growth_factor=2.0, backoff_factor=0.5, enabled=True)
        else:
            torch.amp.GradScaler(
                'cpu', init_scale=2**16, growth_factor=2.0, backoff_factor=0.5, enabled=True)

    def train_one_epoch(self) -> float:
        """
        Trains the model for one epoch.

        Returns
        -------
        float
            Average loss.
        """
        losses = np.zeros(len(self.data_loader), dtype=np.float32)
        progress_bar = tqdm(enumerate(self.data_loader),
                            desc="Training", leave=True, total=len(self.data_loader))
        for i, (moving, fixed) in progress_bar:
            moving = moving.to(self.device)
            fixed = fixed.to(self.device)

            with torch.autocast(self.device.type, enabled=self.use_amp, dtype=torch.bfloat16):
                transformed_moving, displacement_field, internal_loss = self.model(
                    moving, fixed)
                loss = self.sim_loss(transformed_moving, fixed)
                if self.reg_loss is not None:
                    loss += self.reg_loss(displacement_field.displacement_field) / 10
                loss += internal_loss
                loss /= self.accumulation_steps

            if self.grad_scaler is not None and self.use_amp:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % self.accumulation_steps == 0 or i == len(self.data_loader) - 1:
                if self.grad_scaler is not None and self.use_amp:
                    self.grad_scaler.unscale_(self.optimizer)
                    if self.gradient_clipping is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=self.gradient_clipping)
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    if self.gradient_clipping is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=self.gradient_clipping)
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            losses[i] = loss.item() * self.accumulation_steps
            progress_bar.set_postfix(
                loss=losses[i], internal_loss=internal_loss.item() * self.accumulation_steps)
        return losses.mean()

    def train(self, epochs: int) -> np.ndarray:
        """
        Trains the model for the specified number of epochs.

        Parameters
        ----------
        epochs : int
            Number of epochs to train the model.

        Returns
        -------
        np.array
            Average loss for each epoch.
        """
        losses = np.zeros(epochs, dtype=np.float32)
        for i in range(epochs):
            mean_loss = self.train_one_epoch()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(mean_loss)
            losses[i] = mean_loss
            self.early_stopping(mean_loss, self.model, epoch=i)
            if self.early_stopping.stop:
                self.logger.info(
                    'Early stopping criterion reached. Stopping training')
                return losses[:i + 1]
            if self.lr_scheduler is not None:
                self.logger.info(
                    f'Epoch: {i}\nAverage loss: {losses[i]}\n'
                    f'Setting learning rate to: {self.lr_scheduler.get_last_lr()[0]}')
            else:
                self.logger.info(
                    f'Epoch: {i}\nAverage loss: {losses[i]}')
        return losses
