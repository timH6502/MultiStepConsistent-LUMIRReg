import numpy as np
import torch

from utils import init_logger


class EarlyStopping:
    """
    Class that implements early stopping.
    If the training should stop, the stop flag of the instance will be set to True.
    """

    def __init__(self, min_delta: float, patience: int, save_path: str) -> None:
        """
        Initializes the early stopping class.

        Parameters:
        -----------
        min_delta : float
            The minimum change of the loss required for a model to be considered as having improved.
        patience : int
            The number of epochs with no improvement after which training should be stopped.
        save_path : str
            The path where the best model will be saved.
        """
        self.min_delta = min_delta
        self.patience = patience
        self.save_path = save_path
        self.best_loss = np.inf
        self.patience_count = 0
        self.best_epoch = 0
        self.stop = False
        self.logger = init_logger(__name__)

    def __call__(self, loss: float, model: torch.nn.Module, epoch: int) -> None:
        """
        Evaluates the current model's performance and determines whether training should stop early.

        If the provided loss is better (i.e., lower) than the best observed loss so far,
        the minimum delta that has been set, the patience counter will be reset and the model will be saved.

        Parameters:
        -----------
        loss : float
            The loss of the model for the current epoch.
        model : torch.nn.Module
            The model that is trained.
        epoch : int
            The current training epoch
        """
        if loss + self.min_delta < self.best_loss:
            self.best_loss = loss
            self.patience_count = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.save_path)
            self.logger.info(f'Model saved to: {self.save_path}')
        else:
            self.patience_count += 1
            if self.patience_count > self.patience:
                self.stop = True
