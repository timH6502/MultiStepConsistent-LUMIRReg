import json

from pathlib import Path
from typing import Any

import nibabel as nib
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader


class LumirTrainDataset(Dataset):
    """
    Dataset for loading and processing the LUMIR training dataset.
    """

    def __init__(self,
                 image_path: Path = Path('./LUMIR/imagesTr/'),
                 image_size: tuple[int, int, int] = None) -> None:
        """
        Initializes the LumirTrainDataset.

        Parameters
        ----------
        image_path : Path, default=Path('./LUMIR/imagesTr/')
            Path to the directory containing the training images.
        image_size: tuple[int, int, int], default=None
            The desired size of the images. If not None, images will be resized to the given size.
        """
        super().__init__()
        self.image_size = image_size
        self.image_paths = list(image_path.glob('*.nii.gz'))
        self.length = len(self.image_paths)
        assert self.length > 0, 'Incorrect Path'

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the image data for the given index and a random second image.

        Parameters
        ----------
        index : int
            Index of the image to retrieve.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing two tensors:
            - The first tensor is the image data at the specified index.
            - The second tensor is the image data from a randomly selected index.
        """
        other_index = torch.randint(0, self.length, (1,)).item()
        img_data_moving = self.load_image(self.image_paths[index])
        img_data_fixed = self.load_image(self.image_paths[other_index])
        return img_data_moving, img_data_fixed

    def load_image(self, file_name: Path) -> torch.Tensor:
        """
        Load and process the image from the given file path.

        Loads the image data, normalizes it and optionally resizes it.

        Parameters
        ----------
        file_name : Path
            The file path of the image to load.

        Returns
        -------
        torch.Tensor
            A tensor containing the processed image data.
        """
        img_data = torch.from_numpy(
            nib.load(file_name).get_fdata()).squeeze().float()
        img_data = img_data / 255.
        img_data = img_data.unsqueeze(0).unsqueeze(0)
        if self.image_size is not None:
            img_data = torch.nn.Upsample(size=self.image_size)(img_data)
        img_data = img_data[0]
        return img_data

    def as_data_loader(self, batch_size: int, num_workers: int = 0) -> DataLoader:
        """
        Create a data loader for this dataset instance.

        Parameters
        ----------
        batch_size : int
            Batch size for the data loader.
        num_workers : int, default=0
            Number of workers.

        Returns
        -------
        DataLoader
            Data loader from this dataset.
        """
        return DataLoader(self, batch_size=batch_size,
                          shuffle=True,
                          drop_last=True,
                          num_workers=num_workers,
                          pin_memory=True)


class LumirValDataset(Dataset):
    """
    Dataset for loading and processing the LUMIR validation dataset.
    """

    def __init__(self, json_path: Path, img_size: tuple[int, int, int] = None) -> None:
        """
        Initializes the LumirValDataset.

        Parameters
        ----------
        json_path : Path
            Path to the JSON file containing validation image pair information.
        img_size: tuple[int, int, int], default=None
            The desired size of the images. If not None, images will be resized to the given size.
        """
        super().__init__()
        with json_path.open('r', encoding='utf-8') as file:
            data = json.load(file)
        self.df = pd.DataFrame(data['validation'])
        self.df.fixed = json_path.parent / self.df.fixed
        self.df.moving = json_path.parent / self.df.moving
        self.length = self.df.shape[0]
        self.img_size = img_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Retrieves the data for the image pair at the given index from the validation dataset.

        Parameters
        ----------
        index : int
            Index of the image to retrieve.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the following keys:
            - 'img_data_moving': torch.Tensor, the processed moving image.
            - 'img_data_moving_orig': torch.Tensor, the original moving image.
            - 'img_data_fixed': torch.Tensor, the processed fixed image.
            - 'img_data_fixed_orig': torch.Tensor, the original fixed image.
            - 'original_image_size': tuple[int, int, int], the size of the original images.
            - 'save_name': str, the file name for the submission.
        """
        img_data_moving, img_data_moving_orig = self.load_image(
            self.df.iloc[index].moving)
        img_data_fixed, img_data_fixed_orig = self.load_image(
            self.df.iloc[index].fixed)
        save_name = f'disp_{self.df.iloc[index].fixed.name[9:13]}_{self.df.iloc[index].moving.name[9:13]}.nii.gz'

        data = dict(
            img_data_moving=img_data_moving,
            img_data_moving_orig=img_data_moving_orig,
            img_data_fixed=img_data_fixed,
            img_data_fixed_orig=img_data_fixed_orig,
            original_image_size=img_data_fixed_orig.squeeze().shape,
            save_name=save_name
        )
        return data

    def load_image(self, file_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Loads and processes the image from the specified file path.

        The image is normalized to the range [0, 1]. If img_size is not None, the image will be
        resized to this size.

        Parameters
        ----------
        file_name : str
            Path to the file to be loaded.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing the processed and the original image.
        """
        img_data_orig = torch.from_numpy(
            nib.load(file_name).get_fdata()).squeeze().float()
        img_data_orig = img_data_orig / 255.
        img_data_orig = img_data_orig.unsqueeze(0).unsqueeze(0)
        if self.img_size is not None:
            img_data = torch.nn.Upsample(size=self.img_size)(img_data_orig)
        img_data = img_data[0]
        img_data_orig = img_data_orig[0]
        return img_data, img_data_orig

    def as_data_loader(self, batch_size: int, num_workers: int = 0) -> DataLoader:
        """
        Create a data loader for this dataset instance.

        Parameters
        ----------
        batch_size : int
            Batch size for the data loader.
        num_workers : int, default=0
            Number of workers.

        Returns
        -------
        DataLoader
            Data loader from this dataset.
        """
        return DataLoader(self, batch_size=batch_size,
                          shuffle=False,
                          drop_last=False,
                          num_workers=num_workers,
                          pin_memory=True)
