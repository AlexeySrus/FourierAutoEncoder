import cv2
import os
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from pipeline.utils.fft_utils import generate_batt, apply_filter_to_fft
from pytorch_complex_tensor import ComplexTensor


def load_image(path):
    img = cv2.imread(path, 1)
    if img is None:
        print('Can\'t open image: {}'.format(path))
    assert img is not None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class SeriesAndComputingClearDataset(Dataset):
    def __init__(self, series_folders_path, clear_series_path, edges_path, window_size):
        self.series_folders_pathes = [
            os.path.join(series_folders_path, sfp)
            for sfp in os.listdir(series_folders_path)
        ]

        self.clear_series_pathes = [
            os.path.join(clear_series_path, cfp)
            for cfp in os.listdir(clear_series_path)
        ]

        self.edges_series_pathes = [
            os.path.join(edges_path, cfp)
            for cfp in os.listdir(edges_path)
        ]

        assert len(self.series_folders_pathes) == len(
            self.clear_series_pathes)

        self.sort_key = lambda s: int(s.split('_')[-1].split('.')[0])

        self.series_folders_pathes.sort(key=self.sort_key)
        self.clear_series_pathes.sort(key=self.sort_key)
        self.edges_series_pathes.sort(key=self.sort_key)

        self.series_folders_pathes = [
            [os.path.join(sfp, img_name) for img_name in os.listdir(sfp)]
            for sfp in self.series_folders_pathes
        ]

        self.clear_series_pathes = [
            [os.path.join(cfp, img_name) for img_name in os.listdir(cfp)]
            for cfp in self.clear_series_pathes
        ]

        self.edges_series_pathes = [
            [os.path.join(efp, img_name) for img_name in os.listdir(efp)]
            for efp in self.edges_series_pathes
        ]

        for i in range(len(self.series_folders_pathes)):
            self.series_folders_pathes[i].sort(key=self.sort_key)
            self.clear_series_pathes[i].sort(key=self.sort_key)
            self.edges_series_pathes[i].sort(key=self.sort_key)

        self.window_size = window_size

    def get_random_crop(self):
        select_series_index = random.randint(
            0,
            len(self.series_folders_pathes) - 1
        )

        select_image_index = random.randint(
            0,
            len(self.series_folders_pathes[select_series_index]) - 1
        )

        select_image = load_image(
            self.series_folders_pathes[select_series_index][select_image_index]
        )

        clear_image = load_image(
            self.clear_series_pathes[select_series_index][select_image_index]
        )

        edge_image = load_image(
            self.edges_series_pathes[select_series_index][select_image_index]
        )

        x = random.randint(0, select_image.shape[1] - self.window_size - 1)
        y = random.randint(0, select_image.shape[0] - self.window_size - 1)

        return (
            select_image[
               y:y + self.window_size,
               x:x + self.window_size
            ], clear_image[
               y:y + self.window_size,
               x:x + self.window_size
            ].copy(),
            edge_image[
                y:y + self.window_size,
                x:x + self.window_size
            ].copy()
        )

    @staticmethod
    def transformation(img1, img2, img3):
        rot_number = {
            0: cv2.ROTATE_90_CLOCKWISE,
            1: cv2.ROTATE_180,
            2: cv2.ROTATE_90_COUNTERCLOCKWISE
        }

        k = random.randint(0, 3)

        if k <= 2:
            img1 = cv2.rotate(img1, dst=None, rotateCode=rot_number[k])
            img2 = cv2.rotate(img2, dst=None, rotateCode=rot_number[k])
            img3 = cv2.rotate(img3, dst=None, rotateCode=rot_number[k])

        if random.random() > 0.5:
            return img1, img2, img3

        return cv2.flip(img1, 1), cv2.flip(img2, 1), cv2.flip(img3, 1)

    def __len__(self):
        return 50000

    def __getitem__(self, idx):
        img1, img2, img1_edge = self.transformation(*self.get_random_crop())

        fft_img = np.fft.rfft2(
            cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY),
            norm='ortho'
        )

        batt_kernel = generate_batt(img1.shape, 200, 2)
        after_batt_fft = apply_filter_to_fft(fft_img, 1 - batt_kernel)

        return (
            torch.FloatTensor(
                np.concatenate(
                    (
                        np.expand_dims(fft_img.real, 0),
                        np.expand_dims(fft_img.imag, 0)
                    ),
                    0
                )
            ),
            torch.FloatTensor(
                np.concatenate(
                    (
                        np.expand_dims(after_batt_fft.real, 0),
                        np.expand_dims(after_batt_fft.imag, 0)
                    ),
                    0
                )
            )
        )
