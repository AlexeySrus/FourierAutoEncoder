import torch
import argparse
import os
from shutil import copyfile
import yaml
import numpy as np
import torch.nn.functional as F
from pipeline.model.model import Model, get_last_epoch_weights_path
from pipeline.utils.callbacks import (SaveModelPerEpoch, VisPlot,
                                        SaveOptimizerPerEpoch,
                                        VisImageForFFTAE)
from pipeline.utils.dataset_generator import SeriesAndComputingClearDataset
from pipeline.architectures.autoencoder import FFTAE
from torch.utils.data import DataLoader
from pipeline.utils.optimizers import Nadam, RangerAdam as Radam


def parse_args():
    parser = argparse.ArgumentParser(description='Video denoising train script')
    parser.add_argument('--config', required=False, type=str,
                          default='../configuration/train_config.yml',
                          help='Path to configuration yml file.'
                        )
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    return parser.parse_args()


def main():
    args = parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    batch_size = config['train']['batch_size']
    n_jobs = config['train']['number_of_processes']
    epochs = config['train']['epochs']
    window_size = config['model']['window_size']

    if not os.path.isdir(config['train']['save']['model']):
        os.makedirs(config['train']['save']['model'])

    copyfile(
        args.config,
        os.path.join(
            config['train']['save']['model'],
            os.path.basename(args.config)
        )
    )

    optimizers = {
        'adam': torch.optim.Adam,
        'nadam': Nadam,
        'radam': Radam,
        'sgd': torch.optim.SGD,
    }

    fft_shape= np.fft.rfft2(
        np.random.rand(window_size, window_size),
        norm='ortho'
    ).shape

    print('FFT shape: {}'.format(fft_shape))

    model = Model(
        FFTAE(
            fft_shape=fft_shape,
            latent_dim=config['model']['latent_dim']
        ),
        device
    )

    callbacks = []

    callbacks.append(SaveModelPerEpoch(
        os.path.join(
            os.path.dirname(__file__),
            config['train']['save']['model']
        ),
        config['train']['save']['every']
    ))

    callbacks.append(SaveOptimizerPerEpoch(
        os.path.join(
            os.path.dirname(__file__),
            config['train']['save']['model']
        ),
        config['train']['save']['every']
    ))

    if config['visualization']['use_visdom']:
        plots = VisPlot(
            'Video train',
            server=config['visualization']['visdom_server'],
            port=config['visualization']['visdom_port']
        )

        plots.register_scatterplot('train loss per_batch', 'Batch number',
                                   'Loss',
                                   [
                                       'mse between y_pred and y_true',
                                       'mse between y_pred and x'
                                   ])

        plots.register_scatterplot('train validation loss per_epoch', 'Batch number',
                                   'Loss',
                                   [
                                       'mse train loss',
                                       'double mse train loss'
                                   ])

        callbacks.append(plots)

        callbacks.append(
            VisImageForFFTAE(
                'Image visualisation',
                config['visualization']['visdom_server'],
                config['visualization']['visdom_port'],
                config['visualization']['image']['every'],
                scale=config['visualization']['image']['scale'],
                fft_shape=fft_shape
            )
        )

    model.set_callbacks(callbacks)

    start_epoch = 0
    if config['train']['optimizer'] != 'sgd':
        optimizer = optimizers[config['train']['optimizer']](
            model.model.parameters(),
            lr=config['train']['lr'],
            weight_decay=config['train']['weight_decay']
        )
    else:
        optimizer = torch.optim.SGD(
            model.model.parameters(),
            lr=config['train']['lr'],
            weight_decay=config['train']['weight_decay'],
            momentum=0.9,
            nesterov=True

        )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        verbose=True
    )

    if config['train']['load']:
        weight_path, optim_path, start_epoch = get_last_epoch_weights_path(
            os.path.join(
                os.path.dirname(__file__),
                config['train']['save']['model']
            ),
            print
        )

        if weight_path is not None:
            model.load(weight_path)
            # optimizer.load_state_dict(torch.load(optim_path,
            #                                      map_location='cpu'))

    train_data = DataLoader(
        SeriesAndComputingClearDataset(
            series_folders_path=config['dataset']['images_series_folder'],
            clear_series_path=config['dataset']['clear_images_path'],
            edges_path=config['dataset']['edges_images_path'],
            window_size=window_size,
        ),
        batch_size=batch_size,
        num_workers=n_jobs,
        drop_last=True
    )

    model.fit(
        train_data,
        (optimizer, scheduler),
        epochs,
        F.mse_loss,
        init_start_epoch=start_epoch + 1,
        validation_loader=None,
        is_epoch_scheduler=False
    )


if __name__ == '__main__':
    main()
