import argparse
import os

import numpy as np
import skimage
import torch
import yaml
from scipy.io import savemat
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

from utils import set_seed, print_info, Dataset
import zip


def main(args):
    print_info(args)

    ckp_dir = os.path.join(args.log_dir, 'ckp')

    train_cfg_path = os.path.join(args.log_dir, 'train.yml')
    train_cfg = yaml.safe_load(open(train_cfg_path, 'r'))

    model_cfg_path = os.path.join(args.log_dir, "model.yml")
    model_cfg = yaml.safe_load(open(model_cfg_path, 'r'))
    epochs = train_cfg['epochs']
    save_freq = train_cfg['save_freq']

    epoch_list = args.epochs
    if epoch_list is None:
        print(f'generating epochs from log')
        epoch_list = [i for i in range(1, epochs + 1) if i % save_freq == 0]

    save_dir = os.path.join(args.log_dir, "generated")

    cuda = train_cfg['cuda']
    print(f'cuda: {cuda}')

    # fix
    set_seed(train_cfg['seed'])

    # load data
    dataset = Dataset(data=train_cfg['data'], resolution=train_cfg['resolution'], energy=train_cfg['energy'])
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=train_cfg['num_workers'], pin_memory=True)

    for epoch in epoch_list:
        print(f'infer ckp at epoch: {epoch}')
        save_epoch_dir = os.path.join(save_dir, f'epoch{epoch}')

        if not os.path.exists(save_epoch_dir):
            os.makedirs(save_epoch_dir)

        # load ckp
        ckp_path = os.path.join(ckp_dir, f'ckp_epoch{epoch}.pth.tar')
        ckp = torch.load(ckp_path)

        model = getattr(zip, model_cfg['model']['model_name'])(cfg=model_cfg['model'])
        model.load_state_dict(ckp)
        model.eval()

        if cuda:
            model = model.cuda()

        ssim_list = []
        psnr_list = []

        for i, (theta, img) in tqdm(enumerate(data_loader), total=dataset.num_thetas):
            theta = theta.float()
            img = img.float()

            if cuda:
                theta = theta.cuda()

            with torch.no_grad():
                output = model(theta)

            # # max-min norm
            # output = (output - output.min()) / (output.max() - output.min())
            # # standardize
            # output = (output - output.mean()) / output.std()
            # # de-standardize
            # output = output * data.std() + data.mean()

            output = output.detach().cpu()  # [B, C, H, W]
            output = output[0, 0].numpy()

            i_str = str(i).zfill(len(str(dataset.num_thetas)))
            # print(i, data.min(), data.max(), output.min(), output.max())

            skimage.io.imsave(os.path.join(save_epoch_dir, f'{i_str}.tif'), output, check_contrast=False)

            img = img.detach().cpu()  # [B, C, H, W]
            img = img[0, 0].numpy()

            psnr = peak_signal_noise_ratio(image_true=img, image_test=output, data_range=1.0)
            ssim = structural_similarity(im1=img, im2=output, data_range=1.0, channel_axis=None)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

        psnr_list = np.asarray(psnr_list)
        ssim_list = np.asarray(ssim_list)

        print(f'epoch: {epoch}, psnr: {psnr_list.mean()} +- {psnr_list.std()}, ssim: {ssim_list.mean()} +- {ssim_list.std()}')

        # save metric
        savemat(os.path.join(save_dir, f'epoch{epoch}.mat'), {'psnr': psnr_list, 'ssim': ssim_list})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ZIP Inference script')

    # model & optim cfg
    parser.add_argument('--log_dir', type=str, help='log path')

    parser.add_argument('--epochs', nargs='+', help='ckp epochs')

    args = parser.parse_args()

    main(args)
