import argparse
import os
import shutil

import h5py
import numpy as np
import skimage
import torch
import torchvision.transforms.functional as TF
import yaml
from tqdm import tqdm

from utils import print_info, set_seed, imadjust


def main(args):
    print_info(args)


    train_cfg_path = os.path.join(args.log_dir, 'train.yml')
    train_cfg = yaml.safe_load(open(train_cfg_path, 'r'))

    data_path = train_cfg['data']
    assert data_path.endswith('h5'), f'only support h5 file'

    resolution = train_cfg['resolution']

    # fix
    set_seed(train_cfg['seed'])

    epoch_dir = os.path.join(args.log_dir, f'generated/epoch{args.epoch}')

    assert os.path.exists(epoch_dir), f'generated dir does not exist: {epoch_dir}'

    save_dir = os.path.join(args.log_dir, 'reconstructed')

    save_epoch_dir = os.path.join(save_dir, f'epoch{args.epoch}')

    if not os.path.exists(save_epoch_dir):
        os.makedirs(save_epoch_dir)

    save_path = os.path.join(save_dir, f'epoch{args.epoch}.h5')

    # copy
    shutil.copyfile(data_path, save_path)

    mat = h5py.File(data_path, 'r')
    rec = h5py.File(save_path, 'r+')
    print(mat.keys())

    # del
    if 'data' in rec['exchange'].keys():
        del rec['exchange']['data']

    if 'data_dark' in rec['exchange'].keys():
        del rec['exchange']['data_dark']

    if 'data_white' in rec['exchange'].keys():
        del rec['exchange']['data_white']

    # print(mat['exchange']['data'])

    data = mat['exchange']['data'][:]
    print(data.shape)
    target_dtype = data.dtype

    # get
    img_files = list(os.listdir(epoch_dir))
    img_files.sort()
    print(img_files)

    target_data = []
    for i in tqdm(range(len(data))):
        img = data[i]
        # target_dtype = img.dtype
        img = img.astype(float)
        img = torch.from_numpy(img)
        img = TF.center_crop(img, [resolution, resolution])
        img = img.numpy()
        img_ad, img_min, img_max, res = imadjust(img)

        target_img_path = os.path.join(save_epoch_dir, img_files[i])
        target_img = skimage.io.imread(target_img_path)

        target_img = target_img * (img_max - img_min) + img_min + res

        target_img.astype(target_dtype)

        i_str = str(i).zfill(len(str(len(data))))

        skimage.io.imsave(os.path.join(save_epoch_dir, f'{i_str}.tif'), target_img, check_contrast=False)

        target_data.append(target_img)

    target_data = np.asarray(target_data)
    target_data = target_data.astype(target_dtype)

    # print(target_data.shape, target_data.min(), target_data.max())

    def crop_save(key, mat, target_mat):
        data = mat['exchange'][key][:]
        # print(data.shape)

        target_dtype = data.dtype
        data = data.astype(float)
        data = torch.from_numpy(data)
        data = TF.center_crop(data, [resolution, resolution])
        data = data.numpy()
        data = data.astype(target_dtype)
        # print(data.shape)

        # create
        target_mat['exchange'].create_dataset(key, data=data, compression='gzip', shape=data.shape, dtype=target_dtype)

    crop_save('data_dark', mat, rec)
    crop_save('data_white', mat, rec)

    # create
    rec['exchange'].create_dataset('data', data=target_data, compression='gzip', shape=target_data.shape, dtype=target_dtype)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generating the h5 file')

    # model & optim cfg
    parser.add_argument('--log_dir', type=str, help='log path')

    parser.add_argument('--epoch', type=int, help='the generated images at this epoch')

    args = parser.parse_args()

    main(args)
