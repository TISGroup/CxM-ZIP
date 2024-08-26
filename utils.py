import math
import os
import random
import shutil
import sys
from enum import Enum
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
import yaml

import dxchange


def laplacian_loss(input_tensor, target_tensor):
    """
    Compute the Laplacian loss between the input and target tensors.

    Args:
        input_tensor (torch.Tensor): The input tensor, typically the output of a model.
        target_tensor (torch.Tensor): The target tensor, typically the ground truth image.

    Returns:
        torch.Tensor: The Laplacian loss value.
    """
    # Move the weight tensor to the same device as the input tensor
    laplacian_kernel = torch.Tensor([[[[-1, -1, -1],
                                       [-1, 8, -1],
                                       [-1, -1, -1]]]]).to(input_tensor.device)

    # Compute the Laplacian of the input and target tensors
    input_laplacian = F.conv2d(input_tensor, laplacian_kernel, padding=1)
    target_laplacian = F.conv2d(target_tensor, laplacian_kernel, padding=1)

    # Compute the L1 loss between the Laplacian of the input and target
    loss = torch.mean(torch.abs(input_laplacian - target_laplacian))

    return loss


def adjust_lr(optimizer, cur_epoch, all_epoch, cur_iter, data_size, cfg):
    all_iter = all_epoch * data_size
    now_iter = cur_epoch * data_size + cur_iter

    if cfg['optim']['lr_schedule'] == 'warmup_cosine':
        if now_iter < all_iter * cfg['optim']['lr_point']:
            lr_mult = 0.1 + 0.9 * now_iter / (all_iter * cfg['optim']['lr_point'])
        else:
            whole = all_iter - all_iter * cfg['optim']['lr_point']
            cur = now_iter - all_iter * cfg['optim']['lr_point']
            lr_mult = 0.5 * (math.cos(math.pi * cur / whole) + 1.0)
    else:
        raise NotImplementedError

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = cfg['optim']['lr'] * lr_mult

    return cfg['optim']['lr'] * lr_mult


def imadjust(img, in_min=None, in_max=None, out_min=0, out_max=1):
    """
    Adjust the intensity values in an image.

    Parameters:
    I (numpy.ndarray): The input grayscale image.
    in_min (float, optional): The minimum value in the input image to be mapped. If not provided, the 1st percentile value is used.
    in_max (float, optional): The maximum value in the input image to be mapped. If not provided, the 99th percentile value is used.
    out_min (float, optional): The minimum value in the output image. Default is 0.
    out_max (float, optional): The maximum value in the output image. Default is 1.

    Returns:
    numpy.ndarray: The adjusted image.
    """
    # Determine the input min and max values if not provided
    if in_min is None:
        in_min = np.percentile(img, 1)
    if in_max is None:
        in_max = np.percentile(img, 99)

    # Perform the linear mapping
    # J = np.clip((I - in_min) / (in_max - in_min) * (out_max - out_min) + out_min, out_min, out_max)

    # max min

    img_clipped = np.clip(img, in_min, in_max)
    img_res = img - img_clipped
    img_clipped_min = img_clipped.min()
    img_clipped_max = img_clipped.max()
    img_clipped = (img_clipped - img_clipped_min) / (img_clipped_max - img_clipped_min)
    img_clipped = img_clipped * (out_max - out_min) + out_min

    return img_clipped, img_clipped_min, img_clipped_max, img_res


def get_stride_list(h, s):
    assert s in [2, 4, 8, 16, 32]

    if s == 8:
        power_of_two = -5
    elif s == 4:
        power_of_two = -4
    elif s == 2:
        power_of_two = -3
    elif s == 16:
        power_of_two = -6
    elif s == 32:
        power_of_two = -7

    if h % 2 != 0:
        return None  # Invalid input, x must be an even number
    else:
        # power_of_two = -5
        while h % 2 == 0:
            h //= 2  # Divide x by 2 until it's no longer divisible by 2
            power_of_two += 1
        return [4] + [2] * power_of_two


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: str, resolution: int, energy=None):
        super(Dataset, self).__init__()
        assert data.endswith('h5') or data.endswith('txt')
        assert energy is None or data.endswith('.txt'), f'data with multiple energies must be specified energy param: {energy}'

        self.data = data
        self.energy = energy
        self.resolution = resolution

        # h5 file: single energy
        if data.endswith('h5'):
            data = h5py.File(data, 'r')

            thetas = data['exchange']['theta']
            self.thetas_max = np.max(thetas)
            self.thetas_min = np.min(thetas)
            self.num_thetas = len(thetas)

            self.data = data['exchange']['data']
            self.thetas = thetas

        # txt file: multiple energies
        elif data.endswith('txt'):
            assert energy is not None, f'data with multiple energies must be specified energy param: {energy}'
            energies, refs, collects = parse_scan_file(Path(data))
            # print(energies)
            energy_index = energies.index(self.energy)
            self.energy_index = energy_index
            flats, projs, thetas = load_energy_index(energy_index, refs, collects)

            self.thetas_max = np.max(thetas)
            self.thetas_min = np.min(thetas)
            self.num_thetas = len(thetas)

            self.data = projs
            self.thetas = thetas
            pass
        else:
            raise NotImplementedError

    def __len__(self):
        return self.num_thetas

    def __getitem__(self, idx, debug=False):

        if type(self.data) == h5py._hl.dataset.Dataset:
            img = self.data[idx]
        elif type(self.data) == np.ndarray:
            img = self.data[idx]
        else:
            raise NotImplementedError

        theta = self.thetas[idx]

        # normalize
        theta = (theta - self.thetas_min) / (self.thetas_max - self.thetas_min)

        # print(img.shape)
        img = TF.to_tensor(img.astype(float))  # [1, H, W]

        # crop first
        if self.resolution is not None:
            img = TF.center_crop(img, [self.resolution, self.resolution])

        img = img.numpy()
        # adjust
        img, img_min, img_max, res = imadjust(img)
        img = TF.to_tensor(img[0])  # [1, H, W]
        # print(img.shape)

        if debug:
            return theta, img, img_min, img_max, res
        else:
            return theta, img


def load_xrm_list(xrm_list):
    data_stack = None
    metadatas = []
    for i, filename in enumerate(xrm_list):
        data, metadata = dxchange.read_xrm(str(filename))
        if data_stack is None:
            data_stack = np.zeros((len(xrm_list),) + data.shape, data.dtype)
        data_stack[i] = data
        metadatas.append(metadata)
    return data_stack, metadatas


def parse_scan_file(txt_file):
    energies = []
    refs = []
    collects = []
    with open(txt_file, "r") as f:
        for line in f.readlines():
            if line.startswith("sete "):
                energies.append(float(line[5:]))
                refs.append([])
                collects.append([])
            elif line.startswith("collect "):
                filename = line[8:].strip()
                if "_ref_" in filename:
                    refs[-1].append(Path(txt_file).parent / filename)
                else:
                    collects[-1].append(Path(txt_file).parent / filename)
    return energies, refs, collects


def load_energy_index(energy_index, refs, collects):
    flats, _ = load_xrm_list(refs[energy_index])
    projs, metadatas = load_xrm_list(collects[energy_index])
    thetas = [metadata['thetas'][0] for metadata in metadatas]
    return flats, projs, thetas


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    file_path = os.path.join(save_dir, filename)
    torch.save(state, file_path)
    if is_best:
        best_file_path = os.path.join(save_dir, 'model_best.pth.tar')
        shutil.copyfile(file_path, best_file_path)


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_info(args):
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))


def print_cfg(cfg):
    print("CFG:")
    for k, v in cfg.items():
        if type(v) == dict:
            print('\t{}:'.format(k))
            for _k, _v in v.items():
                print('\t\t{}: {}'.format(_k, _v))
        else:
            print('\t{}: {}'.format(k, v))


def save_yaml(dir, args, save_name):
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(os.path.join(dir, save_name), 'w') as outfile:
        yaml.safe_dump(args, outfile, default_flow_style=False)


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
