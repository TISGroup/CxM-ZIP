import argparse
import os
import time
from datetime import datetime

import torch
import torch.nn.functional as F
import yaml
from pytorch_msssim import ssim
from torch.utils.tensorboard import SummaryWriter

import zip
from utils import AverageMeter, set_seed, print_info, save_checkpoint, ProgressMeter, save_yaml, \
    print_cfg, get_stride_list, adjust_lr, laplacian_loss, Dataset


def train(train_loader, model, optimizer, epoch, cuda, writer, print_freq, datasize, cfg, alpha, beta, gamma):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.6f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    for i, (theta, img) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        theta = theta.float()
        img = img.float()

        if cuda:
            theta = theta.cuda()
            img = img.cuda()

        output = model(theta)

        loss = alpha * F.mse_loss(output, img, reduction='none').mean() \
               + beta * (1 - ssim(output, img, data_range=1, size_average=True)) \
               + gamma * laplacian_loss(output, img)

        lr = adjust_lr(optimizer, epoch, cfg["epochs"], i, datasize, cfg)

        # log lr
        # writer.add_scalar('train/lr', lr, i + datasize * (epoch - 1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), img.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/data_time', data_time.avg, epoch)
    writer.add_scalar('train/batch_time', batch_time.avg, epoch)


def main(args):
    # fix seed
    set_seed(args.seed)

    args.datetime = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    cfg = yaml.safe_load(open(args.cfg, 'r'))

    log_dir = args.log_dir

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    ckp_dir = os.path.join(log_dir, 'ckp')
    if not os.path.exists(ckp_dir):
        os.makedirs(ckp_dir)

    # load data
    dataset = Dataset(data=args.data, resolution=args.resolution, energy=args.energy)

    # adapt to resolution
    if not 'stride_list' in cfg['model'].keys():
        _, _data = dataset[0]
        _c, _h, _w = _data.shape
        assert _h == _w, f'image size: {(_h, _w)} must be squared.'
        assert _h % 2 == 0, f"image size: {(_h, _w)} must be divisible by 2 entirely"

        feat_hwc = cfg['model']['feat_hwc']
        s = int(feat_hwc.split('_')[0])

        stride_list = get_stride_list(_h, s)
        cfg['model']['stride_list'] = stride_list
        print(f'data spatial resolution: ({_h}, {_w}), corresponding stride_list: {stride_list}')

    if args.save_freq is None:
        # save last
        args.save_freq = args.epochs
        print(f'args.save_freq is set to: {args.epochs}')

    if args.lr is not None:
        cfg['optim']['lr'] = args.lr
        print(f'lr is reset to: {args.lr} from args')

    print_info(args)
    print_cfg(cfg)

    datasize = len(dataset)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    writer = SummaryWriter(log_dir=log_dir)
    save_yaml(args.log_dir, vars(args), 'train.yml')
    save_yaml(args.log_dir, cfg, 'model.yml')

    model = getattr(zip, cfg['model']['model_name'])(cfg=cfg['model'])

    if args.cuda:
        model = model.cuda()

    optim_cfg = cfg['optim']
    if optim_cfg['optim_type'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=optim_cfg['lr'],
                                     betas=(optim_cfg['beta1'], optim_cfg['beta2']))
    else:
        raise NotImplementedError('optim_type: {} is not implemented.'.format(optim_cfg['optim_type']))

    cfg['epochs'] = args.epochs

    # if the model does not contain sth like BN layers, eval and train mode are almost the same
    model.train()
    for epoch in range(1, args.epochs + 1):

        train(train_loader, model, optimizer, epoch, args.cuda, writer, args.print_freq, datasize, cfg, args.alpha, args.beta, args.gamma)

        if epoch % args.save_freq == 0:
            save_checkpoint(model.state_dict(), False, ckp_dir, f'ckp_epoch{epoch}.pth.tar')

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ZIP Training script')

    # model & optim cfg
    parser.add_argument('--cfg', type=str, default='./cfgs/ZIP_deconv_conv_1024.yaml',
                        help='cfg file path (default: ./cfgs/ZIP_deconv_conv_1024.yaml), '
                             'adapt to any resolution automatically, e.g., 64/128/256/512/1024, etc.')
    # data
    parser.add_argument('--data', type=str,
                        help='h5 data path or txt path')

    parser.add_argument('--energy', type=float,
                        help='energy for data with multiple energy points (default: None)')

    parser.add_argument('--resolution', type=int, default=None,
                        help='resolution for training (default None), if None, use default resolution.')

    # parser.add_argument('--num_frames', type=int, default=None,
    #                     help='num of frames for training (default None), if None, use all frames')

    # training cfg
    parser.add_argument('--cuda', action='store_true',
                        help='use cuda')

    parser.add_argument('--log_dir', default='./runs', type=str, metavar='PATH',
                        help='where checkpoints and logs to be saved (default: ./runs)')

    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--epochs', default=1000, type=int,
                        help='training epochs (default: 1000)')

    parser.add_argument('--batch_size', default=1, type=int,
                        help='training batch size (default: 1)')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='number of workers for data loader (default: 4)')
    parser.add_argument('--print_freq', default=100, type=int,
                        help='print frequency (default: 100)')
    parser.add_argument('--save_freq', type=int, default=None,
                        help='save frequency for checkpoint and denoised results (default: None), if None, save last only')

    parser.add_argument('--alpha', type=float, default=0.5, help='coefficient for L2')
    parser.add_argument('--beta', type=float, default=0.3, help='coefficient for SSIM')
    parser.add_argument('--gamma', type=float, default=0.2, help='coefficient for Laplacian')

    # randomness
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')

    args = parser.parse_args()

    main(args)
