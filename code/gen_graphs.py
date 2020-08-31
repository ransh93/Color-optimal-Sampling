from options.train_options import TrainOptions
from models import create_model

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from util import util
import numpy as np
import progressbar as pb
import shutil

import datetime as dt
import matplotlib.pyplot as plt

if __name__ == '__main__':
    opt = TrainOptions().parse()
    opt.load_model = True
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.display_id = -1  # no visdom display
    opt.phase = 'test'
    opt.dataroot = './dataset/ilsvrc2012/%s/' % opt.phase
    opt.loadSize = 128
    opt.how_many = 50
    opt.aspect_ratio = 1.0
    opt.sample_Ps = [6, ]
    opt.load_model = True
    opt.name = 'rgb_full_128'
    opt.input_nc = 4
    opt.output_nc = 3
    jump_number = 5

    dataset = torchvision.datasets.ImageFolder(opt.dataroot,
                                               transform=transforms.Compose([
                                                   transforms.Resize((opt.loadSize, opt.loadSize)),
                                                   transforms.ToTensor()]))
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=not opt.serial_batches)

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    time = dt.datetime.now()
    str_now = '%02d_%02d_%02d%02d' % (time.month, time.day, time.hour, time.minute)

    shutil.copyfile('./checkpoints/%s/latest_net_G.pth' % opt.name, './checkpoints/%s/%s_net_G.pth' % (opt.name, str_now))
    psnrs = {}
    mses = {}
    values_num = int((100 / jump_number) + 1)

    bar = pb.ProgressBar(max_value = values_num*values_num)
    k = 0

    for R in range(0,101,jump_number):
        for G in range(0,101,jump_number):
            for i, data_raw in enumerate(dataset_loader):
                data_raw[0] = data_raw[0].cuda()
                data_raw[0] = util.crop_mult(data_raw[0], mult=8)
                mask = util.build_mask(data_raw[0].shape, R=R, G=G).cuda()
                data = util.reveal_rgb_values(data_raw, mask, opt)

                model.set_input(data)
                model.test()
                visuals = model.get_current_visuals()

                if not (R,G) in psnrs:
                    psnrs[(R,G)] = np.zeros(opt.how_many)

                if not (R,G) in mses:
                    mses[(R,G)] = np.zeros(opt.how_many)

                mse_value = util.calculate_mse_np(util.tensor2im(visuals['real_rgb']), util.tensor2im(visuals['fake_rgb']))
                psnr_value = util.calculate_psnr_np(util.tensor2im(visuals['real_rgb']), util.tensor2im(visuals['fake_rgb']))
                mses[(R,G)][i] = mse_value
                psnrs[(R,G)][i] = psnr_value

                if i == opt.how_many - 1:
                    break
            k = k + 1
            bar.update(k)

    # Save results
    psnrs_mean = np.zeros((values_num, values_num))
    mses_mean = np.zeros((values_num, values_num))

    for i in range(psnrs_mean.shape[0]):
        for j in range(psnrs_mean.shape[1]):
            r,g = i*jump_number, j*jump_number
            if (r+g) > 100:
                psnrs_mean[i, j] = 0
                mses_mean[i,j] = 3000
            else:
                psnrs_mean[i,j] = np.mean(psnrs[(r,g)])
                mses_mean[i,j] = np.mean(mses[(r,g)])


    import seaborn as sns
    ticks = np.arange(0,101,jump_number)
    plt.figure(num = 1, figsize = (40,40))
    ax = sns.heatmap(psnrs_mean, annot=True, linewidth=1.0, xticklabels=ticks, yticklabels=ticks, square=True, fmt='.2f')
    ax.set_title("Red-Green PSNR Graph")
    plt.savefig('./checkpoints/%s/psnr_sweep_%s.png' % (opt.name, str_now))

    mses_mean = np.log10(mses_mean)
    plt.figure(num = 2, figsize = (40,40))
    ax2 = sns.heatmap(mses_mean, annot=False, linewidth=1.0, xticklabels=ticks, yticklabels=ticks, square=True, cmap="Blues")
    ax2.set_title("Red-Green MSE Graph")
    plt.savefig('./checkpoints/%s/mse_sweep_%s.png' % (opt.name, str_now))
