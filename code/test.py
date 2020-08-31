
import os
from options.train_options import TrainOptions
from models import create_model
from util.visualizer import save_images
from util import html

import string
import torch
import torchvision
import torchvision.transforms as transforms

from util import util
import numpy as np


if __name__ == '__main__':
    sample_ps = [1., .125, .03125]
    to_visualize = ['real_rgb', 'fake_rgb', ]
    S = len(sample_ps)

    opt = TrainOptions().parse()
    opt.load_model = True
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.display_id = -1  # no visdom display
    opt.phase = 'val'
    opt.dataroot = './dataset/ilsvrc2012/%s/' % opt.phase
    opt.serial_batches = True
    opt.aspect_ratio = 1.

    dataset = torchvision.datasets.ImageFolder(opt.dataroot,
                                               transform=transforms.Compose([
                                                   transforms.Resize((opt.loadSize, opt.loadSize)),
                                                   transforms.ToTensor()]))
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=not opt.serial_batches)

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    # statistics
    psnrs = np.zeros((opt.how_many))
    mses = np.zeros((opt.how_many))
    #entrs = np.zeros((opt.how_many, S))
    if opt.load_mask:
        pth = os.path.join(opt.checkpoints_dir, opt.name)
        mask = torch.load(os.path.join(pth, 'mask.pt')).cuda()
    for i, data_raw in enumerate(dataset_loader):
        data_raw[0] = data_raw[0].cuda()
        data_raw[0] = util.crop_mult(data_raw[0], mult=8)

        if not opt.load_mask:
            mask = util.build_mask(data_raw[0].shape, R=30, G=45).cuda()
        
        data = util.reveal_rgb_values(data_raw, mask, opt)

        img_path = [('%08d' % i).replace('.', 'p')]

        model.set_input(data)
        model.test(True)  # True means that losses will be computed
        visuals = util.get_subset_dict(model.get_current_visuals(), to_visualize)

        psnrs[i] = util.calculate_psnr_np(util.tensor2im(visuals['real_rgb']), util.tensor2im(visuals['fake_rgb']))
        mses[i] = util.calculate_mse_np(util.tensor2im(visuals['real_rgb']), util.tensor2im(visuals['fake_rgb']))
        #entrs[i, pp] = model.get_current_losses()['G_entr']

        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

        if i % 5 == 0:
            print('processing (%04d)-th image... ' % i)

        if i == opt.how_many - 1:
            break

    webpage.save()

    # Compute and print some summary statistics
    psnrs_mean = np.mean(psnrs, axis=0)
    psnrs_std = np.std(psnrs, axis=0) / np.sqrt(opt.how_many)

    mses_mean = np.mean(mses, axis=0)
    mses_std = np.std(mses, axis=0) / np.sqrt(opt.how_many)

    print('PSNR mean: %f std: %f' % (psnrs_mean, psnrs_std))
    for x in enumerate(psnrs):
        print(x)

    print('MSE mean: %f std: %f' % (mses_mean, mses_std))
    for x in enumerate(mses):
        print(x)
