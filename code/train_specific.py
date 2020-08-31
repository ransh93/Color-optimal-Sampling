import time
from options.train_options import TrainOptions
from models import create_model
from util.visualizer import Visualizer

import torch
import torchvision
import torchvision.transforms as transforms

from util import util

import os

if __name__ == '__main__':
    opt = TrainOptions().parse()

    opt.dataroot = 'dataset/ilsvrc2012/%s' % opt.phase
    #dirs = os.listdir(opt.dataroot)
    #print(dirs)
    #print('data root %s' % opt.dataroot)
    dataset = torchvision.datasets.ImageFolder(opt.dataroot,
                                               transform=transforms.Compose([
                                                   transforms.RandomChoice([transforms.Resize(opt.loadSize, interpolation=1),
                                                                            transforms.Resize(opt.loadSize, interpolation=2),
                                                                            transforms.Resize(opt.loadSize, interpolation=3),
                                                                            transforms.Resize((opt.loadSize, opt.loadSize), interpolation=1),
                                                                            transforms.Resize((opt.loadSize, opt.loadSize), interpolation=2),
                                                                            transforms.Resize((opt.loadSize, opt.loadSize), interpolation=3)]),
                                                   transforms.RandomChoice([transforms.RandomResizedCrop(opt.fineSize, interpolation=1),
                                                                            transforms.RandomResizedCrop(opt.fineSize, interpolation=2),
                                                                            transforms.RandomResizedCrop(opt.fineSize, interpolation=3)]),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor()]))
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.num_threads))

    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    model.print_networks(True)

    #visualizer = Visualizer(opt)
    total_steps = 0

    R = opt.red_ratio
    G = opt.green_ratio
    mask = None

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay):
        print('epoch %d begin' % epoch)
        epoch_start_time = time.time()
        iter_data_time = time.time()


        iter_start_time = time.time()

        for i, data_raw in enumerate(dataset_loader):
            data_raw[0] = data_raw[0].cuda()

            if mask is None:
                mask = util.build_mask(data_raw[0].shape, R=R, G=G).cuda()
                pth = os.path.join(opt.checkpoints_dir, opt.name)
                fpth = os.path.join(pth, 'mask.pt')
                torch.save(mask, fpth)

            if data_raw[0].shape[0] != mask.shape[0]:
                continue

            data = util.reveal_rgb_values(data_raw, mask, opt)

            if(data is None):
                continue

            model.set_input(data)
            model.optimize_parameters()

        total_steps += opt.batch_size

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

    model.save_networks('latest')
