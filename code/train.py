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

    RG = [(10, 10), (10, 20), (10, 30), (10, 40), (10, 50), (10, 60), (10, 70), (10, 80), (10, 90), (20, 10), (20, 20), (20, 30),
     (20, 40), (20, 50), (20, 60), (20, 70), (30, 10), (30, 20), (30, 30), (30, 40), (30, 50), (30, 60), (40, 10),
     (40, 20), (40, 30), (40, 40), (40, 50), (50, 10), (50, 20), (50, 30), (50, 40), (60, 10), (60, 20), (60, 30),
     (70, 10), (70, 20), (80, 10), (90, 10)]

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay):
        print('epoch %d begin' % epoch)
        epoch_start_time = time.time()
        iter_data_time = time.time()


        iter_start_time = time.time()

        for i, data_raw in enumerate(dataset_loader):
            for (R, G) in RG:
                data_raw[0] = data_raw[0].cuda()
                mask = util.build_mask(data_raw[0].shape, R=R, G=G).cuda()
                data = util.reveal_rgb_values(data_raw, mask, opt)

                #data = util.get_colorization_data(data_raw, opt, p=opt.sample_p)
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
