import torch
from collections import OrderedDict
from util.image_pool import ImagePool
from util import util
from .base_model import BaseModel
from . import networks
import numpy as np


class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.half = opt.half

        self.use_D = self.opt.lambda_GAN > 0

        # specify the training losses you want to print out. The program will call base_model.get_current_losses

        if(self.use_D):
            self.loss_names = ['G_GAN', ]
        else:
            self.loss_names = []

        self.loss_names += ['G_CE', 'G_entr', 'G_entr_hint', ]
        self.loss_names += ['G_L1_max', 'G_L1_mean', 'G_entr', 'G_L1_reg', ]
        self.loss_names += ['G_fake_real', 'G_fake_hint', 'G_real_hint', ]
        self.loss_names += ['0', ]

        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks

        if self.isTrain:
            if(self.use_D):
                self.model_names = ['G', 'D']
            else:
                self.model_names = ['G', ]
        else:  # during test time, only load Gs
            self.model_names = ['G']

        # load/define networks
        #num_in = opt.input_nc + opt.output_nc + 1
        self.netG = networks.define_G(4, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                      use_tanh=True, classification=opt.classification)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if self.use_D:
                self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                              opt.which_model_netD,
                                              opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            # self.criterionL1 = torch.nn.L1Loss()
            self.criterionL1 = networks.L1Loss()
            self.criterionHuber = networks.HuberLoss(delta=1. / opt.ab_norm)

            # if(opt.classification):
            self.criterionCE = torch.nn.CrossEntropyLoss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

            if self.use_D:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)

        if self.half:
            for model_name in self.model_names:
                net = getattr(self, 'net' + model_name)
                net.half()
                for layer in net.modules():
                    if(isinstance(layer, torch.nn.BatchNorm2d)):
                        layer.float()
                print('Net %s half precision' % model_name)

        # initialize average loss values
        self.avg_losses = OrderedDict()
        self.avg_loss_alpha = opt.avg_loss_alpha
        self.error_cnt = 0

        # self.avg_loss_alpha = 0.9993 # half-life of 1000 iterations
        # self.avg_loss_alpha = 0.9965 # half-life of 200 iterations
        # self.avg_loss_alpha = 0.986 # half-life of 50 iterations
        # self.avg_loss_alpha = 0. # no averaging
        for loss_name in self.loss_names:
            self.avg_losses[loss_name] = 0

    def set_input(self, input):
        if(self.half):
            for key in input.keys():
                input[key] = input[key].half()

        self.real_rgb = input['REAL_RGB'].to(self.device)
        self.values = input['V'].to(self.device)
        self.mask_v = input['MASK_V'].to(self.device)

    def forward(self):
        (self.fake_B_class, self.fake_rgb_reg) = self.netG(self.values, self.mask_v)

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_rgb, self.fake_B), 1)) # TODO: what does it mean?
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # self.loss_D_fake = 0

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # self.loss_D_real = 0

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def compute_losses_G(self):
        #mask_avg = torch.mean(self.mask_B_nc.type(torch.cuda.FloatTensor)) + .000001

        self.loss_0 = 0  # 0 for plot

        self.loss_G_L1_reg = 10 * torch.mean(self.criterionL1(self.fake_rgb_reg.type(torch.cuda.FloatTensor),
                                                              self.real_rgb.type(torch.cuda.FloatTensor)))
        self.loss_G = self.loss_G_L1_reg

    def backward_G(self):
        self.compute_losses_G()
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        if(self.use_D):
            # update D
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

            self.set_requires_grad(self.netD, False)

        # update G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    # can be IGNORED for now
    def get_current_visuals(self):
        from collections import OrderedDict
        visual_ret = OrderedDict()

        visual_ret['real_rgb'] = self.real_rgb
        visual_ret['values'] = self.values
        visual_ret['fake_rgb'] = self.fake_rgb_reg

        return visual_ret

    # return training losses/errors. train.py will print out these errors as debugging information
    def get_current_losses(self):
        self.error_cnt += 1
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                self.avg_losses[name] = float(getattr(self, 'loss_' + name)) + self.avg_loss_alpha * self.avg_losses[name]
                errors_ret[name] = (1 - self.avg_loss_alpha) / (1 - self.avg_loss_alpha**self.error_cnt) * self.avg_losses[name]

        # errors_ret['|ab|_gt'] = float(torch.mean(torch.abs(self.real_B[:,1:,:,:])).cpu())
        # errors_ret['|ab|_pr'] = float(torch.mean(torch.abs(self.fake_B[:,1:,:,:])).cpu())

        return errors_ret
