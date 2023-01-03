import argparse
import csv
import math
import os.path
import time
from collections import namedtuple
import configparser

import PIL
import numpy as np
import torch
import torch.nn as nn
import skimage
from skimage.metrics import mean_squared_error

from net import skip
from net.losses import ExclusionLoss
from net.noise import get_noise
from utils.image_io import *

parser = argparse.ArgumentParser()
parser.add_argument('--config', dest='config', default='ihc.cfg', type=str, help='path of configuration file')
args = parser.parse_args()


class Separation(object):
    def __init__(self):
        # read configuration file and prepare input image
        self.configfile = './config/' + args.config
        self.config_name = self.configfile.split('/')[-1].split('.')[0]
        self.process_config()
        self.preprocess_image()

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"new directory is created at {self.output_folder}.")

        # Initialize training parameters
        self.loss_function = None
        self.mses = []
        self.total_loss = None
        self.parameters = None
        self.current_result = None
        self.best_result = None

        # Initialize stain deconvolution parameters
        self.SCF1 = torch.tensor([1.0], device=torch.device("cuda:0"))
        self.SCF2 = torch.tensor([1.0], device=torch.device("cuda:0"))
        self.Con1_net_inputs = None
        self.Con2_net_inputs = None
        self.stain_net1_input = None
        self.stain_net2_input = None
        self.Con1_net = None
        self.Con2_net = None
        self.BG_net = None
        self.Con1_out = None
        self.Con2_out = None
        self.BG_out = None
        self.SeparationResult = namedtuple("SeparationResult", ['Stained1', 'Stained2', 'mse', 'SVec1', 'SVec2'])
        if self.num_stains == 3:
            self.SCF3 = torch.tensor([1.0], device=torch.device("cuda:0"))
            self.Con3_net = None
            self.Con3_net_inputs = None
            self.stain_net3_input = None
            self.Con3_out = None
            self.SeparationResult = namedtuple("SeparationResult",
                                          ['Stained1', 'Stained2', 'Stained3', 'mse', 'SVec1', 'SVec2', 'SVec3'])
        self._init_all()

    def process_config(self):
        self.config = configparser.ConfigParser()
        self.config.read(self.configfile)
        self.input = self.config['system']['input']
        self.input_folder = self.config['system']['input_folder']
        self.output_folder = self.config['system']['output_folder']
        self.output_folder = os.path.join(self.output_folder, f'{self.config_name}')
        self.input_depth = int(self.config['preprocess']['input_depth'])
        self.input_size = int(self.config['preprocess']['input_size'])
        self.roi_selection = self.config['preprocess']['roi_selection']
        self.x_start = int(self.config['preprocess']['x_coor_start'])
        self.y_start = int(self.config['preprocess']['y_coor_start'])
        self.num_stains = int(self.config['stain']['num_stains'])
        self.fixed_color = bool(self.config['stain']['fixed_color'])
        self.default_stain_vector_1 = self.config['stain']['default_stain_vector_1']
        self.default_stain_vector_1 = [float(x.strip('_')) for x in self.default_stain_vector_1.split(',')]
        self.default_stain_vector_2 = self.config['stain']['default_stain_vector_2']
        self.default_stain_vector_2 = [float(x.strip('_')) for x in self.default_stain_vector_2.split(',')]
        self.use_SCF = bool(self.config['stain']['use_spectral_correction_factor'])
        self.iters = int(self.config['network']['iterations'])
        self.keep_color_iters = int(self.config['network']['keep_color_iterations'])
        self.learning_rate = float(self.config['network']['init_learning_rate'])
        self.plot_during_training = bool(self.config['network']['plot_during_training'])
        self.show_every = int(self.config['network']['show_every'])
        self.weight_l1 = float(self.config['loss']['weight_l1'])
        self.weight_exclusion = float(self.config['loss']['weight_exclusion'])
        self.weight_mse = float(self.config['loss']['weight_mse'])
        if self.show_every > self.iters:
            self.show_every = self.iters

    def preprocess_image(self):
        self.image_filename = self.input
        self.image_path = os.path.join(self.input_folder, self.image_filename)
        self.image_name = self.image_filename.split('.')[0]
        # load() function has trouble loading some tiff images, replacing it with skimage.io.imread seems to work
        # i = load(self.image_path)
        i = skimage.io.imread(self.image_path)
        i = Image.fromarray(i)
        self.save_csv = os.path.join(self.output_folder, f'{self.config_name}_out.csv')
        if self.roi_selection == 'resize':
            i_crop = i.resize((self.input_size, self.input_size), resample=PIL.Image.BICUBIC)
        elif self.roi_selection == 'centercrop':
            center = [math.floor(i.height / 2), math.floor(i.width / 2)]
            i_crop = i.crop((center[1] - self.input_size / 2, center[0] - self.input_size / 2,
                             center[1] + self.input_size / 2, center[0] + self.input_size / 2))
        elif self.roi_selection == 'crop':
            i_crop = i.crop((self.x_start, self.y_start, self.x_start + self.input_size, self.y_start + self.input_size))
            self.save_csv = f'{self.config_name}_{self.x_start}_{self.y_start}.csv'
        else:
            i_crop = i
        i_crop.show()
        i_new = pil_to_np(i_crop)
        i_new = RGB_to_OD(i_new)
        self.image = i_new

    ################### Initialization functions ###################
    def _init_all(self):
        # Call initialization functions.
        self._init_images()
        self._init_nets()
        self._init_inputs()
        self._init_parameters()
        self._init_losses()

    def _init_images(self):
        # Create augmented versions of input and convert them to torch tensors.
        self.images = create_augmentations(self.image)
        self.images_torch = [np_to_torch(image).type(torch.cuda.FloatTensor) for image in self.images]
        save_image(self.image_name + "_original", self.images[0], self.output_folder)

    def _init_inputs(self):
        # Initialize input, default is using differnt random noise for different DIP modules
        input_type = 'noise'
        # input_type = 'meshgrid'
        data_type = torch.cuda.FloatTensor
        origin_noise = torch_to_np(get_noise(self.input_depth,
                                             input_type,
                                             (self.images_torch[0].shape[2],
                                              self.images_torch[0].shape[3])).type(data_type).detach())
        self.Con1_net_inputs = [np_to_torch(aug).type(data_type).detach() for aug in
                                create_augmentations(origin_noise)]

        origin_noise = torch_to_np(get_noise(self.input_depth,
                                             input_type,
                                             (self.images_torch[0].shape[2],
                                              self.images_torch[0].shape[3])).type(data_type).detach())
        self.Con2_net_inputs = [np_to_torch(aug).type(data_type).detach() for aug in
                                create_augmentations(origin_noise)]

        origin_noise = torch_to_np(get_noise(self.input_depth,
                                             input_type,
                                             (self.images_torch[0].shape[2],
                                              self.images_torch[0].shape[3])).type(data_type).detach())
        self.stain_net1_input = [np_to_torch(aug).type(data_type).detach() for aug in
                                 create_augmentations(origin_noise)]

        origin_noise = torch_to_np(get_noise(self.input_depth,
                                             input_type,
                                             (self.images_torch[0].shape[2],
                                              self.images_torch[0].shape[3])).type(data_type).detach())
        self.stain_net2_input = [np_to_torch(aug).type(data_type).detach() for aug in
                                 create_augmentations(origin_noise)]

        origin_noise = torch_to_np(get_noise(self.input_depth,
                                             input_type,
                                             (self.images_torch[0].shape[2],
                                              self.images_torch[0].shape[3])).type(data_type).detach())
        self.BG_net_inputs = [np_to_torch(aug).type(data_type).detach() for aug in
                              create_augmentations(origin_noise)]

        if self.num_stains == 3:
            origin_noise = torch_to_np(get_noise(self.input_depth,
                                                 input_type,
                                                 (self.images_torch[0].shape[2],
                                                  self.images_torch[0].shape[3])).type(data_type).detach())
            self.Con3_net_inputs = [np_to_torch(aug).type(data_type).detach() for aug in
                                    create_augmentations(origin_noise)]

            origin_noise = torch_to_np(get_noise(self.input_depth,
                                                 input_type,
                                                 (self.images_torch[0].shape[2],
                                                  self.images_torch[0].shape[3])).type(data_type).detach())
            self.stain_net3_input = [np_to_torch(aug).type(data_type).detach() for aug in
                                     create_augmentations(origin_noise)]

    def _init_parameters(self):
        # Initialize gradients.
        self.parameters = [p for p in self.Con1_net.parameters()] + \
                          [p for p in self.Con2_net.parameters()] + \
                          [p for p in self.BG_net.parameters()]
        self.parameters += [p for p in self.stain1.parameters()]
        self.parameters += [p for p in self.stain2.parameters()]

        if self.use_SCF:
            self.SCF1.requires_grad = True
            self.SCF2.requires_grad = True
            self.parameters += [self.SCF1, self.SCF2]

        if self.num_stains == 3:
            self.parameters += [p for p in self.Con3_net.parameters()]
            self.parameters += [p for p in self.stain3.parameters()]
            if self.use_SCF:
                self.SCF3.requires_grad = True
                self.parameters += [self.SCF3]

    def _init_nets(self):
        # Initialize network structures.
        data_type = torch.cuda.FloatTensor
        pad = 'reflection'
        Con1_net = skip(
            self.input_depth, 1,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.Con1_net = Con1_net.type(data_type)

        Con2_net = skip(
            self.input_depth, 1,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.Con2_net = Con2_net.type(data_type)

        BG_net = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.BG_net = BG_net.type(data_type)

        stain_net1 = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.stain1 = stain_net1.type(data_type)

        stain_net2 = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.stain2 = stain_net2.type(data_type)

        if self.num_stains == 3:
            Con3_net = skip(
                self.input_depth, 1,
                num_channels_down=[8, 16, 32, 64, 128],
                num_channels_up=[8, 16, 32, 64, 128],
                num_channels_skip=[0, 0, 0, 4, 4],
                upsample_mode='bilinear',
                filter_size_down=5,
                filter_size_up=5,
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

            self.Con3_net = Con3_net.type(data_type)

            stain_net3 = skip(
                self.input_depth, 3,
                num_channels_down=[8, 16, 32, 64, 128],
                num_channels_up=[8, 16, 32, 64, 128],
                num_channels_skip=[0, 0, 0, 4, 4],
                upsample_mode='bilinear',
                filter_size_down=5,
                filter_size_up=5,
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

            self.stain3 = stain_net3.type(data_type)

    def _init_losses(self):
        # Initialize loss functions.
        data_type = torch.cuda.FloatTensor
        self.l1_loss = nn.L1Loss().type(data_type)
        self.l1_loss_topk = nn.L1Loss(reduction='none').type(data_type)
        self.mse_loss = nn.MSELoss().type(data_type)
        self.exclusion_loss = ExclusionLoss().type(data_type)

    ################### Optimization functions ###################
    def optimize(self):
        # Optimize the network for one step.
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.iters):
            optimizer.zero_grad()
            # train one step and obtain current results
            self._optimization_closure(j)
            self._obtain_current_result(j)
            if self.plot_during_training:
                self._plot_closure(j)
            optimizer.step()

    def get_stains(self):
        # Normalize stain vectors.
        self.current_stain1_normalized_0 = torch.divide(self.current_stain1[0][0], torch.sqrt(
            torch.pow(self.current_stain1[0][0], 2) + torch.pow(self.current_stain1[0][1], 2) + torch.pow(
                self.current_stain1[0][2], 2)))
        self.current_stain1_normalized_1 = torch.divide(self.current_stain1[0][1], torch.sqrt(
            torch.pow(self.current_stain1[0][0], 2) + torch.pow(self.current_stain1[0][1], 2) + torch.pow(
                self.current_stain1[0][2], 2)))
        self.current_stain1_normalized_2 = torch.divide(self.current_stain1[0][2], torch.sqrt(
            torch.pow(self.current_stain1[0][0], 2) + torch.pow(self.current_stain1[0][1], 2) + torch.pow(
                self.current_stain1[0][2], 2)))

        self.current_stain2_normalized_0 = torch.divide(self.current_stain2[0][0], torch.sqrt(
            torch.pow(self.current_stain2[0][0], 2) + torch.pow(self.current_stain2[0][1], 2) + torch.pow(
                self.current_stain2[0][2], 2)))
        self.current_stain2_normalized_1 = torch.divide(self.current_stain2[0][1], torch.sqrt(
            torch.pow(self.current_stain2[0][0], 2) + torch.pow(self.current_stain2[0][1], 2) + torch.pow(
                self.current_stain2[0][2], 2)))
        self.current_stain2_normalized_2 = torch.divide(self.current_stain2[0][2], torch.sqrt(
            torch.pow(self.current_stain2[0][0], 2) + torch.pow(self.current_stain2[0][1], 2) + torch.pow(
                self.current_stain2[0][2], 2)))

        if self.num_stains == 3:
            self.current_stain3_normalized_0 = torch.divide(self.current_stain3[0][0], torch.sqrt(
                torch.pow(self.current_stain3[0][0], 2) + torch.pow(self.current_stain3[0][1], 2) + torch.pow(
                    self.current_stain3[0][2], 2)))
            self.current_stain3_normalized_1 = torch.divide(self.current_stain3[0][1], torch.sqrt(
                torch.pow(self.current_stain3[0][0], 2) + torch.pow(self.current_stain3[0][1], 2) + torch.pow(
                    self.current_stain3[0][2], 2)))
            self.current_stain3_normalized_2 = torch.divide(self.current_stain2[0][2], torch.sqrt(
                torch.pow(self.current_stain3[0][0], 2) + torch.pow(self.current_stain3[0][1], 2) + torch.pow(
                    self.current_stain3[0][2], 2)))

    def get_stained(self):
        # Generate stained images
        self.Stained1 = np.zeros((1, 3, self.input_size, self.input_size), dtype='float32')
        self.Stained1[0, 0, :, :] = (self.Con1_out[0] * self.current_stain1_normalized_0).detach().cpu()
        self.Stained1[0, 1, :, :] = (self.Con1_out[0] * self.current_stain1_normalized_1).detach().cpu()
        self.Stained1[0, 2, :, :] = (self.Con1_out[0] * self.current_stain1_normalized_2).detach().cpu()
        self.Stained2 = np.zeros((1, 3, self.input_size, self.input_size), dtype='float32')
        self.Stained2[0, 0, :, :] = (self.Con2_out[0] * self.current_stain2_normalized_0).detach().cpu()
        self.Stained2[0, 1, :, :] = (self.Con2_out[0] * self.current_stain2_normalized_1).detach().cpu()
        self.Stained2[0, 2, :, :] = (self.Con2_out[0] * self.current_stain2_normalized_2).detach().cpu()
        self.All = self.SCF1.item() * self.Stained1 + self.SCF2.item() * self.Stained2

        if self.num_stains == 3:
            self.Stained3 = np.zeros((1, 3, self.input_size, self.input_size), dtype='float32')
            self.Stained3[0, 0, :, :] = (self.Con3_out[0] * self.current_stain3_normalized_0).detach().cpu()
            self.Stained3[0, 1, :, :] = (self.Con3_out[0] * self.current_stain3_normalized_1).detach().cpu()
            self.Stained3[0, 2, :, :] = (self.Con3_out[0] * self.current_stain3_normalized_2).detach().cpu()
            self.All += self.SCF3.item() * self.Stained3

        self.All_wb = np.zeros((1, 3, self.input_size, self.input_size), dtype='float32')
        self.All_wb[0, 0, :, :] = self.All[0, 0, :, :] - torch.log10(self.current_BG[0][0]).item()
        self.All_wb[0, 1, :, :] = self.All[0, 1, :, :] - torch.log10(self.current_BG[0][1]).item()
        self.All_wb[0, 2, :, :] = self.All[0, 2, :, :] - torch.log10(self.current_BG[0][2]).item()

    def _get_augmentation(self, iteration):
        # Retrieve augmented images. Choose from 8 versions in #odd iters and use the original image in #even iters.
        if iteration % 2 == 1:
            return 0
        iteration //= 2
        return iteration % 8

    def _optimization_closure(self, step):
        # Calculate current stain deconvolution parameters and backpropagate gradients.
        # get augmentations
        if step == self.iters - 1:
            reg_noise_std = 0
        elif step < 1000:
            reg_noise_std = (1 / 1000.) * (step // 100)
        else:
            reg_noise_std = 1 / 1000.
        aug = self._get_augmentation(step)
        if step == self.iters - 1:
            aug = 0

        # calculate parameters, including stain vectors, concentration maps, background correction factor, and SCFs
        stain_net1_input = self.stain_net1_input[aug] + (self.stain_net1_input[aug].clone().normal_() * reg_noise_std)
        self.current_stain1 = self.stain1(stain_net1_input)[:, :,
                              self.images_torch[aug].shape[2] // 2:self.images_torch[aug].shape[2] // 2 + 1,
                              self.images_torch[aug].shape[3] // 2:self.images_torch[aug].shape[
                                                                       3] // 2 + 1] * 0.9 + 0.05

        stain_net2_input = self.stain_net2_input[aug] + (self.stain_net2_input[aug].clone().normal_() * reg_noise_std)
        self.current_stain2 = self.stain2(stain_net2_input)[:, :,
                              self.images_torch[aug].shape[2] // 2:self.images_torch[aug].shape[2] // 2 + 1,
                              self.images_torch[aug].shape[3] // 2:self.images_torch[aug].shape[
                                                                       3] // 2 + 1] * 0.9 + 0.05

        BG_net_input = self.BG_net_inputs[aug] + (self.BG_net_inputs[aug].clone().normal_() * reg_noise_std)
        self.current_BG = self.BG_net(BG_net_input)[:, :,
                          self.images_torch[aug].shape[2] // 2:self.images_torch[aug].shape[2] // 2 + 1,
                          self.images_torch[aug].shape[3] // 2:self.images_torch[aug].shape[3] // 2 + 1] * 0.9 + 0.05

        Con1_net_input = self.Con1_net_inputs[aug] + (
                self.Con1_net_inputs[aug].clone().normal_() * reg_noise_std)
        Con2_net_input = self.Con2_net_inputs[aug] + (
                self.Con2_net_inputs[aug].clone().normal_() * reg_noise_std)

        self.Con1_out = self.Con1_net(Con1_net_input)

        self.Con2_out = self.Con2_net(Con2_net_input)

        if self.num_stains == 3:
            Con3_net_input = self.Con3_net_inputs[aug] + (
                    self.Con3_net_inputs[aug].clone().normal_() * reg_noise_std)
            self.Con3_out = self.Con3_net(Con3_net_input)
            stain_net3_input = self.stain_net3_input[aug] + (
                        self.stain_net3_input[aug].clone().normal_() * reg_noise_std)
            self.current_stain3 = self.stain3(stain_net3_input)[:, :,
                                  self.images_torch[aug].shape[2] // 2:self.images_torch[aug].shape[2] // 2 + 1,
                                  self.images_torch[aug].shape[3] // 2:self.images_torch[aug].shape[
                                                                           3] // 2 + 1] * 0.9 + 0.05

        # get normalized stain vectors and generate stained images
        self.get_stains()
        self.get_stained()

        self.Con1R = self.SCF1 * self.Con1_out[0] * self.current_stain1_normalized_0
        self.Con1G = self.SCF1 * self.Con1_out[0] * self.current_stain1_normalized_1
        self.Con1B = self.SCF1 * self.Con1_out[0] * self.current_stain1_normalized_2
        self.Con2R = self.SCF2 * self.Con2_out[0] * self.current_stain2_normalized_0
        self.Con2G = self.SCF2 * self.Con2_out[0] * self.current_stain2_normalized_1
        self.Con2B = self.SCF2 * self.Con2_out[0] * self.current_stain2_normalized_2
        if self.num_stains == 2:
            self.Rnorm = self.Con1R + self.Con2R - torch.log10(self.current_BG[0][0])
            self.Gnorm = self.Con1G + self.Con2G - torch.log10(self.current_BG[0][1])
            self.Bnorm = self.Con1B + self.Con2B - torch.log10(self.current_BG[0][2])
        else:
            self.Con3R = self.SCF3 * self.Con3_out[0] * self.current_stain3_normalized_0
            self.Con3G = self.SCF3 * self.Con3_out[0] * self.current_stain3_normalized_1
            self.Con3B = self.SCF3 * self.Con3_out[0] * self.current_stain3_normalized_2
            self.Rnorm = self.Con1R + self.Con2R + self.Con3R - torch.log10(self.current_BG[0][0])
            self.Gnorm = self.Con1G + self.Con2G + self.Con3G - torch.log10(self.current_BG[0][1])
            self.Bnorm = self.Con1B + self.Con2B + self.Con3B - torch.log10(self.current_BG[0][2])

        self.OD = self.images_torch[aug][0][0] + self.images_torch[aug][0][1] + self.images_torch[aug][0][2]
        self.OD_pred = torch.squeeze(self.Rnorm + self.Gnorm + self.Bnorm)

        self.total_loss = self.weight_l1 * self.l1_loss(torch.squeeze(self.Rnorm), self.images_torch[aug][0][0])
        self.total_loss += self.weight_l1 * self.l1_loss(torch.squeeze(self.Gnorm), self.images_torch[aug][0][1])
        self.total_loss += self.weight_l1 * self.l1_loss(torch.squeeze(self.Bnorm), self.images_torch[aug][0][2])
        self.total_loss += self.weight_l1 * self.l1_loss(self.OD, self.OD_pred)

        self.total_loss += self.weight_exclusion * self.exclusion_loss(self.Con1_out.repeat(1, 3, 1, 1), self.Con2_out.repeat(1, 3, 1, 1))

        if self.num_stains == 3:
            self.total_loss += self.weight_exclusion * self.exclusion_loss(self.Con1_out.repeat(1, 3, 1, 1),
                                                       self.Con3_out.repeat(1, 3, 1, 1))
            self.total_loss += self.weight_exclusion * self.exclusion_loss(self.Con2_out.repeat(1, 3, 1, 1),
                                                       self.Con3_out.repeat(1, 3, 1, 1))
            self.total_loss -= 0.01 * self.l1_loss(self.current_stain1, self.current_stain2)
            self.total_loss -= 0.01 * self.l1_loss(self.current_stain1, self.current_stain3)
            self.total_loss -= 0.01 * self.l1_loss(self.current_stain2, self.current_stain3)

        if step < self.keep_color_iters:
            self.total_loss += self.weight_mse * self.mse_loss(self.current_stain1,
                                                  torch.tensor([[[[self.default_stain_vector_1[0]]], [[self.default_stain_vector_1[1]]], [[self.default_stain_vector_1[2]]]]]).type(
                                                      torch.cuda.FloatTensor))
            self.total_loss += self.weight_mse * self.mse_loss(self.current_stain2,
                                                  torch.tensor([[[[self.default_stain_vector_2[0]]], [[self.default_stain_vector_2[1]]], [[self.default_stain_vector_2[2]]]]]).type(
                                                      torch.cuda.FloatTensor))

            self.total_loss += self.weight_mse * self.mse_loss(self.current_BG,
                                                  torch.tensor([[[[1.]], [[1.]], [[1.]]]]).type(torch.cuda.FloatTensor))
            self.total_loss += self.weight_mse * self.mse_loss(self.SCF1, torch.tensor([1.]).type(torch.cuda.FloatTensor))
            self.total_loss += self.weight_mse * self.mse_loss(self.SCF2, torch.tensor([1.]).type(torch.cuda.FloatTensor))
            if self.num_stains == 3:
                self.total_loss += self.weight_mse * self.mse_loss(self.SCF3, torch.tensor([1.]).type(torch.cuda.FloatTensor))
        self.total_loss.backward()

    ################### Output&Visualization functions ###################
    def _obtain_current_result(self, step):
        """
        puts in self.current result the current result.
        also updates the best result
        :return:
        """
        if step == self.iters - 1 or step % 50 == 0 or step % self.show_every == self.show_every - 1:
            Con1_out_np = np.clip(torch_to_np(self.Con1_out), 0, 1)
            Con2_out_np = np.clip(torch_to_np(self.Con2_out), 0, 1)
            image_out_np = np.zeros((1, 3, self.input_size, self.input_size), dtype='float32')
            if self.num_stains == 2:
                image_out_np[0, 0, :, :] = (self.Con1_out[0] * self.current_stain1_normalized_0 + self.Con2_out[0]
                                            * self.current_stain2_normalized_0 - torch.log10(
                            self.current_BG[0][0])).detach().cpu()
                image_out_np[0, 1, :, :] = (self.Con1_out[0] * self.current_stain1_normalized_1 + self.Con2_out[0]
                                            * self.current_stain2_normalized_1 - torch.log10(
                            self.current_BG[0][1])).detach().cpu()
                image_out_np[0, 2, :, :] = (self.Con1_out[0] * self.current_stain1_normalized_2 + self.Con2_out[0]
                                            * self.current_stain2_normalized_2 - torch.log10(
                            self.current_BG[0][2])).detach().cpu()
            else:
                image_out_np[0, 0, :, :] = (self.Con1_out[0] * self.current_stain1_normalized_0 + self.Con2_out[0]
                                            * self.current_stain2_normalized_0 + self.Con3_out[0]
                                            * self.current_stain3_normalized_0 - torch.log10(
                            self.current_BG[0][0])).detach().cpu()
                image_out_np[0, 1, :, :] = (self.Con1_out[0] * self.current_stain1_normalized_1 + self.Con2_out[0]
                                            * self.current_stain2_normalized_1 + self.Con3_out[0]
                                            * self.current_stain3_normalized_1 - torch.log10(
                            self.current_BG[0][1])).detach().cpu()
                image_out_np[0, 2, :, :] = (self.Con1_out[0] * self.current_stain1_normalized_2 + self.Con2_out[0]
                                            * self.current_stain2_normalized_2 + self.Con3_out[0]
                                            * self.current_stain3_normalized_2 - torch.log10(
                            self.current_BG[0][2])).detach().cpu()

            mse = mean_squared_error(self.images[0], image_out_np[0])
            self.mses.append(mse)
            file = open(self.save_csv, 'a+', newline='')
            if self.num_stains == 2:
                self.current_result = self.SeparationResult(Stained1=Con1_out_np, Stained2=Con2_out_np,
                                                       mse=mse, SVec1=self.current_stain1, SVec2=self.current_stain2)
                with file:
                    # identifying header
                    header = ['Step', 'S1R', 'S1G', 'S1B', 'S2R', 'S2G', 'S2B', 'SCF1', 'SCF2', 'BGR', 'BGG', 'BGB']
                    writer = csv.DictWriter(file, fieldnames=header)

                    # writing data row-wise into the csv file
                    writer.writeheader()
                    writer.writerow({'Step': step,
                                     'S1R': self.current_stain1_normalized_0.item(),
                                     'S1G': self.current_stain1_normalized_1.item(),
                                     'S1B': self.current_stain1_normalized_2.item(),
                                     'S2R': self.current_stain2_normalized_0.item(),
                                     'S2G': self.current_stain2_normalized_1.item(),
                                     'S2B': self.current_stain2_normalized_2.item(),
                                     'SCF1': self.SCF1.item(), 'SCF2': self.SCF2.item(),
                                     'BGR': self.current_BG[0][0].item(),
                                     'BGG': self.current_BG[0][1].item(),
                                     'BGB': self.current_BG[0][2].item()})
            else:
                Con3_out_np = np.clip(torch_to_np(self.Con3_out), 0, 1)
                self.current_result = self.SeparationResult(Stained1=Con1_out_np, Stained2=Con2_out_np, Stained3=Con3_out_np,
                                                       mse=mse, SVec1=self.current_stain1, SVec2=self.current_stain2,
                                                       SVec3=self.current_stain3)
                with file:
                    # identifying header
                    header = ['Step', 'S1R', 'S1G', 'S1B', 'S2R', 'S2G', 'S2B', 'S3R', 'S3G', 'S3B', 'SCF1', 'SCF2',
                              'SCF3', 'BGR', 'BGG', 'BGB']
                    writer = csv.DictWriter(file, fieldnames=header)

                    # writing data row-wise into the csv file
                    writer.writeheader()
                    writer.writerow({'Step': step,
                                     'S1R': self.current_stain1_normalized_0.item(),
                                     'S1G': self.current_stain1_normalized_1.item(),
                                     'S1B': self.current_stain1_normalized_2.item(),
                                     'S2R': self.current_stain2_normalized_0.item(),
                                     'S2G': self.current_stain2_normalized_1.item(),
                                     'S2B': self.current_stain2_normalized_2.item(),
                                     'S3R': self.current_stain3_normalized_0.item(),
                                     'S3G': self.current_stain3_normalized_1.item(),
                                     'S3B': self.current_stain3_normalized_2.item(),
                                     'SCF1': self.SCF1.item(), 'SCF2': self.SCF2.item(), 'SCF3': self.SCF3.item(),
                                     'BGR': self.current_BG[0][0].item(),
                                     'BGG': self.current_BG[0][1].item(),
                                     'BGB': self.current_BG[0][2].item()})

            if self.best_result is None or self.best_result.mse < self.current_result.mse:
                self.best_result = self.current_result

    def _plot_closure(self, step):
        print('Iteration {:5d}    Loss {:5f}  MSE_gt: {:f}'.format(step,
                                                                   self.total_loss.item(),
                                                                   self.current_result.mse),
              '\n', end='')
        if step % self.show_every == self.show_every - 1:
            self.save_results(step)

    def finalize(self):
        save_graph(self.image_name + "_mse", self.mses, self.output_folder)

    def save_results(self, step):
        save_image(self.image_name + f"_Con1_{step}", self.current_result.Stained1, self.output_folder)
        save_image(self.image_name + f"_Con2_{step}", self.current_result.Stained2, self.output_folder)
        save_image(self.image_name + f"_Stained1_{step}", self.Stained1[0], self.output_folder)
        save_image(self.image_name + f"_Stained2_{step}", self.Stained2[0], self.output_folder)
        save_image(self.image_name + f"_WB_{step}", self.All_wb[0], self.output_folder)
        save_image(self.image_name + f"_All_{step}", self.All[0], self.output_folder)
        if self.num_stains == 3:
            save_image(self.image_name + f"_Con3_{step}", self.current_result.Stained3, self.output_folder)
            save_image(self.image_name + f"_Stained3_{step}", self.Stained3[0], self.output_folder)
        print(f"Step: {step}, SVec1 and SVec2 values are:")
        print(self.SCF1)
        print(self.SCF2)


if __name__ == "__main__":
    t_start = time.time()
    s = Separation()
    s.optimize()
    t_end = time.time()
    print(f"--- {int((t_end - t_start) / 60)} minutes and {(t_end - t_start) % 60} seconds ---")
    s.finalize()
