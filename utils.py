# -*- coding:utf-8 -*-
""" helper function

author junde
"""

import sys

import numpy

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.utils as vutils
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import autograd
import random
import math
import PIL
import matplotlib.pyplot as plt
import seaborn as sns

import collections
import logging
import math
import os
import time
from datetime import datetime

import dateutil.tz

from typing import Union, Optional, List, Tuple, Text, BinaryIO
import pathlib
import warnings
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
# from lucent.optvis.param.spatial import pixel_image, fft_image, init_image
# from lucent.optvis.param.color import to_valid_rgb
# from lucent.optvis import objectives, transform, param
# from lucent.misc.io import show
from torchvision.models import vgg19
import torch.nn.functional as F
import cfg

import warnings
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch

# from precpt import run_precpt
from models.discriminator import Discriminator
from metrics import Metrics
# from siren_pytorch import SirenNet, SirenWrapper

import shutil
import tempfile

import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR

from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)

args = cfg.parse_args()
device = torch.device('cuda', args.gpu_device)

'''preparation of domain loss'''
# cnn = vgg19(pretrained=True).features.to(device).eval()
# cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
# cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# netD = Discriminator(1).to(device)
# netD.apply(init_D)
# beta1 = 0.5
# dis_lr = 0.0002
# optimizerD = optim.Adam(netD.parameters(), lr=dis_lr, betas=(beta1, 0.999))
'''end'''

def generate_dot_prompts(gt_masks, all_classes, class_names=None, num_points_dict=None):
    """
    Generates dot prompts based on the ground truth masks.

    Args:
        gt_masks (torch.Tensor): Ground truth masks of shape (B, 1, H, W).
        num_classes (int): Number of classes.
        num_points_dict (dict): Dictionary specifying the number of points for each class.
                                E.g., {"background": 0, "bladder": 1, "placenta": 2, "placenta accreta": 4, "myometrium": 4}
    
    Returns:
        points (torch.Tensor): Tensor of shape (B, N, 2) for (x, y) coordinates of points, where N is constant across batches.
        labels (List[List[str]]): List of lists with class names for each point in each batch.
    """
    
    # Default number of points for each class if not specified in `num_points_dict`
    if num_points_dict is None:
        num_points_dict = {"background": 0, 
                           "bladder": 1, 
                           "placenta": 3, 
                           "placenta accreta": 5, 
                           "myometrium": 6, 
                           "fetus": 0, 
                           "infant": 0, 
                           "baby": 0,
                           "fat": 0,
                           "water": 0,
                           "muscle": 0,
                           "amniotic fluid": 0,
                           "uterine": 0,
                            "afterbirth": 0,
                            "vesica": 0,
                            "uterine wall": 0,
                            "pathological tissue": 0,
                            "cervix": 0,
                            "abdominal wall": 0,
                            "spinal column": 0,
                            "liver": 2,
                            "lungs": 2,
                            "kidneys": 2,
                            "bone": 2,
                            "brain": 2}

    batch_size, c, h, w = gt_masks.size()  # (B, 1, H, W)
    gt_masks = gt_masks.squeeze(1)  # Squeeze out the channel dimension
    points = []
    labels = []

    # Class names, ordered correctly
    if class_names is None:
        class_names = ["background", "bladder", "placenta", "placenta accreta", "myometrium"]
    num_classes = len(class_names)

    for b in range(batch_size):
        batch_points = []
        batch_labels = []
        
        # Iterate over all classes
        for class_name in class_names:
            cls = all_classes.index(class_name)
            
            # Get the mask for the current class
            mask = (gt_masks[b] == cls).float()  # Select class-specific pixels
            
            # Find coordinates of all non-zero (class) pixels in the mask
            nonzero_indices = torch.nonzero(mask)  # Shape: (num_nonzero, 2) with (y, x) coordinates
            
            # Get the number of points for this class from the dictionary
            num_points = num_points_dict.get(class_name, 0)  # Default to 0 if not found
            
            if len(nonzero_indices) > 0 and num_points > 0:  # If there are pixels and points required for this class
                # Limit the number of points if there are fewer available pixels than requested
                num_points = min(len(nonzero_indices), num_points)
                
                # Randomly sample indices from the non-zero pixel coordinates
                point_indices = torch.randint(0, len(nonzero_indices), (num_points,))
                
                for idx in point_indices:
                    point = nonzero_indices[idx].squeeze()  # Get the y, x for the point
                    batch_points.append([point[1].item(), point[0].item()])  # Append (x, y) coords
                
                # Add the corresponding class names based on the sampled points
                batch_labels.extend([class_name] * num_points)  # Use class name instead of index for labels

        # Append batch points and labels to the overall lists
        points.append(batch_points)
        labels.append(batch_labels)

    # Convert points to a tensor of shape (B, N, 2)
    points = torch.tensor(points, dtype=torch.long)

    return points, labels

def generate_box_prompts(gt_masks, num_classes=5, num_boxes_dict=None, surround_ratio=1.0):
    """
    Generates box prompts based on the ground truth masks.

    Args:
        gt_masks (torch.Tensor): Ground truth masks of shape (B, 1, H, W).
        num_classes (int): Number of classes.
        num_boxes_dict (dict): Dictionary specifying the number of boxes for each class.
                               E.g., {"background": 0, "bladder": 1, "placenta": 2, "placenta accreta": 4, "myometrium": 4}.
        surround_ratio (float): Ratio (e.g., 1.0 for 100%, 0.75 for 75%) controlling the bounding box tightness.

    Returns:
        points (torch.Tensor): Tensor of shape (B, N, 4) for (x_min, y_min, x_max, y_max) corner coordinates of boxes.
        labels (List[List[str]]): List of lists with class names for each box in each batch.
    """

    # Default number of boxes for each class if not specified in `num_boxes_dict`
    if num_boxes_dict is None:
        num_boxes_dict = {"background": 0, "bladder": 1, "placenta": 1, "placenta accreta": 1, "myometrium": 1}

    batch_size, c, h, w = gt_masks.size()  # (B, 1, H, W)
    gt_masks = gt_masks.squeeze(1)  # Squeeze out the channel dimension
    boxes = []
    labels = []

    # Class names, ordered correctly
    all_classes = ["background", "bladder", "placenta", "placenta accreta", "myometrium"]
    class_names = ["background", "bladder", "placenta", "placenta accreta", "myometrium"]
    num_classes = len(class_names)

    for b in range(batch_size):
        batch_boxes = []
        batch_labels = []

        # Iterate over all classes
        for class_name in class_names:
            cls = all_classes.index(class_name)

            # Get the mask for the current class
            mask = (gt_masks[b] == cls).float()  # Select class-specific pixels

            # Find coordinates of all non-zero (class) pixels in the mask
            nonzero_indices = torch.nonzero(mask)  # Shape: (num_nonzero, 2) with (y, x) coordinates

            # Get the number of boxes for this class from the dictionary
            num_boxes = num_boxes_dict.get(class_name, 0)  # Default to 0 if not found

            if len(nonzero_indices) > 0 and num_boxes > 0:  # If there are pixels and boxes required for this class
                # Limit the number of boxes if there are fewer available regions than requested
                num_boxes = min(1, num_boxes)

                # Calculate the bounding box (x_min, y_min, x_max, y_max) based on the mask
                y_coords, x_coords = nonzero_indices[:, 0], nonzero_indices[:, 1]
                x_min, y_min = x_coords.min().item(), y_coords.min().item()
                x_max, y_max = x_coords.max().item(), y_coords.max().item()

                # Apply surround_ratio to adjust the box size
                box_width = x_max - x_min
                box_height = y_max - y_min
                adjustment_w = int((box_width * (1.0 - surround_ratio)) / 2)
                adjustment_h = int((box_height * (1.0 - surround_ratio)) / 2)

                x_min = max(0, x_min - adjustment_w)
                y_min = max(0, y_min - adjustment_h)
                x_max = min(w - 1, x_max + adjustment_w)
                y_max = min(h - 1, y_max + adjustment_h)

                # Append the box coordinates and label
                batch_boxes.append([x_min, y_min])
                batch_boxes.append([x_max, y_max])
                batch_labels.append(class_name)

        # Append batch boxes and labels to the overall lists
        boxes.append(batch_boxes)
        labels.append(batch_labels)

    # Convert boxes to a tensor of shape (B, N, 2)
    max_boxes = max(len(b) for b in boxes)  # Handle varying number of boxes across batches
    padded_boxes = torch.zeros((batch_size, max_boxes, 2), dtype=torch.long)
    for i, b in enumerate(boxes):
        if len(b) > 0:
            padded_boxes[i, :len(b)] = torch.tensor(b, dtype=torch.long)

    return padded_boxes, labels

def count_parameters(model):
    """
    Count the total parameters (trainable and non-trainable) of a model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    return total_params, trainable_params, non_trainable_params

def show_tensor_image(tensor):
    """
    Display a tensor as an image using matplotlib.
    Args:
    tensor (torch.Tensor): The image tensor to display.
    """
    if tensor.dim() == 3 and tensor.shape[0] == 3:
        # Convert from (C, H, W) to (H, W, C) for RGB Image
        tensor = tensor.permute(1, 2, 0)
    tensor = tensor.cpu().numpy()  # Convert to numpy array and ensure it's on CPU
    tensor = np.clip(tensor, 0, 1)  # Ensure the tensor values are within the [0, 1] range

    plt.imshow(tensor)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()

def get_network(args, net, use_gpu=True, gpu_device=0, distribution=True):
    """ return given network
    """

    if net == 'sam':
        from models.sam import SamPredictor, sam_model_registry
        from models.sam.utils.transforms import ResizeLongestSide

        net = sam_model_registry['vit_b'](args, checkpoint=args.sam_ckpt).to(device)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if use_gpu:
        # net = net.cuda(device = gpu_device)
        if distribution != 'none':
            net = torch.nn.DataParallel(net, device_ids=[int(id) for id in args.distributed.split(',')])
            net = net.to(device=gpu_device)
        else:
            net = net.to(device=gpu_device)

    return net


def get_decath_loader(args):
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_size, args.roi_size, args.chunk),
                pos=1,
                neg=1,
                num_samples=args.num_sample,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
        ]
    )

    data_dir = args.data_path
    split_JSON = "dataset_0.json"

    datasets = os.path.join(data_dir, split_JSON)
    datalist = load_decathlon_datalist(datasets, True, "training")
    val_files = load_decathlon_datalist(datasets, True, "validation")
    train_ds = CacheDataset(
        data=datalist,
        transform=train_transforms,
        cache_num=24,
        cache_rate=1.0,
        num_workers=6,
    )
    train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=args.b, shuffle=True)
    val_ds = CacheDataset(
        data=val_files, transform=val_transforms, cache_num=2, cache_rate=1.0, num_workers=0
    )
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)

    set_track_meta(False)

    return train_loader, val_loader, train_transforms, val_transforms, datalist, val_files


def cka_loss(gram_featureA, gram_featureB):
    scaled_hsic = torch.dot(torch.flatten(gram_featureA), torch.flatten(gram_featureB))
    normalization_x = gram_featureA.norm()
    normalization_y = gram_featureB.norm()
    return scaled_hsic / (normalization_x * normalization_y)


def random_box(multi_rater):
    max_value = torch.max(multi_rater[:,0,:,:], dim=0)[0]
    max_value_position = torch.nonzero(max_value)

    x_coords = max_value_position[:, 0]
    y_coords = max_value_position[:, 1]


    x_min = int(torch.min(x_coords))
    x_max = int(torch.max(x_coords))
    y_min = int(torch.min(y_coords))
    y_max = int(torch.max(y_coords))


    x_min = random.choice(np.arange(x_min-10,x_min+11))
    x_max = random.choice(np.arange(x_max-10,x_max+11))
    y_min = random.choice(np.arange(y_min-10,y_min+11))
    y_max = random.choice(np.arange(y_max-10,y_max+11))

    return x_min, x_max, y_min, y_max

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


@torch.no_grad()
def make_grid(
        tensor: Union[torch.Tensor, List[torch.Tensor]],
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = False,
        value_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
        **kwargs
) -> torch.Tensor:
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if "range" in kwargs.keys():
        warning = "range will be deprecated, please use value_range instead."
        warnings.warn(warning)
        value_range = kwargs["range"]

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None:
            assert isinstance(value_range, tuple), \
                "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, low, high):
            img.clamp(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid


@torch.no_grad()
def save_image(
        tensor: Union[torch.Tensor, List[torch.Tensor]],
        fp: Union[Text, pathlib.Path, BinaryIO],
        format: Optional[str] = None,
        **kwargs
) -> None:
    """
    Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)


def create_logger(log_dir, phase='train'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def set_log_dir(root_dir, exp_name):
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    # set log path
    exp_path = os.path.join(root_dir, exp_name)
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = exp_path + '_' + timestamp
    os.makedirs(prefix)
    path_dict['prefix'] = prefix

    # set checkpoint path
    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path

    log_path = os.path.join(prefix, 'Log')
    os.makedirs(log_path)
    path_dict['log_path'] = log_path

    # set sample image path for fid calculation
    sample_path = os.path.join(prefix, 'Samples')
    os.makedirs(sample_path)
    path_dict['sample_path'] = sample_path

    plot_path = os.path.join(prefix, 'Plots')
    os.makedirs(plot_path)
    path_dict['plot_path'] = plot_path

    return path_dict


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))


class RunningStats:
    def __init__(self, WIN_SIZE):
        self.mean = 0
        self.run_var = 0
        self.WIN_SIZE = WIN_SIZE

        self.window = collections.deque(maxlen=WIN_SIZE)

    def clear(self):
        self.window.clear()
        self.mean = 0
        self.run_var = 0

    def is_full(self):
        return len(self.window) == self.WIN_SIZE

    def push(self, x):

        if len(self.window) == self.WIN_SIZE:
            # Adjusting variance
            x_removed = self.window.popleft()
            self.window.append(x)
            old_m = self.mean
            self.mean += (x - x_removed) / self.WIN_SIZE
            self.run_var += (x + x_removed - old_m - self.mean) * (x - x_removed)
        else:
            # Calculating first variance
            self.window.append(x)
            delta = x - self.mean
            self.mean += delta / len(self.window)
            self.run_var += delta * (x - self.mean)

    def get_mean(self):
        return self.mean if len(self.window) else 0.0

    def get_var(self):
        return self.run_var / len(self.window) if len(self.window) > 1 else 0.0

    def get_std(self):
        return math.sqrt(self.get_var())

    def get_all(self):
        return list(self.window)

    def __str__(self):
        return "Current window values: {}".format(list(self.window))


def iou(outputs: np.array, labels: np.array):
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return iou.mean()


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device=input.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


'''parameter'''


def para_image(w, h=None, img=None, mode='multi', seg=None, sd=None, batch=None,
               fft=False, channels=None, init=None):
    h = h or w
    batch = batch or 1
    ch = channels or 3
    shape = [batch, ch, h, w]
    param_f = fft_image if fft else pixel_image
    if init is not None:
        param_f = init_image
        params, maps_f = param_f(init)
    else:
        params, maps_f = param_f(shape, sd=sd)
    if mode == 'multi':
        output = to_valid_out(maps_f, img, seg)
    elif mode == 'seg':
        output = gene_out(maps_f, img)
    elif mode == 'raw':
        output = raw_out(maps_f, img)
    return params, output


def to_valid_out(maps_f, img, seg):  # multi-rater
    def inner():
        maps = maps_f()
        maps = maps.to(device=img.device)
        maps = torch.nn.Softmax(dim=1)(maps)
        final_seg = torch.multiply(seg, maps).sum(dim=1, keepdim=True)
        return torch.cat((img, final_seg), 1)
        # return torch.cat((img,maps),1)

    return inner


def gene_out(maps_f, img):  # pure seg
    def inner():
        maps = maps_f()
        maps = maps.to(device=img.device)
        # maps = torch.nn.Sigmoid()(maps)
        return torch.cat((img, maps), 1)
        # return torch.cat((img,maps),1)

    return inner


def raw_out(maps_f, img):  # raw
    def inner():
        maps = maps_f()
        maps = maps.to(device=img.device)
        # maps = torch.nn.Sigmoid()(maps)
        return maps
        # return torch.cat((img,maps),1)

    return inner


class CompositeActivation(torch.nn.Module):

    def forward(self, x):
        x = torch.atan(x)
        return torch.cat([x / 0.67, (x * x) / 0.6], 1)
        # return x


def cppn(args, size, img=None, seg=None, batch=None, num_output_channels=1, num_hidden_channels=128, num_layers=8,
         activation_fn=CompositeActivation, normalize=False, device="cuda:0"):
    r = 3 ** 0.5

    coord_range = torch.linspace(-r, r, size)
    x = coord_range.view(-1, 1).repeat(1, coord_range.size(0))
    y = coord_range.view(1, -1).repeat(coord_range.size(0), 1)

    input_tensor = torch.stack([x, y], dim=0).unsqueeze(0).repeat(batch, 1, 1, 1).to(device)

    layers = []
    kernel_size = 1
    for i in range(num_layers):
        out_c = num_hidden_channels
        in_c = out_c * 2  # * 2 for composite activation
        if i == 0:
            in_c = 2
        if i == num_layers - 1:
            out_c = num_output_channels
        layers.append(('conv{}'.format(i), torch.nn.Conv2d(in_c, out_c, kernel_size)))
        if normalize:
            layers.append(('norm{}'.format(i), torch.nn.InstanceNorm2d(out_c)))
        if i < num_layers - 1:
            layers.append(('actv{}'.format(i), activation_fn()))
        else:
            layers.append(('output', torch.nn.Sigmoid()))

    # Initialize model
    net = torch.nn.Sequential(OrderedDict(layers)).to(device)

    # Initialize weights
    def weights_init(module):
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.normal_(module.weight, 0, np.sqrt(1 / module.in_channels))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    net.apply(weights_init)
    # Set last conv2d layer's weights to 0
    torch.nn.init.zeros_(dict(net.named_children())['conv{}'.format(num_layers - 1)].weight)
    outimg = raw_out(lambda: net(input_tensor), img) if args.netype == 'raw' else to_valid_out(
        lambda: net(input_tensor), img, seg)
    return net.parameters(), outimg


def get_siren(args):
    wrapper = get_network(args, 'siren', use_gpu=args.gpu, gpu_device=torch.device('cuda', args.gpu_device),
                          distribution=args.distributed)
    '''load init weights'''
    checkpoint = torch.load('./logs/siren_train_init_2022_08_19_21_00_16/Model/checkpoint_best.pth')
    wrapper.load_state_dict(checkpoint['state_dict'], strict=False)
    '''end'''

    '''load prompt'''
    checkpoint = torch.load('./logs/vae_standard_refuge1_2022_08_21_17_56_49/Model/checkpoint500')
    vae = get_network(args, 'vae', use_gpu=args.gpu, gpu_device=torch.device('cuda', args.gpu_device),
                      distribution=args.distributed)
    vae.load_state_dict(checkpoint['state_dict'], strict=False)
    '''end'''

    return wrapper, vae


def siren(args, wrapper, vae, img=None, seg=None, batch=None, num_output_channels=1, num_hidden_channels=128,
          num_layers=8,
          activation_fn=CompositeActivation, normalize=False, device="cuda:0"):
    vae_img = torchvision.transforms.Resize(64)(img)
    latent = vae.encoder(vae_img).view(-1).detach()
    outimg = raw_out(lambda: wrapper(latent=latent), img) if args.netype == 'raw' else to_valid_out(
        lambda: wrapper(latent=latent), img, seg)
    # img = torch.randn(1, 3, 256, 256)
    # loss = wrapper(img)
    # loss.backward()

    # # after much training ...
    # # simply invoke the wrapper without passing in anything

    # pred_img = wrapper() # (1, 3, 256, 256)
    return wrapper.parameters(), outimg


'''adversary'''


def render_vis(
        args,
        model,
        objective_f,
        real_img,
        param_f=None,
        optimizer=None,
        transforms=None,
        thresholds=(256,),
        verbose=True,
        preprocess=True,
        progress=True,
        show_image=True,
        save_image=False,
        image_name=None,
        show_inline=False,
        fixed_image_size=None,
        label=1,
        raw_img=None,
        prompt=None
):
    if label == 1:
        sign = 1
    elif label == 0:
        sign = -1
    else:
        print('label is wrong, label is', label)
    if args.reverse:
        sign = -sign
    if args.multilayer:
        sign = 1

    '''prepare'''
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y, %H:%M:%S")

    netD, optD = pre_d()
    '''end'''

    if param_f is None:
        param_f = lambda: param.image(128)
    # param_f is a function that should return two things
    # params - parameters to update, which we pass to the optimizer
    # image_f - a function that returns an image as a tensor
    params, image_f = param_f()

    if optimizer is None:
        optimizer = lambda params: torch.optim.Adam(params, lr=5e-1)
    optimizer = optimizer(params)

    if transforms is None:
        transforms = []
    transforms = transforms.copy()

    # Upsample images smaller than 224
    image_shape = image_f().shape

    if fixed_image_size is not None:
        new_size = fixed_image_size
    elif image_shape[2] < 224 or image_shape[3] < 224:
        new_size = 224
    else:
        new_size = None
    if new_size:
        transforms.append(
            torch.nn.Upsample(size=new_size, mode="bilinear", align_corners=True)
        )

    transform_f = transform.compose(transforms)

    hook = hook_model(model, image_f)
    objective_f = objectives.as_objective(objective_f)

    if verbose:
        model(transform_f(image_f()))
        print("Initial loss of ad: {:.3f}".format(objective_f(hook)))

    images = []
    try:
        for i in tqdm(range(1, max(thresholds) + 1), disable=(not progress)):
            optimizer.zero_grad()
            try:
                model(transform_f(image_f()))
            except RuntimeError as ex:
                if i == 1:
                    # Only display the warning message
                    # on the first iteration, no need to do that
                    # every iteration
                    warnings.warn(
                        "Some layers could not be computed because the size of the "
                        "image is not big enough. It is fine, as long as the non"
                        "computed layers are not used in the objective function"
                        f"(exception details: '{ex}')"
                    )
            if args.disc:
                '''dom loss part'''
                # content_img = raw_img
                # style_img = raw_img
                # precpt_loss = run_precpt(cnn, cnn_normalization_mean, cnn_normalization_std, content_img, style_img, transform_f(image_f()))
                for p in netD.parameters():
                    p.requires_grad = True
                for _ in range(args.drec):
                    netD.zero_grad()
                    real = real_img
                    fake = image_f()
                    # for _ in range(6):
                    #     errD, D_x, D_G_z1 = update_d(args, netD, optD, real, fake)

                    # label = torch.full((args.b,), 1., dtype=torch.float, device=device)
                    # label.fill_(1.)
                    # output = netD(fake).view(-1)
                    # errG = nn.BCELoss()(output, label)
                    # D_G_z2 = output.mean().item()
                    # dom_loss = err
                    one = torch.tensor(1, dtype=torch.float)
                    mone = one * -1
                    one = one.cuda(args.gpu_device)
                    mone = mone.cuda(args.gpu_device)

                    d_loss_real = netD(real)
                    d_loss_real = d_loss_real.mean()
                    d_loss_real.backward(mone)

                    d_loss_fake = netD(fake)
                    d_loss_fake = d_loss_fake.mean()
                    d_loss_fake.backward(one)

                    # Train with gradient penalty
                    gradient_penalty = calculate_gradient_penalty(netD, real.data, fake.data)
                    gradient_penalty.backward()

                    d_loss = d_loss_fake - d_loss_real + gradient_penalty
                    Wasserstein_D = d_loss_real - d_loss_fake
                    optD.step()

                # Generator update
                for p in netD.parameters():
                    p.requires_grad = False  # to avoid computation

                fake_images = image_f()
                g_loss = netD(fake_images)
                g_loss = -g_loss.mean()
                dom_loss = g_loss
                g_cost = -g_loss

                if i % 5 == 0:
                    print(f' loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')
                    print(f'Generator g_loss: {g_loss}')
                '''end'''

            '''ssim loss'''

            '''end'''

            if args.disc:
                loss = sign * objective_f(hook) + args.pw * dom_loss
                # loss = args.pw * dom_loss
            else:
                loss = sign * objective_f(hook)
                # loss = args.pw * dom_loss

            loss.backward()

            # #video the images
            # if i % 5 == 0:
            #     print('1')
            #     image_name = image_name[0].split('\\')[-1].split('.')[0] + '_' + str(i) + '.png'
            #     img_path = os.path.join(args.path_helper['sample_path'], str(image_name))
            #     export(image_f(), img_path)
            # #end
            # if i % 50 == 0:
            #     print('Loss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
            #       % (errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            optimizer.step()
            if i in thresholds:
                image = tensor_to_img_array(image_f())
                # if verbose:
                #     print("Loss at step {}: {:.3f}".format(i, objective_f(hook)))
                if save_image:
                    na = image_name[0].split('\\')[-1].split('.')[0] + '_' + str(i) + '.png'
                    na = date_time + na
                    outpath = args.quickcheck if args.quickcheck else args.path_helper['sample_path']
                    img_path = os.path.join(outpath, str(na))
                    export(image_f(), img_path)

                images.append(image)
    except KeyboardInterrupt:
        print("Interrupted optimization at step {:d}.".format(i))
        if verbose:
            print("Loss at step {}: {:.3f}".format(i, objective_f(hook)))
        images.append(tensor_to_img_array(image_f()))

    if save_image:
        na = image_name[0].split('\\')[-1].split('.')[0] + '.png'
        na = date_time + na
        outpath = args.quickcheck if args.quickcheck else args.path_helper['sample_path']
        img_path = os.path.join(outpath, str(na))
        export(image_f(), img_path)
    if show_inline:
        show(tensor_to_img_array(image_f()))
    elif show_image:
        view(image_f())
    return image_f()


def tensor_to_img_array(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, [0, 2, 3, 1])
    return image


def view(tensor):
    image = tensor_to_img_array(tensor)
    assert len(image.shape) in [
        3,
        4,
    ], "Image should have 3 or 4 dimensions, invalid image shape {}".format(image.shape)
    # Change dtype for PIL.Image
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 4:
        image = np.concatenate(image, axis=1)
    Image.fromarray(image).show()


def export(tensor, img_path=None):
    # image_name = image_name or "image.jpg"
    c = tensor.size(1)
    # if c == 7:
    #     for i in range(c):
    #         w_map = tensor[:,i,:,:].unsqueeze(1)
    #         w_map = tensor_to_img_array(w_map).squeeze()
    #         w_map = (w_map * 255).astype(np.uint8)
    #         image_name = image_name[0].split('/')[-1].split('.')[0] + str(i)+ '.png'
    #         wheat = sns.heatmap(w_map,cmap='coolwarm')
    #         figure = wheat.get_figure()    
    #         figure.savefig ('./fft_maps/weightheatmap/'+str(image_name), dpi=400)
    #         figure = 0
    # else:
    if c == 3:
        vutils.save_image(tensor, fp=img_path)
    else:
        image = tensor[:, 0:3, :, :]
        w_map = tensor[:, -1, :, :].unsqueeze(1)
        image = tensor_to_img_array(image)
        w_map = 1 - tensor_to_img_array(w_map).squeeze()
        # w_map[w_map==1] = 0
        assert len(image.shape) in [
            3,
            4,
        ], "Image should have 3 or 4 dimensions, invalid image shape {}".format(image.shape)
        # Change dtype for PIL.Image
        image = (image * 255).astype(np.uint8)
        w_map = (w_map * 255).astype(np.uint8)

        Image.fromarray(w_map, 'L').save(img_path)


class ModuleHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = None
        self.features = None

    def hook_fn(self, module, input, output):
        self.module = module
        self.features = output

    def close(self):
        self.hook.remove()


def hook_model(model, image_f):
    features = OrderedDict()

    # recursive hooking function
    def hook_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                features["_".join(prefix + [name])] = ModuleHook(layer)
                hook_layers(layer, prefix=prefix + [name])

    hook_layers(model)

    def hook(layer):
        if layer == "input":
            out = image_f()
        elif layer == "labels":
            out = list(features.values())[-1].features
        else:
            assert layer in features, f"Invalid layer {layer}. Retrieve the list of layers with `lucent.modelzoo.util.get_model_layers(model)`."
            out = features[layer].features
        assert out is not None, "There are no saved feature maps. Make sure to put the model in eval mode, like so: `model.to(device).eval()`. See README for example."
        return out

    return hook


def vis_image(imgs, pred_masks, gt_masks, predicted_class , type_label, save_path, reverse=False, points=None):
    b, c, h, w = pred_masks.size()
    dev = pred_masks.get_device()
    row_num = min(b, 4)

    if torch.max(pred_masks) > 1 or torch.min(pred_masks) < 0:
        pred_masks = torch.sigmoid(pred_masks)

    if reverse == True:
        pred_masks = 1 - pred_masks
        gt_masks = 1 - gt_masks
    if c == 2:
        pred_disc, pred_cup = pred_masks[:, 0, :, :].unsqueeze(1).expand(b, 3, h, w), pred_masks[:, 1, :, :].unsqueeze(
            1).expand(b, 3, h, w)
        gt_disc, gt_cup = gt_masks[:, 0, :, :].unsqueeze(1).expand(b, 3, h, w), gt_masks[:, 1, :, :].unsqueeze(
            1).expand(b, 3, h, w)
        tup = (
            imgs[:row_num, :, :, :], pred_disc[:row_num, :, :, :], pred_cup[:row_num, :, :, :],
            gt_disc[:row_num, :, :, :],
            gt_cup[:row_num, :, :, :])
        # compose = torch.cat((imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:]),0)
        compose = torch.cat((pred_disc[:row_num, :, :, :], pred_cup[:row_num, :, :, :], gt_disc[:row_num, :, :, :],
                             gt_cup[:row_num, :, :, :]), 0)
        vutils.save_image(compose, fp=save_path, nrow=row_num, padding=10)
    else:
        imgs = torchvision.transforms.Resize((h, w))(imgs)
        if imgs.size(1) == 1:
            imgs = imgs[:, 0, :, :].unsqueeze(1).expand(b, 3, h, w)
        pred_masks = pred_masks[:, 0, :, :].unsqueeze(1).expand(b, 3, h, w)
        gt_masks = gt_masks[:, 0, :, :].unsqueeze(1).expand(b, 3, h, w)
        if points != None:
            for i in range(b):
                if args.thd:
                    p = np.round(points.cpu() / args.roi_size * args.out_size).to(dtype=torch.int)
                else:
                    p = np.round(points.cpu() / args.image_size * args.out_size).to(dtype=torch.int)
                # gt_masks[i,:,points[i,0]-5:points[i,0]+5,points[i,1]-5:points[i,1]+5] = torch.Tensor([255, 0, 0]).to(dtype = torch.float32, device = torch.device('cuda:' + str(dev)))
                gt_masks[i, 0, p[i, 0] - 5:p[i, 0] + 5, p[i, 1] - 5:p[i, 1] + 5] = 0.5
                gt_masks[i, 1, p[i, 0] - 5:p[i, 0] + 5, p[i, 1] - 5:p[i, 1] + 5] = 0.1
                gt_masks[i, 2, p[i, 0] - 5:p[i, 0] + 5, p[i, 1] - 5:p[i, 1] + 5] = 0.4
        tup = (imgs[:row_num, :, :, :], pred_masks[:row_num, :, :, :], gt_masks[:row_num, :, :, :])
        # compose = torch.cat((imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:]),0)
        compose = torch.cat(tup, 0)
        vutils.save_image(compose, fp=save_path, nrow=row_num, padding=10)

    return

import torchvision
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont

def visual_image(imgs, pred_masks, gt_masks, save_path, predicted_class, type_label, reverse=False, points=None):
    b, c, h, w = pred_masks.size()
    row_num = min(b, 4)  # Limiting the number of rows to display

    if torch.max(pred_masks) > 1 or torch.min(pred_masks) < 0:
        pred_masks = torch.sigmoid(pred_masks)

    if reverse:
        pred_masks = 1 - pred_masks
        gt_masks = 1 - gt_masks

    imgs = torchvision.transforms.Resize((h, w))(imgs)
    imgs = imgs[:, 0, :, :].unsqueeze(1).expand(b, 3, h, w)
    pred_masks = pred_masks[:, 0, :, :].unsqueeze(1).expand(b, 3, h, w)
    gt_masks = gt_masks[:, 0, :, :].unsqueeze(1).expand(b, 3, h, w)

    if points != None:
        for i in range(b):
            if args.thd:
                p = np.round(points.cpu() / args.roi_size * args.out_size).to(dtype=torch.int)
            else:
                p = np.round(points.cpu() / args.image_size * args.out_size).to(dtype=torch.int)
            # gt_masks[i,:,points[i,0]-5:points[i,0]+5,points[i,1]-5:points[i,1]+5] = torch.Tensor([255, 0, 0]).to(dtype = torch.float32, device = torch.device('cuda:' + str(dev)))
            gt_masks[i, 0, p[i, 0] - 5:p[i, 0] + 5, p[i, 1] - 5:p[i, 1] + 5] = 0.5
            gt_masks[i, 1, p[i, 0] - 5:p[i, 0] + 5, p[i, 1] - 5:p[i, 1] + 5] = 0.1
            gt_masks[i, 2, p[i, 0] - 5:p[i, 0] + 5, p[i, 1] - 5:p[i, 1] + 5] = 0.4

    # Convert tensors to PIL Images for more complex manipulations
    imgs_pil = [TF.to_pil_image(img.cpu()) for img in imgs]
    pred_masks_pil = [TF.to_pil_image(mask.cpu()) for mask in pred_masks]
    gt_masks_pil = [TF.to_pil_image(mask.cpu()) for mask in gt_masks]

    # Create a new PIL Image for each set to annotate and combine
    combined_images = []
    for i in range(row_num):
        # Draw text onto the original image
        img_draw = ImageDraw.Draw(imgs_pil[i])
        font = ImageFont.load_default()
        if predicted_class != "NaN":
            label_text = f'Predicted: {predicted_class[i].item()}, Actual: {type_label[i].item()}'
            text_color = 'red' if predicted_class[i] != type_label[i] else 'green'
            img_draw.text((10, 10), label_text, font=font, fill=text_color, font_size=40)

        # Combine image, pred_mask, and gt_mask side by side
        combined = Image.new('RGB', (w * 3, h))
        combined.paste(imgs_pil[i], (0, 0))
        combined.paste(pred_masks_pil[i], (w, 0))
        combined.paste(gt_masks_pil[i], (w * 2, 0))

        combined_images.append(TF.to_tensor(combined))

    # Stack all combined images to create a batch tensor
    combined_images_tensor = torch.stack(combined_images)

    # Save the combined image batch
    torchvision.utils.save_image(combined_images_tensor, save_path, nrow=1, padding=10)

    return

import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def generate_class_colors(classes):
    """
    Generate a dictionary mapping each unique class name in 'classes' to a random RGB color.
    """
    unique_classes = set([cls for batch_classes in classes for cls in batch_classes])
    class_colors = {cls: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for cls in unique_classes}
    return class_colors

def vis_mri_image(imgs, pred_masks, gt_masks, save_path, classes, class_colors=None, reverse=False, points=None, point_labels=None, alpha=0.5):
    """
    Visualize the MRI image with Ground Truth and Predicted Masks as transparent overlays.
    
    Args:
        imgs: Tensor of shape (B, C, H, W) - Input MRI images.
        pred_masks: Tensor of shape (B, H, W) - Predicted masks (class indices per pixel).
        gt_masks: Tensor of shape (B, H, W) - Ground truth masks (class indices per pixel).
        class_colors: Dictionary mapping class indices to RGB colors for visualization.
        save_path: Where to save the image.
        reverse: If true, reverse the mask values.
        points: Optional points to overlay on the image.
        point_labels: Optional labels corresponding to the points for color coding.
        alpha: Float (0 to 1), transparency factor for overlay masks.
    """
    if class_colors is None:
        class_colors = generate_class_colors(classes)

    b, c, h, w = imgs.size()  # Input dimensions
    row_num = min(b, 4)  # Limit the number of rows to display

    if reverse:
        pred_masks = 1 - pred_masks
        gt_masks = 1 - gt_masks

    # Resize images for consistency
    imgs = torchvision.transforms.Resize((h, w))(imgs)

    # Create figure for displaying images
    if row_num == 1:
        fig, axs = plt.subplots(1, 3, figsize=(24, 6))
    else:
        fig, axs = plt.subplots(row_num, 3, figsize=(24, 6 * row_num))

    if row_num == 1:
        axs = [axs]

    for i in range(row_num):
        # Convert MRI image to PIL format
        img_pil = TF.to_pil_image(imgs[i].cpu())
        
        # Generate color-coded masks for ground truth and predictions
        gt_overlay = create_overlay(img_pil, gt_masks[i].cpu(), class_colors, alpha)
        pred_overlay = create_overlay(img_pil, pred_masks[i].cpu(), class_colors, alpha)

        # Plot the MRI image
        axs[i][0].imshow(img_pil)
        axs[i][0].set_title("MRI Image", fontsize=12)
        axs[i][0].axis('off')

        # Plot the Ground Truth overlay
        axs[i][1].imshow(gt_overlay)
        axs[i][1].set_title("Ground Truth Overlay", fontsize=12)
        axs[i][1].axis('off')

        # Plot the Predicted overlay
        axs[i][2].imshow(pred_overlay)
        axs[i][2].set_title("Predicted Overlay", fontsize=12)
        axs[i][2].axis('off')

        # Generate a legend dynamically based on the classes in this batch item
        legend_patches = [Patch(color=np.array(class_colors[class_name]) / 255, label=class_name) for class_name in classes[i] if class_name in class_colors]
        fig.legend(handles=legend_patches, loc='upper center', ncol=len(classes[i]))

    # Save the resulting image
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def create_overlay(base_image, mask, classes, class_colors, alpha):
    """
    Create a semi-transparent overlay of the mask on the base image.
    
    Args:
        base_image: PIL image of the MRI.
        classes: List of defined label set in this task
        mask: 2D tensor of the mask with class indices.
        class_colors: Dictionary mapping class indices to RGB colors.
        alpha: Float (0 to 1), transparency factor for blending.
    
    Returns:
        PIL image with the mask overlay.
    """
    # Convert base image to RGB (if not already)
    base_image = base_image.convert("RGB")
    base_array = np.array(base_image)

    # Create a blank overlay
    overlay = np.zeros_like(base_array)

    # Assign colors to the mask regions
    # Create a mapping from class names to label indices
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_colors.keys())}

    # Assuming `mask` is a tensor with class indices
    for i, class_name in enumerate(classes):
        color = class_colors[class_name]
        class_idx = class_to_idx[class_name]  # Get the corresponding label number
        overlay[mask == i] = color  # Apply the color to the mask
    # Blend the base image and the overlay
    blended = np.clip(base_array * (1 - alpha) + overlay * alpha, 0, 255).astype(np.uint8)

    # Convert back to PIL for visualization
    return Image.fromarray(blended)


def vis_mri_pred(imgs, pred_masks, save_path, classes, class_colors=None, reverse=False, points=None, point_labels=None, alpha=0.5):
    """
    Visualize the MRI image with Ground Truth and Predicted Masks as transparent overlays.
    
    Args:
        imgs: Tensor of shape (B, C, H, W) - Input MRI images.
        pred_masks: Tensor of shape (B, H, W) - Predicted masks (class indices per pixel).
        gt_masks: Tensor of shape (B, H, W) - Ground truth masks (class indices per pixel).
        class_colors: Dictionary mapping class indices to RGB colors for visualization.
        save_path: Where to save the image.
        reverse: If true, reverse the mask values.
        points: Optional points to overlay on the image.
        point_labels: Optional labels corresponding to the points for color coding.
        alpha: Float (0 to 1), transparency factor for overlay masks.
    """
    if class_colors is None:
        class_colors = generate_class_colors(classes)

    b, c, h, w = imgs.size()  # Input dimensions
    row_num = min(b, 4)  # Limit the number of rows to display

    if reverse:
        pred_masks = 1 - pred_masks

    # Resize images for consistency
    imgs = torchvision.transforms.Resize((h, w))(imgs)

    for i in range(row_num):
        # Convert MRI image to PIL format
        img_pil = TF.to_pil_image(imgs[i].cpu())
        
        pred_overlay = create_overlay(img_pil, pred_masks[i].cpu(), class_colors, alpha)

        pred_overlay.save(save_path)
        
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import ImageDraw
import torchvision.transforms.functional as TF

def calculate_dice_coefficient(pred_mask, gt_mask):
    """
    Calculate the Dice coefficient between the predicted mask and ground truth mask.
    Args:
        pred_mask: Tensor of shape (H, W) - Predicted binary mask (0 or 1).
        gt_mask: Tensor of shape (H, W) - Ground truth binary mask (0 or 1).
    Returns:
        Dice coefficient value as a float.
    """
    intersection = torch.sum(pred_mask * gt_mask)
    union = torch.sum(pred_mask) + torch.sum(gt_mask)
    dice_coefficient = (2 * intersection) / (union + 1e-6)  # Adding small epsilon to avoid division by zero
    return dice_coefficient.item()

def vis_mri_image_with_no_prompt(
    imgs,
    gt_masks, 
    pred_masks, 
    pred_masks_no_prompt, 
    save_path, 
    classes, 
    class_colors=None, 
    reverse=False, 
    points=None, 
    point_labels=None, 
    boxes=None, 
    box_labels=None, 
    alpha=0.5
):
    """
    Visualize the MRI image with Ground Truth and Predicted Masks (with and without prompts) as transparent overlays.
    Args:
        imgs: Tensor of shape (B, C, H, W) - Input MRI images.
        pred_masks: Tensor of shape (B, H, W) - Predicted masks (class indices per pixel) with point prompts.
        gt_masks: Tensor of shape (B, H, W) - Ground truth masks (class indices per pixel).
        pred_masks_no_prompt: Tensor of shape (B, H, W) - Predicted masks without using point prompts.
        classes: List of lists, where each inner list contains class names relevant to each image in the batch.
        class_colors: Dictionary mapping class indices to RGB colors for visualization.
        save_path: Where to save the image.
        reverse: If true, reverse the mask values.
        points: Optional points to overlay on the image.
        point_labels: Optional labels corresponding to the points for color coding.
        boxes: Optional boxes to overlay, tensor of shape (B, N, 4), where each box is (x_min, y_min, x_max, y_max).
        box_labels: Optional labels corresponding to the boxes for color coding.
        alpha: Float (0 to 1), transparency factor for overlay masks.
    """
    # Generate colors automatically based on all classes in the batch
    if class_colors is None:
        class_colors = generate_class_colors(classes)

    num_classes = len(classes[0])
    
    b, c, h, w = imgs.size()  # Input dimensions
    row_num = min(b, 4)  # Limit the number of rows to display

    if reverse:
        pred_masks = 1 - pred_masks
        gt_masks = 1 - gt_masks
        pred_masks_no_prompt = 1 - pred_masks_no_prompt

    # Resize images for consistency
    imgs = torchvision.transforms.Resize((h, w))(imgs)

    mask_pred_m = F.one_hot(pred_masks, num_classes=num_classes).permute(0, 3, 1, 2).float()
    mask_pred_m_no_prompt = F.one_hot(pred_masks_no_prompt, num_classes=num_classes).permute(0, 3, 1, 2).float()

    # Create figure for displaying images
    if row_num == 1:
        fig, axs = plt.subplots(1, 4, figsize=(32, 6))  # Single row if batch size is 1
    else:
        fig, axs = plt.subplots(row_num, 4, figsize=(32, 6 * row_num))  # Multiple rows if batch size > 1

    if row_num == 1:
        axs = [axs]  # Wrap it in a list so it's iterable

    for i in range(row_num):
        # Convert MRI image to PIL for display
        img_pil = TF.to_pil_image(imgs[i].cpu())

        unique_pred = torch.unique(pred_masks, return_counts=True)
        unique_gt = torch.unique(gt_masks, return_counts=True)

        # Generate color-coded overlays for ground truth and predicted masks
        gt_overlay = create_overlay(img_pil, gt_masks[i].cpu(), classes[i], class_colors, alpha)
        pred_overlay = create_overlay(img_pil, pred_masks[i].cpu(), classes[i], class_colors, alpha)
        pred_no_prompt_overlay = create_overlay(img_pil, pred_masks_no_prompt[i].cpu(), classes[i], class_colors, alpha)

        metrics_calculator = Metrics(classes=classes[i])
        # Calculate Dice coefficient for both predicted masks (with and without prompts)
        dice_with_prompt = metrics_calculator.dice_score(mask_pred_m[i].unsqueeze(0), gt_masks[i].unsqueeze(0))
        dice_no_prompt = metrics_calculator.dice_score(mask_pred_m_no_prompt[i].unsqueeze(0), gt_masks[i].unsqueeze(0))

        # Draw points if provided
        if points is not None and point_labels is not None:
            draw = ImageDraw.Draw(img_pil)
            point_coords = points[i].cpu().numpy()  # Get point coordinates for this image
            point_labels_batch = point_labels[i]  # Access labels directly as nested list of strings

            for point, label in zip(point_coords, point_labels_batch):
                x, y = point
                if x != -1 and y != -1:  # Ignore padded points
                    color = class_colors.get(label, (255, 255, 255))  # Default to white if not found
                    # Draw a large star (cross) at the point location
                    draw.line((x - 10, y, x + 10, y), fill=color, width=2)
                    draw.line((x, y - 10, x, y + 10), fill=color, width=2)

        # Draw boxes if provided
        if boxes is not None and box_labels is not None:
            box_coords = boxes[i].cpu().numpy()
            box_labels_batch = box_labels[i]
            draw = ImageDraw.Draw(img_pil)
            for (x_min, y_min), (x_max, y_max), label in zip(box_coords[::2], box_coords[1::2], box_labels_batch):
                # Draw the box with significant visibility
                if x_min != -1 and y_min != -1 and x_max != -1 and y_max != -1:  # Ignore padded boxes
                    color = class_colors.get(label, (255, 255, 255))  # Default to white if not found
                    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=4)

        # Display the images in the appropriate subplot axes
        axs[i][0].imshow(img_pil)  # Display original MRI image
        axs[i][0].set_title("MRI Image", fontsize=12)
        axs[i][0].axis('off')

        axs[i][1].imshow(gt_overlay)  # Display ground truth overlay
        axs[i][1].set_title("Ground Truth Overlay", fontsize=12)
        axs[i][1].axis('off')

        axs[i][2].imshow(pred_overlay)  # Display predicted mask with prompts overlay
        axs[i][2].set_title(f"Predicted Masks (With Prompt)\nDice: {dice_with_prompt:.4f}", fontsize=12)
        axs[i][2].axis('off')

        axs[i][3].imshow(pred_no_prompt_overlay)  # Display predicted mask without prompts overlay
        axs[i][3].set_title(f"Predicted Masks (No Prompt)\nDice: {dice_no_prompt:.4f}", fontsize=12)
        axs[i][3].axis('off')

        # Generate a legend dynamically based on the classes in this batch item
        legend_patches = [Patch(color=np.array(class_colors[class_name]) / 255, label=class_name) for class_name in classes[i] if class_name in class_colors]
        fig.legend(handles=legend_patches, loc='upper center', ncol=len(classes[i]))

    # Save the resulting image
    plt.savefig(save_path)
    plt.close()


def vis_mri_image_compliation(o_imgs, masked_imgs, pred_imgs, save_path, reverse=False, points=None):
    """
    Visualize the MRI image, Masked Image, and Predicted Original Image side by side.
    
    imgs: Tensor of shape (B, C, H, W) - Original MRI images.
    masked_imgs: Tensor of shape (B, C, H, W) - Masked MRI images (with parts hidden).
    pred_imgs: Tensor of shape (B, C, H, W) - Predicted MRI images (the reconstructed output from the model).
    save_path: Path to save the visualized image.
    reverse: If true, reverse the mask values (if needed).
    points: Optional points to overlay on the image.
    """

    b, c, h, w = o_imgs.size()  # Input dimensions
    row_num = min(b, 4)  # Limit the number of rows to display

    if reverse:
        masked_imgs = 1 - masked_imgs
        pred_imgs = 1 - pred_imgs

    # Resize images for consistency (optional if not needed)
    imgs = torchvision.transforms.Resize((h, w))(o_imgs)

    masked_imgs = torchvision.transforms.Resize((h, w))(masked_imgs)
    pred_imgs = torchvision.transforms.Resize((h, w))(pred_imgs)

    # Create figure for displaying images
    if row_num == 1:
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # Single row if batch size is 1
    else:
        fig, axs = plt.subplots(row_num, 3, figsize=(18, 6 * row_num))  # Multiple rows if batch size > 1

    # If axs is a single row (batch size = 1), it will not be an array of arrays
    if row_num == 1:
        axs = [axs]  # Wrap it in a list so it's iterable

    # Now axs should work as a grid of axes
    for i in range(row_num):
        # Convert MRI images (original, masked, predicted) to PIL for display
        img_pil = TF.to_pil_image(imgs[i].cpu())  # Original image
        masked_img_pil = TF.to_pil_image(masked_imgs[i].cpu())  # Masked image
        pred_pil = TF.to_pil_image(pred_imgs[i].cpu())  # Predicted image

        # Display the images
        axs[i][0].imshow(img_pil)  # Display original MRI image
        axs[i][0].set_title("Original MRI Image", fontsize=12)
        axs[i][0].axis('off')

        axs[i][1].imshow(masked_img_pil)  # Display masked image
        axs[i][1].set_title("Masked MRI Image", fontsize=12)
        axs[i][1].axis('off')

        axs[i][2].imshow(pred_pil)  # Display predicted image
        axs[i][2].set_title("Predicted Original Image", fontsize=12)
        axs[i][2].axis('off')

    # Save the resulting image
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def color_code_mask(mask, class_colors, classes):
    """
    Color code a mask where each pixel is a class index.
    class_colors: A dictionary mapping class names to RGB values.
    classes: List of class names to use for this specific mask.
    """
    # Initialize a blank RGB mask with the same height and width as the input mask
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # Iterate over classes and assign colors based on class_colors
    for idx, class_name in enumerate(classes):
        if class_name in class_colors:
            color = class_colors[class_name]
            colored_mask[mask == idx] = color  # Apply the color to the corresponding class in the mask
    
    return colored_mask


import torchvision.transforms.functional as TF

def visualize_feature_map(feature_map, save_path, title="Feature Map", num_channels=6):
    """
    Visualizes the first 'num_channels' of the feature map.
    Args:
        feature_map: A tensor of shape (B, C, H, W).
        title: Title of the plot.
        num_channels: Number of channels to visualize.
    """
    feature_map = feature_map[0].cpu().detach().numpy()  # Assuming B=1 for visualization
    num_channels = min(num_channels, feature_map.shape[0])  # Max channels to display

    fig, axarr = plt.subplots(1, num_channels, figsize=(num_channels * 3, 3))
    
    for idx in range(num_channels):
        axarr[idx].imshow(feature_map[idx], cmap='viridis')
        axarr[idx].set_title(f"Channel {idx}")
        axarr[idx].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



def visualize_feature_map_greyscale(original_img, feature_map, save_path, title="Feature Map", num_channels=6):
    feature_map = feature_map.squeeze(0)
    
    # Summing feature map channels to create the greyscale representation
    gray_scale = torch.sum(feature_map, 0)
    gray_scale = gray_scale.cpu().detach().numpy()

    # Convert original image to CPU if necessary
    original_img = original_img.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # If the image is a tensor

    # Set up the figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # Plot the original image on the left
    axs[0].imshow(original_img)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    # Plot the greyscale feature map on the right
    axs[1].imshow(gray_scale, cmap='viridis')
    axs[1].set_title(title)
    axs[1].axis("off")

    # Save the figure
    plt.savefig(save_path, bbox_inches='tight')
    plt.close


def overlay_mask_on_image(image, mask, color=(255, 165, 0), alpha=0.1):
    """Overlay a semi-transparent mask on the image."""
    overlay = Image.new('RGBA', image.size, color + (0,))
    mask = mask.convert("L")
    overlay.paste(Image.new('RGBA', image.size, color + (int(255 * alpha),)), (0, 0), mask)
    return Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')

def visual_image_with_overlay(imgs, pred_masks, save_path, predicted_class, predicted_class_logit):
    b, c, h, w = pred_masks.size()
    row_num = min(b, 4)  # Limiting the number of rows to display

    # Ensure pred_masks are in range [0, 1]
    if torch.max(pred_masks) > 1 or torch.min(pred_masks) < 0:
        pred_masks = torch.sigmoid(pred_masks)

    # Resize images to match masks' dimensions
    imgs = torchvision.transforms.Resize((h, w))(imgs)
    imgs = imgs[:, 0, :, :].unsqueeze(1).expand(b, 3, h, w)
    
    pred_masks = pred_masks[:, 0, :, :].unsqueeze(1).expand(b, 1, h, w)  # Only keep one channel for mask

    # Convert tensors to PIL Images for more complex manipulations
    imgs_pil = [TF.to_pil_image(img.cpu()) for img in imgs]
    pred_masks_pil = [TF.to_pil_image(mask.cpu()) for mask in pred_masks]

    combined_images = []
    for i in range(row_num):
        # Overlay the mask on the image
        combined = overlay_mask_on_image(imgs_pil[i], pred_masks_pil[i])

        # Draw text onto the combined image
        draw = ImageDraw.Draw(combined)
        font = ImageFont.truetype("SimSun.ttf", 40)  # Use a larger font size
        label_text = f'预测: {"良性" if predicted_class[i] == 0 else "恶性"} ({predicted_class_logit:.2f})'
        text_color = 'red' if predicted_class[i] == 1 else 'green'
        text_size = draw.textlength(label_text, font=font)
        draw.text((combined.width - text_size - 10, 10), label_text, font=font, fill=text_color)

        combined_images.append(TF.to_tensor(combined))

    # Stack all combined images to create a batch tensor
    combined_images_tensor = torch.stack(combined_images)

    # Save the combined image batch
    torchvision.utils.save_image(combined_images_tensor, save_path, nrow=1, padding=10)

    return

def plot_confusion_matrix(TT, TF, FT, FF, plot_path, epoch):
    matrix = np.array([[len(TT), len(FT)],
                       [len(TF), len(FF)]])
    total = matrix.sum()
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Update title to include total amount
    ax.set_title(f'Confusion Matrix - Total: {total}')
    ax.set(xticks=np.arange(matrix.shape[1]),
           yticks=np.arange(matrix.shape[0]),
           xticklabels=['0', '1'], yticklabels=['0', '1'],
           ylabel='Actual label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations with better contrast
    fmt = 'd'
    thresh = matrix.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, format(matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if matrix[i, j] > thresh else "black")

    fig.tight_layout()
    cmat_name = os.path.join(plot_path, str(epoch) + '_confusion_matrix.png')
    fig.savefig(cmat_name, dpi=300, bbox_inches='tight')


def plot_metrics(plot_path, train_loss, val_loss, train_iou, val_iou, train_dice, val_dice, epochs, val_freq):
    """
    Plots the training and validation loss, IoU, and Dice coefficient over epochs.
    Validation metrics are plotted according to the validation frequency.

    Args:
    - train_loss (list): List of training losses over epochs.
    - val_loss (list): List of validation losses.
    - train_iou (list): List of training IoU scores over epochs.
    - val_iou (list): List of validation IoU scores.
    - train_dice (list): List of Dice coefficients for training epochs.
    - val_dice (list): List of Dice coefficients for validation epochs.
    - epochs (int): Total number of epochs.
    - val_freq (int): Frequency of validation per number of epochs.
    """
    epoch_range = range(1, epochs + 1)
    val_epochs = range(1, epochs + 1, val_freq)  # Validation happens every val_freq epochs

    plt.figure(figsize=(12, 10))

    # Plotting training and validation loss
    plt.subplot(3, 1, 1)
    plt.plot(epoch_range, train_loss, 'b-', marker='o', label='Training Loss')
    plt.plot(val_epochs, val_loss, 'r-', marker='o', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plotting training and validation IoU
    plt.subplot(3, 1, 2)
    plt.plot(epoch_range, train_iou, 'b-', marker='o', label='Training IoU')
    plt.plot(val_epochs, val_iou, 'r-', marker='o', label='Validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.title('Training and Validation IoU')
    plt.legend()

    # Plotting training and validation Dice coefficient
    plt.subplot(3, 1, 3)
    plt.plot(epoch_range, train_dice, 'b-', marker='o', label='Training Dice Coefficient')
    plt.plot(val_epochs, val_dice, 'g-', marker='o', label='Validation Dice Coefficient')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.title('Training and Validation Dice Coefficient')
    plt.legend()

    metric_name = os.path.join(plot_path, str(epochs) + '_metrics_plot.png')
    plt.savefig(metric_name, dpi=300, format='png', bbox_inches='tight')
    plt.close()
  
from scipy.ndimage import binary_dilation

def enlarge_mask(mask, radius):
    """
    Enlarges the mask by a given radius using a binary dilation operation.
    
    Args:
        mask (torch.Tensor): A binary mask of shape (B, 1, H, W).
        radius (int): The radius by which to enlarge the mask.
    
    Returns:
        torch.Tensor: The enlarged mask.
    """
    # Ensure mask is a binary mask
    mask = mask > 0.5

    # Convert to numpy array for processing
    mask_np = mask.cpu().numpy()

    # Create a structuring element (disk) for dilation
    struct_element = np.ones((2*radius+1, 2*radius+1))

    # Apply binary dilation
    enlarged_mask_np = np.array([binary_dilation(m[0], structure=struct_element) for m in mask_np])

    # Convert back to torch tensor
    enlarged_mask = torch.tensor(enlarged_mask_np, dtype=torch.float32).unsqueeze(1).to(mask.device)

    return enlarged_mask

def eval_seg(pred, true_mask_p, threshold):
    '''
    threshold: a int or a tuple of int
    masks: [b,2,h,w]
    pred: [b,2,h,w]
    '''
    b, c, h, w = pred.size()
    if c == 2:
        iou_d, iou_c, disc_dice, cup_dice = 0, 0, 0, 0
        for th in threshold:
            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:, 0, :, :].numpy().astype('int32')
            cup_pred = vpred_cpu[:, 1, :, :].numpy().astype('int32')

            disc_mask = gt_vmask_p[:, 0, :, :].squeeze(1).cpu().numpy().astype('int32')
            cup_mask = gt_vmask_p[:, 1, :, :].squeeze(1).cpu().numpy().astype('int32')

            '''iou for numpy'''
            iou_d += iou(disc_pred, disc_mask)
            iou_c += iou(cup_pred, cup_mask)

            '''dice for torch'''
            disc_dice += dice_coeff(vpred[:, 0, :, :], gt_vmask_p[:, 0, :, :]).item()
            cup_dice += dice_coeff(vpred[:, 1, :, :], gt_vmask_p[:, 1, :, :]).item()

        return iou_d / len(threshold), iou_c / len(threshold), disc_dice / len(threshold), cup_dice / len(threshold)
    else:
        eiou, edice = 0, 0
        for th in threshold:
            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:, 0, :, :].numpy().astype('int32')

            disc_mask = gt_vmask_p[:, 0, :, :].squeeze(1).cpu().numpy().astype('int32')

            '''iou for numpy'''
            eiou += iou(disc_pred, disc_mask)

            '''dice for torch'''
            edice += dice_coeff(vpred[:, 0, :, :], gt_vmask_p[:, 0, :, :]).item()

        return eiou / len(threshold), edice / len(threshold)


def eval_seg(pred, true_mask_p, num_classes):
    '''
    pred: [1, 1, H, W] - Predicted class indices
    true_mask_p: [1, 1, H, W] - Ground truth class indices
    num_classes: Total number of classes (including background)
    '''
    assert pred.shape == true_mask_p.shape, "Pred and True mask must have the same shape"
    
    # Remove the channel dimension since it's [1, 1, H, W]
    pred = pred.squeeze(1)
    true_mask_p = true_mask_p.squeeze(1)
    
    iou_per_class = []
    dice_per_class = []
    
    for cls in range(num_classes):
        # Create binary masks for each class
        pred_cls = (pred == cls).float()
        true_cls = (true_mask_p == cls).float()
        
        intersection = (pred_cls * true_cls).sum()
        union = pred_cls.sum() + true_cls.sum()
        
        dice = (2 * intersection) / (union + 1e-6)  # Add epsilon to avoid division by zero
        dice_per_class.append(dice.item())
        
        # IoU calculation
        union = (pred_cls + true_cls).clamp(0, 1).sum()
        iou = intersection / (union + 1e-6)
        iou_per_class.append(iou.item())
    
    return iou_per_class, dice_per_class

def overall_dice_iou(pred, target, num_classes):
    dice_scores = []
    iou_scores = []
    
    for cls in range(1, num_classes):  # Exclude background class (0)
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()

        # Intersection and Union
        intersection = torch.sum(pred_cls * target_cls)
        union = torch.sum(pred_cls) + torch.sum(target_cls)

        # Dice
        dice = (2 * intersection) / (union + 1e-8)
        dice_scores.append(dice.item())

        # IoU
        union_area = torch.sum((pred_cls + target_cls) > 0).float()
        iou = intersection / (union_area + 1e-8)
        iou_scores.append(iou.item())

    # Compute global Dice and IoU across all classes (average)
    overall_dice = sum(dice_scores) / num_classes
    overall_iou = sum(iou_scores) / num_classes
    
    return overall_iou, overall_dice 

# @objectives.wrap_objective()
def dot_compare(layer, batch=1, cossim_pow=0):
    def inner(T):
        dot = (T(layer)[batch] * T(layer)[0]).sum()
        mag = torch.sqrt(torch.sum(T(layer)[0] ** 2))
        cossim = dot / (1e-6 + mag)
        return -dot * cossim ** cossim_pow

    return inner


def init_D(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def pre_d():
    netD = Discriminator(3).to(device)
    # netD.apply(init_D)
    beta1 = 0.5
    dis_lr = 0.00002
    optimizerD = optim.Adam(netD.parameters(), lr=dis_lr, betas=(beta1, 0.999))
    return netD, optimizerD


def update_d(args, netD, optimizerD, real, fake):
    criterion = nn.BCELoss()

    label = torch.full((args.b,), 1., dtype=torch.float, device=device)
    output = netD(real).view(-1)
    # Calculate loss on all-real batch
    errD_real = criterion(output, label)
    # Calculate gradients for D in backward pass
    errD_real.backward()
    D_x = output.mean().item()

    label.fill_(0.)
    # Classify all fake batch with D
    output = netD(fake.detach()).view(-1)
    # Calculate D's loss on the all-fake batch
    errD_fake = criterion(output, label)
    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    # Compute error of D as sum over the fake and the real batches
    errD = errD_real + errD_fake
    # Update D
    optimizerD.step()

    return errD, D_x, D_G_z1


def calculate_gradient_penalty(netD, real_images, fake_images):
    eta = torch.FloatTensor(args.b, 1, 1, 1).uniform_(0, 1)
    eta = eta.expand(args.b, real_images.size(1), real_images.size(2), real_images.size(3)).to(device=device)

    interpolated = (eta * real_images + ((1 - eta) * fake_images)).to(device=device)

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = netD(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                              grad_outputs=torch.ones(
                                  prob_interpolated.size()).to(device=device),
                              create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return grad_penalty


def random_click(mask, point_labels=1, inout=1):
    indices = np.argwhere(mask == inout)
    return indices[np.random.randint(len(indices))]


def generate_click_prompt(img, msk, pt_label=1):
    # return: prompt, prompt mask
    """
    带提示的训练 我们在新的数据集上调整带提示的SAM。这一过程与SAM基本相同，但做了如下修改：
    对于点击提示，正点击表示前景区域，负点击表示背景区域。我们采用随机和迭代点击采样相结合的策略对该提示进行训练。
    具体来说，我们首先使用随机抽样进行初始化，然后使用迭代抽样程序添加一些点击。
    迭代采样策略类似于与真实用户的交互，因为在实践中，每一次新的点击都会被置于网络利用之前的点击集合生成的预测的错误区域。
    我们按照文献[22]生成随机抽样，并按照文献[24]模拟迭代抽样。

    我们在SAM中使用了不同的文本提示训练策略。
    在SAM中，作者使用由CLIP生成的目标对象作物的图像嵌入作为图像嵌入，以接近其在CLIP中的相应文本描述或定义。
    然而，由于CLIP几乎没有在医学图像数据集上进行过训练，因此它很难将器官/组织与CLIP中的文字描述或定义联系起来。
    相反，我们首先从ChatGPT中随机生成几个包含目标定义的自由文本（如视盘、脑肿瘤）作为关键词，然后使用CLIP提取文本的嵌入作为训练的提示。
    一个自由文本可能包含多个目标，在这种情况下，我们用所有目标对应的掩码对模型进行监督。
    """
    pt_list = []
    msk_list = []
    b, c, h, w, d = msk.size()
    msk = msk[:, 0, :, :, :]  # 压缩channel 4维
    for i in range(d):
        pt_list_s = []
        msk_list_s = []
        for j in range(b):
            msk_s = msk[j, :, :, i]  # b, h, w, d 每个slice 2维
            indices = torch.nonzero(msk_s)  # 原mask中非零点的索引 2维
            if indices.size(0) == 0:
                # generate a random array between [0-h, 0-h]:
                random_index = torch.randint(0, h, (2,)).to(device=msk.device)  # 一维向量：坐标[a, b]
                new_s = msk_s
            else:
                random_index = random.choice(indices)  # 任选一点 一维向量：坐标[a, b]
                label = msk_s[random_index[0], random_index[1]]  # 制作label
                new_s = torch.zeros_like(msk_s)
                # convert bool tensor to int
                new_s = (msk_s == label).to(dtype=torch.float)  # 二维
                # new_s[msk_s == label] = 1
            pt_list_s.append(random_index)  # 多个一维tensor拼接
            msk_list_s.append(new_s)  # 多个二维tensor拼接
        pts = torch.stack(pt_list_s, dim=0)
        msks = torch.stack(msk_list_s, dim=0)  # 三维
        pt_list.append(pts)  # 多个二维tensor拼接
        msk_list.append(msks)
    pt = torch.stack(pt_list, dim=-1)  # 3
    msk = torch.stack(msk_list, dim=-1)  # 四维

    msk = msk.unsqueeze(1)  # 五维

    return img, pt, msk  # [b, c, h, w, d], [b, n==2, d], [b, c, h, w, d]
    # [batch_size, channel, height, width, depth(通道数 × 每个通道所占位数)]
    # n代表prompt点数, 初始一个, 迭代生成一个
