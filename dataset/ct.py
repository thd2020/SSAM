""" train and test dataset

author jundewu
"""
import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate
from utils import random_click
import random
from monai.transforms import LoadImaged, Randomizable, LoadImage
from torchvision.transforms import ToTensor
import csv
import random

class CTDataset(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training',prompt = 'click', plane = False):
        if mode == 'Training':
            df = pd.read_csv(os.path.join(data_path, 'pas_training.csv'), encoding='UTF-8')
        else:
            df = pd.read_csv(os.path.join(data_path, 'pas_testing.csv'), encoding='UTF-8')

        self.name_list = df.iloc[:, 0].tolist()  # Image paths
        self.mask_list = df.iloc[:, 1].tolist()  # Mask paths
        self.type_list = df.iloc[:, 2].tolist()
        self.data_path = data_path
        self.mode = mode
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        """Get the images"""
        image_path = os.path.join(self.data_path, self.name_list[index])
        image = Image.open(image_path).convert('RGB')
        resize_transform = transforms.Resize((self.img_size, self.img_size))

        # Load corresponding masks
        masks = []
        for i, mask_name in enumerate([self.mask_list[index]]):
            mask_path = os.path.join(self.data_path, mask_name)
            mask = Image.open(mask_path).convert('L')  # Convert mask to grayscale
            mask_tensor = resize_transform(transforms.ToTensor()(mask))  # Convert mask to tensor

            # Assign class index to the mask, add it to the list
            class_mask = (mask_tensor > 0).long() * (i + 1)
            masks.append(class_mask)

        # Combine all masks into a single tensor by taking the max across the masks
        combined_mask = torch.stack(masks, dim=0).max(dim=0)[0]  # Combines into a single mask with class indices

        # Resize images and masks using torchvision.transforms
        image = resize_transform(image)

        # Convert image (PIL) to a tensor
        image_tensor = ToTensor()(image)
        
        # Generate four random points (replace `random_click` with your custom logic if needed)
        # pt = [(random.randint(0, self.img_size), random.randint(0, self.img_size)) for _ in range(4)]

        type_label = self.type_list[index]  # Placeholder, update logic as per your need
        p_label = 1  # Default binary label (e.g., binary classification)

        # Return the dictionary
        return {
            'image': image_tensor,
            'label': combined_mask,
            'typeLabel': type_label,
            'p_label': p_label,
            # 'pt': pt,
            'image_meta_dict': {'filename_or_obj': os.path.basename(image_path)}
        }
        