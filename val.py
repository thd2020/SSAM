import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import cfg
from dataset.ct import CTDataset
import function
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from dataset import *
from torch.utils.data import DataLoader, random_split
from dataset.ctorg import CTOrgTorchDataset
from dataset.placenta import MRIDataset, MRIMaskDataset
from utils import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

if __name__ == '__main__':
    '''initialization'''
    args = cfg.parse_args()

    GPUdevice = torch.device('cuda', args.gpu_device)

    net = get_network(args, 
                      args.net, 
                      use_gpu=args.gpu, 
                      gpu_device=GPUdevice, 
                      distribution=args.distributed)

    optimizer = optim.Adam(net.parameters(), 
                           lr=args.lr, betas=(0.9, 0.999), 
                           eps=1e-08, 
                           weight_decay=0, 
                           amsgrad=False)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                          step_size=10, 
                                          gamma=0.5)  # learning rate decay

    '''load pretrained model'''
    def sam_load_pretrained(net, args):
        print(f'=> resuming from {args.sam_ckpt}')
        assert os.path.exists(args.sam_ckpt)
        checkpoint_file = os.path.join(args.sam_ckpt)
        loc = 'cuda:{}'.format(args.gpu_device)
        checkpoint = torch.load(checkpoint_file, map_location=loc)

        if args.primitive:
            state_dict = checkpoint
            logger = None
            start_epoch = 0
            best_dice = 0
        else:
            state_dict = checkpoint['state_dict']
            args.path_helper = checkpoint['path_helper']
            logger = create_logger(args.path_helper['log_path'])
            start_epoch = checkpoint['epoch']
            best_dice = checkpoint['best_dice']

        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k in net.state_dict():
                if v.shape == net.state_dict()[k].shape:
                    filtered_state_dict[k] = v
        # filtered_state_dict = state_dict
        net.load_state_dict(filtered_state_dict, strict=False)
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
        return logger, start_epoch, best_dice
    if args.sam_ckpt != None and args.pretrain:
        logger, start_epoch, best_dice = sam_load_pretrained(net, args)

    '''setting log dir'''
    args.path_helper = set_log_dir('logs', args.exp_name)
    log_path = args.path_helper['log_path']
    log_file = os.path.join(log_path, "train_logs.txt")
    writer = SummaryWriter(log_path)
    logger = create_logger(log_path)
    logger.info(args)

    '''segmentation data'''
    transform_train = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size))
    ])

    transform_train_msk = transforms.Compose([
        transforms.Resize((args.mask_size, args.mask_size))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size))
    ])

    transform_test_msk = transforms.Compose([
        transforms.Resize((args.mask_size, args.mask_size))
    ])

    if args.dataset == 'isic':
        dataset = ISIC2016(args, 
                        args.data_path, 
                        transform=transform_train,
                        transform_msk=transform_train_msk, 
                        mode='Testing')

    elif args.dataset == 'mri':
        dataset = MRIDataset(args, 
                            args.data_path, 
                            transform=transform_train,
                            transform_msk=transform_train_msk, 
                            mode='Testing')

    elif args.dataset == 'mri_masked':
        dataset = MRIMaskDataset(args, 
                                args.data_path,
                                transform=transform_train,
                                transform_msk=transform_train_msk,
                                mode='Testing')
        
    elif args.dataset == 'ct':
        dataset = CTDataset(args, 
                                args.data_path,
                                transform=transform_train,
                                transform_msk=transform_train_msk,
                                mode='Testing')
        
    elif args.dataset == "ctorg":
        dataset = CTOrgTorchDataset(args, args.data_path)
        # Specify the output file path
        # output_file = "volume_data.json"
        # # Write the data to the file
        # with open(output_file, "w") as f:
        #     json.dump(dataset.data, f, indent=4)  # `indent=4` makes the output human-readable

    dataloader = DataLoader(dataset, batch_size=args.b, num_workers=1, pin_memory=True)
    
'''begain valuation'''
best_acc = 0.0
best_tol = 1e4

if args.mod == 'sam_adpt':
    net.eval()
    tol, metrics = function.validation_sam(args, dataloader, 0, start_epoch, net)
    logger.info(f'Total score: {tol}, metrics: {metrics} || @ epoch {start_epoch}.')
    log_path = args.path_helper['log_path']
    log_file = os.path.join(log_path, "train_logs.txt")

    with open(log_file, 'a') as f:
        log_message = f'Total score: {tol}, metrics: {metrics} || @ epoch {start_epoch}.'
        f.write(log_message)