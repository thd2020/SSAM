import json
import os
import cfg
import time
import torch
from dataset.ct import CTDataset
import function
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard.writer import SummaryWriter
from conf import settings
from utils import *
from dataset import *
from dataset.placenta import MRIDataset, MRIMaskDataset
from dataset.ctorg import CTOrgTorchDataset, SliceSampler

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
        start_epoch = 0

        if args.primitive:
            state_dict = checkpoint
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
    if args.sam_ckpt != None and args.pretrain:
        sam_load_pretrained(net, args)

    '''setting log dir'''
    args.path_helper = set_log_dir('logs', args.exp_name)
    log_path = args.path_helper['log_path']
    log_file = os.path.join(log_path, "train_logs.txt")
    writer = SummaryWriter(log_path)
    logger = create_logger(log_path)
    logger.info(args)

    '''define training folds'''
    kfold = KFold(n_splits=args.num_folds, shuffle=True, random_state=42)

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

    if args.dataset == 'mri':
        dataset = MRIDataset(args, 
                            args.data_path, 
                            transform=transform_train,
                            transform_msk=transform_train_msk, 
                            mode='Training')

    elif args.dataset == 'mri_masked':
        dataset = MRIMaskDataset(args, 
                                args.data_path,
                                transform=transform_train,
                                transform_msk=transform_train_msk,
                                mode='Training')
        
    elif args.dataset == 'ct':
        dataset = CTDataset(args, 
                                args.data_path,
                                transform=transform_train,
                                transform_msk=transform_train_msk,
                                mode='Training')
        
    elif args.dataset == "ctorg":
        dataset = CTOrgTorchDataset(args, args.data_path)
        # Specify the output file path
        # output_file = "volume_data.json"
        # # Write the data to the file
        # with open(output_file, "w") as f:
        #     json.dump(dataset.data, f, indent=4)  # `indent=4` makes the output human-readable

    '''begin training'''
    if args.dataset == 'ctorg':
        indices = list(dataset.data.keys())
    else:
        indices = np.arange(len(dataset))
    best_dice_folds = []  # To track the best dice scores for each fold
    t_v = kfold.split(indices)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        if args.dataset == 'ctorg':
        # Map train_idx and val_idx to the actual indices
            train_idx = [indices[i] for i in train_idx]
            val_idx = [indices[i] for i in val_idx]

        start_epoch = 0

        train_loss = []
        train_dice = []
        train_iou = []

        val_loss = []
        val_dice = []
        val_iou = []

        best_acc = 0.0
        best_dice = 0.0

        print(f"Processing Fold {fold + 1}/{args.num_folds} ")
        with open(log_file, 'a') as f:
            log_message = f"Processing Fold {fold + 1}/{args.num_folds}"
            f.write(log_message)
        if args.dataset == 'ctorg':
            train_sampler = SliceSampler(train_idx, dataset.data)
            val_sampler = SliceSampler(val_idx, dataset.data)
        else:
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=args.b, num_workers=1, pin_memory=True)
        val_loader = DataLoader(dataset, sampler=val_sampler, batch_size=args.b, num_workers=1, pin_memory=True)
        for epoch in range(args.fold_epoch):
            if args.mod == 'sam_adpt':
                args.epoch = epoch

                net.train()
                time_start = time.time()
                loss, iou, dice = function.train_sam(args, net, optimizer, train_loader, fold, epoch, writer)
                train_loss.append(loss)
                train_iou.append(iou)
                train_dice.append(dice)
                logger.info(f'Train loss: {loss}, iou: {iou}, dice: {dice} || @ epoch {epoch}.')
                with open(log_file, 'a') as f:
                    log_message = f'Train loss: {loss}, IoU: {iou}, Dice: {dice}|| @ epoch {epoch}.\n'
                    f.write(log_message)
                # Log the values for TensorBoard
                writer.add_scalar('Loss/train', loss, epoch)
                writer.add_scalar('IoU/train', iou, epoch)
                writer.add_scalar('Dice/train', dice, epoch)
                # Force write to disk
                writer.flush()
                time_end = time.time()
                print('time_for_epoch_training ', time_end - time_start)

                if epoch and epoch % args.val_freq == 0:
                    args.epoch = epoch
                    net.eval()
                    tol, metrics = function.validation_sam(args, val_loader, fold, epoch, net)
                    edice = metrics['mDice']
                    eiou = metrics['mIoU']
                    val_dice.append(edice)
                    val_loss.append(tol)
                    val_iou.append(eiou)
                    logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice}, metrics: {metrics} || @ epoch {epoch}.')
                    with open(log_file, 'a') as f:
                        log_message = f'Total score: {tol}, IOU: {eiou}, DICE: {edice}, metrics: {metrics} || @ epoch {epoch}.'
                        f.write(log_message)

                    writer.add_scalar('Loss/val', tol, epoch)
                    writer.add_scalar('IoU/val', eiou, epoch)
                    writer.add_scalar('Dice/val', edice, epoch)
                    # Force write to disk
                    writer.flush()

                    if args.distributed != 'none':
                        sd = net.module.state_dict()
                    else:
                        sd = net.state_dict()

                    print('dice, best_dice:', edice, best_dice)
                    # plot_metrics(args.path_helper['plot_path'], train_loss, val_loss, train_iou, val_iou, train_dice, val_dice, epoch, args.val_freq)

                    if edice > best_dice:
                        best_dice = edice
                        is_best = True if len(best_dice_folds) == 0 else edice > np.max(best_dice_folds)
                        save_checkpoint({
                            'epoch': epoch + 1,
                            'model': args.net,
                            'state_dict': sd,
                            'optimizer': optimizer.state_dict(),
                            'best_dice': best_dice,
                            'path_helper': args.path_helper,
                        }, 
                        is_best, 
                        args.path_helper['ckpt_path'], 
                        filename=f"best_checkpoint_fold{fold}.pth")
                        print('save best model to ', args.path_helper['ckpt_path'])
                    else:
                        is_best = False
        best_dice_folds.append(best_dice)
        sam_load_pretrained(net, args) # initiation for next fold

    # After all folds are complete
    mean_dice = np.mean(best_dice_folds)
    std_dice = np.std(best_dice_folds)
    max_dice = np.max(best_dice_folds)
    logger.info(f"5-Fold Cross-Validation Results - Mean Dice: {mean_dice:.4f}, Std Dice: {std_dice:.4f}, Best Dice: {max_dice:.4f}")
    with open(log_file, 'a') as f:
        log_message = f"5-Fold Cross-Validation Results - Mean Dice: {mean_dice:.4f}, Std Dice: {std_dice:.4f}, Best Dice: {max_dice:.4f}"
        f.write(log_message)
    writer.close()