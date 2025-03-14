# -*- coding:utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import cfg
from utils import *
from tqdm import tqdm
from einops import rearrange
from collections import defaultdict
from torch.utils.tensorboard.writer import SummaryWriter
from metrics import Metrics

torch.backends.cudnn.benchmark = True

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)

dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

# all_classes = ["background", "bladder", "placenta", "placenta accreta", "myometrium"]
# all_classes = ["background", "liver", "bladder", "lungs", "kidneys", "bone", "brain"]
all_classes = args.all_classes
# classes = [["background", "bladder", "placenta", "placenta accreta", "myometrium"]]
# classes = [["background", "liver", "bladder", "lungs", "kidneys", "bone", "brain"]]
classes = [args.classes]
# class_colors={
#         'background': (0, 0, 0),  # Background
#         'bladder': (255, 0, 0),  # Bladder
#         'placenta': (0, 255, 0),  # Placenta
#         'placenta accreta': (0, 0, 255),  # Placenta Accreta Area
#         'myometrium': (255, 255, 0),  # Uterine Myometrium
#         'fetus': (255, 0, 255),
#         'infant': (125, 125, 0),
#         'baby': (125, 255, 73),
#         'fat': (237,145,33),
#         'water': (153,51,250),
#         'muscle': (0,199,140),
#         'afterbirth': (0, 124, 123),
#         'vesica': (23, 52, 114),
#         'uterine wall': (85, 211, 9),
#         'pathological tissue': (235, 1, 192),
#         'amniotic fluid': (14, 51, 122),
#         'cervix': (45, 11, 124),
#         'abdominal wall': (125, 12, 12),
#         'spinal column':(41, 203, 62),
#         }
class_colors={
            'background': (0, 0, 0),
            "placenta": (128, 128, 0),
            'liver': (255, 0, 0),
            'bladder': (0, 255, 0),
            'lungs': (0, 0, 255),
            'kidneys': (255, 255, 0),
            'bone': (255, 0, 255),
            'brain': (0, 255, 255)
        }

def get_classification_label_from_path(image_path):
    # The label is indicated by the second last directory name in the path
    # E.g., "train0/wangxinxiu_CT1201905280381/LABEL/286.png" -> "train0"
    class_folder = image_path.split('/')[0]  # Split the path and get the first component
    class_label = int(class_folder[-1])  # Extract the last character and convert to integer
    return class_label


def train_sam(args,
              net: nn.Module,
              optimizer,
              train_loader,
              fold,
              epoch,
              writer,
              schedulers=None,
              show=150):
    
    epoch_loss = 0
    total_dice = 0
    total_iou = 0
    ind = 0

    # Train mode
    net.train()
    optimizer.zero_grad()

    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    count = 0
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        metrics_caculator = Metrics(classes=all_classes)
        num_classes = len(all_classes)

        if args.mod == 'sam_adpt':
           for n, value in net.image_encoder.named_parameters(): 
                    if "Adapter" not in n:
                        value.requires_grad = False
                    else:
                        value.requires_grad = True

        for ind, pack in enumerate(train_loader):
            imgs = pack['image'].to(dtype=torch.float32, device=device)
            gt_masks = pack['label'].to(dtype=torch.float32, device=device)
            if True:
                point_coords, point_labels = generate_dot_prompts(gt_masks, all_classes, class_names=all_classes)
            else:
                point_coords = pack['pt']
                point_labels = pack['p_label']
            img_name = pack['image_meta_dict']['filename_or_obj']

            # Forward pass
            unique = torch.unique(imgs, return_counts=True)
            if args.dataset == 'ctorg':
                imgs = imgs.repeat(1, 3, 1, 1)
            imge = net.image_encoder(imgs)
            se, de, te = net.prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=None,
                texts=classes
            )
            pred = net.mask_decoder(
                image_embeddings=imge,
                text_embeddings=te,
                image_pe=net.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de
            )
            pred_no_prompt = net.mask_decoder(
                image_embeddings=imge,
                text_embeddings=te,
                image_pe=net.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=None,
                dense_prompt_embeddings=de
            )

            # Convert predicted logits to class indices
            pred_masks = torch.argmax(torch.softmax(pred, dim=1), dim=1)
            pred_masks_no_prompt = torch.argmax(torch.softmax(pred_no_prompt, dim=1), dim=1)
            gt_masks = gt_masks.squeeze(0).long()

            # Calculate Dice and IoU for the batch
            mask_pred_m = F.one_hot(pred_masks, num_classes=num_classes).permute(0, 3, 1, 2).float()
            mask_pred_m_no_p = F.one_hot(pred_masks_no_prompt, num_classes=num_classes).permute(0, 3, 1, 2).float()
            ordered_gt_masks = gt_masks  # Assuming your ground truth is already in the correct format

            # Compute loss and metrics for this batch
            class_weights = [0.1] + [1] * (num_classes - 1)
            class_weights = torch.tensor(class_weights).to(pred.device)
            loss = F.cross_entropy(pred, gt_masks, class_weights)
            dice = metrics_caculator.dice_score(mask_pred_m, ordered_gt_masks)
            dice_no_prompt = metrics_caculator.dice_score(mask_pred_m_no_p, ordered_gt_masks)
            iou = metrics_caculator.iou_score(mask_pred_m, ordered_gt_masks)

            step = (epoch + 1) * ind
            writer.add_scalar('loss/step', loss.item(), step)
            writer.add_scalar('dice/step', dice, step)
            writer.add_scalar('dice_no_prompt/step', dice_no_prompt, step)
            writer.add_scalar('iou/step', iou, step)

            # Combined loss: weighted sum of cross-entropy and Dice loss
            loss = loss

            # Update the model
            pbar.set_postfix(loss=loss.item())
            epoch_loss += loss.item()
            total_dice += dice
            total_iou += iou
            loss.backward()

            # print("Layers with gradients and their magnitudes:")
            # for name, param in net.named_parameters():
            #     if param.requires_grad:
            #         if param.grad is not None:
            #             grad_magnitude = param.grad.abs().mean().item()  # Compute mean magnitude of gradients
            #             print(f"Layer: {name} | Grad Magnitude: {grad_magnitude:.6f}")
            #         else:
            #             print(f"Layer: {name} | Grad is None")

            optimizer.step()
            optimizer.zero_grad()

            # if ind == 1000:
            #     sd = net.state_dict()
            #     save_checkpoint({
            #                 'epoch': epoch + 1,
            #                 'model': args.net,
            #                 'state_dict': sd,
            #                 'optimizer': optimizer.state_dict(),
            #                 'best_dice': 0,
            #                 'path_helper': args.path_helper,
            #             }, 
            #             is_best=False, 
            #             output_dir=args.path_helper['ckpt_path'], 
            #             filename=f"best_checkpoint_fold{0}")
            #     print('save best model to ', args.path_helper['ckpt_path'])

            if ind % args.vis_train == 0:
                namecat = 'Train'
                for na in img_name:
                    namecat = namecat + na.split('/')[-1].split('.')[0] + '+'
                vis_mri_image_with_no_prompt(imgs,
                                            ordered_gt_masks,
                                            pred_masks, 
                                            pred_masks_no_prompt, 
                                            os.path.join(args.path_helper['sample_path'], namecat + '_' + str(ind) + 'fold_' + str(fold) + 'epoch_' + str(epoch) + '.jpg'),
                                            classes,
                                            class_colors=class_colors,
                                            points=point_coords,
                                            point_labels=point_labels,
                                            boxes=None,
                                            box_labels=None)
            pbar.update()

        # After the loop, calculate the average Dice and IoU across all batches
        avg_dice = total_dice / len(train_loader)
        avg_iou = total_iou / len(train_loader)

    return epoch_loss / len(train_loader), avg_iou, avg_dice

def validation_sam(args, 
                   val_loader, 
                   fold,
                   epoch, 
                   net: nn.Module, 
                   clean_dir=True):
    # eval mode
    net.eval()
    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    tot = 0 # total loss
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        metrics_list = []

        for ind, pack in enumerate(val_loader):

            imgs = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            gt_masks = pack['label'].to(dtype=torch.float32, device=GPUdevice)
            img_name = pack['image_meta_dict']['filename_or_obj']
            if True:
                point_coords, point_labels = generate_dot_prompts(gt_masks, all_classes, class_names=classes[0])
            else:
                point_coords = pack['pt']
                point_labels = pack['p_label']
            box_coords, box_labels = generate_box_prompts(gt_masks)

            metrics_caculator = Metrics(classes = classes[0])
            num_classes = len(classes[0])
            valid_indices = []
            for predict_class in classes[0]:
                valid_indices.append(all_classes.index(predict_class))

            with torch.no_grad():
                imge = net.image_encoder(imgs)
                se, de, te= net.prompt_encoder(
                    points=(point_coords, point_labels),
                    boxes=None,
                    masks=None,
                    texts=classes
                )
                pred = net.mask_decoder(
                    image_embeddings=imge,
                    text_embeddings=te,
                    image_pe=net.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de
                )
                pred_no_prompt = net.mask_decoder(
                    image_embeddings=imge,
                    text_embeddings=te,
                    image_pe=net.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=None,
                    dense_prompt_embeddings=de
                )

                # Convert predicted logits to class indices
                pred_masks = torch.argmax(torch.softmax(pred, dim=1), dim=1)
                pred_masks_no_prompt = torch.argmax(torch.softmax(pred_no_prompt, dim=1), dim=1)
                gt_masks = gt_masks.squeeze(1).long() # [B, H, W]

                # For varied label counts
                if num_classes != len(all_classes):
                    gt_masks[~np.isin(gt_masks.cpu(), valid_indices)] = 0
                    unique_values = torch.unique(gt_masks)
                    value_mapping = {v.item(): i for i, v in enumerate(unique_values)}
                    flattened_gt_masks = gt_masks.view(-1)
                    ordered_gt_masks = torch.tensor([value_mapping[val.item()] for val in flattened_gt_masks])
                    ordered_gt_masks = ordered_gt_masks.view(gt_masks.shape).to('cuda')
                else:
                    ordered_gt_masks = gt_masks

                mask_pred_m = F.one_hot(torch.softmax(pred, dim=1).argmax(dim=1), num_classes=num_classes).permute(0, 3, 1, 2).float()
                mask_pred_m_no_prompt = F.one_hot(torch.softmax(pred_no_prompt, dim=1).argmax(dim=1), num_classes=num_classes).permute(0, 3, 1, 2).float()
                # pred_all.append(mask_pred_m)
                # mask_all.append(ordered_gt_masks)

                class_weights = [0.1]
                for predict_class in classes[0][1:]:
                    class_weights.append(1)
                class_weights = torch.tensor(class_weights).to(pred.device)
                loss = F.cross_entropy(pred, ordered_gt_masks, class_weights) 
                tot += loss.item()
                
                metrics = metrics_caculator.compute_all_metrics(mask_pred_m, gt_masks)
                metrics_list.append(metrics)

                namecat = 'Test'
                for na in img_name:
                    img_name = na.split('/')[-1].split('.')[0]
                    namecat = namecat + img_name + '+'
                '''vis images'''
                if ind > -1:
                    if ind % args.vis_val == 0:
                        vis_mri_image_with_no_prompt(imgs, 
                                                    ordered_gt_masks,
                                                    pred_masks, 
                                                    pred_masks_no_prompt, 
                                                    os.path.join(args.path_helper['sample_path'], namecat + '_' + str(ind) + '_fold' + str(fold) + '_epoch_' + str(epoch) + '.jpg'),
                                                    classes, 
                                                    class_colors=class_colors,
                                                    points=point_coords,
                                                    point_labels=point_labels,
                                                    boxes=None,
                                                    box_labels=None)
                        
                    # vis_mri_pred(imgs, pred_masks_no_prompt, 
                    # os.path.join('predict_no_prompt', namecat[4:-1] + '.jpg'), classes, 
                    # class_colors={
                    # 'background': (0, 0, 0),  # Background
                    # 'bladder': (255, 0, 0),  # Bladder
                    # 'placenta': (0, 255, 0),  # Placenta
                    # 'placenta accreta': (0, 0, 255),  # Placenta Accreta Area
                    # 'myometrium': (255, 255, 0),  # Uterine Myometrium
                    # }, points=point_coords, point_labels=point_labels)

                    # vis_mri_pred(imgs, pred_masks, 
                    # os.path.join('predict', namecat[4:-1] + '.jpg'), classes, 
                    # class_colors={
                    # 'background': (0, 0, 0),  # Background
                    # 'bladder': (255, 0, 0),  # Bladder
                    # 'placenta': (0, 255, 0),  # Placenta
                    # 'placenta accreta': (0, 0, 255),  # Placenta Accreta Area
                    # 'myometrium': (255, 255, 0),  # Uterine Myometrium
                    # }, points=point_coords, point_labels=point_labels)

                    # vis_mri_pred(imgs, gt_masks, 
                    # os.path.join('gt_masks', namecat[4:-1] + '.jpg'), classes, 
                    # class_colors={
                    # 'background': (0, 0, 0),  # Background
                    # 'bladder': (255, 0, 0),  # Bladder
                    # 'placenta': (0, 255, 0),  # Placenta
                    # 'placenta accreta': (0, 0, 255),  # Placenta Accreta Area
                    # 'myometrium': (255, 255, 0),  # Uterine Myometrium
                    # }, points=point_coords, point_labels=point_labels)

        pbar.update()

    # Initialize a defaultdict to store cumulative sums
    cumulative_metrics = defaultdict(float)

    # Sum all metrics
    for metrics in metrics_list:
        for key, value in metrics.items():
            cumulative_metrics[key] += value

    # Compute averages
    metrics = {key: value / len(metrics_list) for key, value in cumulative_metrics.items()}

    return tot / n_val, metrics