import torch
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff
from collections import defaultdict
import numpy as np

class Metrics:
    def __init__(self, classes):
        """
        Initialize the Metrics class.
        
        Args:
            classes (list): label set
        """
        self.classes = classes
        self.num_classes = len(classes)

    def dice_score(self, pred, gt):
        """
        Calculate Dice score for each class.
        """
        dice_scores = []
        for cls in range(self.num_classes):
            pred_cls = (pred[:, cls] > 0.5).float()  # Ensure binary predictions
            gt_cls = (gt == cls).float()

            intersection = (pred_cls * gt_cls).sum((1, 2))
            union = pred_cls.sum((1, 2)) + gt_cls.sum((1, 2))

            # Handle empty masks (both predicted and ground truth are empty)
            dice = torch.where(union == 0, torch.tensor(1.0), (2. * intersection) / (union + 1e-5))
            dice_scores.append(dice.mean().item())

        return torch.tensor(dice_scores).mean().item()  # Optionally weight by class or batch size


    def single_class_dice_score(self, pred, gt):
        """
        Calculate Dice score for a single class.
        """
        pred_cls = pred[:, 0]  # Assumes single class in channel dimension
        intersection = (pred_cls * gt).sum((1, 2))
        union = pred_cls.sum((1, 2)) + gt.sum((1, 2))
        dice = torch.where(union == 0, torch.tensor(1.0, device=pred.device), (2. * intersection) / (union + 1e-5))
        return dice.mean().item()


    def iou_score(self, pred, gt):
        """
        Calculate IoU score for each class.
        """
        iou_scores = []
        for cls in range(self.num_classes):
            pred_cls = pred[:, cls]
            gt_cls = (gt == cls).float()
            intersection = (pred_cls * gt_cls).sum((1, 2))
            union = pred_cls.sum((1, 2)) + gt_cls.sum((1, 2)) - intersection
            iou = torch.where(union == 0, torch.tensor(1.0, device=pred.device), intersection / (union + 1e-5))
            iou_scores.append(iou.mean().item())
        return torch.tensor(iou_scores).mean().item()

    def sensitivity(self, pred, gt):
        """
        Calculate sensitivity (recall) for each class.
        """
        sensitivities = []
        for cls in range(self.num_classes):
            pred_cls = pred[:, cls]
            gt_cls = (gt == cls).float()
            true_positive = (pred_cls * gt_cls).sum((1, 2))
            false_negative = ((1 - pred_cls) * gt_cls).sum((1, 2))
            sensitivity = true_positive / (true_positive + false_negative + 1e-5)
            sensitivities.append(sensitivity.mean().item())
        return torch.tensor(sensitivities).mean().item()

    def specificity(self, pred, gt):
        """
        Calculate specificity for each class.
        """
        specificities = []
        for cls in range(self.num_classes):
            pred_cls = pred[:, cls]
            gt_cls = (gt == cls).float()
            true_negative = ((1 - pred_cls) * (1 - gt_cls)).sum((1, 2))
            false_positive = (pred_cls * (1 - gt_cls)).sum((1, 2))
            specificity = true_negative / (true_negative + false_positive + 1e-5)
            specificities.append(specificity.mean().item())
        return torch.tensor(specificities).mean().item()

    def hausdorff_distance(self, pred, gt):
        """
        Calculate Hausdorff Distance (HD) for each class.
        """
        hd_distances = []
        for cls in range(self.num_classes):
            pred_cls = pred[:, cls]
            gt_cls = (gt == cls).float()
            for b in range(pred.shape[0]):
                pred_points = torch.nonzero(pred_cls[b]).cpu().numpy()
                gt_points = torch.nonzero(gt_cls[b]).cpu().numpy()
                if pred_points.size and gt_points.size:
                    hd = max(directed_hausdorff(pred_points, gt_points)[0],
                             directed_hausdorff(gt_points, pred_points)[0])
                else:
                    hd = np.inf  # Assign high value if one of the objects is missing
                hd_distances.append(hd)
        return np.mean(hd_distances)

    def compute_all_metrics(self, pred, gt):
        """
        Compute all metrics and return as a dictionary.
        
        Args:
            pred (torch.Tensor): Predicted masks with shape [b, c, h, w].
            gt (torch.Tensor): Ground truth masks with shape [b, h, w].

        Returns:
            dict: Dictionary of all computed metrics.
        """
        # Ensure pred is one-hot encoded
        if pred.shape[1] != self.num_classes:
            raise ValueError("Pred shape should be [b, num_classes, h, w]")

        # Convert predictions to binary for each class (thresholding is typically not needed for one-hot)
        pred_bin = (pred > 0.5).float()
        
        metrics = {
            "mDice": self.dice_score(pred_bin, gt),
            "mIoU": self.iou_score(pred_bin, gt),
            "Sensitivity": self.sensitivity(pred_bin, gt),
            "Specificity": self.specificity(pred_bin, gt),
            "Hausdorff Distance": self.hausdorff_distance(pred_bin, gt),
        }
        
        # Dice for each class
        for cls, cls_name in enumerate(self.classes):
            metrics[f'{cls_name} Dice'] = self.single_class_dice_score(pred_bin[:, cls:cls+1], (gt == cls).float())

        return metrics