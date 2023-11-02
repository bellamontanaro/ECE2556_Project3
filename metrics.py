import torch
import torch.nn as nn
from medpy import metric
import torch.nn.functional as F
import numpy as np
    
class SegmentationMetrics(nn.Module):
    def __init__(self, pred_mask, true_mask):
        super(SegmentationMetrics, self).__init__()
        
        # pred_mask = F.sigmoid(pred_mask)
        pred_mask = torch.sigmoid(pred_mask)
        pred_mask= pred_mask.detach().cpu().numpy()
        pred_mask[pred_mask > 0.5] = 1
        pred_mask[pred_mask <= 0.5] = 0
        self.pred_mask = pred_mask
        true_mask = true_mask.cpu().numpy()
        true_mask[true_mask > 0.5] = 1
        true_mask[true_mask <= 0.5] = 0
        self.true_mask = true_mask
        
    
    def f1_score(self):
        """
        Calculate F1 Score, also known as Dice Coefficient.
        """
        dice_score = metric.binary.dc(self.pred_mask, self.true_mask)
        return dice_score
    
    def hausdorff_distance(self):
        """
        Calculate Hausdorff Distance.
        """
        hausdorff_distance = metric.binary.hd(self.pred_mask, self.true_mask)
        return hausdorff_distance
    
    def hausdorff_95(self):
        """
        Calculate 95th percentile of Hausdorff Distance.
        """
        if 0 == np.count_nonzero(self.pred_mask):
            print("pred_mask is empty")
            return 0
        hausdorff_95 = metric.binary.hd95(self.pred_mask, self.true_mask)
        return hausdorff_95
    
    def asd(self):
        """
        Calculate Average Surface Distance.
        """
        
        if 0 == np.count_nonzero(self.pred_mask):
            print("pred_mask is empty")
            return 0
        asd = metric.binary.asd(self.pred_mask, self.true_mask)
        return asd
    
if __name__ == "__main__":
    # test segmentation metrics
    pred = torch.rand(4, 1, 256, 256)
    target = torch.rand(4, 1, 256, 256)
    seg_metrics = SegmentationMetrics(pred, target)
    f1 = seg_metrics.f1_score()
    print(f1)