import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        loss = (1 - ((2. * intersection + self.smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + self.smooth)))
        return loss.mean()
    
class bce_dice_loss(nn.Module):
    def __init__(self):
        super(bce_dice_loss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        # pred = F.sigmoid(pred)
        pred = torch.sigmoid(pred)
        dice_loss = self.dice(pred, target)
        return bce_loss, dice_loss
    
if __name__ == "__main__":
    # test bce dice loss
    pred = torch.rand(4, 1, 256, 256)
    target = torch.rand(4, 1, 256, 256)
    bce_dice_loss = bce_dice_loss()
    loss = bce_dice_loss(pred, target)
    print(loss)
