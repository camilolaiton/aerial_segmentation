import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class DiceLoss(nn.Module):
    """DiceLoss.

    .. seealso::
        Milletari, Fausto, Nassir Navab, and Seyed-Ahmad Ahmadi. "V-net: Fully convolutional neural networks for
        volumetric medical image segmentation." 2016 fourth international conference on 3D vision (3DV). IEEE, 2016.

    Args:
        smooth (float): Value to avoid division by zero when images and predictions are empty.

    Attributes:
        smooth (float): Value to avoid division by zero when images and predictions are empty.
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target, weights=None):
        loss = 0.

        if weights is not None:
            
            for c in range(len(weights)):
                iflat = prediction.reshape(-1)
                tflat = target.reshape(-1)
                intersection = (iflat * tflat).sum()

                w = weights[c]
                loss += w*(1 - ((2. * intersection + self.smooth) /
                             (iflat.sum() + tflat.sum() + self.smooth)))
        else:
            iflat = prediction.reshape(-1)
            tflat = target.reshape(-1)
            intersection = (iflat * tflat).sum()

            # if (weights is not None):
            #     intersection = torch.mean(weights * intersection)

            loss = (2.0 * intersection + self.smooth) / (iflat.sum() + tflat.sum() + self.smooth)

        return 1 - loss

ALPHA = 0.5 # < 0.5 penalises FP more, > 0.5 penalises FN more
CE_RATIO = 0.5 #weighted contribution of modified CE loss compared to Dice loss

class FocalDiceLoss(nn.Module):
    """FocalDiceLoss.

    .. seealso::
        Wong, Ken CL, et al. "3D segmentation with exponential logarithmic loss for highly unbalanced object sizes."
        International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2018.

    Args:
        beta (float): Value from 0 to 1, indicating the weight of the dice loss.
        gamma (float): Value from 0 to 5, Control between easy background and hard ROI
            training examples. If set to 0, equivalent to cross-entropy.
        alpha (float): Value from 0 to 1, usually corresponding to the inverse of class frequency to address class
            imbalance.

    Attributes:
        beta (float): Value from 0 to 1, indicating the weight of the dice loss.
        gamma (float): Value from 0 to 5, Control between easy background and hard ROI
            training examples. If set to 0, equivalent to cross-entropy.
        alpha (float): Value from 0 to 1, usually corresponding to the inverse of class frequency to address class
            imbalance.
    """
    def __init__(self, beta=1, gamma=2, alpha=0.25):
        super().__init__()
        self.beta = beta
        self.focal = FocalLoss(gamma, alpha)
        self.dice = DiceLoss()

    def forward(self, input, target, weights=None):
        dc_loss = - self.dice(input, target, weights)
        fc_loss = self.focal(input, target)

        # used to fine tune beta
        # with torch.no_grad():
        #     print('DICE loss:', dc_loss.cpu().numpy(), 'Focal loss:', fc_loss.cpu().numpy())
        #     log_dc_loss = torch.log(torch.clamp(dc_loss, 1e-7))
        #     log_fc_loss = torch.log(torch.clamp(fc_loss, 1e-7))
        #     print('Log DICE loss:', log_dc_loss.cpu().numpy(), 'Log Focal loss:', log_fc_loss.cpu().numpy())
        #     print('*'*20)

        loss = torch.log(torch.clamp(fc_loss, 1e-7)) - self.beta * torch.log(torch.clamp(dc_loss, 1e-7))

        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

def ce_loss(true, logits, weights, ignore=255):
    """Computes the weighted multi-class cross-entropy loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        weight: a tensor of shape [C,]. The weights attributed
            to each class.
        ignore: the class index to ignore.
    Returns:
        ce_loss: the weighted multi-class cross-entropy loss.
    """
    ce_loss = F.cross_entropy(
        logits.float(),
        true.long(),
        ignore_index=ignore,
        weight=weights,
    )
    return ce_loss

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.__name__ = 'DiceBCELoss'

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() +
                                                        targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE
    
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()
        self.__name__ = 'IoULoss'

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

class WCrossEntropy(nn.Module):
    def __init__(self, weight=None, size_average=True, DEVICE='cuda'):
        super(WCrossEntropy, self).__init__()
        self.__name__ = 'WCEntropy'
        self.weight = None
        if weight!= None:
            self.weight = torch.tensor(weight).to(DEVICE)

    def forward(self, inputs, targets):
        # From tensor [B, C, H, W] to tensor [B, H, W]        
        targets = torch.argmax(targets, dim = 1)
        loss = nn.CrossEntropyLoss(weight=self.weight)
        output = loss(inputs, targets )
        return output