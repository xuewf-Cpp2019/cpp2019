import torch
from torch.autograd import Function, Variable
import torch.nn as nn
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


def dice_coeff(input, target, cuda_device=0):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda(cuda_device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)
'''
def one_hot_encoder(y_predict, y_true):
    
    #transform the ground truth channel into the same shape as the model output
    #y_predict: B*C*H*W
    #y_true: B*H*W
    #:return: y_true: B*C*H*W
    encoder_target = y_predict.data.clone().zero_()
    y_true = encoder_target.scatter_(1, y_true.unsqueeze(1), 1)
    return y_true
'''
class CE_Dice_loss(nn.Module):
    def __init__(self):
        super(CE_Dice_loss, self).__init__()

    def forward(self, ED_seg, ES_seg, gt):
        criterion_CE = nn.CrossEntropyLoss()
        criterion_Dice = DiceLoss()

        CE_loss_ED = criterion_CE(ED_seg, gt[:, 0, ])
        CE_loss_ES = criterion_CE(ES_seg, gt[:, 1, ])

        ED_seg = torch.max(ED_seg, dim=1)[1]
        ES_seg = torch.max(ES_seg, dim=1)[1]
        tot_Dice_Loss_ED = 0
        for i in range(4):
            diceLoss = criterion_Dice((ED_seg == i).float(), (gt[:, 0, ] == i).float())
            tot_Dice_Loss_ED += diceLoss
        tot_Dice_Loss_ES = 0
        for i in range(4):
            diceLoss = criterion_Dice((ES_seg == i).float(), (gt[:, 1, ] == i).float())
            tot_Dice_Loss_ES += diceLoss
        return (CE_loss_ED+CE_loss_ES)/2, (tot_Dice_Loss_ED+tot_Dice_Loss_ES)/2


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1e-6
        input_flat = Variable(input.view(N, -1))
        target_flat = Variable(target.view(N, -1))
        intersection = input_flat * target_flat
        dice = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = Variable(1 - dice.sum() / N, requires_grad=True)
        return loss
'''
class MulticlassDiceLoss(nn.Module):
    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, seg, gt, weights=None):
        dice_loss = DiceLoss()
        seg = torch.max(seg, dim=1)[1]
        totalLoss = 0
        for i in range(4):
            diceLoss = dice((seg==i).float(), (gt==i).float())
            totalLoss += diceLoss
        return totalLoss
'''
def dice(input, target):
    eps = 1e-6
    dsc = torch.zeros(input.shape[0],)
    for i in range(input.shape[0]):
        inter = torch.dot(input[i,].view(-1), target[i,].view(-1))
        union = torch.sum(input[i,]) + torch.sum(target[i,]) + eps
        dsc[i,] = (2 * inter.float() + eps) / union.float()
    return dsc