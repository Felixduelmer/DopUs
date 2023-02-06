import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_2D(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss


# PyTorch
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        smooth = 1
        n_class = 2
        dice_loss = 0.0
        for class_idx in range(n_class):
            # For background class
            if class_idx == 0:
                img_A = 1 - inputs.view(-1)
                img_B = 1 - targets.view(-1)
            else:
                img_A = inputs.view(-1)
                img_B = targets.view(-1)
            dice_loss += 1 - ((2.0 * torch.sum(img_A * img_B) + smooth) /
                              (torch.sum(torch.pow(img_A, 2)) + torch.sum(torch.pow(img_B, 2)) + smooth))

        return dice_loss / n_class


if __name__ == '__main__':
    from torch.autograd import Variable

    depth = 3
    batch_size = 2
    y = Variable(torch.LongTensor(batch_size, 1, 1, 2, 2).random_() %
                 depth).cuda()  # 4 classes,1x3x3 img
    x = Variable(torch.randn(y.size()).float()).cuda()
    dicemetric = DiceLoss()
    print(dicemetric(x, y))
