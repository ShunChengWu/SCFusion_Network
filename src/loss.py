import torch
import torch.nn as nn
import torchvision.models as models


class LogNLLLoss(nn.Module):
    """
    Similar to CrossEntropyLoss. But take input already processed by SoftMax
    """
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(LogNLLLoss, self).__init__()
        self.loss = nn.NLLLoss(weight,size_average,ignore_index, reduce,reduction)
    def __call__(self, output, target):
        output = torch.clamp(output,1e-5,output.size(1)-1-1e-5)
        return self.loss(torch.log(output),target)
        

class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss


def soft_f1_score(y_pred:torch.Tensor, y_true:torch.Tensor, epsilon = 1e-12) -> torch.Tensor:
    '''
    https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d
    '''
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    #precision = tp / (tp + fp + epsilon)
    #recall = tp / (tp + fn + epsilon)
    #f1 = 2* (precision*recall) / (precision + recall + epsilon)
    
    soft_f1_1 = 2*tp / (2*tp + fn + fp + epsilon)
    # return soft_f1_1
    soft_f1_0 = 2*tn / (2*tn + fn + fp + epsilon)
    return 0.5 * (soft_f1_1+soft_f1_0)
    

def SoftF1Loss(y_pred:torch.Tensor, y_true:torch.Tensor, weight=None) -> torch.Tensor:
    assert y_pred.ndim >= 2
    assert y_true.dtype == torch.long
    assert y_pred.ndim == y_true.ndim # assert y_pred.ndim == y_true.ndim+1
    class_num = y_pred.shape[1] if y_pred.shape[0] == y_true.shape[0] else y_pred.shape[0]
    
    if weight is not None:
        assert(weight.nelement() == class_num)
        weight = weight.view(class_num,-1)
    assert y_true.nelement() == y_pred.nelement() # assert y_true.nelement() == y_pred.nelement()/class_num # dimension should match
        
    # if y_pred.is_cuda:
        # score = torch.zeros([class_num]).cuda()
        # score = torch.zeros(0.).cuda()
    # else:
        # score = torch.zeros([class_num])
        # score = torch.tensor(0.)
    
    score = 0
    if y_pred.ndim == 2:
        assert y_true.ndim == 1
        score = soft_f1_score(y_pred, y_true)
    else:
        assert(y_pred.shape[1] == class_num)

        B = y_pred.shape[0]
        y_pred_flat = y_pred.view(B, class_num, -1)
        y_true_flat = y_true.view(B, class_num, -1)

        # for b in range(B):
        if weight is not None:
            for c in range(class_num):
                score += -torch.log( soft_f1_score(y_pred_flat[:,c,:], y_true_flat[:,c,:]) + 1e-16)* weight[c] * 1.0/ B
        else:
            score = soft_f1_score(y_pred_flat, y_true_flat)
        # for c in range(class_num):
        #     score[c] = soft_f1_score(y_pred[:,c,:], (y_true==c).float())
            
    # return score.sum()
    return score

    if weight is not None:
        return (-torch.log(score)).mean()
        return ((1-score)*weight).sum()
    else:
        return (1-score).sum()