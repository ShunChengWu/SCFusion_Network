import torch.nn as nn
import torch

###
#        Losses
###
class VariationalLoss(nn.Module):
    def __init__(self):
        super(VariationalLoss, self).__init__()
    def __call__(self,z_mu, z_log_sigma):
        return -0.5 * torch.sum(1.0+2.0*z_log_sigma-torch.mul(z_mu,z_mu)-
                                torch.exp(2.0*z_log_sigma))
    
    
        
class ReconstructionLoss(nn.Module):
    def __init__(self, lambda_gamma=0.97):
        super(ReconstructionLoss, self).__init__()
        self.lambda_gamma = lambda_gamma
    def __call__(self, predict, target, weight):
        minus_lambda = (1-self.lambda_gamma)
        minus_gt = (1-target)
        loss = torch.sum(
            -torch.sum(
                self.lambda_gamma*target*torch.log(1e-6+predict)+
                minus_lambda*minus_gt*torch.log(1e-6+1-predict), dim=[0,2,3,4]
            ) * weight
        )
        return loss
    
class GeometricSemanticSceneCompletionLoss(nn.Module):
    def __init__(self, lambda_gamma=0.97):
        super(GeometricSemanticSceneCompletionLoss, self).__init__()
        self.lambda_gamma = lambda_gamma
    def __call__(self, predict, full_gt, weight, mask=None):
        minus_lambda = (1-self.lambda_gamma)
        minus_gt = (1-full_gt)
        if mask is None:
            loss = torch.sum(
                -torch.sum(
                    ((self.lambda_gamma*full_gt*torch.log(1e-6+predict))*2+
                    minus_lambda*minus_gt*torch.log(1e-6+1-predict)),dim=[2,3,4]
                ) * weight
            )
        else:
            loss = torch.sum(
            -torch.sum(
                ((self.lambda_gamma*full_gt*torch.log(1e-6+predict))*2+
                minus_lambda*minus_gt*torch.log(1e-6+1-predict))*mask,dim=[2,3,4]
            ) * weight
        )
        return loss
    
class StandardGANLoss(nn.Module):
    def __init__(self):
        super(StandardGANLoss, self).__init__()
        self.BCE = nn.BCEWithLogitsLoss()
    def __call__(self,input):
        return torch.mean(self.BCE(input, torch.ones_like(input)))