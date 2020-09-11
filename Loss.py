from torch import nn
import torch
class Loss():
    def __init__(self):
        pass
    
    def MSE_loss(self,outputNet, labels, utils, n_classes=10):
        sigmoid = nn.Sigmoid()
        clf_criterion = nn.MSELoss(reduction='mean')
        clf_loss = clf_criterion(sigmoid(outputNet),utils.one_hot_matrix(labels,n_classes))
        return clf_loss
    
    def CE_loss(self,outputNet, labels):
        clf_criterion = torch.nn.CrossEntropyLoss()
        clf_loss = clf_criterion(outputNet,labels)
        return clf_loss