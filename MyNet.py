from rete_riconoscimento_umani.resnet import resnet32
import torch
from torch import nn
import copy
import torch.optim as optim

class MyNet():
    def __init__(self, n_classes = 10):
        self.net = resnet32(num_classes=n_classes)
        self.net.linear = nn.Linear(64,n_classes)
        self.net.linear.weight = torch.nn.init.kaiming_normal_(self.net.linear.weight)

    
    def update_network(self,best_net,n_classes,init_weights,type='not_cosine'):
        self.prev_net = copy.deepcopy(best_net)
        prev_weights = copy.deepcopy(best_net.linear.weight)
        if type == 'not_cosine':
            prev_bias = copy.deepcopy(best_net.linear.bias)
            self.net.linear = nn.Linear(64,n_classes)
            self.net.linear.weight.data[:n_classes-self.batch_classes] = prev_weights
            self.net.linear.weight.data[n_classes-self.batch_classes:n_classes] = init_weights
            self.net.linear.bias.data[:n_classes-self.batch_classes] = prev_bias
        else:
            prev_sigma = copy.deepcopy(self.net.linear.sigma)
            self.net.linear = CosineLinear(64,n_classes)
            self.net.linear.weight.data[:n_classes-self.batch_classes] = prev_weights
            self.net.linear.weight.data[n_classes-self.batch_classes:n_classes] = init_weights
            self.net.linear.sigma.data = prev_sigma
        return self.prev_net,self.net

    def get_old_outputs(self,images,labels,n_old_classes = None,type='not_cosine'):
        self.prev_net.train(False)
        if type == 'cosine':
            features,output = self.prev_net(images)
            return features[:],output
        elif type == 'rebalancing':
            self.balancing_net.train(False)
            output = self.balancing_net(images)
            return output[:,n_old_classes:]
        else:
            output = self.prev_net(images)
            return output
    
    def get_old_features_cosine(self,images,labels):
        self.prev_net.train(False)
        feature_map,_ = self.prev_net(images)
        return feature_map

    def prepare_training(self,LR,MOMENTUM,WEIGHT_DECAY,STEP_SIZE,GAMMA,typeScheduler,type='normal'):    
        parameters_to_optimize = self.net.parameters()
        if type == 'cosine':
            optimizer = optim.SGD(parameters_to_optimize,lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        elif type == 'normal':
            optimizer = optim.SGD(parameters_to_optimize,lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        if typeScheduler == 'multistep':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, STEP_SIZE, gamma=GAMMA)
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')

        return (optimizer,scheduler)
    
    def freeze_conv(self):
        for i,child in enumerate(self.net.children()):
            if(i==5):
                break
            for param in child.parameters():
                param.requires_grad = False
        
        return self.net

    def freeze_neurons(self,n_old_classes):
        for param in self.net.linear.parameters():
            param.grad[:n_old_classes]=0
        
        return self.net

    def unfreeze_conv(self):
        for i,child in enumerate(self.net.children()):
            if(i==5):
                break
            for param in child.parameters():
                param.requires_grad = True
        
        return self.net