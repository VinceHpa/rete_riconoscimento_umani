from torchvision.datasets import CIFAR100
from torch.utils.data import Subset 
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import  DataLoader
import numpy as np
import random
from torchvision import transforms

# splitting the 100 classes in [num_groups] groups
# and the indexes of the images belonging to those classes as well
def get_n_splits(dataset, n_groups,random_state=41):
  
  available_labels = list(range(100))
  n_classes_group = int(100 / n_groups)
  labels = []
  indexes = [[] for i in range(n_groups)]
  random.seed(random_state)

  for index in range(n_groups):
    labels_sample = random.sample(available_labels,n_classes_group)
    labels.append(labels_sample)
    available_labels = list(set(available_labels) - set(labels_sample))

  for index in range(len(dataset)):
    label = dataset.__getitem__(index)[1]
    for i in range(n_groups):
      if labels[i].__contains__(label):
        indexes[i].append(index)
        break

  return indexes,labels



# IncrementalCIFAR class stores the CIFAR100 dataset and some info helpful for the 
# incremental learning process: the splitting of the groups and of the indexes
train_transform = transforms.Compose([
                                      transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(), # Turn PIL Image to torch.Tensor
                                      transforms.Normalize( (0.4501, 0.4296, 0.3882), (0.2479, 0.2407, 0.2350))]) # Normalizes tensor with mean and standard deviation

# Define transforms for the evaluation phase
eval_transform = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize( (0.4501, 0.4296, 0.3882), (0.2479, 0.2407, 0.2350))])
  


class MyCIFAR100():
  def __init__(self, root, n_groups = 10, train=True, transform=None, target_transform=None, download=False, random_state=653):
        self.dataset = CIFAR100(root, train=train, transform = None, target_transform=None, download=download)
        self.ind_targets = list()
        self.transform = transform
        self.target_transform = lambda target : self.sorted_labels.index(target)
        self.random_state = random_state
        self.indexes_split,self.labels_split = get_n_splits(self.dataset, n_groups,random_state = self.random_state)
        self.sorted_labels = []
        
        
        for l in self.labels_split:
            self.sorted_labels += l
    
  def get_train_val_test(self):
    X = np.zeros((len(self.ind_targets),2))
    
    correspondence={}
    for i in range(len(self.ind_targets)):
        correspondence[str(i)] = self.ind_targets[i][0]
        
    sss = StratifiedShuffleSplit(n_splits=1,test_size= 0.3,random_state=0)
    targets=[]
    indices=[]
    for ind,label in self.ind_targets:
        targets.append(label)
        indices.append(ind)
        
    for train_val_fasullo, test_fasullo in sss.split(X,np.array(targets)):
      train_val = []
      for elem in train_val_fasullo:
          train_val.append(correspondence[str(elem)])
      test=[]
      for elem in test_fasullo:
          test.append(correspondence[str(elem)])

      sss_2 = StratifiedShuffleSplit(n_splits=1,test_size= 0.2,random_state=0)
      X = np.zeros((len(train_val),2))

      for train_fasullo, val_fasullo in sss_2.split(X,np.array(targets)[train_val_fasullo]):
        train = []
        for elem in train_fasullo:
          train.append(correspondence[str(elem)])
        print(set(train)-set(indices))
        val=[]
        for elem in val_fasullo:
          val.append(correspondence[str(elem)])
        
      return Subset(self,train), Subset(self,val), Subset(self,test)

  def create_dataLoaders(self,train,test,val, BATCH_SIZE):
    train_dataloader =  DataLoader(train,batch_size=BATCH_SIZE,drop_last=True,num_workers=4,shuffle=True)
    val_dataloader = DataLoader(val,batch_size=BATCH_SIZE,drop_last=False,num_workers=4)
    test_dataloader = DataLoader(test,batch_size=BATCH_SIZE,drop_last=False,num_workers=4)
    return train_dataloader,val_dataloader,test_dataloader

  def trasform(self,index):
    if(self.transform):
        image = self.transform(self.dataset[index][0])
    if(self.target_transform):
        target = self.target_transform(self.dataset[index][1])
    
    return image,target,index


  def __getitem__(self,index):
    
    if(self.transform):
        image = self.transform(self.dataset[index][0])
    if(self.target_transform):
        target = self.target_transform(self.dataset[index][1])

    return image,target,index