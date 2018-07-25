# coding: utf-8
import torch
import time as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms


# Prepare data
train_samples = datasets.ImageFolder('data/training', transforms.ToTensor())
test_samples = datasets.ImageFolder('data/testing', transforms.ToTensor())

# Load data
train_set = DataLoader(train_samples, batch_size=170, shuffle=True, num_workers=0)
test_set = DataLoader(test_samples, batch_size=170, shuffle=False, num_workers=0)


class Cnn(nn.Module):
    
    def __init__(self):
        
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, padding=1)
        self.fc1 = nn.Linear(5*5*50, 4)
        self.fc2 = nn.Linear(4, 10)
        
    def forward(self,x):
        x = F.sigmoid(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.sigmoid(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 5*5*50)
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create the Cnn
model = Cnn().to(device)

# Create the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1,  momentum=0.1, weight_decay=0)

# Establish our loss funcion (NLLLoss needs log_softmax outputs)
criterion = nn.NLLLoss()

def train(model, optimizer, criterion):
    
    model.train() # training mode
    
    running_loss = 0
    running_corrects = 0
    
    for x,y in train_set:
        
        x=x.to(device)
        y=y.to(device)
        
        optimizer.zero_grad() # make the gradients 0 
        output = model(x) # forward pass
        _, preds = torch.max(output, 1)
        
        loss = criterion(output, y) # calculate the loss value
        
        loss.backward() # compute the gradients
        optimizer.step() # uptade network parameters 
                
        # statistics 
        running_loss += loss.item() * x.size(0)
        running_corrects += torch.sum(preds==y).item()
            
    epoch_loss = running_loss / len(train_samples) # mean epoch loss 
    epoch_acc = running_corrects / len(train_samples) # mean epoch accuracy
    
    return epoch_loss, epoch_acc


def test(model, optimizer, criterion):
    

    model.eval() # evaluation mode
    
    running_loss = 0
    running_corrects = 0
    
    # we do not need to compute the gradients in eval mode
    with torch.no_grad(): 

        for x,y in test_set:
            
            x=x.to(device)
            y=y.to(device)
        
            output = model(x) # forward pass
            _, preds = torch.max(output, 1)

            loss = criterion(output, y) # calculate the loss value
           
            # statistics 
            running_loss += loss.item() * x.size(0)
            running_corrects += torch.sum(preds==y).item()
            
    epoch_loss = running_loss / len(test_samples) # mean epoch loss
    epoch_acc = running_corrects / len(test_samples) # mean epoch accuracy
    
    
    return epoch_loss, epoch_acc


for epoch in range(10):

    start = t.time()

    train_loss, train_acc = train(model, optimizer, criterion)
    print('-' * 74)
    print('| End of epoch: {:3d} | Time: {:.2f}s | Train loss: {:.3f} | Train acc: {:.3f}|'
          .format(epoch + 1, t.time() - start, train_loss, train_acc))
    
    test_loss, test_acc = test(model, optimizer, criterion)
    print('-' * 74)
    print('| End of epoch: {:3d} | Time: {:.2f}s | Test loss: {:.3f} | Test acc: {:.3f}|'
          .format(epoch + 1, t.time() - start, test_loss, test_acc))

