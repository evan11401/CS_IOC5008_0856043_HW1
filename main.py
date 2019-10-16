from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy


#hyperparameter
batch_si = 16
learning_rate = 1e-3
epochs = 200
print('hyperparameter ready')

#data preprocess
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'all': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

#load data
data_dir = 'data/cs-ioc5008-hw1'
image_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                          data_transforms['train'])

#calculating the number of all classes to use WeightedRandomSampler
numbers = [0,0,0,0,0,0,0,0,0,0,0,0,0]
for inputs, labels in image_dataset:
    numbers[labels] += 1

#calculating the distribution of all classes
weight_per_class = [0.] * 13
N = float(sum(numbers))
for i in range(13):                                                   
    weight_per_class[i] = N/float(numbers[i]) 
weights = [0.]*len(image_dataset)
for i, (inputs, labels) in enumerate(image_dataset):
    weights[i] = weight_per_class[labels]
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights),replacement=True)                     
 
#data loaders
dataloaders = {'train': torch.utils.data.DataLoader(
                image_dataset,
                batch_size=batch_si,
                sampler=sampler
                )
              }

class_names = image_dataset.classes
dataset_sizes = len(image_dataset)

#using GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('train data loading ready')

#using resnet152 and pretrained=True
model_ft = models.resnet152(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 13)
model_ft = model_ft.to(device)

#freezing all layers but layer3, layer4
for name, child in model_ft.named_children():
   if name in ['layer3', 'layer4']:
       for param in child.parameters():
           param.requires_grad = True
   else:
       for param in child.parameters():
           param.requires_grad = False

print('model ready')

#using CrossEntropyLoss loss function | SGD weights updating method | CosineAnnealingLR for learning rate decay and restart
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=0.9)
exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft,T_max=10)

#deciding train method
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.train()  
       
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)                   
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        scheduler.step()
        epoch_loss = running_loss / dataset_sizes
        epoch_acc = running_corrects.double() / dataset_sizes

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            'train ', epoch_loss, epoch_acc))

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

print('train function ready')

#running train
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=epochs)

#loading test data
data_dir = 'data\\cs-ioc5008-hw1\\test'
test_dataset = datasets.ImageFolder(data_dir,data_transforms['all'])
testdataloader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_si,
                                          shuffle=False, num_workers=2)

print('test data loading ready')

#deciding test model
def test_model(model):
    was_training = model.training
    model.eval()
    import csv
    with open('0856043.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['id','label'])
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(testdataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                for j,(_) in enumerate(preds):
                    sample_fname, _ = test_dataset.samples[i*batch_si+j]
                    print(sample_fname[30:-4],class_names[preds[j].item()])                    
                    writer.writerow([sample_fname[30:-4], class_names[preds[j].item()]])

            model.train(mode=was_training)

#making predict
test_model(model_ft)

#saving model
torch.save(model_ft, 'resnet152')

print('model saved')