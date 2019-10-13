import argparse
# Imports here
import time
import copy
import numpy as np
import pandas as pd
from timeit import default_timer as timer
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim import lr_scheduler
from collections import OrderedDict


def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Creates parse 
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, default='flowers/', 
                        help='Path o data directory')
    parser.add_argument('--arch', type=str, default='vgg16', 
                        help='chosen model')
    parser.add_argument('--save_dir', type=str, default='/',
                        help='trained model save dir')
    #--learning_rate 0.01 --hidden_units 512 --epochs 20
    parser.add_argument('--epochs', type=int, default= 20,
                        help='no of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=512,
                        help='no of epochs to train')
    parser.add_argument('--gpu', type=int, default=1,
                        help='whether to train on GPU')

    # returns parsed argument collection
    return parser.parse_args()

def train(model, criterion, optimizer, scheduler, dataloaders, datasizes, gpu, num_epochs=10):
    """
    Trains the model 
    Parameters:
     model - the trained model
     criterion - the type model arch fine tuned with
     optimizer -  type of optimizer to use
     scheduler
     dataloaders - the dataloders
     datasizes - size of the data for calculating accuracy
     num_epochs - how many epochs to run
    Returns:
     model - a trained model
    """
    start_time = time.time()
    # device = 'cuda'
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # if gpu:
    # model.to('cuda')
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  
            else:
                model.eval()  

            running_loss = 0.0
            running_corrects = 0
        
            for inputs, labels in dataloaders[phase]:
                if gpu:
                    inputs = inputs.to('cuda')
                    labels = labels.to('cuda')

                # zero the parameter gradients
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)   
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / datasizes[phase]
            epoch_acc = running_corrects.double() / datasizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    end_time = time.time()
    total_time_spend = end_time - start_time
    print('Training complete {:.0f}m {:.0f}s'.format(
        total_time_spend // 60, total_time_spend % 60))
    print('val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


def validation(model, data, dataloaders, cuda=False):
    """
    Runs the trained model agains validation data and prints the accuracies.
    Parameters:
        model - the trained model
        data - the type of data to run on: valid, test
        cuda -  whether to run on GPU or not
    Returns:
        None - does not returns anything
    """
    model.eval()
    model.to('cuda')
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloaders[data]):
            if cuda:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                
            # run forward pass
            outputs = model.forward(inputs)
            # maximum probability, max value
            _, predicted = outputs.max(dim=1)
            equals = predicted == labels.data
            print("Validation accuracies below")
            print(equals.float().mean())
            
    print("Validation done")

def save_model(model, arch, image_datasets, save_dir):
    
    """
    Saves the trained model into a directory
    Parameters:
     model - the trained model
     arch - the type model arch fine tuned with
     image_dataset -  to get the class indexes
    Returns:
     None - does not returns anything
    """
    model.class_to_idx = image_datasets['train'].class_to_idx
    model.cpu()
    
    model_save_dir = "%s/checkpoint.pth" %(save_dir) 
    torch.save({'arch': arch,
                'state_dict': model.state_dict(), 
                'class_to_idx': model.class_to_idx}, 
                model_save_dir)
    
    print("model saved to %s" %(model_save_dir))


def get_dataloaders(data_dir):
    """
    Creates transforms and generate dataloaders
    Parameters:
     dat_dir - The directory of the training data
    Returns:
     dataloaders - The dataloaders
     data_sizes - size of the data
     image_datasets - image datasets
    """
        
        
    # data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    data_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=256),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    # Validation does not use augmentation
    'valid':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }

    # DONE: Load the datasets with ImageFolder
    # image_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)
    image_datasets = {
        'train':
        datasets.ImageFolder(root=train_dir, transform=data_transforms['train']),
        'valid':
        datasets.ImageFolder(root=valid_dir, transform=data_transforms['valid']),
        'test':
        datasets.ImageFolder(root=test_dir, transform=data_transforms['test']),
    }

    # Done: Using the image datasets and the trainforms, define the dataloaders
    # dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True)
    }

    data_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
    
    return dataloaders, data_sizes, image_datasets


def build_network(model, hidden_units):
    # DONE: Build and train your network
    
    """
    Fine tune and return a model
    Parameters:
     model - the model architechture, vgg16 or densenet121
    Returns:
     dataloaders - The dataloaders
     data_sizes - size of the data
     image_datasets - image datasets
    """
    from collections import OrderedDict
    
    if model == 'vgg16':
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(25088, hidden_units)),
                ('relu1', nn.ReLU()),
                ('dropout',nn.Dropout(0.2)),
                ('fc2', nn.Linear(hidden_units, 102)),
                ('output',nn.LogSoftmax(dim=1))
            ]))
        
    elif model == 'densenet121':
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(1024, hidden_units)),
                ('relu', nn.ReLU()),
                ('fc2', nn.Linear(hidden_units, 102)),
                ('output', nn.LogSoftmax(dim=1))
            ]))
    else:
        print('Model architecuture not supported, try vgg16 or densenet121')
        exit(0)
    model.classifier = classifier
    
    
    # Visualize trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    
    return model

def main():
    input_args = get_input_args()
    print(input_args)
    dataloaders, datasizes, image_datasets = get_dataloaders(input_args.data_dir)
    
    
    # Get the fine tuned model 
    model = build_network(input_args.arch, input_args.hidden_units)
    
    criteria = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=input_args.learning_rate)
    sched = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    # epochs=1
    
                                       
    if input_args.gpu==1:
        model = model.to('cuda')
        gpu = True
    else:
        gpu = False
    
    # train and fit the model
    model_fit = train(model, criteria, optimizer, sched, dataloaders, datasizes, gpu, input_args.epochs)
    print("training done")
    # run validation
    validation(model_fit, 'test', dataloaders, True)
    
    # save model
    save_model(model_fit, input_args.arch, image_datasets, input_args.save_dir)
    
    
    
# Call to main function to run the program
if __name__ == "__main__":
    main()