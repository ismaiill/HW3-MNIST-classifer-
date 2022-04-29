from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import  ConcatDataset
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from numpy import linalg as LA


'''
This code is adapted from two sources:
(i) The official PyTorch MNIST example (https://github.com/pytorch/examples/blob/master/mnist/main.py)
(ii) Starter code from Yisong Yue's CS 155 Course (http://www.yisongyue.com/courses/cs155/2020_winter/)
'''

class fcNet(nn.Module):
    '''
    Design your model with fully connected layers (convolutional layers are not
    allowed here). Initial model is designed to have a poor performance. These
    are the sample units you can try:
        Linear, Dropout, activation layers (ReLU, softmax)
    '''
    def __init__(self):
        # Define the units that you will use in your model
        # Note that this has nothing to do with the order in which operations
        # are applied - that is defined in the forward function below.
        super(fcNet, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=20)
        self.fc2 = nn.Linear(20, 10)
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Define the sequence of operations your model will apply to an input x
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output


class ConvNet(nn.Module):
    '''
    Design your model with convolutional layers.
    '''
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


class Net(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(10, 8, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(200, 81)
        self.fc2 = nn.Linear(81, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output





def train(args, model, device, train_loader, optimizer, epoch,losstrk,error):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()   # Set the model to training mode
    
    correct = 0
    train_num = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
      
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)                # Make predictions
        loss = F.nll_loss(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step
        
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        #if train_num==50560:
        #    losstrk.append(loss.item())
        #    error.append(correct)
            
        train_num += len(data)
        
        
        if batch_idx % args.log_interval == 0:
        
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \tLoss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item(), correct, train_num,
        100. * correct / train_num)
                
                )
        

def test(model, device, test_loader,testlosstrk,testerror):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    predictedarray=[]
    targetarray=[]
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)
            
            
            # code to print confusion matrix
            
            #for i in range(0,len(pred)):
            #    predictedarray.append(pred[i].item()) # predicted array
            #    targetarray.append(target[i].item())  # target array
            
            # code to print where the algorithm made a mistake
            
            #for k in range(0,len(data) ):
            #    if pred[k].item() != target[k].item():
            #       plt.imshow(data[k][0].numpy(), cmap='gray')   
            #      plt.show()
                    
      
    test_loss /= test_num
    testlosstrk.append(test_loss)
    testerror.append((test_num-correct) /test_num)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_num,
        100. * correct / test_num))
    
    #print(confusion_matrix(predictedarray, targetarray)) # print confusion matrix
    
def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=60, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')
    parser.add_argument('--load-model', type=str,
                        help='model file path')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# Evaluate on the official test set   
    
    if args.evaluate:
        assert os.path.exists(args.load_model)

        # Set the test model
        model = fcNet().to(device)
        model.load_state_dict(torch.load(args.load_model))

        test_dataset = datasets.MNIST('/Users/Ismail/Documents/CS148/Homeworks/HW 3/', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        test(model, device, test_loader)

        return
       
# Evaluate on the official test (WITHOUT COMMAND LINE)

    Done_training= False
    
    if Done_training:
        
        
        testlosstrk =[]
        testerror=[]
        model = Net().to(device)
        model.load_state_dict(torch.load('/Users/Ismail/Documents/CS148/Homeworks/HW 3/caltech-ee148-spring2020-hw03/mnist_model.pt'))
        
        test_dataset = datasets.MNIST('/Users/Ismail/Documents/CS148/Homeworks/HW 3/', train=False,
        transform=transforms.Compose([
        transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
        
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
        
        
        #Evaluate on the test set
        
        test(model, device, test_loader,testlosstrk,testerror) 
            
        Embedding_2D=False # Code to plot the embedding
        
        
        if Embedding_2D:
            # Extracting feature vector
            
            
            def get_features(name):
                def hook(model, input, output):
                    features[name] = output.detach()
                return hook
            
            # Register hook (just before the last layer)
            
            model.fc1.register_forward_hook(get_features('feat'))
            
            # extract feature vector before the last layer
            
            feature_vector=[]
            target_vector=[]
            features={}
            
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                feature_vector.append(features['feat'].cpu().numpy())
                target_vector.append(target)
                
            # concatenate all the arrays
            
            feature_vector=np.concatenate(feature_vector)
            target_vector=np.concatenate(target_vector)
            
            # inspect the shape
            
            print(target_vector)
            
            # USE TSNE to visualize in 2D
            
            tsne = TSNE(n_components=2, random_state=0)
            
            feature_embd = tsne.fit_transform(feature_vector)
            
            
            labels=[0,1,2,3,4,5,6,7,8,9]
            target_ids = range(len(labels))
    
            plt.figure(figsize=(6, 5))
            colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
            for i, c, label in zip(target_ids, colors, labels):
                plt.scatter(feature_embd[target_vector == i, 0], feature_embd[target_vector == i, 1], c=c, label=label)
            plt.legend()
            plt.show()
            
            # code to find the neighbors inside the embedded space
            
        Euclidian_metric_finder = True
        
        
        
        if Euclidian_metric_finder:
            
            def get_features(name):
                def hook(model, input, output):
                    features[name] = output.detach()
                return hook
            
            # Register hook (just before the last layer)
            
            model.fc1.register_forward_hook(get_features('feat'))
            
            features={}
            
            
            
            # convert the entire test set into feature vectors and concadetane data
            
            feature_vector_testset=[]
            choose_feature_vector=[]
            concatenate_data=[]
            
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                feature_vector_testset.append(features['feat'].cpu().numpy())
                
                for k in range(0,10):
                    concatenate_data.append(data[k])
                
            
                
            feature_vector_testset=np.concatenate(feature_vector_testset)
            
            I_0 = feature_vector_testset[500] # choose a feature vector (e.g first image of the test set)
            
            # plot I_0
            
            #plt.grid(False)
            #plt.imshow(concatenate_data[500][0].numpy(), cmap='gray')   
            
            
            # substract the two arrays I_0 and feature_vector
            
            coord_distance=np.subtract(I_0,feature_vector_testset)
            
            # compute the norm of the coord_distance
            
            Euc_distance=[]
            
            
            for i in range (len(coord_distance)):
                 Euc_distance.append (LA.norm(coord_distance[i]))
            
           # find the indices of the 8 closest images in the euclidian metrix
           
            print(np.argsort(Euc_distance)[:9]) # we don't count the anchor image, so we need :9

            # Plot the the closest images to I_0:
                
            for j in range(len(np.argsort(Euc_distance)[:9])):
                #print(np.argsort(Euc_distance)[:9][j])
                plt.grid(False)

                plt.imshow(concatenate_data[np.argsort(Euc_distance)[:9][j]][0].numpy(), cmap='gray')   
                plt.show()



# Default training dataset (60 000)
    
    train_dataset = datasets.MNIST('/Users/Ismail/Documents/CS148/Homeworks/HW 3/', train=True, download=False,
                transform=transforms.Compose([       # Data preprocessing
                    transforms.ToTensor(),
                     # Add data augmentation here
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))
# Transformed training dataset (60 000)

    Blurtrain_dataset = datasets.MNIST('/Users/Ismail/Documents/CS148/Homeworks/HW 3/', train=True, download=False,
                transform=transforms.Compose([       # Data preprocessing
                    transforms.ToTensor(),
                    transforms.GaussianBlur(kernel_size=(1,17), sigma=(0.1, 2.0)),# Add data augmentation here
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))
    
# Augmented training set
    
    Aug_train_dataset=ConcatDataset([train_dataset, Blurtrain_dataset])


    # You can assign indices for training/validation or use a random subset for
    # training by using SubsetRandomSampler. Right now the train and validation
    # sets are built from the same indices - this is bad! Change it so that
    # the training and validation sets are disjoint and have the correct relative sizes.
    
    
    subset_indices_train = range(9000,60000)
    subset_indices_augmentedtrain = range(9000,120000)
    subset_indices_valid = range(0,9000)
    
# load training dataset

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(subset_indices_train)
    )
    
# load augmented training dataset instead (set True if train on augmenged dataset)
    
    augmented= False
    
    if augmented:
        
        train_loader = torch.utils.data.DataLoader(
            Aug_train_dataset, batch_size=args.batch_size,
            sampler=SubsetRandomSampler(subset_indices_augmentedtrain)
        )
    
# load validation dataset
    
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.test_batch_size,
        sampler=SubsetRandomSampler(subset_indices_valid)
    )
    

# Load your model [fcNet, ConvNet, Net]
    model = Net().to(device)

# Try different optimzers here [Adam, SGD, RMSprop]
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

# Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

# Training loop
    
    Original_training_loop = True
    
    if Original_training_loop:
    
        trainlosstrk=[]
        testlosstrk=[]
        trainerror=[]
        testerror=[]
        
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch,trainlosstrk, trainerror)
            test(model, device, val_loader,testlosstrk,testerror)
            scheduler.step()    # learning rate scheduler
    
        torch.save(model.state_dict(), "mnist_model.pt")
                    
        
# plot loss vs epochs for training 

        #plt.plot(list(range(1,args.epochs + 1)),losstrk)
    
# plot loss vs epochs for test 
    
        #plt.plot(list(range(1,args.epochs + 1)),testlosstrk)
    
    

             
    
# TRAINING error on different number of training examples
    
    Vald_different_size_training_loop = False
    
    
    if  Vald_different_size_training_loop:
        
        N=[3187,6375, 12750, 25500] # 1/2, 1/4, 1/8, 1/16 training examples
        valdE=[] # validation error
        for i in range (0,4):
            indices = range(9000,9000+ N[i])
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size,
                sampler=SubsetRandomSampler(indices)
            )
            trainlosstrk=[]
            testlosstrk=[]
            trainerror=[]
            testerror=[]
            
            for epoch in range(1, args.epochs + 1):
                train(args, model, device, train_loader, optimizer, epoch,trainlosstrk, trainerror)
                test(model, device, val_loader,testlosstrk,testerror)
                scheduler.step()    # learning rate scheduler
                
                if epoch== args.epochs:
                    valdE.append(testerror[epoch-1]) # save error of the last epoch for the the training set i
                    print(testerror[epoch-1])
        
        # plot vald error vs # training example:
    
        N=np.log(N)    # log scale
        valdE=np.log(valdE)  # log scale
        plt.plot(N,valdE) # plot on log-log scale
        plt.xlabel("Number of training example (log scale)")
        plt.ylabel("training error (log scale) ")
        
# TEST error on different number of training examples
    
    Test_different_size_training_loop= False
    
    # load official test dataset
        
    test_dataset = datasets.MNIST('/Users/Ismail/Documents/CS148/Homeworks/HW 3/', train=False,
    transform=transforms.Compose([
    transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
        
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
    
    
    if Test_different_size_training_loop:
        
        N=[3187,6375, 12750, 25500] # 1/2, 1/4, 1/8, 1/16 training examples
        testE=[] # test error
        for i in range (0,4):
            indices = range(9000,9000+ N[i])
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size,
                sampler=SubsetRandomSampler(indices)
            )
            trainlosstrk=[]
            testlosstrk=[]
            trainerror=[]
            testerror=[]
            
            for epoch in range(1, args.epochs + 1):
                train(args, model, device, train_loader, optimizer, epoch,trainlosstrk, trainerror)
                test(model, device, test_loader,testlosstrk,testerror)
                scheduler.step()    # learning rate scheduler
                
                if epoch== args.epochs:
                    testE.append(testerror[epoch-1]) # save error of the last epoch for the the training set i
                    print(testerror[epoch-1])
        
        # plot vald error vs # training example:
    
        N=np.log(N)    # log scale
        testE=np.log(testE)  # log scale
        plt.plot(N,testE) # plot on log-log scale
        plt.xlabel("Number of training example (log scale)")
        plt.ylabel("test error (log scale) ") 
        
        
# Analyse the network

    Analyse_the_model= False
    
    if Analyse_the_model:
    
        # load the model and dictionary    
        model = Net().to(device) 
        model.load_state_dict(torch.load('/Users/Ismail/Documents/CS148/Homeworks/HW 3/caltech-ee148-spring2020-hw03/mnist_model.pt'))
        
        # Print a kernel from the first layer
    
        print(list(model.parameters()) [0][3][0].detach())  
        
        # Visualize a kernel from the first layer
        
        plt.imshow(list(model.parameters()) [0][8][0].detach().numpy(), cmap='gray')   
            

# You may optionally save your model at each epoch here

    #if args.save_model:
    #    torch.save(model.state_dict(), "mnist_model.pt")


if __name__ == '__main__':
    main()
