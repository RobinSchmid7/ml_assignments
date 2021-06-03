
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
from PIL import Image

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt


####################################
LOAD_PREPROCESSED = True
LOAD_PREPARED_TRAINING_DATA = False
####################################
CHANNELS = 3
WIDTH = 150
HEIGHT = 150
####################################
BATCHSIZE = 64
EPOCHS = 1
LEARNING_RATE = 1e-4
EMBEDDING_DIM = 128
####################################
HANDOUT_PATH = "/home/marvin/Downloads/IML_handout_task4/"  
####################################

# set random seed
np.random.seed(42)
torch.manual_seed(42)

###########################################################################
################################ FUNCTIONS ################################
###########################################################################
#preprocessing and loading the dataset
class SiameseDataset():
    def __init__(self,training_csv=None,training_dir=None,transform=None):
        # used to prepare the labels and images path
        self.train_df=pd.read_csv(os.path.join(training_dir, training_csv), sep=' ', dtype=str)
        self.train_df.columns =["A","B","C"]
        self.train_dir = training_dir    
        self.transform = transform

    def __getitem__(self,index):
        # getting the image path
        imageA_path=os.path.join(self.train_dir+'food/',str(self.train_df.iat[index,0]))+'.jpg'
        imageB_path=os.path.join(self.train_dir+'food/',str(self.train_df.iat[index,1]))+'.jpg'
        imageC_path=os.path.join(self.train_dir+'food/',str(self.train_df.iat[index,2]))+'.jpg'
        # Loading the image
        imgA = Image.open(imageA_path)
        imgB = Image.open(imageB_path)
        imgC = Image.open(imageC_path)
        imgA = imgA.convert("RGB")
        imgB = imgB.convert("RGB")
        imgC = imgC.convert("RGB")
        # Apply image transformations
        if self.transform is not None:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)
            imgC = self.transform(imgC)

        return imgA, imgB, imgC
        
    def __len__(self):
        return len(self.train_df)


#create a siamese network
class SiameseNetwork(nn.Module):
    '''define a siamese network'''
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Setting up the Sequential of CNN Layers
        self.model = models.vgg11(pretrained=True)
        # replace the last fully connected layer
        # the output of the fully connected layer is the embedding of the image (128 components)
        self.model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, EMBEDDING_DIM),
            nn.BatchNorm1d(128)
            )

        # fix the weights of the original model
        for param in self.model.parameters():
            param.requires_grad = False

        # make the dense layers trainable
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward_once(self, inputs):
        # Forward pass 
        embedding = self.model(inputs)
        return embedding

    def forward(self, input1, input2, input3):
        # forward pass of input 1
        #output1 = self.forward_once(input1)
        # forward pass of input 2
        #output2 = self.forward_once(input2)
        # forward pass of input 3
        #output3 = self.forward_once(input3)
        output1 = self.model(input1)
        output2 = self.model(input2)
        output3 = self.model(input3)
        return output1, output2, output3

        
class TripletLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=0.1):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, x2):
        # euclidian distance A-B
        diff1 = x0 - x1
        dist1 = torch.sum(torch.pow(diff1, 2), 1)
        dist1 = torch.sqrt(dist1)

        # euclidian distance A-C
        diff2 = x0 - x2
        dist2 = torch.sum(torch.pow(diff2, 2), 1)
        dist2 = torch.sqrt(dist2)

        # get the triplet loss
        mdist = self.margin + dist1 - dist2
        dist = torch.log(torch.add(torch.exp(mdist),1))
        loss = torch.div(torch.sum(dist), BATCHSIZE)
        return loss

###########################################################################
############################### MAIN SCRIPT ###############################
###########################################################################

# Load the the dataset from raw image folders
train_dataset = SiameseDataset(training_csv='train_triplets.txt',training_dir=HANDOUT_PATH,
                                        transform=transforms.Compose([transforms.Resize((WIDTH,HEIGHT)),
                                                                      transforms.ToTensor()]))
train_dataloader = DataLoader(train_dataset,batch_size=BATCHSIZE,shuffle=True,num_workers=4)                                                                     

# Declare Siamese Network
SiamNet = SiameseNetwork()
# Decalre Loss Function
criterion = TripletLoss()
# Declare Optimizer
optimizer = torch.optim.Adam(SiamNet.parameters())
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SiamNet = SiamNet.to(device)

# ================
# Perform training
# ================
SiamNet.train()
loss_vec=[] 
counter_vec=[]
iteration_number = 0
for epoch in range(1,EPOCHS+1):
    for i, data in tqdm(enumerate(train_dataloader,1)):

        img0, img1, img2 = data
        img0, img1, img2 = img0.to(device), img1.to(device), img2.to(device)
        # zero the gradients
        optimizer.zero_grad()
        # produce predictions
        embedding1,embedding2,embedding3 = SiamNet(img0,img1,img2)
        # compute loss
        loss = criterion(embedding1,embedding2,embedding3)
        # backpropagate
        loss.backward()
        # update weights
        optimizer.step()  
        iteration_number += 1
        print(loss.item())
    print("Epoch {}\n Current loss {}\n".format(epoch,loss.item()))
    iteration_number += 10
    counter_vec.append(iteration_number)
    loss_vec.append(loss.item())
plt.plot(counter_vec, loss_vec)   
torch.save(SiamNet.state_dict(), "model.pt")
print("Model Saved Successfully") 

# ================
# Predict test set
# ================
test_dataset = SiameseDataset(training_csv='test_triplets.txt',training_dir=HANDOUT_PATH,
                                        transform=transforms.Compose([transforms.Resize((WIDTH,HEIGHT)),
                                                                      transforms.ToTensor()]))
test_dataloader = DataLoader(test_dataset,num_workers=4,batch_size=BATCHSIZE,shuffle=False)
