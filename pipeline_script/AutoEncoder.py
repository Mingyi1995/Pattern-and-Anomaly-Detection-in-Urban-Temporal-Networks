#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import random

from sklearn.mixture import GaussianMixture 
import datetime
import operator
import os

from IPython.display import clear_output


# In[64]:


RecordWritingPath = ''
TransportationDataPath = '/Users/hemingyi/Documents/capstone/post/transportation/'
EventDataPath = '/Users/hemingyi/Dropbox/quziyin/event data/update/'
# comboPath = '/Users/hemingyi/Documents/capstone/combo/'
# dataFile = TransportationDataPath+city+'EdgeYearwiseAggregated.csv'


# In[65]:


def preprocessComm(df_comm):
    '''
    1. This function converts the raw community data in Long Format to Wide.
    2. Also, scales all the edge weight using minmax scaling
    
    Parameter:
        df_comm:
        df_comm is dataframe with raw data of inter-community ridership.
        df_comm.columns: ['start_id', 'end_id', 'date', 'amount']

    Returns:
        Dataframe with dates as row, communitypairs as columns and value is 
        ridership for given communitypairs on the given date
    '''
    # convert to datetime
    print('processing date')
    df_comm['date'] = pd.to_datetime(df_comm['date'])
    # create a ID
    print('processing id')
    df_comm['start_id'] = df_comm['start_id'].astype('str')
    df_comm['end_id'] = df_comm['end_id'].astype('str')
    df_comm['ID'] = df_comm['start_id'] +','+ df_comm['end_id']
    # convert data from long to wide
    print('dataframe pivot')
    df_main = df_comm.pivot(index = 'date', columns = 'ID', values ='amount')
    df_main = df_main.sort_index()
    # fill missing value
    df_main.fillna(0, inplace = True)
    # scale all the columns
    df_main = pd.DataFrame(minmax_scale(df_main.values))
    
    return df_main

class autoencoder(nn.Module):
    '''
    The autoencoder architecture
    ''' 
    def __init__(self,inputD,encoding_dim):
        super(autoencoder, self).__init__()
        
        ## ENCODER
        self.encoder = nn.Sequential()
        
        self.encoder.add_module("enc_0", nn.Linear(inputD,encoding_dim[0]))
        self.encoder.add_module("relu_0", nn.Tanh())
          
        for l in range(1,len(encoding_dim)):
            self.encoder.add_module("enc_"+str(l), nn.Linear(encoding_dim[l-1],encoding_dim[l]))
            self.encoder.add_module("encrelu_"+str(l), nn.Tanh())
        
        ## DECODER
        self.decoder = nn.Sequential()
        
        for l in range(len(encoding_dim)-1,0,-1):
            self.decoder.add_module("dec_"+str(l), nn.Linear(encoding_dim[l],encoding_dim[l-1]))
            self.decoder.add_module("decrelu_"+str(l), nn.Tanh())
            
        self.decoder.add_module("dec_0", nn.Linear(encoding_dim[0],inputD))
        self.decoder.add_module("decrelu_0", nn.Sigmoid())
        
        ## WEIGHTS
        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def init_weights(self,m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def representation(self, x):
        x = self.encoder(x)
        return x
    
    
def augmentData(df_comm):
    '''
    This function takes the preprocessed df_comm DataFrame as input.
    Augments the input by adding noise and increasing the number of 
    observation by 20 times.
    
    Parameters:
        df_comm: Wide format df_comm obtained after preprocessing
        The columns should be community pairs and rows should be
        dates/observation.
    
    Returns:
        The ouput consist the a pair of 2 DataFrames
        agTrain_setI = The augmented training set with noise
        agTrain_setO = The corresponding clean ser.
    '''
    # Initiate output sets
    agTrain_setI = df_comm
    agTrain_setO = df_comm
    
    # Augment output sets
    for i in range(20):
        agTrain_setI = np.vstack([agTrain_setI , df_comm + np.random.uniform(low=0.0, high=0.05, size=(df_comm.shape[0],df_comm.shape[1]))])
        agTrain_setO = np.vstack([agTrain_setO , df_comm])
        
    return agTrain_setI, agTrain_setO


def trainModel(agTrain_setI, agTrain_setO):
    
    print('4.a: Converting data to tensor and defining dataloader.')
    batch_size = 40
    agTrain_setI = torch.tensor(agTrain_setI).float()
    agTrain_setO = torch.tensor(agTrain_setO).float()
    train_tensor = torch.utils.data.TensorDataset(agTrain_setI, agTrain_setO)
    train_loader = torch.utils.data.DataLoader(dataset = train_tensor, batch_size = batch_size, shuffle = True)
    
    print('4.b: Defining hyperparameter.')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 0.0001
    encoding_dim = [1024,512,256,128,64,32,16]
    batch_size = 40
    num_epochs = 1000
    criterion = nn.BCELoss()
    model = autoencoder(agTrain_setI.shape[1],encoding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    print('4.c: The training begins. Warning: Watching the model train is addictive.')

    # empty lists to store losses
    bce_loss = []
    mse_loss = []

    # epoch train
    for epoch in range(num_epochs):
        MSE_loss = []
        BCE_loss = []
        for data in train_loader:
            X, _ = data
            X = X.to(device)
            # ===================forward=====================
            output = model(X)
            loss = criterion(output, X)
            MSE_loss.append(nn.MSELoss()(output, X).item())
            BCE_loss.append(nn.BCELoss()(output, X).item())

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        #plt.plot(epoch + 1, loss.item(), '.' )
        print('epoch [{}/{}], loss:{:.4f}, MSE_loss:{:.4f}'
            .format(epoch + 1, num_epochs, np.mean(BCE_loss), np.mean(MSE_loss)))
        mse_loss.append(np.mean(MSE_loss))
        bce_loss.append(np.mean(BCE_loss))
    
    print('4.d: Woah!! if things ran smoothly so far. Rest will be easy.')
    return model

def autoencoderOutput(df_comm):
    
    print('1. Importing Libraries.')
    import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.utils.data
    
    # print('2. Preprocessing input.')
    # df_comm = preprocessComm(df_comm)
    
    print('3. Augmenting training data.')
    agTrain_setI, agTrain_setO = augmentData(df_comm)
    
    print('4. Model Training: Now things have gone deep. Sit back wait and relax.')
    modelAutoencoder = trainModel(agTrain_setI, agTrain_setO)
    
    print('5. Here is the output. Enjoy!')
    train_set = torch.tensor(df_comm.values).float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_output = modelAutoencoder.representation(train_set.to(device)).cpu().detach().numpy()
    
    return tr_output




def pipeline(city, raw, standard):
    print('Initialize')
    data =  pd.read_csv(TransportationDataPath+'Output/'+raw+'/OriginSize/'+standard+'/'+city+raw+standard+'.csv', date_parser='date')
    date = data['date']
    del data['date']
    dataTs = pd.DataFrame(autoencoderOutput(data))
    dataTs['date'] = date
    dataTs.to_csv(TransportationDataPath+'Output/'+raw+'/AE/'+standard+'/'+city+raw+standard+'.csv', index=False)

city = input('input city: ')
raw = input('input raw: Comm, IO ')
standard = input('standard: Normalize, Whiten, Both ')
pipeline(city, raw, standard)








