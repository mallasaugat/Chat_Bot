
from nltk_utils import preprocess_sentence, vectorize
import numpy as np
import pandas as pd
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import Net

with open('intents.json','r', encoding='utf-8') as f:
    intents = json.load(f)

patterns = np.array([]) # sentences written by users
tags = np.array([]) # sentences' categories

for intent in intents['intents']:
    for pattern in intent['patterns']:
        pattern = preprocess_sentence(pattern)
        patterns = np.append(patterns, pattern)
        tags = np.append(tags, intent['tag'])
patterns = patterns.reshape((-1,1))
tags = tags.reshape((-1,1))     
xy = np.concatenate((patterns,tags),axis=1)
unique_tags = sorted(set(tags.reshape(-1)))

df = pd.DataFrame(xy,columns=['Pattern','Tag'])

X_train, y_train = vectorize(df)

print(X_train.shape[1])

class ChatDataset(Dataset):
    def __init__(self):
        self.x_data = X_train
        self.y_data = y_train
        self.n_samples = len(X_train)
        
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]
    
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
n_inputs = X_train.shape[1]
n_outputs = len(set(y_train))
        
# Setting hyperparameters

n_epochs = 10000
lr = 0.001 # learning rate defining the importance of each step during the parameters' optimization
hidden1 = 8 # defining the number of neurons in hidden layers
hidden2 = 8
batch_size = 8

model = Net([n_inputs, hidden1, hidden2, n_outputs])

train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(n_epochs):
    for x, y in train_loader:
        yhat = model(x.float())
        y = y.to(dtype=torch.long)
        loss = criterion(yhat, y)
        optimizer.zero_grad()
        loss.backward() # compute gradient of the loss with respect to all the learnable parameters
        optimizer.step() # update parameters value
    if (epoch+1)%100 == 0:
        print(f'Epoch {epoch+1}/{n_epochs} total loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

data = {
        'n_inputs':n_inputs,
        'hidden1':hidden1,
        'hidden2':hidden2,
        'n_outputs': n_outputs,
        'model_state': model.state_dict(),
        'tags': unique_tags
        }

FILE = "data.pth"
torch.save(data, FILE)

print(f'Training complete. File has been saved to {FILE}')