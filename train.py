import numpy as np
import random
import sys
from subprocess import *
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))



# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 10000
batch_size = 154
learning_rate = 1e-3
input_size = len(X_train[0])
hidden_size = 154
output_size = len(tags)
print(output_size, input_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
try:
    index = 1
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            
            # Forward pass
            outputs = model(words)
            # if y would be one-hot, we must apply
            # labels = torch.max(labels, 1)[1]
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if (epoch+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            print("")
            print(f'Epoch {epoch+1} complete')
            print("")
            index += 1
except:
    raise Exception("Null Dataset")


final_loss = f'{loss.item():.4f}'


while True:
    if final_loss > " 3.0000":
        print("Retraining the model results not pretty good excepted", 0.0000, 'but')
        print(f'final loss: {loss.item():.4f}')
        # Retrain the model
        try:
            index =+ 1
            for epoch in range(num_epochs):
                for (words, labels) in train_loader:
                    words = words.to(device)
                    labels = labels.to(dtype=torch.long).to(device)
                    
                    # Forward pass
                    outputs = model(words)
                    # if y would be one-hot, we must apply
                    # labels = torch.max(labels, 1)[1]
                    loss = criterion(outputs, labels)
                    
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                if (epoch+1) % 100 == 0:
                    print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
                    print("")
                    print(f'Epoch {epoch+1} complete')
                    print("")
                    index += 1
        except:
            raise Exception("Null Dataset")
        if final_loss > " 3.0000":
            break
        else:
            continue
    else:
        break

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

print(f'final loss: {loss.item():.4f}')

FILE = "nlp.pth"
torch.save(data, FILE)

print(" ")
print(f'training complete. file saved to {FILE}')
