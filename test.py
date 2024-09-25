import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
stemmer=PorterStemmer()
import json
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader

from main import NeuralNet

import os
with open('words.json','r') as f:
    intents=json.load(f)


all_words=[]
tags=[]
xy=[]
    
#nltk.download('punkt') do that also lol

def tokenise(word):
    return nltk.word_tokenize(word)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenised_sentence,words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    sentence_words=[stem(w) for w in tokenised_sentence]
    bag=np.zeros(len(words),dtype=np.float32)
    
    for idx,w in enumerate(words):
        if w in sentence_words:
            bag[idx]=1
    return bag
            
    

for intent in intents['intents']:
    tag=intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w=tokenise(pattern)
        all_words.extend(w)
        xy.append((w,tag))  
    
ignore_words=['?','!','.',',']
all_words=[stem(w) for w in all_words if w not in ignore_words]
all_words=sorted(set(all_words))
tags=sorted(set(tags))

X_train=[]
y_train=[]

for pattern_sentence,tag in xy:
    bag=bag_of_words(pattern_sentence,all_words)
    X_train.append(bag)
    
    label=tags.index(tag)
    y_train.append(label)
    

X_train=np.array(X_train)
y_train=np.array(y_train)

num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
output_size = len(tags)
hidden_size=8

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
                          num_workers=os.cpu_count(),persistent_workers=True)
    
    
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to('cpu')

import torch.nn as nn




#print(stem_words)
#print(a)
if __name__ == '__main__':
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to('cpu')
            labels = labels.to(dtype=torch.long).to('cpu')
            
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


    print(f'final loss: {loss.item():.4f}')
    
    data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
    }
    
    FILE = "model.pth"
    torch.save(data, FILE)

    print(f'training complete. file saved to {FILE}')

    



