#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import glob
import time
import pickle
import pandas as pd

from tokenizers import BertWordPieceTokenizer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# In[ ]:


device = "cuda" if torch.cuda.is_available() else "cpu"


# In[ ]:


cross_path = '../data/cross_validation_data'
trans_path = '../data/transformed'
model_path = '../data/bert-case'


# In[ ]:


file_path = glob.glob(os.path.join(trans_path, '*'))


# In[ ]:


cross_val_path = glob.glob(os.path.join(cross_path, '*', '*'))


# In[ ]:


class DatasetPairs(Dataset):
    def __init__(self, file_path, cross_val_paths, model_path):
        self.dataset = self.read_pickle(file_path)
        self.split_dict = self.get_id_cross_val(cross_val_paths)
        self.tokenizer = self.get_tokenizer(model_path)
        self.splited_data(k=1)
        self.set_split(split='train')
    
    def read_pickle(self, path):
        data = pickle.load(open(path, 'rb'))
        
        return data
    
    def read_csv(self, path):
        d_data = pd.read_csv(path, sep='\t')
        return d_data
    
    def get_id_cross_val(self, paths):
        data_dict = {}
        path_dict = dict((file.split('/')[-2], file) for file in paths)
        for k, path in path_dict.items():
            train = self.read_csv(path)
            id_train = train.id.tolist()
            
            path = path.replace('train.csv', 'test.csv')
            test = self.read_csv(path)
            id_test = test.id.tolist()
            
            data_dict[int(k)] = (id_train, id_test)
            
        return data_dict
    
    def get_tokenizer(self, path):
        tokenizer = BertWordPieceTokenizer(os.path.join(path, 'vocab.txt'))
        return tokenizer
    
    def splited_data(self, k):
        id_train, id_test = self.split_dict[k]
        train = self.dataset[self.dataset.id.isin(id_train)]
        test = self.dataset[self.dataset.id.isin(id_test)]
        
        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)
        
        self.data_dict = {'train': (train, len(train)), 'test': (test, len(test))}
        
    def set_split(self, split='train'):
        self.data, self.length = self.data_dict[split]
    
    def __getitem__(self, idx):
        q1 = self.data.loc[idx, "question1"]
        q2 = self.data.loc[idx, "question2"]
        
        x_raw = self.tokenizer.encode(q1, q2)
        x = x_raw.ids
        y  = self.data.loc[idx, "is_duplicate"]
        
        x = torch.LongTensor(x)
        y = torch.LongTensor([y])
        
        return (x, y)
    
    def __len__(self):
        return self.length


# In[ ]:


class QuoraClassifier(nn.Module):
    def __init__(self, num_vocab, emb_size, hid_size, num_class):
        super(QuoraClassifier, self).__init__()
        self.emb = nn.Embedding(num_vocab, emb_size)
        self.lstm = nn.LSTM(emb_size, hid_size, batch_first=True)
        self.fc = nn.Linear(hid_size, num_class)
        
    def forward(self, input_):
        out = self.emb(input_)
        out, (h, c) = self.lstm(out)
        out = out[:, -1, :]
        out = self.fc(out)
        
        return out


# In[ ]:


dataset = DatasetPairs(file_path[0], cross_val_path, model_path)


# In[ ]:


num_vocab = dataset.tokenizer.get_vocab_size()
emb_size = 512
hid_size = 512
num_class = 2
batchsize = 256


# In[ ]:


model = QuoraClassifier(num_vocab, emb_size, hid_size, num_class)
model = model.to(device)


# In[ ]:


parameters = sum(p.numel() for p in model.parameters())
print(f'model has {parameters:,} trainable parameters')


# In[ ]:


optimizer = optim.Adam(model.parameters(), lr = 1e-3)
criterion = nn.CrossEntropyLoss()


# In[ ]:


def compute_accuracy(y, y_pred):
    y_label = y_pred.argmax(dim=1)
    n_correct = torch.eq(y, y_label).sum().item()
    accuracy = (n_correct / len(y_label)) * 100
    
    return accuracy


# In[ ]:


def compute_time(start, end):
    duration = end - start
    m = int(duration / 60)
    s = int(duration % 60)
    
    return m, s


# In[ ]:


def padding(data):
    x_list = []
    y_list = []
    for x, y in data:
        x_list.append(x)
        y_list.append(y)
        
    x_pad = pad_sequence(x_list, batch_first=True)
    y_pad = pad_sequence(y_list, batch_first=True)
    
    return x_pad, y_pad


# In[ ]:


history = {"running_loss": [], "running_loss_v": [], "running_accu": [], "running_accu_v": []}


# In[ ]:


for epoch in range(1, 51):
    
    running_loss = 0
    running_loss_v = 0
    running_accu = 0
    running_accu_v = 0
    
    start = time.time()
    
    dataset.set_split("train")
    data_gen = DataLoader(dataset, batch_size=batchsize, collate_fn=padding)
    model.train()
    for batch_index, (x, y) in enumerate(data_gen, 1):
        optimizer.zero_grad()
        x = x.to(device)
        y = y.squeeze().to(device)
        
        out = model(x)
        
        loss = criterion(out, y)
        loss_ = loss.item()
        running_loss += (loss_ - running_loss) / batch_index
        
        accuracy = compute_accuracy(y, out)
        running_accu += (accuracy-running_accu) / batch_index
        
        loss.backward()
        optimizer.step()
        
    dataset.set_split("test")
    data_gen = DataLoader(dataset, batch_size=batchsize, collate_fn=padding)
    model.eval()
    for batch_index, (x, y) in enumerate(data_gen, 1):
        x = x.to(device)
        y = y.squeeze().to(device)
        
        out = model(x)
        
        loss = criterion(out, y)
        loss_ = loss.item()
        running_loss_v += (loss_ - running_loss_v) / batch_index
        
        accuracy = compute_accuracy(y, out)
        running_accu_v += (accuracy - running_accu_v) / batch_index
        
    end = time.time()
    m, s = compute_time(start, end)
    
    print(f'{epoch} | {m}m {s}s')
    print(f'\ttrain loss: {running_loss:.2f} | train accuracy: {running_accu:.2f}')
    print(f'\tval loss: {running_loss_v:.2f} | val accuracy: {running_accu_v:.2f}')
    
    history["running_loss"].append(running_loss)
    history["running_loss_v"].append(running_loss_v)
    history["running_accu"].append(running_accu)
    history["running_accu_v"].append(running_accu_v)


# In[ ]:


pickle.dump(history, open("../reports/model_5_v1.pkl", 'wb'))

