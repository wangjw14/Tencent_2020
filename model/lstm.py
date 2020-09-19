# coding: UTF-8
import torch
import torch.nn as nn
import numpy as np
import os, time, pickle 
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.config = args  
        self.fc1 = nn.Linear() 
        self.bn1 = nn.
        self.fc2 = nn.Linear() 
        self.lstm = nn.LSTM() 
        self.fc3 = nn.Linear()
        self.fc4 = nn.Linear() 
        
        self.dropout = nn.Dropout(args.dropout_prob)
        self.fc_age = nn.Linear(self.num_filters * len(self.filter_sizes), 10)
        self.fc_gender = nn.Linear(self.num_filters * len(self.filter_sizes), 2)

        
    def forward(self,age, gender, dense_features, text_features):
        out = text_features.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        age_logits = self.fc_age(out)
        gender_logits = self.fc_gender(out) 
        if age is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(age_logits, age.long()) + loss_fct(gender_logits,gender.long()) 
            return loss 
        else:
            age_probs = torch.softmax(age_logits,-1)
            gender_probs = torch.softmax(gender_logits,-1)
            return age_probs, gender_probs




        self.norm= nn.BatchNorm1d(args.out_size)
        self.dense = nn.Linear(args.out_size, args.linear_layer_size[0])
        self.norm_1= nn.BatchNorm1d(args.linear_layer_size[0])
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.dense_1 = nn.Linear(args.linear_layer_size[0], args.linear_layer_size[1])  
        self.norm_2= nn.BatchNorm1d(args.linear_layer_size[1])
        self.out_proj = nn.Linear(args.linear_layer_size[1], args.num_label)

    def forward(self, features, **kwargs):
        x = self.norm(features)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(self.norm_1(x))
        x = self.dropout(x)
        x = self.dense_1(x)
        x = torch.relu(self.norm_2(x))
        x = self.dropout(x)        
        x = self.out_proj(x)
        return x
