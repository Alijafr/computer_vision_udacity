import torch
import torch.nn as nn
import torchvision.models as models

import torch.nn.functional as F
import numpy as np
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size,num_layers=1):
        ''' Initialize the layers of this model.'''
        super().__init__()
    
        self.num_layers =num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        # Embedding layer that turns words into a vector of a specified size
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # The LSTM takes embedded word vectors (of a specified size) as input
        # and outputs hidden states of size hidden_dim
        self.lstm = nn.LSTM(input_size=embed_size, \
                            hidden_size=hidden_size, 
                            num_layers=num_layers,
                            bias=True, 
                            batch_first=True, 
                            dropout=0,  
                            bidirectional=False, 
                           )
                             
        self.fc = nn.Linear(hidden_size,vocab_size)
        #self.dropout = nn.Dropout(p=0.3)
        self.softmax= nn.Softmax(dim=2)
        
    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        # Implement function
        train_on_gpu = torch.cuda.is_available()
        # initialize hidden state with zero weights, and move to GPU if available
        weight = next(self.parameters()).data
        if (train_on_gpu):
            hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().cuda(),
                  weight.new(self.num_layers, batch_size, self.hidden_size).zero_().cuda())
        else:
            hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                      weight.new(self.num_layers, batch_size, self.hidden_size).zero_())
        return hidden

    def forward(self, features, captions):
        """ Define the feedforward behavior of the model """
        batch_size= features.size(0)
        #remove <end> so that the RNN does not produce a prediciton for it
        captions = captions[:, :-1] # (batch_size, catption_size)
          
        
        # Initialize the hidden state
        self.hidden = self.init_hidden(batch_size) 
                
         #use embeding layer for the captions
        embeddings = self.embed(captions)#old size: (batch_size,caption_length), new_size:(batch_size,caption_length-1,embed_size), -1 because we deleted <end>
        
        #the first input will be the features of the image, 
        #then we need to use captoin to generate the rest of the input of the RNN
        #therefore, we need to concatinate the features and captions (without <end>)
        #features dimension is (batch_size,embed_size) --> need unsqueeze to be able to concatinate with embeding
        #we will add the feature to the caption length as the start to be the first input
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)  #new size: (batch_size, captoin_length,embed_size), after adding the features
        
        #apply the rnn
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden) # lstm_out shape : (batch_size, caption length, hidden_size)

        # Fully connected layer
        out = self.fc(lstm_out) # outputs shape : (batch_size, caption length, vocab_size)
        #out = self.softmax(out)
        return out
    
    
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        #list to hold the predicited words
        words_idx=[]
        #initialize hidden 
        hidden = self.init_hidden(1)
        
        for i in range(max_len):
            lstm_out, hidden = self.lstm(inputs,hidden) # result (1,1,hidden_size)
            out=self.softmax(self.fc(lstm_out)) # (1,1,vocab_size)
            _,top_word = torch.topk(out,1,dim=2) #(1,1,1)
            
            words_idx.append(top_word.cpu().numpy().squeeze(0).squeeze(0).item())
            
            if top_word ==1:# 1 means it is <end>
                break
            inputs = self.embed(top_word.squeeze(0))# will result in (1,1,embed_size), without the squeeze --> (1,1,1,embed_size)
            
            
        return words_idx
            
            
        
        
        