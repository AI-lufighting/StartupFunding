#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
@course: IS6912
@author: ZHANG Xiaolu
@Student ID: 57318465
@Organization: Department of Information Systems, Business College, CityU
"""

import torch
import torch.nn as nn
import numpy as np
from alphabet import Alphabet


class TextCNN(nn.Module):
    def __init__(self, config, alphabet : Alphabet):
        super(TextCNN, self).__init__()
        self.config = config
        self.embeddings = nn.Embedding(alphabet.size(), config.emb_dim)
        # self.embeddings.weight.requires_grad = False
        if config['train_mode'] == 'static':
            self.embeddings = self.embeddings.from_pretrained(torch.from_numpy(alphabet.pretrained_emb))
        elif config['train_mode'] == 'fine-tuned':
            self.embeddings.weight.data.copy_(torch.from_numpy(alphabet.pretrained_emb))

        filters = config['filters']
        self.cnn = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1, config['output_channels'], [w, config.emb_dim]),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1)) for w in filters])

        self.linear = nn.Linear(config['output_channels'] * len(filters), 10, bias=True)

        self.linear_basic_1 = nn.Linear(51, 10, bias=True)
        # self.linear_basic_2 = nn.Linear(20, 10, bias=True)

        self.linear_final = nn.Linear(20, 2, bias=True)
        self.outfinal = nn.Softmax(dim=1)

        # device1 = torch.device('cuda')

        self.para1 = nn.Parameter(torch.tensor([0.5,0.5]), requires_grad=True).to('cuda')
        





        self.dropout = nn.Dropout(config['dropout'])
        self.relu = nn.ReLU()
        self.scale = np.sqrt(3.0 / config.emb_dim)
        self.apply(self._init_esim_weights)

    def _init_esim_weights(self, module):
        """
        Initialise the weights of the ESIM model.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight) # random 
            nn.init.constant_(module.bias.data, 0.0) # 0.0-> 85.45, 0.1-> 85.28
        elif isinstance(module, nn.Conv2d):
            nn.init.uniform_(module.weight.data, -0.1, 0.1) # 81.71
            nn.init.constant_(module.bias.data, 0.0) # 无所谓
        elif isinstance(module, nn.LSTM):
            nn.init.xavier_uniform_(module.weight_ih_l0.data)
            nn.init.orthogonal_(module.weight_hh_l0.data)
            nn.init.constant_(module.bias_ih_l0.data, 0.0)
            nn.init.constant_(module.bias_hh_l0.data, 0.0)
            hidden_size = module.bias_hh_l0.data.shape[0] // 4
            module.bias_hh_l0.data[hidden_size:(2 * hidden_size)] = 1.0

            if (module.bidirectional):
                nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
                nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
                nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
                nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
                module.bias_hh_l0_reverse.data[hidden_size:(2 * hidden_size)] = 1.0
        if isinstance(module, nn.Embedding) and self.config['train_mode'] == 'random':
            nn.init.uniform_(module.weight, -0.1, 0.1) # 81.71

    def forward(self, input_,input_basic): 
        # input: (batch_size, sentence_length)
        # input: (batch_size, sentence_length, emb_dim)
        # print('###$$$')
        # print(input_.shape)
        # print('#####$$$$')
        input_ = self.embeddings(input_).unsqueeze(1)
        # print('###')
        # print(input_.shape)
        # print('#####')
        # batch_size, output_channel, 1
        cnn = [conv(input_) for conv in self.cnn]
        
        hidden_cnn = torch.cat(cnn, 1).squeeze(2).squeeze(2)
        hidden_cnn = self.dropout(hidden_cnn)

        hidden_cnn = self.relu(self.linear(hidden_cnn))

        hidden_basic_1 = self.dropout(self.relu(self.linear_basic_1(input_basic)))
        # hidden_basic_2 = self.linear_basic_2(hidden_basic_1)
        hidden_fused = torch.cat([hidden_cnn,hidden_basic_1],1)

        # out1 = self.para1[0]*hidden_basic_2+self.para1[1]*hidden_cnn
        output = self.linear_final(hidden_fused)
        # output =self.outfinal(output)
        return output

