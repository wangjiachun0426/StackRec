import torch
import torch.nn as nn
import argparse
from utils import normalize
from torch.nn.parameter import Parameter

from Modules import *
class TransformerLayer_Alpha(nn.Module):
    def __init__(self, hidden_size, num_heads, model_para, reader, layerid, layer_num, dropout_rate=0.5):
        super(TransformerLayer_Alpha, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.SelfAttention = multihead_attention_alpha(model_para, reader, layerid, layer_num, hidden_size, num_units=self.hidden_size,
                                                 num_heads=self.num_heads, dropout_rate=self.dropout_rate,
                                                 causality=True, with_qk=False)
        self.ff = feedforward_alpha(model_para, reader, layerid, layer_num, num_units=[self.hidden_size, self.hidden_size], dropout_rate=self.dropout_rate)

    def forward(self, input):
        x = self.SelfAttention(queries=input, keys=input)
        out = self.ff(x)
        return out


class SASRec_Alpha(nn.Module):
    def __init__(self, model_para, device='gpu'):
        super(SASRec_Alpha, self).__init__()
        self.model_para = model_para
        self.load_model = model_para['load_model']
        self.method = model_para['method']

        self.hidden_size = model_para['hidden_factor']
        self.item_num = int(model_para['item_size'])
        self.max_len = model_para['seq_len']
        self.device = torch.device(device)
        self.num_blocks = model_para['num_blocks']
        self.num_heads = model_para['num_heads']
        self.dropout_rate = model_para['dropout']

        self.item_embeddings = nn.Embedding(
            num_embeddings=self.item_num,
            embedding_dim=self.hidden_size,
        )
        self.pos_embeddings = nn.Embedding(
            num_embeddings=self.max_len,
            embedding_dim=self.hidden_size,
        )
        
        self.reader = None
        if self.load_model:
            self.model_path = model_para['model_path']
            self.reader = torch.load(self.model_path)
            self.item_embeddings.weight = Parameter(self.reader['item_embeddings.weight'])
            self.pos_embeddings.weight = Parameter(self.reader['pos_embeddings.weight'])
            print("load item_embeddings.weight")
            print("load pos_embeddings.weight")
        else:    
            # init embedding
            nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
            nn.init.normal_(self.pos_embeddings.weight, 0, 0.01)

        rb = [TransformerLayer_Alpha(self.hidden_size, self.num_heads, self.model_para, self.reader, layerid, self.num_blocks, dropout_rate=self.dropout_rate) for layerid in range(self.num_blocks)]

        self.transformers = nn.Sequential(*rb)

        #dropout
        self.dropout = nn.Dropout(self.dropout_rate)

        #layer norm
        self.layer_norm_pre = nn.LayerNorm(self.hidden_size)


        #softmax Layer
        self.final = nn.Linear(self.hidden_size, self.item_num)
        #
        # if self.load_model:
        #     self.final.weight = Parameter(self.reader['final.weight'])
        #     self.final.bias = Parameter(self.reader['final.bias'])
        #     print("load final.weight")
        #     print("load final.bias")
        
    def forward(self, inputs, onecall=True):
        input_emb = self.item_embeddings(inputs)
        pos_emb_input = torch.cat(inputs.size(0)*[torch.arange(start=0,end=inputs.size(1)).to(self.device).unsqueeze(0)])
        pos_emb_input = pos_emb_input.long()
        pos_emb = self.pos_embeddings(pos_emb_input)
        x = input_emb + pos_emb

        x = self.dropout(x)

        x = self.layer_norm_pre(x)

        x = self.transformers(x)

        if onecall:
            x = x[:, -1, :].view(-1, self.hidden_size) # [batch_size, hidden_size]
        else:
            x = x.view(-1, self.hidden_size) # [batch_size*seq_len, hidden_size]

        out = self.final(x)
        return out
