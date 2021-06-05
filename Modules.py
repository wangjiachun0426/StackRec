import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import normalize
from torch.nn.parameter import Parameter
class multihead_attention_alpha(nn.Module):
    def __init__(self, model_para, reader, layer_id, layer_num, hidden_size, num_units=None, num_heads=8, dropout_rate=0, causality=True,
                 with_qk=False):
        super(multihead_attention_alpha, self).__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        self.with_qk = with_qk
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.hidden_size, num_units)
        self.fc2 = nn.Linear(self.hidden_size, num_units)
        self.fc3 = nn.Linear(self.hidden_size, num_units)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = nn.LayerNorm(num_units)
        self.rez = nn.Parameter(torch.zeros(1))

        self.method = model_para['method']

        if model_para["load_model"]:
            if model_para['method'] == 'stackC':
                if layer_id >= layer_num - 6:
                    relative_layer_id = layer_id - 6
                else:
                    relative_layer_id = layer_id
            elif model_para['method'] == 'stackA':
                if layer_id >= layer_num - 6:
                    relative_layer_id = int((layer_id - 6) // 2 + 6)
                else:
                    relative_layer_id = layer_id
            else:
                print("method is wrong!!!!!!!!!!!!!!")
            relative_layer_id = str(relative_layer_id)


            
            initial_name = "transformers." + relative_layer_id + ".SelfAttention.fc1.weight"
            self.fc1.weight = Parameter(reader[initial_name])
            initial_name = "transformers." + relative_layer_id + ".SelfAttention.fc1.bias"
            self.fc1.bias = Parameter(reader[initial_name])
            print("load selfattention fc1 weight", layer_id, "   from   ", relative_layer_id)
            print("load selfattention fc1 bias", layer_id, "   from   ", relative_layer_id)

            initial_name = "transformers." + relative_layer_id + ".SelfAttention.fc2.weight"
            self.fc2.weight = Parameter(reader[initial_name])
            initial_name = "transformers." + relative_layer_id + ".SelfAttention.fc2.bias"
            self.fc2.bias = Parameter(reader[initial_name])
            print("load selfattention fc2 weight", layer_id, "   from   ", relative_layer_id)
            print("load selfattention fc2 bias", layer_id, "   from   ", relative_layer_id)
            
            initial_name = "transformers." + relative_layer_id + ".SelfAttention.fc3.weight"
            self.fc3.weight = Parameter(reader[initial_name])
            initial_name = "transformers." + relative_layer_id + ".SelfAttention.fc3.bias"
            self.fc3.bias = Parameter(reader[initial_name])
            print("load selfattention fc3 weight", layer_id, "   from   ", relative_layer_id)
            print("load selfattention fc3 bias", layer_id, "   from   ", relative_layer_id)
            

    def forward(self, queries, keys):
        if self.num_units is None:
            self.num_units = queries.size(-1)
        # Linear projections

        Q = self.fc1(queries)  # (N, T_q, C)
        K = self.fc2(keys)  # (N, T_k, C)
        V = self.fc3(keys)  # (N, T_k, C)

        # Split and concat
        q_split = int(Q.size(2) / self.num_heads)
        k_split = int(K.size(2) / self.num_heads)
        v_split = int(V.size(2) / self.num_heads)
        Q_ = torch.cat(torch.split(Q, q_split, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.split(K, k_split, dim=2), dim=0)  # (h*N, T_k, C/h)
        V_ = torch.cat(torch.split(V, v_split, dim=2), dim=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = torch.matmul(Q_, K_.permute(0, 2, 1))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.size(-1) ** 0.5)

        # Key Masking
        #key_masks = torch.sign(torch.abs(torch.sum(keys, -1)))  # (N, T_k)
        #key_masks = torch.cat(self.num_heads * [key_masks])  # (h*N, T_k)
        #key_masks = torch.cat(queries.size(1) * [key_masks.unsqueeze(1)], dim=1)  # (h*N, T_q, T_k)

        #paddings = torch.ones_like(outputs) * (-2 ** 32 + 1)
        #outputs = torch.where(torch.eq(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if self.causality:
            diag_vals = torch.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = torch.tril(diag_vals)  # (T_q, T_k)
            masks = torch.cat(outputs.size(0) * [tril.unsqueeze(0)])  # (h*N, T_q, T_k)

            paddings = torch.ones_like(masks) * (-2 ** 32 + 1)
            outputs = torch.where(torch.eq(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = self.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
#        query_masks = torch.sign(torch.abs(torch.sum(queries,-1))) # (N, T_q)
#        query_masks = torch.cat(self.num_heads*[query_masks]) # (h*N, T_q)
#        query_masks = torch.cat(keys.size(1)*[query_masks.unsqueeze(-1)], dim=2) # (h*N, T_q, T_k)
#        outputs *= query_masks # broadcasting. (N, T_q, C)

        # Dropouts

        outputs = self.dropout(outputs)

        # Weighted sum
        outputs = torch.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        o_split = int(outputs.size(0) / self.num_heads)
        outputs = torch.cat(torch.split(outputs, o_split, dim=0), dim=2)  # (N, T_q, C)

        # Residual connection
        outputs = queries + outputs * self.rez

        # Normalize
        outputs = self.layer_norm(outputs)  # (N, T_q, C)

        if self.with_qk:
            return Q, K
        else:
            return outputs


class feedforward_alpha(nn.Module):

    def __init__(self, model_para, reader, layer_id, layer_num, num_units, dropout_rate=0.5):
        super(feedforward_alpha, self).__init__()
        self.inner_cnn = nn.Conv1d(num_units[0], num_units[0], 1)
        self.readout_cnn = nn.Conv1d(num_units[0], num_units[1], 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(num_units[1])
        self.rez = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        residual = inputs
        x = inputs.transpose(1, 2)  # [N, C, T_q]
        x = F.relu(self.inner_cnn(x))
        x = self.dropout(x)
        x = self.readout_cnn(x)
        x = x.transpose(1, 2)  # [N, C, T_q]
        x = self.dropout(x)
        x = residual + x * self.rez
        outputs = self.layer_norm(x)
        return outputs
