
import torch.nn as nn


class MLPModule(nn.Module):
    def __init__(self, d_in, d_out, d_hidden, n_layers, activation, dropout=0.0):
        super(MLPModule, self).__init__()
        self.n_layers = n_layers
        if self.n_layers == 1:
            self.linear = nn.Linear(d_in, d_out)
        else:
            self.d_hidden = d_hidden
            self.layer_list = nn.ModuleList()
            self.in_proj = nn.Linear(d_in, self.d_hidden)
            for _ in range(self.n_layers-2):
                self.layer_list.append(nn.Linear(self.d_hidden, self.d_hidden))
            # print(self.d_hidden_feats, d_out_feats)
            self.out_proj = nn.Linear(self.d_hidden, d_out)
            self.act = activation
            self.hidden_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        if self.n_layers == 1:
            return self.linear(x)
        else:
            x = self.act(self.in_proj(x))
            x = self.hidden_dropout(x)
            for i in range(self.n_layers-2):
                x = self.act(self.layer_list[i](x))
                x = self.hidden_dropout(x)
            y = self.out_proj(x)
            return y