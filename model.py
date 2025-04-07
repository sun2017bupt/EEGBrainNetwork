import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
import torch.nn.init as init
from einops import rearrange, repeat
from tqdm import tqdm
from tgmodel import TimeGraph
import math
    
# Start this paper's first model block, different scale conv for different scale catch on EEG Signal
class Conv_layer(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Conv_layer, self).__init__()
        self.conv1 = nn.Conv1d(input_channel, output_channel, kernel_size=3, dilation=1)
        self.conv2 = nn.Conv1d(output_channel,output_channel, kernel_size=3, dilation=2)
        self.conv3 = nn.Conv1d(output_channel, output_channel,kernel_size=3,dilation=4)
        self.conv4 = nn.Conv1d(output_channel, output_channel,kernel_size=3,dilation=8)
        self.pool1 = nn.MaxPool1d(kernel_size=3)
        self.bn1 = nn.BatchNorm1d(16)

        self.dropout = nn.Dropout(0.6)

        self.fc = nn.Linear(25, 32) # 36
        self.fc1 = nn.Linear(16*32,128)

        # after one block
        self.conv5 = nn.Conv1d(1, output_channel, kernel_size=3, dilation=1)
        self.conv6 = nn.Conv1d(16,output_channel, kernel_size=3, dilation=2)
        self.conv7 = nn.Conv1d(16, output_channel,kernel_size=3,dilation=4)
        self.conv8 = nn.Conv1d(16, output_channel,kernel_size=3,dilation=8)
        self.pool2 = nn.MaxPool1d(kernel_size=3)
        self.bn2 = nn.BatchNorm1d(16)

        self.fc2 = nn.Linear(400, 128) # 1728, 576, 16*28
        ## New add
        self.fc3 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.6)

    def forward(self,x):
        # dealing chaning shape during training
        batch = x.shape[0]
        if len(x.shape) == 3:
            # print(f"first conv blk x shape:{x.shape}")
            batch = x.shape[0]
            out1 = F.gelu(self.conv1(x))
            out2 = F.gelu(self.conv2(out1))
            out3 = F.gelu(self.conv3(out2))
            out4 = F.gelu(self.conv4(out3))
            out = torch.cat([out1,out2,out3,out4],dim=-1)
            out = self.pool1(out)
            out = self.fc(out)
            out = out.view(batch, -1)
            out = self.dropout(out)
            out = self.fc1(out)
        else:
            # print(f"other conv blk x shape:{x.shape}")
            x = x.unsqueeze(1)
            out1 = F.gelu(self.conv5(x))
            out2 = F.gelu(self.conv6(out1))
            out3 = F.gelu(self.conv7(out2))
            out4 = F.gelu(self.conv8(out3))
            out = torch.cat([out1,out2,out3,out4],dim=-1)
            out = self.pool2(out)
            out = out.view(batch,-1)
            out = self.dropout(out)
            if out.shape[-1] == 2:
                out = out.flatten()
            out = self.fc2(out)
            out = self.fc3(out)

        return out
    
# for attention mechanism
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model%2:
            pe[:, 1::2] = torch.cos((position * div_term)[:,:-1])
        else:
            pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.shape[0]
        # print(f'x shape:{x.shape}')
        if len(x.shape) == 3:
            pos_encoding = self.pe[:seq_len, :]
        else:
            pos_encoding = self.pe[:seq_len,:].squeeze()

        return x + pos_encoding.to(x.device)
    
# multi head attention with residual connection
class Attention_layer1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Attention_layer1, self).__init__()

        self.fc1 = nn.Linear(input_dim,output_dim,bias=True)

        self.attn = nn.MultiheadAttention(output_dim,8,0.6)
        self.fc2 = nn.Linear(output_dim,output_dim)

        self.add_norm = AddNorm(output_dim,0.6)
        self.dropout = nn.Dropout(0.6)
        self.add_norm1 = AddNorm(output_dim,0.6)

        self.positional = PositionalEncoding(input_dim,5000)
    
    def forward(self, data):
        x_pe = self.positional(data).squeeze()
        x = self.fc1(x_pe)
        attn_out,_ = self.attn(x,x,x)
        attn_out = self.dropout(attn_out)
        out = self.add_norm(x,attn_out)

        return out

# residual connection
class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

# Model block
class BlockMixture(nn.Module):
    def __init__(self,input_dim,output_dim,input_channel,output_channel,input_size,hidden_size):
        super(BlockMixture, self).__init__()
        self.attention1 = Attention_layer1(input_dim, output_dim)
        self.conv1 = Conv_layer(input_channel, output_channel)
        self.fc4 = nn.Linear(32*7, 128)
        self.fc_a = nn.Linear(64,128)
        self.fc6 = nn.Linear(128, 128)

        self.add_norm = AddNorm(128,0.6)

        
        self.lstm1 = nn.RNN(input_size, hidden_size,3)

        self.batch_norm1 = nn.BatchNorm1d(32)

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.fc7 = nn.Linear(hidden_size//2,hidden_size//2)
        self.fc3 = nn.Linear(hidden_size, hidden_size//2)
        self.fc_t = nn.Linear(hidden_size//2, hidden_size//2)
        self.fc8 = nn.Linear(hidden_size//2, hidden_size//2)
        self.fc9 = nn.Linear(hidden_size//2,hidden_size//4)
        self.fc5 = nn.Linear(64,128)

        self.add_norm1 = AddNorm(hidden_size,0.6)
        self.add_norm2 = AddNorm(hidden_size//2,0.6)
        self.add_norm3 = AddNorm(hidden_size//4,0.6)

        self._init_weights()
    
    def _init_weights(self):
        init.xavier_uniform_(self.attention1.fc1.weight)
        init.xavier_uniform_(self.attention1.fc1.weight)

        init.kaiming_uniform_(self.conv1.conv1.weight)
        if self.conv1.conv1.bias is not None:
            init.constant_(self.conv1.conv1.bias, 0.0)

        for name, param in self.lstm1.named_parameters():
            if 'bias' in name:
                init.constant_(param,0.0)
            elif 'weight' in name:
                init.xavier_uniform_(param)
        
        init.xavier_uniform_(self.fc1.weight)
        init.constant_(self.fc1.bias, 0)
        init.xavier_uniform_(self.fc2.weight)
        init.constant_(self.fc2.bias, 0)
        init.xavier_uniform_(self.fc3.weight)
        init.constant_(self.fc3.bias, 0)
        init.xavier_uniform_(self.fc4.weight)
        init.constant_(self.fc4.bias, 0)
        init.xavier_uniform_(self.fc5.weight)
        init.constant_(self.fc5.bias, 0)
        init.xavier_uniform_(self.fc6.weight)
        init.constant_(self.fc6.bias, 0)
        init.xavier_uniform_(self.fc7.weight)
        init.constant_(self.fc7.bias, 0)
        init.xavier_uniform_(self.fc8.weight)
        init.constant_(self.fc8.bias, 0)
        init.xavier_uniform_(self.fc9.weight)
        init.constant_(self.fc9.bias, 0)

    def forward(self,data):
        batch = data.shape[0]
        x = self.attention1(data)
        x = self.conv1(x)
        x1 = data
        if len(x1.shape) == 3:
            x1 = data.reshape(batch,-1)
            x1 = self.fc4(x1)
        elif len(x1.shape) == 2:
            x1 = self.fc_a(x1)
            x = self.fc6(x)

        x = F.gelu(self.add_norm(x1,x))
        x = self.add_norm(x1,x)
        x1 = x1.unsqueeze(1)
        x1,_ = self.lstm1(x1)
        x1 = x1.squeeze()

        x1 = F.gelu(x1 + self.fc1(x1))
        x1 = x1 + self.fc1(x1)
        x1 = self.fc2(x1)
    
        x = torch.cat([x1,x],dim=-1)

        x = self.fc3(x)
        x = F.gelu(x + self.fc8(x))
        x = x + self.fc8(x)
        x = self.fc9(x)
        return x

class MyMixture(nn.Module):
    def __init__(self,input_dim,output_dim,input_channel,output_channel,input_size,hidden_size,in_channels, out_channels,K):
        super(MyMixture, self).__init__()

        # feature process
        self.block_mixture1 = BlockMixture(input_dim,output_dim,input_channel,output_channel,input_size,hidden_size)
        self.block_mixture2 = BlockMixture(hidden_size//4,output_dim,input_channel,output_channel,input_size,hidden_size)
        self.block_mixture3 = BlockMixture(hidden_size//4,output_dim,input_channel,output_channel,input_size,hidden_size)


        # time graph process
        self.tg = TimeGraph(in_channels, out_channels,K)
        self.fc = nn.Linear(32*32,32*16)
        # self.fc6 = nn.Linear(32*7,8)
        # self.fc = nn.Linear(32*32,8)
        self.fc_e = nn.Linear(32*16, 32*8)
        # only tg
        # self.fc_e = nn.Linear(32*2, 2)
        # self.fc_e = nn.Linear(32, 1)
        # self.fc_11 = nn.Linear(2,2)
        # self.fc_e = nn.Linear(32*16, 32*8)
        self.fc_a = nn.Linear(32*8, 32*2)
        self.fc4 = nn.Linear(64,64)
        self.fc5 = nn.Linear(64,128)

        # output linear
        self.fc1 = nn.Linear(hidden_size//2,hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2,hidden_size//4)
        # wtg linear
        # self.fc1 = nn.Linear(hidden_size//4,hidden_size//4)
        # self.fc2 = nn.Linear(hidden_size//4,hidden_size//4)
        # ntg linear
        # self.fc1 = nn.Linear(hidden_size//4,hidden_size//4)
        # self.fc2 = nn.Linear(hidden_size//4,hidden_size//4)
        # classifical bands

        self.fc3 = nn.Linear(hidden_size//4,2)
        self.fc3 = nn.Linear(hidden_size//4,2)
        self.dropout = nn.Dropout(0.6)

        self.bn = nn.BatchNorm1d(512)

        # self._init_weights()
    
    def _init_weights(self):
        init.xavier_uniform_(self.fc1.weight)
        init.constant_(self.fc1.bias, 0)
        init.xavier_uniform_(self.fc2.weight)
        init.constant_(self.fc2.bias, 0)
        init.xavier_uniform_(self.fc3.weight)
        init.constant_(self.fc3.bias, 0)
        init.xavier_uniform_(self.fc4.weight)
        init.constant_(self.fc4.bias, 0)

        init.xavier_uniform_(self.fc.weight)
        init.constant_(self.fc.bias, 0)

        init.xavier_uniform_(self.fc_e.weight)
        init.constant_(self.fc_e.bias, 0)

        init.xavier_uniform_(self.fc_a.weight)
        init.constant_(self.fc_a.bias, 0)

        init.xavier_uniform_(self.fc_11.weight)
        init.constant_(self.fc_11.bias, 0)

    
    def forward(self, base_data,harm_data, graph,step):
        n = base_data.size(0)
        x = base_data
        
        # Stack encoding part
        x = self.block_mixture1(x)
        x = self.block_mixture2(x)
        x = self.block_mixture3(x)

        # independ time graph part
        tg_out = self.tg(harm_data,graph,step)

        if len(tg_out.shape) == 2:
            tg_out = tg_out.flatten()
            x = x.flatten()
        else:
            tg_out = tg_out.view(n,-1)

        tg_out = self.bn(self.fc(tg_out))
        # print(f"tg out:{tg_out.shape}")
        tg_out = self.fc_e(tg_out)
        tg_out = F.gelu(self.fc_a(tg_out))

        x = x + F.gelu(self.fc4(x))
        # if not self.training:
        #     print("Model is in eval mode, saving variable.")
        #     torch.save(x,'/home/sjf/eegall/nolimits/FACED/valence_spatialtemporal.pt')
        #     print("The target variable is successfully saved.")


        # combine time graph and encoding
        all_out = torch.cat([tg_out, x],dim=-1)
        # print(f"allout shape:{all_out.shape}")
        all_out = self.fc2(all_out + self.fc1(all_out))
        # if not self.training:
        #     print("Model is in eval mode, saving variable.")
        #     torch.save(all_out,'/home/sjf/eegall/nolimits/FACED/valence_combinedall.pt')
        #     print("The target variable is successfully saved.")
        all_out = self.fc3(all_out)

        return all_out
    
    # # only tg
    # def forward(self, base_data,harm_data, graph,step):
    #     n = base_data.size(0)
    #     x = base_data

    #     # # independ time graph part
    #     tg_out = self.tg(harm_data,graph,step)
    #     tg_out = tg_out.view(n,-1)
    #     all_out = self.fc_e(tg_out)
    #     all_out = all_out + self.fc_11(all_out)
    #     # all_out = F.dropout(all_out,p=0.6)
    #     # print(f"all out shape:{tg_out.shape}")
    #     # all_out = all_out.view(n,-1)
    #     # print(f"tg out shape:{tg_out.shape}")
    #     # all_out = self.fc_11(all_out)
        
    #     return all_out
    # ## no tg
    # def forward(self, base_data,harm_data, graph,step):
    #     n = base_data.size(0)
    #     x = base_data

    #     # Stack encoding part
    #     x = self.block_mixture1(x)
    #     x = self.block_mixture2(x)
    #     x = self.block_mixture3(x)
    #     x = self.dropout(x + F.gelu(self.fc4(x)))

    #     # print(f"x shape:{x.shape}")
    #     all_out = self.fc2(x + self.fc1(x))
    #     all_out = self.fc3(all_out)

    #     return all_out