from math import sqrt
import numpy as np
import sys
import torch
from scipy.signal import stft
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.data import Batch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.init as init
from einops import rearrange, repeat
from tqdm import tqdm
# Image and Real Convolution part
import torch
import torch.nn as nn

class ComplexConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv, self).__init__()
        # 分别定义实部和虚部的卷积层
        self.real_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.imag_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

    def forward(self, real_input, imag_input):
        # 对实部和虚部分别进行卷积操作
        real_real = self.real_conv(real_input)
        imag_imag = self.imag_conv(imag_input)
        real_imag = self.real_conv(imag_input)
        imag_real = self.imag_conv(real_input)
        
        # 组合卷积结果得到新的实部和虚部
        real_output = real_real - imag_imag
        imag_output = real_imag + imag_real
        
        return real_output, imag_output

class ComplexReLU(nn.Module):
    def __init__(self):
        super(ComplexReLU, self).__init__()
        # self.relu = nn.LeakyReLU()
        self.relu = nn.LeakyReLU()
        # nn.GELU()
    
    def forward(self, real, imag):
        return self.relu(real), self.relu(imag)

class ComplexBatchNorm1d(nn.Module):
    def __init__(self, num_features):
        super(ComplexBatchNorm1d, self).__init__()
        self.real_bn = nn.BatchNorm1d(num_features)
        self.imag_bn = nn.BatchNorm1d(num_features)
    
    def forward(self, real, imag):
        return self.real_bn(real), self.imag_bn(imag)

class ComplexConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ComplexConvBlock, self).__init__()
        self.conv = ComplexConv(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = ComplexBatchNorm1d(out_channels)
        self.relu = ComplexReLU()
    
    def forward(self, real, imag):
        real, imag = self.conv(real, imag)
        real, imag = self.bn(real, imag)
        real, imag = self.relu(real, imag)
        return real, imag

class ComplexModel(nn.Module):
    def __init__(self):
        super(ComplexModel,self).__init__()
        self.block1 = ComplexConvBlock(33, 32, kernel_size=3, padding=1)
        self.block2 = ComplexConvBlock(32, 32, kernel_size=3, padding=1)
        self.block3 = ComplexConvBlock(32, 32, kernel_size=3, padding=1)

        self.fc_real = nn.Linear(33, 32)
        self.fc_imag = nn.Linear(33, 32)
        self.fc_connect = nn.Linear(64, 32)

        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias,0)
        init.xavier_uniform_(self.fc_real.weight)
        init.constant_(self.fc_real.bias, 0)
        init.xavier_uniform_(self.fc_imag.weight)
        init.constant_(self.fc_imag.bias, 0)

        init.xavier_uniform_(self.fc_connect.weight)
        init.constant_(self.fc_connect.bias, 0)

    def forward(self, real, imag):
        real, imag = self.block1(real, imag)
        # print(f"real shape:{real.shape}, imag shape:{imag.shape}")
        real, imag = self.block2(real, imag)
        real, imag = self.block3(real, imag)
        # print(f"real shape:{real.shape}, imag shape:{imag.shape}")
        real_output = self.fc_real(real)
        imag_output = self.fc_imag(imag)
        # output = real_output + imag_output
        output = torch.cat([real_output, imag_output], dim=-1)
        output = self.fc_connect(output)
        # print(f"output shape:{output.shape}")
        return output

# Different scale conv part
class NBeatsBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, dilation_base=2):
        super(NBeatsBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_size, output_size, kernel_size, dilation=1)
        self.conv2 = nn.Conv1d(output_size, output_size, kernel_size, dilation=dilation_base)
        self.conv3 = nn.Conv1d(output_size, output_size, kernel_size, dilation=dilation_base ** 2)
        self.conv4 = nn.Conv1d(output_size, output_size, kernel_size, dilation=dilation_base ** 3)
        self.fc = nn.Linear(4 * 32, 32)
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = x+self.relu(self.conv1(x))
        out2 = self.relu(self.conv2(out1))
        out3 = self.relu(self.conv3(out2))
        out4 = self.relu(self.conv4(out3))
        out = torch.cat([out1, out2, out3, out4], dim=-1)
        # print(f"out shape:{out.shape}")
        out = self.fc(out)
        return out

class NBeatsModel(nn.Module):
    def __init__(self, input_size, output_size, blocks, kernel_size, dilation_base=2):
        super(NBeatsModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.blocks = nn.ModuleList([NBeatsBlock(input_size, output_size, kernel_size, dilation_base) for _ in range(blocks)])
        self.fc = nn.Linear(output_size,16)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            x = x.permute(0,2,1)
            for block in self.blocks:
                x = x+block(x)
            x = x.permute(0,2,1).squeeze()
        else:
            x = x.permute(0,2,1)  # 将输入形状转换为 (feature, batch, seq_len)
            for block in self.blocks:
                x = x + block(x)
            x = x.permute(0,2,1)  # 将输出形状转换为 (batch, seq_len, feature)
        x = self.fc(x)
        return x

# handling the time series graphs
class TimeGraph(nn.Module):
    def __init__(self,in_channels, out_channels,K,bias=True):
        super(TimeGraph, self).__init__()

        self.weight = nn.Parameter(torch.FloatTensor(in_channels*5, out_channels))
        self.nb = NBeatsModel(32,32,1,1,2)
        self.cmc = ComplexModel()
        self.linear = nn.Linear(32,32)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        else:
            self.register_parameter('bias',None)

        self.ln1 = nn.Linear(40*16,64)
        # self.bn = nn.BatchNorm1d(32)
        # self.reset_parameter()
    
    def reset_parameter(self):
        nn.init.xavier_uniform_(self.weight)
        init.xavier_uniform_(self.linear.weight)
        init.xavier_uniform_(self.ln1.weight)
        init.constant_(self.linear.bias, 0)
        init.constant_(self.ln1.bias, 0)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def get_laplacian(self, adj, normalize):
        """
        the graph without self loop
        """
        # sym matrix
        graph = 0.5*(adj + adj.T)
        
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1/2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L

    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian
        """
        N = laplacian.size(0)
        laplacian = laplacian.unsqueeze(0)
        first_laplacian = torch.zeros([1,N,N],device=laplacian.device, dtype=torch.float)
        second_laplacian = laplacian
        third_laplacian = ((2 * torch.matmul(laplacian, second_laplacian)) - first_laplacian)
        forth_laplacian = (2 * torch.matmul(laplacian, third_laplacian) - second_laplacian)
        fifth_laplacian = (2 * torch.matmul(laplacian, forth_laplacian) - third_laplacian)
        multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian,fifth_laplacian], dim=0)
        # multi_order_laplacian = first_laplacian+second_laplacian+third_laplacian+forth_laplacian
        return multi_order_laplacian
    
    def forward(self, x, graph, step):
        if len(x.shape) == 2:
            # print(f'valadation x shape:{x.shape}')
            num_nodes = x.shape[0]
            laplacian = self.get_laplacian(graph,normalize=True)
            mul_lap = self.cheb_polynomial(laplacian)
            # graph convolution by chebv polynomial
            x_conv = torch.matmul(mul_lap,x)
            x_conv = x_conv.view(num_nodes,-1)
            output = 1. * torch.mm(x_conv,self.weight)
            # Fou = self.create_fourier_matrix(1).type(torch.float)
            output = self.nb(output)
        else:
            time_length, num_nodes, _ = x.size()
            # print(f"in training x shape:{x.shape}")
            all_out = []
            for i in range(time_length):
                laplacian = self.get_laplacian(graph[i],normalize=True)
                mul_lap = self.cheb_polynomial(laplacian)
                x_conv = torch.matmul(mul_lap,x[i])
                x_conv = x_conv.view(num_nodes,-1)
                output = torch.mm(x_conv,self.weight)
                all_out.append(output)
            
            all_out = torch.cat(all_out,dim=0)
            # print(f"all out shape:{all_out.shape}")
            freq_out = all_out.reshape(time_length,num_nodes,-1)
            # print(f"freq out shape:{freq_out.shape}")
            # freq_out = self.nb(freq_out)
            # print(f"freq out shape:{freq_out.shape}")
            # if (step+1) % 100 == 0:
            #     filename = str(step) + 'timefourier.pt'
            #     filepath = '/home/sjf/eegall/dataex/'+filename
            #     torch.save(freq_out.detach().cpu(),filepath)                
            #     print("success save target variable.")
            #     sys.exit('Program interrupted after saving the target variable.')
            
            # freq_out = all_out.reshape(time_length,-1)
            # # # print(f"freq out shape:{freq_out.shape}")
            # window = torch.hann_window(64).to(freq_out.device)
            # freq_out1 = torch.stft(freq_out,n_fft=64,hop_length=32,win_length=64, window=window, return_complex=True)
            
            # print(f"freq out shape:{freq_out1.shape}")
            # freq_out1 = freq_out1[:,:32,:32]
            # print(f'freq out shape:{freq_out.shape}')
            # freq_out1 = torch.fft.fft(freq_out, dim=0, norm='ortho')
            # print(f"freqout1 shape:{freq_out1.shape}")
            # freq_out = freq_out.reshape(time_length,-1)
            # output = torch.cat((freq_out1.real, freq_out1.imag),dim=-1)
            
            # output = self.cmc(freq_out1.real, freq_out1.imag)
            # print(f"output shape:{output.shape}")
            # if (step+1) % 100 == 0:
            #     filename = str(step) + 'deapcmctimefourier.pt'
            #     filepath = '/home/sjf/eegall/dataex/'+filename
            #     torch.save(output.detach().cpu(),filepath)                
            #     print("success save target variable.")
            #     sys.exit('Program interrupted after saving the target variable.')
            # print(f"freq out value:{freq_out}")
            # freq_out = torch.complex(freq_out,torch.zeros_like(freq_out))
            # print(f"freq_out shape:{freq_out.shape} Fou shape:{Fou.shape}")
            # Fou = Fou.to(freq_out.device)
            # # print(f"freqout device:{freq_out.device} Foudevice:{Fou.device}")
            # all_out = torch.matmul(Fou, freq_out)
            # # print(f"all_out imag{type(all_out.imag)}")
            # all_real = all_out.real
            # all_img = all_out.imag.float()
            # # print(f"all_real type:{type(all_real)}, all_img type:{type(all_img)}")
            # output = torch.cat((all_real, all_img),dim=-1)
            # print(f"output shape:{output.shape}")
            # output = output.reshape(time_length,-1)
            # output = freq_out + self.linear(output)
            # print(f"output shape:{output.shape}")
            # output = output.view(time_length,num_nodes,-1)
            # print(f"output shape:{output.shape}")
            # # print(f"all_out shape{all_out.shape}")
            # output = self.nb(output)
            # r_output,i_output = self.cmc(output[:,:,:32],output[:,:,32:])
            # output = torch.cat((r_output, i_output),dim=-1)
            # print(f"output shape:{output.shape}")
            # output = output + self.linear(output)

            # output = output.view(time_length,num_nodes,-1).squeeze()
            # if (step+1) % 100 == 0:
            #     print("Model is in eval mode, saving variable.")
            #     torch.save(output,'/home/sjf/eegall/nolimits/FACED/valence_explainoutput.pt')
            #     print("The target variable is successfully saved.")

        return freq_out
