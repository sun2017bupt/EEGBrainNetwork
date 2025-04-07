import _pickle as cPickle
import numpy as np
import torch
import os
from scipy.signal import stft
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.data import Batch
import torch.nn.functional as F
import pickle

# Function to load data from each participant file
def read_eeg_signal_from_file(filename):
    x = pickle._Unpickler(open(filename, 'rb'))
    x.encoding = 'latin1'
    p = x.load()
    return p

def data_calibrate(data):
    """
    数据标定
    :param data: 原始数据
    :return: 标定后的数据
    """
    fs = 128
    baseline_time = 3  # 基线数据的时间长度
    # 将 3s 基线时间与 60s 数据分开
    baseline_data, normal_data = np.split(data, [baseline_time * fs], axis=-1)
    # 将基线数据重复 20 次，补成 60s
    baseline_data = np.concatenate([baseline_data] * 20, axis=-1)
    # 用 60s 数据减去基线数据，去除噪声
    return normal_data
    # return normal_data - baseline_data

def set_label(labels):
    """
    打标签
    :param labels: 标签
    :return: 处理后的标签
    """
    return torch.tensor(np.where(labels < 5, 0, 1), dtype=torch.long)  # 小于 5 的元素改为 0，大于等于 5 的改为 1

def data_divide(data, label):
    """
    数据分割
    :param data: 标定后的数据
    :param label: 标签
    :return: 分割后的数据和标签
    """
    fs = 128
    window_size = 3  # 窗口大小
    step = 2  # 窗口滑动的步长
    num = (60 - window_size) // step + 1  # 分割成的段数

    divided_data = []
    for i in range(0, num * step, step):
        segment = data[:, :, i * fs: (i + window_size) * fs]
        divided_data.append(segment)
    divided_data = np.vstack(divided_data)

    divided_label = np.vstack([label] * num)

    return divided_data, divided_label

def calculate_de(psd):
    variance = np.var(psd, axis=-2, ddof=1) + 1e-5
    de = 0.5 * np.log(2 * np.pi * np.e * variance)
    return de

def base_homo_select(eeg_data,sample_rate,num_exp, num_channel):
    signal = eeg_data
    fs = sample_rate
    order = 8
    f, t, zxx = stft(signal, fs=128, window='hann', nperseg=64, noverlap=0, nfft=256, scaling='psd')
    power = np.power(np.abs(zxx),2)
    base_freq_idx = np.abs(power).argmax(axis=2)
    base_freq = np.mean(f[base_freq_idx], axis=-1)
    length = base_freq.shape[0]
    # 找到谐波
    harm_freq = np.empty((length, num_channel, order))
    for i in range(length):
        for j in range(num_channel):
            base_f = base_freq[i][j]
            for k in range(8):
                harmonic_f = base_f*(k+2)
                harmonic_idx = np.argmin(np.abs(f-harmonic_f))
                harm_freq[i,j,k] = f[harmonic_idx]
    
    # Base Freq
    print(f"Base freq for every channel and every second: {base_freq.shape}")
    # Harmonic Freq
    print(f"Harmonic freq for every second and every channel, and with 2-9 order harmonic freq: {harm_freq.shape}")
    return base_freq, f, harm_freq, zxx

def feature_extract(base_freq, f, harm_freq, zxx):
    # Total withoud distinguish base freq and hramonic freq
    power = np.power(np.abs(zxx),2)
    channel_num = base_freq.shape[-1]
    alpha = 1e-5

    # Find the range of base freq for every channel
    base_flow_list = []
    base_fhigh_list = []
    for i in range(channel_num):
        std = np.std(base_freq[:,i])
        mean = np.mean(base_freq[:,i])
        fhigh = int(mean+std)
        flow = int(mean-std)
        base_flow_list.append(flow)
        base_fhigh_list.append(fhigh)

    # sepfreq = np.max(base_fhigh_list)
    # print(f"Channel:{i} Sep freq: {sepfreq}")
    harm_flow_list = []
    harm_fhigh_list = []
    harm_freq = np.mean(harm_freq, axis=-1)
    for i in range(channel_num):
        std = np.std(harm_freq[:,i])
        mean = np.mean(harm_freq[:,i])
        fhigh = int(mean + std)
        flow = int(mean - std)
        harm_flow_list.append(flow)
        harm_fhigh_list.append(fhigh)
    # Base
    base_de_features = []
    harmon_de_features = []
    for i in range(channel_num):
        base_flow = base_flow_list[i]
        base_fhigh = base_fhigh_list[i]
        if base_flow < 0.5:
            base_flow = 0.5
        if base_fhigh < base_flow or np.abs(base_fhigh - base_flow)<2:
            base_fhigh = base_flow + 4
        if base_fhigh > max(f):
            base_fhigh = max(f) // 2
        
        index1 = np.where(f == base_flow)[0][0]
        index2 = np.where(f == base_fhigh)[0][0]
        # if index1 > index2:
        #     print(f"Channel:{i} index1:{index1}, index2:{index2}, base_fhigh:{base_fhigh},base_flow:{base_flow}")
        #     break
        psd = power[:,i,index1:index2,:]+alpha
        base_de = calculate_de(psd)
        base_de_features.append(base_de)
        print(f"Base: Channel:{i} freq: max:{base_fhigh}min:{base_flow}")

        ### Harm part
        harm_flow = harm_flow_list[i]
        harm_fhigh = harm_fhigh_list[i]
        
        if harm_flow < base_fhigh or harm_flow == 0:
            harm_flow = base_fhigh
            if harm_fhigh < harm_flow:
                harm_fhigh = harm_flow + 4
        if np.abs(harm_fhigh-harm_flow) < 2:
            harm_fhigh = harm_flow + 5
        if harm_fhigh > max(f):
            harm_fhigh = max(f)
        
        index1 = np.where(f == harm_flow)[0][0]
        index2 = np.where(f == harm_fhigh)[0][0]
        psd = power[:,i,index1:index2,:]+alpha
        harm_de = calculate_de(psd)
        harmon_de_features.append(harm_de)
        print(f"Harmon: Channel:{i} freq: max:{harm_fhigh} min:{harm_flow}")
    base_de_features = np.array(base_de_features)
    base_de_features = np.transpose(base_de_features, (1,0,2))
    # print(f"base_de :{base_de_features.shape}")
    harmon_de_features = np.array(harmon_de_features)
    # print(f"harm_de : {harmon_de_features.shape}")
    harmon_de_features = np.transpose(harmon_de_features,(1,0,2))

    print(f"Created Base And Harmon Features with shape: {base_de_features.shape} BASE:max:{np.max(base_freq)}min:{np.min(base_freq)} Harm:max:{np.max(harm_freq)}min:{np.min(harm_freq)}")
    return base_de_features, harmon_de_features

def data_process(data,sample_rate,exp_num,channel_num,time,labels):
    data = data_calibrate(data)
    # print(data.shape)
    data, labels = data_divide(data, labels)
    # print(f"labels shape:{labels}")
    base_freq, f, harm_freq, zxx = base_homo_select(data,sample_rate,exp_num, channel_num)
    base_de_features, harmon_de_features = feature_extract(base_freq, f, harm_freq, zxx)
    labels = set_label(labels)
    # print(f"labels:{labels}")
    return base_de_features, harmon_de_features, labels

# X (32, 760, 40, 7) ; 32主体，每个主体有760(19*40)个样本，40个通道，7s构成向量
def phase_sync(de_features):
    n_channels, n_samples = de_features.shape
    phase_sync_matrix = np.zeros((n_channels, n_channels))
    alpha = 1e-5
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            phase_diff = np.angle(np.exp(1j * (de_features[i] - de_features[j])))
            imag = np.abs(np.mean(np.exp(1j * phase_diff)).imag)
            # 计算相位相似度
            phase_sync_matrix[i, j] = np.log(imag+alpha) / np.log(0.5)
    return phase_sync_matrix

# get the graph adjacency matrix by phase sync for base and harm signal
def phase_graph(base_features, harm_features):
    num_subject, num_sample, num_channel, seconds = base_features.shape
    base_graph = np.empty((num_subject,num_sample,num_channel,num_channel))
    harm_graph = np.empty((num_subject,num_sample,num_channel,num_channel))
    for i in range(num_subject):
        for j in range(num_sample):
            base_psy = phase_sync(base_features[i][j])
            harm_psy = phase_sync(harm_features[i][j])
            base_graph[i][j] = base_psy
            harm_graph[i][j] = harm_psy
    return base_graph, harm_graph

if __name__ == '__main__':
    sample_rate = 128
    exp_num = 40
    channel_num = 40
    time = 60

    raw_dir = "/DEAP/data_preprocessed_python/data_preprocessed_python/raw/"
    file_names = os.listdir(raw_dir)
    all_base_de_features = []
    all_harmon_de_features = []
    all_labels = []
    for filename in file_names:
        filepath = "/DEAP/data_preprocessed_python/data_preprocessed_python/raw/"+str(filename)
        trial = read_eeg_signal_from_file(filepath)
        data = trial['data']
        labels = trial['labels']
        print(f"******* Processing on file {filename} ********")
        base_de_features, harmon_de_features, labels = data_process(data,sample_rate,exp_num,channel_num,time,labels)
        
        all_base_de_features.append(base_de_features)
        all_harmon_de_features.append(harmon_de_features)
        all_labels.append(labels)
        # 做到这步之后，保存三个all的变量就行了


    all_base_de_features = torch.tensor(all_base_de_features)
    all_harmon_de_features = torch.tensor(all_harmon_de_features)
    all_labels=torch.stack(all_labels)

    torch.save(all_base_de_features,'/eegall/data/DEAP/all_ori13base_de_features.pt')
    torch.save(all_harmon_de_features,'eegall/data/DEAP/all_ori13harmon_de_features.pt')
    torch.save(all_labels,'/eegall/data/DEAP/all_ori13labels.pt')

    base_graph, harm_graph = phase_graph(all_base_de_features, all_harmon_de_features)
    torch.save(base_graph,'/eegall/data/DEAP/ori13base_graph.pt')
    torch.save(harm_graph,'/eegall/data/DEAP/ori13harm_graph.pt')
