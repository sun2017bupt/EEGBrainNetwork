import _pickle as cPickle
import numpy as np
import torch
import os
import scipy.io
from scipy.signal import stft
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.data import Batch
import torch.nn.functional as F
import pickle
from scipy.signal import resample
# from model import *

def resample_data(data, original_rate, target_rate, duration):
    # data = np.array(data)
    num_channels, original_samples = data.shape
    target_samples = int(target_rate * duration)
    resampled_data = np.zeros((num_channels, target_samples))
    
    for i in range(num_channels):
        resampled_data[i] = resample(data[i], target_samples)
    
    return resampled_data

def read_eeg_signal_from_file(filename):
    x = pickle._Unpickler(open(filename, 'rb'))
    x.encoding = 'latin1'
    p = x.load()
    return p

def data_divide(data, label):
    fs = 128
    window_size = 6
    step = 2
    num = (280 - window_size) // step + 1
    labels = list(label)

    divided_data = []
    for i in range(0, num * step, step):
        segment = data[:, :, i * fs: (i + window_size) * fs]
        divided_data.append(segment)
    divided_data = np.vstack(divided_data)
    labels = labels*num
    labels = np.array(labels)
    return divided_data, labels

def data_normalize(data):
    mean = np.mean(data, axis=2, keepdims=True)
    std = np.std(data, axis=2, keepdims=True)
    nomarlized_data = (data - mean) / std
    return nomarlized_data

def data_calibrate(data):
    fs = 128
    baseline_time = 3
    baseline_data, normal_data = np.split(data, [baseline_time * fs], axis=-1)
    baseline_data = np.concatenate([baseline_data] * 9, axis=-1)
    return normal_data - baseline_data

def calculate_de(psd):
    # power = np.power(np.abs(zxx),2)
    # variance for frequencies
    variance = np.var(psd, axis=-2, ddof=1)+1e-5
    de = 0.5 * np.log(2 * np.pi * np.e * variance)
    return de


def base_homo_select(eeg_data,sample_rate,num_exp, num_channel):
    signal = eeg_data
    fs = sample_rate
    order = 8
    f, t, zxx = stft(signal, fs=128, window='hann', nperseg=128, noverlap=0, nfft=256, scaling='psd')
    print(f"after stft the zxx shape:{zxx.shape}")
    window = 55
    base_freq = np.empty((window*num_exp,num_channel))
    base_freq_idx = np.abs(zxx).argmax(axis=2)
    for i in range(window*num_exp):
        for j in range(num_channel):
            base_freq[i][j] = np.mean(f[base_freq_idx[i][j]])
    # 找到谐波
    harm_freq = np.empty((window*num_exp, num_channel, order))
    for i in range(window*num_exp):
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
    alpha = 1e-10

    # Find the range of base freq for every channel
    base_flow_list = []
    base_fhigh_list = []
    for i in range(channel_num):
        std = np.std(base_freq[:,i])
        mean = np.mean(base_freq[:,i])
        base_fhigh = int(mean+3*std)
        base_flow = int(mean-3*std)
        base_flow_list.append(base_flow)
        base_fhigh_list.append(base_fhigh)
    base_flow = int(sum(base_flow_list) / len(base_flow_list))
    base_fhigh = int(sum(base_fhigh_list) / len(base_fhigh_list))
    if base_flow < 0.5:
        base_flow = 0.5
    if base_fhigh < base_flow or np.abs(base_fhigh - base_flow)<2:
        base_fhigh = base_flow + 4
    if base_fhigh > max(f):
        base_fhigh = max(f) // 2

    # # Harmonic freq psd
    harm_m = []
    harm_s = []
    for i in range(channel_num):
        a = []
        b = []
        for j in range(8):
            m = np.mean(harm_freq[:,i,j])
            s = np.std(harm_freq[:,i,j])
            a.append(m)
            b.append(s)
        a = np.mean(a)
        b = np.mean(b)
        harm_m.append(a)
        harm_s.append(b)
    mean = np.mean(harm_m)
    std = np.mean(harm_s)
    harm_fhigh = int(mean+3*std)
    harm_flow = int(mean-3*std)

    if harm_flow < base_fhigh or harm_flow == 0:
        harm_flow = base_fhigh
        if harm_fhigh < harm_flow:
            harm_fhigh = harm_flow + 4
    if np.abs(harm_fhigh-harm_flow) < 2:
        harm_fhigh = harm_flow + 4
    if harm_fhigh > max(f):
        harm_fhigh = max(f)

    # freq_select = [base_flow, base_fhigh, harm_flow, harm_fhigh]
    # freq_select = sorted(freq_select)

    # base_flow = freq_select[0]
    # if freq_select[1] == 0:
    #     base_fhigh = freq_select[2]
    #     harm_flow = freq_select[2]
    #     harm_fhigh = freq_select[3]
    # elif freq_select[2] == freq_select[3]:
    #     base_fhigh = freq_select[1]
    #     harm_flow = freq_select[1]
    #     harm_fhigh = freq_select[2]
    # else:
    #     base_fhigh = freq_select[1]
    #     harm_flow = freq_select[2]
    #     harm_fhigh = freq_select[3]

    index1 = np.where(f == base_flow)[0][0]
    index2 = np.where(f == base_fhigh)[0][0]

    # # divid = fhigh-flow
    # if divid == 0:
    #     divid += 1
    # psd = np.sum(power[:,:,index1:index2,:]+alpha,axis=2) / divid
    # base_de = np.log2(psd)
    # print(f"psd shape:{psd.shape} index:{index1},{index2}")
    psd = power[:,:,index1:index2,:]+alpha
    base_de = calculate_de(psd)
    # print(f"base_de shape:{base_de.shape}")
    base_de_features = base_de

    ### Harm part
    index1 = np.where(f == harm_flow)[0][0]
    index2 = np.where(f == harm_fhigh)[0][0]
    # divid = fhigh-flow
    # if divid == 0:
    #     divid += 1
    # psd = np.sum(power[:,:,index1:index2,:]+alpha,axis=2) / divid
    # harm_de = np.log2(psd)
    psd = power[:,:,index1:index2,:]+alpha
    harm_de = calculate_de(psd)
    harmon_de_features = harm_de

    return base_de_features, harmon_de_features

def phase_sync(de_features):
    n_channels, n_samples = de_features.shape
    phase_sync_matrix = np.zeros((n_channels, n_channels))
    alpha = 1e-10
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            phase_diff = np.angle(np.exp(1j * (de_features[i] - de_features[j])))
            imag = np.abs(np.mean(np.exp(1j * phase_diff)).imag)
            # 计算相位相似度
            phase_sync_matrix[i, j] = np.log(imag+alpha) / np.log(0.5)
    return phase_sync_matrix

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

def data_process(data,sample_rate,exp_num,channel_num):
    labels = scipy.io.loadmat('/home/sjf/brainnet/SEED/SEED/label.mat')
    labels = labels['label']
    labels = np.repeat(labels,3,axis=0)
    labels = labels.flatten()
    print(f"in data process data shape:{data.shape} labels:{labels.shape}")
    base_de_features = []
    harmon_de_features = []
    all_labels = []
    for i in range(15):
        divided_data, alabels = data_divide(data[i], labels)
        all_labels.append(alabels)
        # all_data.append(divided_data)
        print(f"data shape:{divided_data.shape} labels:{labels.shape}")
        base_freq, f, harm_freq, zxx = base_homo_select(divided_data,sample_rate,exp_num, channel_num)
        print(f"*******Starting extract features*******")
        base_de, harmon_de = feature_extract(base_freq, f, harm_freq, zxx)
        base_de_features.append(base_de)
        harmon_de_features.append(harmon_de)
    all_labels = np.array(all_labels)
    base_de_features = np.array(base_de_features)
    harmon_de_features = np.array(harmon_de_features)
    # print(f"base de fea:{base_de_features.shape}")
    return base_de_features, harmon_de_features, all_labels

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    raw_dir = "/home/sjf/brainnet/SEED/SEED/Preprocessed_EEG/"
    file_names = os.listdir(raw_dir)
    all_base_de_features = []
    all_harmon_de_features = []
    all_labels = []
    raw_eeg = [None]*15
    prefixnumber = -1
    for filename in file_names:
        file_path = raw_dir+str(filename)
        prefix = filename.split("_")[0]
        prefixnumber = int(prefix)
        print(f"file:{filename}'s number {prefixnumber}")
        mat_data = scipy.io.loadmat(file_path)
        trial = []
        raw_sampel = []
        
        prefile = list(mat_data.keys())[3][:-1]
        print(f"prefiel is :{prefile} matdata keys:{mat_data.keys()}")
        for i in range(1,16):
            file = prefile+str(i)
            # raw_sampel.append(mat_data[file].shape[-1])
            trial.append(np.array(mat_data[file]))
        if raw_eeg[prefixnumber-1] is None:
            raw_eeg[prefixnumber-1] = [trial]
        else:
            raw_eeg[prefixnumber-1].append(trial)
        print(f"******* Processing on file {filename} ********")
    print(f"*******Start resampling data*******")
    resamped_data = [None]*15
    for i in range(15):
        for j in range(3):
            for k in range(15):
                resdata = resample_data(raw_eeg[i][j][k],200,128,280)
                if resamped_data[i] is None:
                    resamped_data[i] = [resdata]
                else:
                    resamped_data[i].append(resdata)
    resamped_data = np.array(resamped_data)
    print(f"resampled data shape is : {resamped_data.shape}")
    sample_rate = 128
    exp_num = 45
    channel_num = 62
    all_base_de_features, all_harmon_de_features, all_labels = data_process(resamped_data,sample_rate,exp_num,channel_num)
    # all_base_de_features.append(base_de_features)
    # all_harmon_de_features.append(harmon_de_features)
    # all_labels.append(labels)

    all_base_de_features = torch.tensor(all_base_de_features, dtype=torch.float)
    all_harmon_de_features = torch.tensor(all_harmon_de_features, dtype=torch.float)
    all_labels=torch.tensor(all_labels, dtype=torch.float)

    torch.save(all_base_de_features,'/home/sjf/eegall/data/SEED/all_seed_rebase_de_features.pt')
    torch.save(all_harmon_de_features,'/home/sjf/eegall/data/SEED/all_seed_reharmon_de_features.pt')
    torch.save(all_labels,'/home/sjf/eegall/data/SEED/all_seed_relabels.pt')
    print("********Start graph construct********")
    base_graph, harm_graph = phase_graph(all_base_de_features, all_harmon_de_features)
    torch.save(base_graph,'/home/sjf/eegall/data/SEED/seed_rebase_graph.pt')
    torch.save(harm_graph,'/home/sjf/eegall/data/SEED/seed_reharm_graph.pt')
