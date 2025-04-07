import torch
from torch.utils.data import Dataset, DataLoader
import os

class CustomEEGDataset(Dataset):
    def __init__(self, base_x_path, harm_x_path, labels_path, base_graph_path, harm_graph_path):
        self.base_x = torch.load(base_x_path).float()
        self.harm_x = torch.load(harm_x_path).float()
        self.labels = torch.load(labels_path)
        self.base_graph = torch.tensor(torch.load(base_graph_path),dtype=torch.float)
        self.harm_graph = torch.tensor(torch.load(harm_graph_path),dtype=torch.float)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        base_x = self.base_x[idx]
        harm_x = self.harm_x[idx]
        label = self.labels[idx]
        base_graph = self.base_graph[idx]
        harm_graph = self.harm_graph[idx]
        
        return base_x, harm_x, label, base_graph, harm_graph

# 文件路径
base_x_path = '/home/sjf/eegall/data/DEAP/all_base1_de_features.pt'
harm_x_path = '/home/sjf/eegall/data/DEAP/all_harmon1_de_features.pt'
labels_path = '/home/sjf/eegall/data/DEAP/all_1labels.pt'
base_graph_path = '/home/sjf/eegall/data/DEAP/base1_graph.pt'
harm_graph_path = '/home/sjf/eegall/data/DEAP/harm1_graph.pt'

# 创建数据集
dataset = CustomEEGDataset(base_x_path, harm_x_path, labels_path, base_graph_path, harm_graph_path)

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

# 检查数据加载速度
import time

start_time = time.time()
for i, data in enumerate(dataloader):
    if i == 10:  # 只循环10个批次以测试加载速度
        break
    print([d.shape for d in data])  # 打印每个元素的形状
end_time = time.time()

print(f"10 batches loaded in {end_time - start_time:.2f} seconds")
