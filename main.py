from model import MyMixture
from fremodel import FreMyMixture
import matplotlib.pyplot as plt
import os

from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data import random_split


from sklearn.model_selection import KFold
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from customloss import SmoothCrossEntropyLoss
import argparse
import random
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import sys
from torch.nn.utils import clip_grad_norm_

parser = argparse.ArgumentParser()
parser.add_argument('--loss', default='sce', type=str, help='Enter your lossfn')
parser.add_argument('--graph', default='base', type=str, help='Enter your datarange')
parser.add_argument('--e',default=0.1, type=float, help='Enter your min loss range')
parser.add_argument('--lr',default=0.00005, type=float, help='Enter your learning rate')
parser.add_argument('--alpha', default=0.0008,type=float, help='Enter your weight decay')
parser.add_argument('--batch',default=38,type=int,help='Enter batch size')
parser.add_argument('--maxiter',default=500,type=int,help='Enter max iteration')
parser.add_argument('--kfold',default=10, type=int, help='Enter K-Fold number')
parser.add_argument('--limit', default=1, type=int,help='Enter whether use limit')
parser.add_argument('--data',default='DEAP', type=str, help='Enter Dataset name')
parser.add_argument('--seed',default=74, type=int, help='Enter random seed for all')
parser.add_argument('--valpage',default=6, type=int, help='Enter validation batch for validating')
parser.add_argument('--ablation',default=0,type=int,help='Enter whether do the ablation study')
parser.add_argument('--norm', default=0, type=int, help='Enter whether need norm data')
parser.add_argument('--abtype',default='W',type=str,help='Enter what the type ablation WTG NTG')
parser.add_argument('--freq', default='BH', type=str, help='Enter whether to use the base harm freq.')
parser.add_argument('--freph',default='no',type=str,help='Enter which graph to use')
parser.add_argument('--ckpoint',default=0,type=int,help='Enter where to continue the exp')
parser.add_argument('--tylabel', default='Valence',type=str,help='Enter which label')
parser.add_argument('--modeldir', default=1, type=int,help='Enter model dir')

args = parser.parse_args()

params = {
    'batch_size': args.batch,
    'max_iteration': args.maxiter,  # maximum number of iterations
    'k_fold': args.kfold,
}


params['e'] = args.e
params['lr'] = args.lr
params['alpha'] = args.alpha


def set_seed(seed):
    # Python内置random模块的种子
    random.seed(seed)
    
    # NumPy的种子
    np.random.seed(seed)
    
    # PyTorch的种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(args.seed)
def train_accuracy(output, target):
    with torch.no_grad():
        predictions = torch.argmax(output, dim=-1)
        correct = (predictions == target).sum().item()
        return correct / target.size(0)

def freq_train(limit, model, device, ng1_data, ng2_data, ng3_data, ng4_data, g_data, train_graph, train_labels, loss_fn, optimizer):
    # Show the train data length
    print(f"Training... train_data length:{len(ng1_data)}")
    # import torch.nn as nn
    # move all the data to the same device
    ng1_data = torch.tensor(ng1_data,dtype=torch.float).to(device)
    ng2_data = torch.tensor(ng2_data,dtype=torch.float).to(device)
    ng3_data = torch.tensor(ng3_data,dtype=torch.float).to(device)
    ng4_data = torch.tensor(ng4_data,dtype=torch.float).to(device)
    g_data = torch.tensor(g_data,dtype=torch.float).to(device)
    train_graph = torch.tensor(train_graph,dtype=torch.float).to(device)
    train_labels = torch.tensor(train_labels,dtype=torch.float).to(device)
    # initialize
    model.to(device)
    model.train()
    batch_size = params['batch_size']
    loss_record = []
    acc_record = []

    step = 0
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=7000, gamma=0.1)

    flag = True
# with tqdm(total=params['max_iteration']) as pbar:
    # print(f"base_train_data :{base_train_data}")
    while flag:
        for i in range(0, len(ng1_data), batch_size):
            # select batch for training
            ng1_batch = ng1_data[i:i+batch_size].to(device)
            ng2_batch = ng2_data[i:i+batch_size].to(device)
            ng3_batch = ng3_data[i:i+batch_size].to(device)
            ng4_batch = ng4_data[i:i+batch_size].to(device)
            g_batch = g_data[i:i+batch_size].to(device)
            if ng1_batch.shape[0] == 1:
                break

            graph = train_graph[i:i+batch_size].to(device)
            label = train_labels[i:i+batch_size].to(device)
            # print(f"base_batch:{base_batch}")
            output = model(ng1_batch,ng2_batch,ng3_batch,ng4_batch,g_batch, graph, step)
            # print(f"output:{output} label:{label}")
            loss = loss_fn(output, label)
            loss_record.append(loss.item())
            acc = train_accuracy(output, label)
            acc_record.append(acc)
            if step % 100 == 0:
                print(f"step: {step}, Loss: {loss.item()} Acc:{acc}")
            
            # stop condition
            if limit:
                if loss < params['e']:
                    flag = False
                    break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            # scheduler.step()
            # pbar.update(1)
            if step >= params['max_iteration']:
                flag = False
                break
    # pbar.close()
    print('training successfully ended.')
    return model,loss_record,acc_record



# Train Function
def train(limit, model, device, base_train_data,harm_train_data, train_graph, train_labels, loss_fn, optimizer):
    # Show the train data length
    print(f"Training... train_data length:{len(base_train_data)}")
    import torch.nn as nn
    # move all the data to the same device
    for base_data in base_train_data:
        base_data.to(device)
    for harm_data in harm_train_data:
        harm_data.to(device)
    torch.autograd.set_detect_anomaly(True)
    # initialize
    model.to(device)
    model.train()
    batch_size = params['batch_size']
    loss_record = []
    acc_record = []

    step = 0
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=7000, gamma=0.1)

    flag = True
    # with tqdm(total=params['max_iteration']) as pbar:
        # print(f"base_train_data :{base_train_data}")
    while flag:
        for i in range(0, len(base_train_data), batch_size):
            # select batch for training
            base_batch = base_train_data[i:i+batch_size]
            harm_batch = harm_train_data[i:i+batch_size]
            if base_batch.shape[0] == 1:
                break

            graph = train_graph[i:i+batch_size].to(device)
            label = train_labels[i:i+batch_size].to(device)
            # print(f"base_batch:{base_batch}")
            output = model(base_batch, harm_batch, graph, step)
            # print(f"output:{output} label:{label}")
            loss = loss_fn(output, label)
            # print(f"the loss is that: {loss.item()}")
            loss_record.append(loss.item())
            acc = train_accuracy(output, label)
            acc_record.append(acc)
            if step % 100 == 0:
                print(f"step: {step}, Loss: {loss.item()} Acc:{acc}")
            
            # stop condition
            if limit:
                if loss < params['e']:
                    flag = False
                    break
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            step += 1
            # scheduler.step()
            # pbar.update(1)
            if step >= params['max_iteration']:
                flag = False
                break
    # pbar.close()
    print('training successfully ended.')
    return model,loss_record,acc_record

def validate(model, device, base_val_data, harm_val_data, val_graph, val_labels,batch_size):
    print('validating...')
    print(f"validate data length:{len(base_val_data)}")
    model.to(device)
    model.eval()
    base_val_data = torch.tensor(base_val_data)
    harm_val_data = torch.tensor(harm_val_data)
    val_graph = torch.tensor(val_graph)
    val_labels = torch.tensor(val_labels)

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    step=0

    with torch.no_grad():
        for i in range(0, len(base_val_data), batch_size):
            base_batch = base_val_data[i:i+batch_size].to(device)
            if base_batch.shape[0] != batch_size:
                break
            harm_batch = harm_val_data[i:i+batch_size].to(device)
            graph = val_graph[i:i+batch_size].to(device)
            label = val_labels[i:i+batch_size].to(device)

            output = model(base_batch, harm_batch, graph, step)
            
            result = torch.argmax(output, dim=-1)
            # record all the metrics
            for i in range(len(result)):
                if result[i] == 0 and result[i] == label[i]:
                    TP += 1
                if result[i] == 0 and result[i] != label[i]:
                    FP += 1
                if result[i] == 1 and result[i] == label[i]:
                    TN += 1
                if result[i] == 1 and result[i] != label[i]:
                    FN += 1

    acc = (TP + TN) / (TP + TN + FP + FN)

    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)

    if precision + recall == 0:
        f_score = 0
    else:
        f_score = 2 * precision * recall / (precision + recall)

    print(f'acc: {acc}')
    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'F_score: {f_score}')
    # torch.save(val_labels,'/home/sjf/eegall/nolimits/FACED/valence_val_labels.pt')
    return acc, precision, recall, f_score

def freq_validate(model, device, ng1_data, ng2_data, ng3_data, ng4_data, g_data, val_graph, val_labels, batch_size):
    print('validating...')
    print(f"validate data length:{len(ng1_data)}")
    model.to(device)
    model.eval()
    ng1_data = torch.tensor(ng1_data, dtype=torch.float).to(device)
    ng2_data = torch.tensor(ng2_data, dtype=torch.float).to(device)
    ng3_data = torch.tensor(ng3_data, dtype=torch.float).to(device)
    ng4_data = torch.tensor(ng4_data, dtype=torch.float).to(device)
    g_data = torch.tensor(g_data, dtype=torch.float).to(device)

    val_graph = torch.tensor(val_graph)
    val_labels = torch.tensor(val_labels)

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    step=0

    with torch.no_grad():
        for i in range(0, len(ng1_data), batch_size):
            ng1_batch = ng1_data[i:i+batch_size].to(device)
            ng2_batch = ng2_data[i:i+batch_size].to(device)
            ng3_batch = ng3_data[i:i+batch_size].to(device)
            ng4_batch = ng4_data[i:i+batch_size].to(device)
            g_batch = g_data[i:i+batch_size].to(device)
            
            if ng1_batch.shape[0] != batch_size:
                break
            graph = val_graph[i:i+batch_size].to(device)
            label = val_labels[i:i+batch_size].to(device)

            output = model(ng1_batch, ng2_batch, ng3_batch, ng4_batch, g_batch, graph,step)
            
            result = torch.argmax(output, dim=-1)
            # record all the metrics
            for i in range(len(result)):
                if result[i] == 0 and result[i] == label[i]:
                    TP += 1
                if result[i] == 0 and result[i] != label[i]:
                    FP += 1
                if result[i] == 1 and result[i] == label[i]:
                    TN += 1
                if result[i] == 1 and result[i] != label[i]:
                    FN += 1

    acc = (TP + TN) / (TP + TN + FP + FN)

    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)

    if precision + recall == 0:
        f_score = 0
    else:
        f_score = 2 * precision * recall / (precision + recall)

    print(f'acc: {acc}')
    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'F_score: {f_score}')

    return acc, precision, recall, f_score

#origin
# def binary_sampling(data,harm_data,graph, label):
#     """
#     对二分类数据集进行重采样，保证数据平衡
#     """
#     num = len(data)  # number of samples

#     index_0 = torch.nonzero(label == 0, as_tuple=True)[0]
#     num_0 = len(index_0)

#     index_1 = torch.nonzero(label == 1, as_tuple=True)[0]
#     num_1 = len(index_1)
#     print([num_0, num_1])

#     if abs(num_0 - num_1) < num * 0.15:
#         return data,harm_data,graph, label
#     elif num_0 == 0 or num_1 == 0:
#         return data,harm_data,graph, label
#     else:
#         selected_index = []
#         shorter_index, shorter_num = (index_0, num_0) if num_0 < num_1 else (index_1, num_1)
#         longer_index, longer_num = (index_0, num_0) if num_0 > num_1 else (index_1, num_1)
#         for i in range(longer_num):
#             selected_index.append(longer_index[i])
#             selected_index.append(shorter_index[i % shorter_num])

#         selected_data = []
#         selected_harm_data = []
#         selected_graph = []
#         for i in selected_index:
#             selected_data.append(data[i])
#             selected_harm_data.append(harm_data[i])
#             selected_graph.append(graph[i])
        
#         selected_label = label[selected_index]
#         return selected_data, selected_harm_data, selected_graph, selected_label
# DEAP only
def binary_sampling(data,harm_data,graph, label):
    """
    对二分类数据集进行重采样，保证数据平衡
    """
    num = len(data)  # number of samples

    index_0 = torch.nonzero(label == 0, as_tuple=True)[0]
    num_0 = len(index_0)

    index_1 = torch.nonzero(label == 1, as_tuple=True)[0]
    num_1 = len(index_1)
    print([num_0, num_1])

    if abs(num_0 - num_1) < num * 0.15:
        return data,harm_data,graph, label
    elif num_0 == 0 or num_1 == 0:
        return data,harm_data,graph, label
    else:
        selected_index = []
        shorter_index, shorter_num = (index_0, num_0) if num_0 < num_1 else (index_1, num_1)
        longer_index, longer_num = (index_0, num_0) if num_0 > num_1 else (index_1, num_1)
        for i in range(longer_num):
            selected_index.append(longer_index[i])
            selected_index.append(shorter_index[i % shorter_num])

        selected_data = []
        selected_harm_data = []
        selected_graph = []
        for i in selected_index:
            selected_data.append(data[i])
            selected_harm_data.append(harm_data[i])
            selected_graph.append(graph[i])
        
        selected_label = label[selected_index]
        selected_data = np.array(selected_data)
        selected_harm_data = np.array(selected_harm_data)
        selected_graph = np.array(selected_graph)
        selected_label = np.array(selected_label)
        return selected_data, selected_harm_data, selected_graph, selected_label

def sliding_window_inference(page, base_val_data, harm_val_data, val_graph,val_labels, batch_size):
    batch_size = len(base_val_data) // page
    return base_val_data, harm_val_data, val_graph,val_labels, batch_size

def freq_sliding_window_inference(page, delta_val_data, theta_val_data,alpha_val_data,beta_val_data, gamma_val_data, val_graph,val_labels, batch_size):
    batch_size = len(delta_val_data) // page
    return delta_val_data, theta_val_data,alpha_val_data,beta_val_data, gamma_val_data, val_graph,val_labels, batch_size

def contains_non_int(arr):
    return any(not isinstance(x, (int, np.integer)) for x in arr)

# Cross_validation
def cross_validation(args, subject, base_data, harm_data, graph, labels, seed, device):
    splits = KFold(n_splits=params['k_fold'], shuffle=True, random_state=seed)
    input_dim = 7
    output_dim = 32
    input_channel = 32
    output_channel = 16
    input_size = 128
    hidden_size = 256

    in_channels = 7
    out_channels = 32
    K = 4
    acc_list = []
    f_score_list = []
    precision_list = []
    recall_list = []
    fold_models = []
    all_loss = []
    all_train_acc = []
    # print(f"in cross val base_data {base_data}")
    # DEAP only
    base_data, harm_data, graph, labels = binary_sampling(base_data, harm_data, graph, labels)

    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(base_data)))):
        print(f'******fold {fold+1}******\n')
        # data preparation for learning
        base_train_data = []
        for i in train_idx:
            base_train_data.append(base_data[i])
        # train_idx = train_idx.astype(int)
        # print(f"train_idx:{train_idx} type train_idx{type(train_idx)} len graph:{graph.shape}")
        # print(contains_non_int(train_idx)) 
        # print(f"base_train_data:{base_data}")
        train_labels = labels[train_idx]
        train_graph = graph[train_idx]

        harm_train_data = []
        for i in train_idx:
            harm_train_data.append(harm_data[i])

        # base_train_data, harm_train_data, train_graph, train_labels = binary_sampling(base_train_data,harm_train_data,train_graph,train_labels)
        train_labels = torch.tensor(np.array(train_labels)).to(device)
        train_graph = torch.tensor(np.array(train_graph)).to(device)
        base_train_data = torch.tensor(np.array(base_train_data)).to(device)
        harm_train_data = torch.tensor(np.array(harm_train_data)).to(device)
        # print(f"base_train_data : {base_train_data}")

        # Val data
        base_val_data = []
        for i in val_idx:
            base_val_data.append(base_data[i])
        val_labels = labels[val_idx]
        base_val_data = np.array(base_val_data)
        base_val_data = torch.tensor(base_val_data)
        val_graph = graph[val_idx]

        harm_val_data = []
        for i in val_idx:
            harm_val_data.append(harm_data[i])
        harm_val_data = np.array(harm_val_data)
        harm_val_data = torch.tensor(harm_val_data)

        # Model define
        model_dir = '/home/sjf/eegall/intermodel/'
        if args.norm:
            model_path = "norm-"+"lr"+str(args.lr)+args.data+str(args.kfold)+'_'+str(args.batch)+'_'+str(args.maxiter) + '_' +args.loss + args.graph + 'seed'+str(args.seed)+'fold_best_model.pth'
        else:
            model_path = "lr"+str(args.lr)+args.data+str(args.kfold)+'_'+str(args.batch)+'_'+str(args.maxiter) + '_' +args.loss + args.graph + 'seed'+str(args.seed)+'fold_best_model.pth'
        
        if args.tylabel == 'Valence':
            model_path = "Valence-"+model_path
        elif args.tylabel == 'Arousal':
            model_path = "Arousal-"+model_path
        
        if args.ablation:
            model_path = 'Ablation'+model_path

        if args.freq == 'BH':
            pass
        else:
            model_path = 'CF'+model_path
        if args.modeldir:
            model_dir = model_dir+'modelsave'+str(args.modeldir)+'/'
        model_path = model_dir + model_path

        if os.path.exists(model_path):
            print(f"in cross val model path:{model_path}")
            model = MyMixture(input_dim,output_dim,input_channel,output_channel,input_size,hidden_size,in_channels, out_channels,K)
            model.load_state_dict(torch.load(model_path))
        else:
            print("*******Initializing new model*******")
            model = MyMixture(input_dim,output_dim,input_channel,output_channel,input_size,hidden_size,in_channels, out_channels,K)

        
        # different loss smooth loss is customized by introducing a smoothing factor
        if args.loss == 'ce':
            loss_fn = nn.CrossEntropyLoss()
        else:
            loss_fn = SmoothCrossEntropyLoss(smoothing=0.5)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['alpha'])

        # start training
        trained_model, loss_record, acc_record = train(args.limit, model, device, base_train_data, harm_train_data, train_graph, train_labels, loss_fn, optimizer)
        # record every loss
        all_loss.append(loss_record)
        all_train_acc.append(acc_record)
        # record every fold models
        fold_models.append(trained_model)
        # metric
        base_val_data,harm_val_data, val_graph,val_labels, batch_size = sliding_window_inference(args.valpage,base_val_data, harm_val_data, val_graph,val_labels, args.batch)

        acc, precision, recall, f_score = validate(trained_model, device, base_val_data,harm_val_data, val_graph,val_labels,batch_size)
        acc_list.append(acc)
        f_score_list.append(f_score)
        precision_list.append(precision)
        recall_list.append(recall)

        model_index = acc_list.index(max(acc_list))
        fold_best_model = fold_models[model_index]
        torch.save(fold_best_model.state_dict(),model_path)

    avg_acc = np.array(acc_list).mean()
    avg_f_score = np.array(f_score_list).mean()
    max_acc = np.array(acc_list).max()
    max_f_score = np.array(f_score_list).max()
    avg_recall = np.array(recall_list).mean()
    max_recall = np.array(recall_list).max()
    avg_precision = np.array(precision_list).mean()
    max_precision = np.array(precision_list).max()
    print(f'subject {subject} Avgacc: {avg_acc} Avgfscore: {avg_f_score} \n Max acc:{max_acc}, Max f score:{max_f_score} Avg Recall:{avg_recall} Max Recall:{max_recall} Avg Precision:{avg_precision} Max Precision:{max_precision}')
    # 将结果保存为txt文件
    if args.norm:
        filename = "norm-"+"lr"+str(args.lr)+args.data+str(args.kfold)+'_'+str(args.batch)+'_'+str(args.maxiter) + '_' +args.loss + args.graph + 'seed'+str(args.seed)+'_results.txt'
    else:
        filename = "lr"+str(args.lr)+args.data+str(args.kfold)+'_'+str(args.batch)+'_'+str(args.maxiter) + '_' +args.loss + args.graph + 'seed'+str(args.seed)+'_results.txt'
    if args.tylabel == 'Valence':
        filename = "Valence-"+filename
    elif args.tylabel == 'Arousal':
        filename = "Arousal-"+filename
    if args.ablation:
        filename = 'Ablation-'+filename
    if args.abtype == 'WTG':
        filename = 'WTG'+filename
        # print('ok')
    elif args.abtype == 'NTG':
        filename = 'NTG'+filename
    if args.freq == 'BH':
        pass
    else:
        filename = "CF"+filename
    if args.modeldir:
        filename = str(args.modeldir)+filename
    if args.limit:
        filepath = '/home/sjf/eegall/withlimits/'+filename
    else:
        filepath = '/home/sjf/eegall/nolimits/'+filename

    # all loss record
    los_path = '/home/sjf/eegall/lossre/'
    loss_name = "lr"+str(args.lr)+args.data+str(args.kfold)+'_'+str(args.batch)+'_'+str(args.maxiter) + '_' +args.loss + args.graph + 'seed'+str(args.seed)
    if args.tylabel == 'Valence':
        loss_name = "Valence-"+loss_name
    elif args.tylabel == 'Arousal':
        loss_name = "Arousal-"+loss_name
    loss_path = los_path+loss_name
    if args.data == 'DEAP':
        loss_path = loss_path + 'nwDEAPloss.pt'
    elif args.data == 'FACED':
        loss_path = loss_path + 'nwFACEDloss.pt'
    
    torch.save(all_loss,loss_path)
    # sys.exit('all loss have been saved.')

    with open(filepath, 'a') as f:
        f.write(f'\nsubject {subject}: Avgacc: {avg_acc}, Avgfscore: {avg_f_score}, Avg Recall:{avg_recall}, Avg Precision:{avg_precision} \n')
        f.write(f'Max acc:{max_acc}, Max f score:{max_f_score}, Max Recall:{max_recall},  Max Precision:{max_precision}\n')

    if args.norm:
        tnac_filename = "norm-"+"lr"+str(args.lr)+args.data+str(args.kfold)+'_'+str(args.batch)+'_'+str(args.maxiter) + '_' +args.loss + args.graph + 'seed'+str(args.seed)+'_acc.pt'
    else:
        tnac_filename = "lr"+str(args.lr)+args.data+str(args.kfold)+'_'+str(args.batch)+'_'+str(args.maxiter) + '_' +args.loss + args.graph + 'seed'+str(args.seed)+'_acc.pt'
    
    if args.tylabel == 'Valence':
        tnac_filename = "Valence-" + tnac_filename
    elif args.tylabel == 'Arousal':
        tnac_filename = "Arousal-" + tnac_filename

    if args.freq == 'BH':
        pass
    else:
        tnac_filename = "CF"+tnac_filename
    tnac_path = "/home/sjf/eegall/results/"+tnac_filename
    torch.save(all_train_acc, tnac_path)
  

    return fold_best_model, max_acc, avg_acc, avg_f_score, max_f_score

# Cross_validation
def freq_cross_validation(args, subject, ng1_data, ng2_data, ng3_data, ng4_data, g_data, graph, labels, seed, device):
    splits = KFold(n_splits=params['k_fold'], shuffle=True, random_state=seed)
    input_dim = 7
    output_dim = 32
    input_channel = 32
    output_channel = 16
    input_size = 128
    hidden_size = 256

    in_channels = 7
    out_channels = 32
    K = 4
    acc_list = []
    f_score_list = []
    precision_list = []
    recall_list = []
    fold_models = []
    all_loss = []
    all_train_acc = []
    # print(f"in cross val base_data {base_data}")
    # if args.freph == 'delta':
    #     theta_data, alpha_data,
    # # DEAP only
    # base_data, harm_data, graph, labels = binary_sampling(base_data, harm_data, graph, labels)

    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(ng1_data)))):
        print(f'******fold {fold+1}******\n')
        # data preparation for learning
        ng1_train_data = []
        ng2_train_data = []
        ng3_train_data = []
        ng4_train_data = []
        g_train_data = []
        for i in train_idx:
            ng1_train_data.append(ng1_data[i])
            ng2_train_data.append(ng2_data[i])
            ng3_train_data.append(ng3_data[i])
            ng4_train_data.append(ng4_data[i])
            g_train_data.append(g_data[i])

        train_labels = labels[train_idx]
        train_graph = graph[train_idx]

        # harm_train_data = []
        # for i in train_idx:
        #     harm_train_data.append(harm_data[i])

        # base_train_data, harm_train_data, train_graph, train_labels = binary_sampling(base_train_data,harm_train_data,train_graph,train_labels)
        train_labels = torch.tensor(np.array(train_labels)).to(device)
        train_graph = torch.tensor(np.array(train_graph)).to(device)
        ng1_train_data = torch.tensor(np.array(ng1_train_data)).to(device)
        ng2_train_data = torch.tensor(np.array(ng2_train_data)).to(device)
        ng3_train_data = torch.tensor(np.array(ng3_train_data)).to(device)
        ng4_train_data = torch.tensor(np.array(ng4_train_data)).to(device)
        g_train_data = torch.tensor(np.array(g_train_data)).to(device)

        # print(f"base_train_data : {base_train_data}")

        # Val data
        ng1_val_data = []
        ng2_val_data = []
        ng3_val_data = []
        ng4_val_data = []
        g_val_data = []
        for i in val_idx:
            ng1_val_data.append(ng1_data[i])
            ng2_val_data.append(ng2_data[i])
            ng3_val_data.append(ng3_data[i])
            ng4_val_data.append(ng4_data[i])
            g_val_data.append(g_data[i])
        val_labels = labels[val_idx]
        val_graph = graph[val_idx]

        val_labels = torch.tensor(np.array(val_labels)).to(device)
        val_graph = torch.tensor(np.array(val_graph)).to(device)

        ng1_val_data = torch.tensor(np.array(ng1_val_data)).to(device)
        ng2_val_data = torch.tensor(np.array(ng2_val_data)).to(device)
        ng3_val_data = torch.tensor(np.array(ng3_val_data)).to(device)
        ng4_val_data = torch.tensor(np.array(ng4_val_data)).to(device)
        g_val_data = torch.tensor(np.array(g_val_data)).to(device)


        # Model define
        model_dir = '/home/sjf/eegall/intermodel/'
        if args.norm:
            model_path = "norm-"+"lr"+str(args.lr)+args.data+str(args.kfold)+'_'+str(args.batch)+'_'+str(args.maxiter) + '_' +args.loss + args.graph + 'seed'+str(args.seed)+'fold_best_model.pth'
        else:
            model_path = "lr"+str(args.lr)+args.data+str(args.kfold)+'_'+str(args.batch)+'_'+str(args.maxiter) + '_' +args.loss + args.graph + 'seed'+str(args.seed)+'fold_best_model.pth'
        
        if args.tylabel == 'Valence':
            model_path = "Valence-"+model_path
        elif args.tylabel == 'Arousal':
            model_path = "Arousal-"+model_path
        
        if args.ablation:
            model_path = 'Ablation'+model_path

        if args.freq == 'BH':
            pass
        else:
            model_path = 'CF'+model_path
        if args.modeldir:
            model_dir = model_dir+'modelsave'+str(args.modeldir)+'/'
        model_path = model_dir + model_path

        if os.path.exists(model_path):
            print(f"in fre cross model path:{model_path}")
            model = FreMyMixture(input_dim,output_dim,input_channel,output_channel,input_size,hidden_size,in_channels, out_channels,K)
            model.load_state_dict(torch.load(model_path))
        else:
            print("*******Initializing new model*******")
            model = FreMyMixture(input_dim,output_dim,input_channel,output_channel,input_size,hidden_size,in_channels, out_channels,K)

        
        # different loss smooth loss is customized by introducing a smoothing factor
        if args.loss == 'ce':
            loss_fn = nn.CrossEntropyLoss()
        else:
            loss_fn = SmoothCrossEntropyLoss(smoothing=0.5)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['alpha'])

        # start training
        trained_model, loss_record, acc_record = freq_train(args.limit, model, device, ng1_train_data, ng2_train_data,ng3_train_data,ng4_train_data, g_train_data, train_graph, train_labels, loss_fn, optimizer)
        # record every loss
        all_loss.append(loss_record)
        all_train_acc.append(acc_record)
        # record every fold models
        fold_models.append(trained_model)
        # metric
        ng1_val_data, ng2_val_data,ng3_val_data,ng4_val_data, g_val_data, val_graph,val_labels, batch_size = freq_sliding_window_inference(args.valpage,ng1_val_data, ng2_val_data,ng3_val_data,ng4_val_data, g_val_data, val_graph,val_labels, args.batch)

        acc, precision, recall, f_score = freq_validate(trained_model, device, ng1_val_data, ng2_val_data,ng3_val_data,ng4_val_data, g_val_data, val_graph,val_labels,batch_size)
        acc_list.append(acc)
        f_score_list.append(f_score)
        precision_list.append(precision)
        recall_list.append(recall)

        model_index = acc_list.index(max(acc_list))
        fold_best_model = fold_models[model_index]
        torch.save(fold_best_model.state_dict(),model_path)

    avg_acc = np.array(acc_list).mean()
    avg_f_score = np.array(f_score_list).mean()
    max_acc = np.array(acc_list).max()
    max_f_score = np.array(f_score_list).max()
    avg_recall = np.array(recall_list).mean()
    max_recall = np.array(recall_list).max()
    avg_precision = np.array(precision_list).mean()
    max_precision = np.array(precision_list).max()

    print(f'subject {subject} Avgacc: {avg_acc} Avgfscore: {avg_f_score} \n Max acc:{max_acc}, Max f score:{max_f_score} Avg Recall:{avg_recall} Max Recall:{max_recall} Avg Precision:{avg_precision} Max Precision:{max_precision}')
    # 将结果保存为txt文件
    if args.norm:
        filename = "norm-"+"lr"+str(args.lr)+args.data+str(args.kfold)+'_'+str(args.batch)+'_'+str(args.maxiter) + '_' +args.loss + args.graph + 'seed'+str(args.seed)+'_results.txt'
    else:
        filename = "lr"+str(args.lr)+args.data+str(args.kfold)+'_'+str(args.batch)+'_'+str(args.maxiter) + '_' +args.loss + args.graph + 'seed'+str(args.seed)+'_results.txt'
    if args.ablation:
        filename = 'Ablation-'+filename
    if args.abtype == 'WTG':
        filename = 'WTG'+filename
        # print('ok')
    elif args.abtype == 'NTG':
        filename = 'NTG'+filename
    if args.freq == 'BH':
        pass
    else:
        filename = "FRE"+"CF"+filename
    filename = str(args.freph) + filename
    if args.limit:
        filepath = '/home/sjf/eegall/withlimits/'+filename
    else:
        filepath = '/home/sjf/eegall/nolimits/'+filename

    with open(filepath, 'a') as f:
        f.write(f'\nsubject {subject}: Avgacc: {avg_acc}, Avgfscore: {avg_f_score}, Avg Recall:{avg_recall}, Avg Precision:{avg_precision} \n')
        f.write(f'Max acc:{max_acc}, Max f score:{max_f_score}, Max Recall:{max_recall},  Max Precision:{max_precision}\n')

    if args.norm:
        tnac_filename = "norm-"+"lr"+str(args.lr)+args.data+str(args.kfold)+'_'+str(args.batch)+'_'+str(args.maxiter) + '_' +args.loss + args.graph + 'seed'+str(args.seed)+'_acc.pt'
    else:
        tnac_filename = "lr"+str(args.lr)+args.data+str(args.kfold)+'_'+str(args.batch)+'_'+str(args.maxiter) + '_' +args.loss + args.graph + 'seed'+str(args.seed)+'_acc.pt'
    if args.freq == 'BH':
        pass
    else:
        tnac_filename = "FRE"+"CF"+tnac_filename
    tnac_filename = str(args.freph)+tnac_filename
    tnac_path = "/home/sjf/eegall/results/"+tnac_filename
    torch.save(all_train_acc, tnac_path)

    return fold_best_model, max_acc, avg_acc, avg_f_score, max_f_score


# scaled eeg signal
def scaled_eeg(base_x, harm_x):
    # base feature
    base_subject_means = torch.mean(base_x, axis=(1, 2))
    base_subject_stds = torch.std(base_x, axis=(1, 2))
    # harm feature
    harm_subject_means = torch.mean(base_x, axis=(1, 2))
    harm_subject_stds = torch.std(base_x, axis=(1, 2))

    base_global_mean = torch.mean(base_subject_means, axis=0)
    base_global_std = torch.mean(base_subject_stds, axis=0)
    harm_global_mean = torch.mean(harm_x, axis=0)
    harm_global_std = torch.mean(harm_subject_stds, axis=0)


    base_normal = (base_x - base_subject_means[:, np.newaxis, np.newaxis, :]) / base_subject_stds[:, np.newaxis, np.newaxis, :]
    harm_normal = (harm_x - harm_subject_means[:, np.newaxis, np.newaxis, :]) / harm_subject_stds[:, np.newaxis, np.newaxis, :]

    base_standard = base_normal*base_global_std + base_global_mean
    harm_standard = harm_normal*harm_global_std + harm_global_mean
    return base_standard, harm_standard




if __name__ == '__main__':
    if args.data == 'DEAP':
        print("--------- DEAP DATA ---------\n")
        if args.freq == 'BH':
            base_x = torch.load('/home/sjf/eegall/data/DEAP/all_base1_de_features.pt').float()
            harm_x = torch.load('/home/sjf/eegall/data/DEAP/all_harmon1_de_features.pt').float()
            all_labels = torch.load('/home/sjf/eegall/data/DEAP/all_1labels.pt')
            base_graph = torch.load('/home/sjf/eegall/data/DEAP/base1_graph.pt')
            harm_graph = torch.load('/home/sjf/eegall/data/DEAP/harm1_graph.pt')
            base_graph = torch.tensor(base_graph, dtype=torch.float)
            harm_graph = torch.tensor(harm_graph, dtype=torch.float)
        else:
            all_features = torch.load('/home/sjf/eegall/data/DEAP/all_fre_features.pt')
            delta_features = all_features[:,0,:,:,:]
            theta_features = all_features[:,1,:,:,:]
            alpha_features = all_features[:,2,:,:,:]
            beta_features = all_features[:,3,:,:,:]
            gamma_features = all_features[:,4,:,:,:]
            all_labels = torch.load('/home/sjf/eegall/data/DEAP/all_fre_deaplabels.pt')
            delta_graph = torch.load('/home/sjf/eegall/data/DEAP/delta_graph.pt')
            theta_graph = torch.load('/home/sjf/eegall/data/DEAP/theta_graph.pt')
            alpha_graph = torch.load('/home/sjf/eegall/data/DEAP/alpha_graph.pt')
            beta_graph = torch.load('/home/sjf/eegall/data/DEAP/beta_graph.pt')
            gamma_graph = torch.load('/home/sjf/eegall/data/DEAP/gamma_graph.pt')
            delta_graph = torch.tensor(delta_graph, dtype=torch.float)
            theta_graph = torch.tensor(theta_graph, dtype=torch.float)
            alpha_graph = torch.tensor(alpha_graph, dtype=torch.float)
            beta_graph = torch.tensor(beta_graph, dtype=torch.float)
            gamma_graph = torch.tensor(gamma_graph, dtype=torch.float)



    elif args.data=='FACED':
        print("--------- FACED DATA ---------\n")
        if args.freq == 'BH':
            # base_x = torch.load('/home/sjf/eegall/data/FACED/all_nwrebase_de_features.pt')
            # harm_x = torch.load('/home/sjf/eegall/data/FACED/all_nwreharmon_de_features.pt')
            # all_labels = torch.load('/home/sjf/eegall/data/FACED/all_nwrelabels.pt')
            # base_graph = torch.load('/home/sjf/eegall/data/FACED/nwrebase_graph.pt')
            # harm_graph = torch.load('/home/sjf/eegall/data/FACED/nwreharm_graph.pt')
            base_x = torch.load('/home/sjf/eegall/data/all_8nwrebase_de_features.pt')
            harm_x = torch.load('/home/sjf/eegall/data/all_8nwreharmon_de_features.pt')
            all_labels = torch.load('/home/sjf/eegall/data/all_8nwrelabels.pt')
            base_graph = torch.load('/home/sjf/eegall/data/8nwrebase_graph.pt')
            harm_graph = torch.load('/home/sjf/eegall/data/8nwreharm_graph.pt')

            base_graph = torch.tensor(base_graph, dtype=torch.float)
            harm_graph = torch.tensor(harm_graph, dtype=torch.float)



        else:
            all_features = torch.load('/home/sjf/eegall/data/FACED/all_fre_features.pt')
            delta_features = all_features[:,0,:,:,:]
            theta_features = all_features[:,1,:,:,:]
            alpha_features = all_features[:,2,:,:,:]
            beta_features = all_features[:,3,:,:,:]
            gamma_features = all_features[:,4,:,:,:]
            all_labels = torch.load('/home/sjf/eegall/data/FACED/all_fre_facedlabels.pt')
            delta_graph = torch.load('/home/sjf/eegall/data/FACED/delta_graph.pt')
            theta_graph = torch.load('/home/sjf/eegall/data/FACED/theta_graph.pt')
            alpha_graph = torch.load('/home/sjf/eegall/data/FACED/alpha_graph.pt')
            beta_graph = torch.load('/home/sjf/eegall/data/FACED/beta_graph.pt')
            gamma_graph = torch.load('/home/sjf/eegall/data/FACED/gamma_graph.pt')
            delta_graph = torch.tensor(delta_graph, dtype=torch.float)
            theta_graph = torch.tensor(theta_graph, dtype=torch.float)
            alpha_graph = torch.tensor(alpha_graph, dtype=torch.float)
            beta_graph = torch.tensor(beta_graph, dtype=torch.float)
            gamma_graph = torch.tensor(gamma_graph, dtype=torch.float)


    # data info
    print("*********** ALL Loaded data ***************\n")
    if args.freq == 'BH':
        print(f"base_x: {base_x.shape}, harm_x: {harm_x.shape} all_labels: {all_labels.shape} \n base_graph: {base_graph.shape} harm_graph: {harm_graph.shape}\n")
    else:
        print(f"all_features: {all_features.shape}, delta_graph: {delta_graph.shape} all_labels: {all_labels.shape}")


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    all_max_acc = []
    all_avg_acc = []
    all_max_fs = []
    all_avg_fs = []
    if args.limit:
        print("this is with limit version")
    else:
        print("this is without limit version")
    if args.graph != 'harm':
        print("Uing base graph and base feature for Time Graph part and harm feature for encoding!")
    if args.freq == 'BH':
        num_subjects = base_x.shape[0]
    else:
        num_subjects = delta_features.shape[0]

    # Fou = create_fourier_matrix(args.batch)
     # Model define
    model_dir = '/home/sjf/eegall/intermodel/'
    if args.norm:
        model_path = "norm-"+"lr"+str(args.lr)+args.data+str(args.kfold)+'_'+str(args.batch)+'_'+str(args.maxiter) + '_' +args.loss + args.graph + 'seed'+str(args.seed)+'fold_best_model.pth'
    else:
        model_path = "lr"+str(args.lr)+args.data+str(args.kfold)+'_'+str(args.batch)+'_'+str(args.maxiter) + '_' +args.loss + args.graph + 'seed'+str(args.seed)+'fold_best_model.pth'
    
    if args.tylabel == 'Valence':
        model_path = "Valence-"+model_path
    elif args.tylabel == 'Arousal':
        model_path = "Arousal-"+model_path
    
    if args.ablation:
        model_path = 'Ablation'+model_path

    if args.freq == 'BH':
        pass
    else:
        model_path = 'CF'+model_path
    if args.modeldir:
        model_dir = model_dir+'modelsave'+str(args.modeldir)+'/'
        if os.path.exists(model_dir):
            pass
        else:
            os.makedirs(model_dir)
    model_path = model_dir + model_path
    model_path = model_dir + model_path
    print(f"before val model path:{model_path}")
    subject_model = []
    begin_num = args.ckpoint
    for i in range(begin_num,num_subjects):
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"subject_{i-1}'s fold model has been deleted.")

        print(f"******** mix subject_{i} ********\n")
        if args.freq == 'BH':
            base_data = base_x[i,:,:32,:]
            harm_data = harm_x[i,:,:32,:]
            # print(f"base_data : {base_data}")
            if args.data == 'DEAP':
                if args.tylabel == 'Valence':
                    use_label = torch.tensor(all_labels[i,:,0],dtype=int)
                elif args.tylabel == 'Arousal':
                    use_label = torch.tensor(all_labels[i,:,1],dtype=int)
            elif args.data == 'FACED':
                if args.tylabel == 'Valence':
                    use_label = torch.tensor(all_labels[i,:,0],dtype=int)
                elif args.tylabel == 'Arousal':
                    use_label = torch.tensor(all_labels[i,:,1],dtype=int)

            if args.graph == 'harm':
                use_graph = harm_graph[i,:,:32,:32]
                train_models, max_acc, avg_acc, avg_f_score, max_f_score = cross_validation(args, i, base_data, harm_data, use_graph, use_label,seed=76, device=device)
            else:
                use_graph = base_graph[i,:,:32,:32]
                train_models, max_acc, avg_acc, avg_f_score, max_f_score = cross_validation(args, i, harm_data, base_data, use_graph, use_label,seed=76, device=device)
        else:
            delta_data = delta_features[i,:,:32,:]
            theta_data = theta_features[i,:,:32,:]
            alpha_data = alpha_features[i,:,:32,:]
            beta_data = beta_features[i,:,:32,:]
            gamma_data = gamma_features[i,:,:32,:]
            if args.data == 'DEAP':
                if args.tylabel == 'Valence':
                    use_label = torch.tensor(all_labels[i,:,0],dtype=int)
                elif args.tylabel == 'Arousal':
                    use_label = torch.tensor(all_labels[i,:,1],dtype=int)
            elif args.data == 'FACED':
                if args.tylabel == 'Valence':
                    use_label = torch.tensor(all_labels[i,:,0],dtype=int)
                elif args.tylabel == 'Arousal':
                    use_label = torch.tensor(all_labels[i,:,1],dtype=int)
            if args.freph == 'delta':
                g_data = delta_data
                ng1_data = theta_data
                ng2_data = alpha_data
                ng3_data = beta_data
                ng4_data = gamma_data
                use_graph = delta_graph[i,:,:32,:32]
                train_models, max_acc, avg_acc, avg_f_score, max_f_score = freq_cross_validation(args, i, ng1_data, ng2_data,ng3_data,ng4_data,g_data,use_graph,use_label,seed=76, device=device)
            elif args.freph == 'theta':
                g_data = theta_data
                ng1_data = delta_data
                ng2_data = alpha_data
                ng3_data = beta_data
                ng4_data = gamma_data
                use_graph = theta_graph[i,:,:32,:32]
                train_models, max_acc, avg_acc, avg_f_score, max_f_score = freq_cross_validation(args, i, ng1_data, ng2_data,ng3_data,ng4_data,g_data,use_graph,use_label,seed=76, device=device)
            if args.freph == 'alpha':
                g_data = alpha_data
                ng1_data = theta_data
                ng2_data = delta_data
                ng3_data = beta_data
                ng4_data = gamma_data
                use_graph = alpha_graph[i,:,:32,:32]
                train_models, max_acc, avg_acc, avg_f_score, max_f_score = freq_cross_validation(args, i, ng1_data, ng2_data,ng3_data,ng4_data,g_data,use_graph,use_label,seed=76, device=device)
            if args.freph == 'beta':
                g_data = beta_data
                ng1_data = theta_data
                ng2_data = alpha_data
                ng3_data = delta_data
                ng4_data = gamma_data
                use_graph = beta_graph[i,:,:32,:32]
                train_models, max_acc, avg_acc, avg_f_score, max_f_score = freq_cross_validation(args, i, ng1_data, ng2_data,ng3_data,ng4_data,g_data,use_graph,use_label,seed=76, device=device)
            if args.freph == 'gamma':
                g_data = gamma_data
                ng1_data = theta_data
                ng2_data = alpha_data
                ng3_data = beta_data
                ng4_data = delta_data
                use_graph = gamma_graph[i,:,:32,:32]
                train_models, max_acc, avg_acc, avg_f_score, max_f_score = freq_cross_validation(args, i, ng1_data, ng2_data,ng3_data,ng4_data,g_data,use_graph,use_label,seed=76, device=device)

        all_max_acc.append(max_acc)
        all_avg_acc.append(avg_acc)
        all_max_fs.append(max_f_score)
        all_avg_fs.append(avg_f_score)
        subject_model.append(train_models)

    print(f"all subject avg max acc:{sum(all_max_acc) / len(all_max_acc)}\n all subject avg avg acc:{sum(all_avg_acc)/len(all_avg_acc)}\n avg max fs:{sum(all_max_fs)/len(all_max_fs)}\n avg avg fs:{sum(all_avg_fs)/len(all_avg_fs)}")
    if args.norm:
        filename1 = 'norm-'+'lr'+str(args.lr)+args.data+str(args.kfold)+'_'+str(args.batch)+'_'+str(args.maxiter) + '_' +args.loss + args.graph + 'seed'+str(args.seed)+'_results.txt'
    else:
        filename1 = 'lr'+str(args.lr)+args.data+str(args.kfold)+'_'+str(args.batch)+'_'+str(args.maxiter) + '_' +args.loss + args.graph + 'seed'+str(args.seed)+'_results.txt'
    
    if args.tylabel == 'Valence':
        filename1 = "Valence-"+filename1
    elif args.tylabel == 'Arousal':
        filename1 = "Arousal-"+filename1
    
    if args.ablation:
        filename1 = 'Ablation-'+filename1
        
    if args.abtype == 'WTG':
        filename1 = 'WTG'+filename1
    elif args.abtype == 'NTG':
        filename1 = 'NTG'+filename1

    if args.freq == 'BH':
        pass
    else:
        filename1 = "CF"+filename1
    if args.modeldir:
        filename1 = str(args.modeldir)+filename1
    # filename1 = str(args.freph) + filename1
    if args.limit:
        filepath = '/home/sjf/eegall/withlimits/'+filename1
    else:
        filepath = '/home/sjf/eegall/nolimits/'+filename1

    with open(filepath, 'a') as f:
        if len(all_max_acc) == 0:
            f.write(f"exist zero predictions!!!")

        else:
            f.write(f"all subject avg max acc:{sum(all_max_acc) / len(all_max_acc)}\n all subject avg avg acc:{sum(all_avg_acc)/len(all_avg_acc)}\n avg max fs:{sum(all_max_fs)/len(all_max_fs)}\n avg avg fs:{sum(all_avg_fs)/len(all_avg_fs)}")
        # f.write(f'Max acc:{max_acc}, Max f score:{max_f_score}\n')
    
    if args.limit:
        directory = '/home/sjf/eegall/withlimits/'
    else:
        directory = '/home/sjf/eegall/nolimits/'

    if args.data == 'DEAP':
        filename = args.data+str(args.kfold)+'_'+str(args.batch)+'_'+str(args.maxiter) + '_' +args.loss + args.graph+'seed'+str(args.seed) + '_trained_models.pt'
    elif args.data == 'FACED':
        filename = args.data+str(args.kfold)+'_'+str(args.batch)+'_'+str(args.maxiter) + '_' +args.loss + args.graph+'seed'+str(args.seed) + '_trained_models.pt'
    if args.ablation:
        filename = 'Ablation-'+filename
    if args.abtype == 'WTG':
        filename = 'WTG'+filename
    elif args.abtype=='NTG':
        filename = 'NTG'+filename
    if args.tylabel == 'Valence':
        filename = "Valence-"+filename
    elif args.tylabel == 'Arousal':
        filename = "Arousal-"+filename
    filename = str(args.freph)+filename
    directory = directory + args.data

    filepath = os.path.join(directory,filename)
    torch.save(subject_model,filepath)


