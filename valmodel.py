from model import MyMixture
import matplotlib.pyplot as plt
import os

from torch.utils.data import Dataset, DataLoader

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

    return acc, precision, recall, f_score
#origin
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
        return selected_data, selected_harm_data, selected_graph, selected_label

def sliding_window_inference(page, base_val_data, harm_val_data, val_graph,val_labels, batch_size):
    batch_size = len(base_val_data) // page
    return base_val_data, harm_val_data, val_graph,val_labels, batch_size
def contains_non_int(arr):
    return any(not isinstance(x, (int, np.integer)) for x in arr)
# Cross_validation
def cross_validation(subject,base_data, harm_data, graph, labels, seed, device):
    splits = KFold(n_splits=10, shuffle=True, random_state=seed)
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
    # DEAP only
    # base_data, harm_data, graph, labels = binary_sampling(base_data, harm_data, graph, labels)

    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(base_data)))):
        print(f'******fold {fold+1}******\n')
        # data preparation for learning
        base_train_data = []
        for i in train_idx:
            base_train_data.append(base_data[i])
        # train_idx = train_idx.astype(int)
        # print(f"train_idx:{train_idx} type train_idx{type(train_idx)} len graph:{graph.shape}")
        # print(contains_non_int(train_idx)) 
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
        model_path = '/eegall/withlimits/FACED/FACED10_13_3000_scebaseseed74_trained_models.pt'
        model = MyMixture(input_dim,output_dim,input_channel,output_channel,input_size,hidden_size,in_channels, out_channels,K)
        model.load_state_dict(torch.load(model_path)[0][0])

        # metric
        base_val_data,harm_val_data, val_graph,val_labels, batch_size = sliding_window_inference(6,base_val_data, harm_val_data, val_graph,val_labels, batch_size=13)

        acc, precision, recall, f_score = validate(model, device, base_val_data,harm_val_data, val_graph,val_labels,batch_size)
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
    print(f'subject {subject} Avgacc: {avg_acc} Avgfscore: {avg_f_score} \n Max acc:{max_acc}, Max f score:{max_f_score}')
    # # 将结果保存为txt文件
    # if args.norm:
    #     filename = "norm-"+"lr"+str(args.lr)+args.data+str(args.kfold)+'_'+str(args.batch)+'_'+str(args.maxiter) + '_' +args.loss + args.graph + 'seed'+str(args.seed)+'_results.txt'
    # else:
    #     filename = "lr"+str(args.lr)+args.data+str(args.kfold)+'_'+str(args.batch)+'_'+str(args.maxiter) + '_' +args.loss + args.graph + 'seed'+str(args.seed)+'_results.txt'
    # if args.ablation:
    #     filename = 'Ablation-'+filename
    # if args.abtype == 'WTG':
    #     filename = 'WTG'+filename
    #     # print('ok')
    # elif args.abtype == 'NTG':
    #     filename = 'NTG'+filename
    # if args.limit:
    #     filepath = '/eegall/withlimits/'+filename
    # else:
    #     filepath = '/eegall/nolimits/'+filename
    # # all loss record
    # loss_path = '/eegall/lossre/'
    # if args.data == 'DEAP':
    #     loss_path = loss_path + 'DEAPloss.pt'
    # elif args.data == 'FACED':
    #     loss_path = loss_path + 'FACEDloss.pt'
    
    # torch.save(all_loss,loss_path)
    

    # with open(filepath, 'a') as f:
    #     f.write(f'\nsubject {subject} Avgacc: {avg_acc} Avgfscore: {avg_f_score}\n')
    #     f.write(f'Max acc:{max_acc}, Max f score:{max_f_score}\n')

    return

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
    # if args.data == 'DEAP':
    #     print("--------- DEAP DATA ---------\n")
    #     # base_x = torch.load('/eegall/data/DEAP/all_base1_de_features.pt').float()
    #     # harm_x = torch.load('/eegall/data/DEAP/all_harmon1_de_features.pt').float()
    #     # all_labels = torch.load('/eegall/data/DEAP/all_1labels.pt')
    #     # base_graph = torch.load('/eegall/data/DEAP/base1_graph.pt')
    #     # harm_graph = torch.load('/eegall/data/DEAP/harm1_graph.pt')

    #     # base_graph = torch.tensor(base_graph, dtype=torch.float)
    #     # harm_graph = torch.tensor(harm_graph, dtype=torch.float)
    #     base_x = torch.load('/eegall/data/DEAP/all_base1_de_features.pt').float()
    #     harm_x = torch.load('/eegall/data/DEAP/all_harmon1_de_features.pt').float()
    #     all_labels = torch.load('/eegall/data/DEAP/all_1labels.pt')
    #     base_graph = torch.load('/eegall/data/DEAP/base1_graph.pt')
    #     harm_graph = torch.load('/eegall/data/DEAP/harm1_graph.pt')

    #     # 10 folds some can achieve 98, but avg 96, this is potential to be a good choice
    #     # base_x = torch.load('/eegall/data/DEAP/all_base1_de_features.pt').float()
    #     # harm_x = torch.load('/eegall/data/DEAP/all_harmon1_de_features.pt').float()
    #     # all_labels = torch.load('/eegall/data/DEAP/all_1labels.pt')
    #     # base_graph = torch.load('/eegall/data/DEAP/base1_graph.pt')
    #     # harm_graph = torch.load('/eegall/data/DEAP/harm1_graph.pt')
    #     # if args.norm:
    #     #     base_x, harm_x = scaled_eeg(base_x, harm_x)
    #     # else:
    #     #     pass


    #     base_graph = torch.tensor(base_graph, dtype=torch.float)
    #     harm_graph = torch.tensor(harm_graph, dtype=torch.float)


        
    # elif args.data=='FACED':
    print("--------- FACED DATA ---------\n")
    # base_x = torch.load('/eegall/data/all_base3_de_features.pt')
    # harm_x = torch.load('/eegall/data/all_harmon3_de_features.pt')
    # # base_x, harm_x = scaled_eeg(base_x, harm_x)
    # all_labels = torch.load('/eegall/data/all_3labels.pt')
    # base_graph = torch.load('/eegall/data/base3_graph.pt')
    # harm_graph = torch.load('/eegall/data/harm3_graph.pt')
    base_x = torch.load('/eegall/data/FACED/all_faced_rebase_de_features.pt')
    harm_x = torch.load('/eegall/data/FACED/all_faced_reharmon_de_features.pt')
    # base_x, harm_x = scaled_eeg(base_x, harm_x)
    all_labels = torch.load('/eegall/data/FACED/all_faced_relabels.pt')
    base_graph = torch.load('/eegall/data/FACED/faced_rebase_graph.pt')
    harm_graph = torch.load('/eegall/data/FACED/faced_reharm_graph.pt')

    base_graph = torch.tensor(base_graph, dtype=torch.float)
    harm_graph = torch.tensor(harm_graph, dtype=torch.float)


    # data info
    print("*********** ALL Loaded data ***************\n")
    print(f"base_x: {base_x.shape}, harm_x: {harm_x.shape} all_labels: {all_labels.shape} \n base_graph: {base_graph.shape} harm_graph: {harm_graph.shape}\n")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    all_max_acc = []
    all_avg_acc = []
    all_max_fs = []
    all_avg_fs = []
    # if args.limit:
    #     print("this is with limit version")
    # else:
    #     print("this is without limit version")
    # if args.graph != 'harm':
    #     print("Uing base graph and base feature for Time Graph part and harm feature for encoding!")
    
    num_subjects = base_x.shape[0]

    # Fou = create_fourier_matrix(args.batch)
    # model_dir = '/eegall/intermodel/'
    # if args.norm:
    #     model_path = "norm"+str(args.norm)+"lr"+str(args.lr)+args.data+str(args.kfold)+'_'+str(args.batch)+'_'+str(args.maxiter) + '_' +args.loss + args.graph + 'seed'+str(args.seed)+'fold_best_model.pth'
    # else:
    #     model_path = "lr"+str(args.lr)+args.data+str(args.kfold)+'_'+str(args.batch)+'_'+str(args.maxiter) + '_' +args.loss + args.graph + 'seed'+str(args.seed)+'fold_best_model.pth'

    # if args.ablation:
    #     model_path = 'Ablation'+model_path
    # if args.abtype == 'WTG':
    #     model_path = 'WTG'+model_path
    #     # print('ok')
    # elif args.abtype == 'NTG':
    #     model_path = 'NTG'+model_path
    
    # model_path = model_dir + model_path
    
    for i in range(num_subjects):
        # if os.path.exists(model_path):
        #     os.remove(model_path)
        #     print(f"subject_{i-1}'s fold model has been deleted.")

        print(f"******** mix subject_{i} ********\n")
        base_data = base_x[i,:,:32,:]
        harm_data = harm_x[i,:,:32,:]
        # if args.data == 'DEAP':
        #     use_label = torch.tensor(all_labels[i,:,0],dtype=int)
        # elif args.data == 'FACED':
        use_label = torch.tensor(all_labels[i,:],dtype=int)

        # if args.graph == 'harm':
        #     use_graph = harm_graph[i,:,:32,:32]
        #     train_models, max_acc, avg_acc, avg_f_score, max_f_score = cross_validation(args, i, base_data,harm_data, use_graph, use_label,seed=76, device=device)
        # else:
        use_graph = base_graph[i,:,:32,:32]
        train_models, max_acc, avg_acc, avg_f_score, max_f_score = cross_validation(i, harm_data,base_data, use_graph, use_label,seed=76, device=device)
        all_max_acc.append(max_acc)
        all_avg_acc.append(avg_acc)
        all_max_fs.append(max_f_score)
        all_avg_fs.append(avg_f_score)
    print(f"all subject avg max acc:{sum(all_max_acc) / len(all_max_acc)}\n all subject avg avg acc:{sum(all_avg_acc)/len(all_avg_acc)}\n avg max fs:{sum(all_max_fs)/len(all_max_fs)}\n avg avg fs:{sum(all_avg_fs)/len(all_avg_fs)}")
    # if args.norm:
    #     filename1 = 'norm-'+'lr'+str(args.lr)+args.data+str(args.kfold)+'_'+str(args.batch)+'_'+str(args.maxiter) + '_' +args.loss + args.graph + 'seed'+str(args.seed)+'_results.txt'
    # else:
    #     filename1 = 'lr'+str(args.lr)+args.data+str(args.kfold)+'_'+str(args.batch)+'_'+str(args.maxiter) + '_' +args.loss + args.graph + 'seed'+str(args.seed)+'_results.txt'
    # if args.ablation:
    #     filename1 = 'Ablation-'+filename1
    # if args.abtype == 'WTG':
    #     filename1 = 'WTG'+filename1
    # elif args.abtype == 'NTG':
    #     filename1 = 'NTG'+filename1
    # if args.limit:
    #     filepath = '/eegall/withlimits/'+filename1
    # else:
    #     filepath = '/eegall/nolimits/'+filename1

    # with open(filepath, 'a') as f:
    #     f.write(f"all subject avg max acc:{sum(all_max_acc) / len(all_max_acc)}\n all subject avg avg acc:{sum(all_avg_acc)/len(all_avg_acc)}\n avg max fs:{sum(all_max_fs)/len(all_max_fs)}\n avg avg fs:{sum(all_avg_fs)/len(all_avg_fs)}")
    #     # f.write(f'Max acc:{max_acc}, Max f score:{max_f_score}\n')
    
    # if args.limit:
    #     directory = '/eegall/withlimits/'
    # else:
    #     directory = '/eegall/nolimits/'

    # if args.data == 'DEAP':
    #     filename = args.data+str(args.kfold)+'_'+str(args.batch)+'_'+str(args.maxiter) + '_' +args.loss + args.graph+'seed'+str(args.seed) + '_trained_models.pt'
    # elif args.data == 'FACED':
    #     filename = args.data+str(args.kfold)+'_'+str(args.batch)+'_'+str(args.maxiter) + '_' +args.loss + args.graph+'seed'+str(args.seed) + '_trained_models.pt'
    # if args.ablation:
    #     filename = 'Ablation-'+filename
    # if args.abtype == 'WTG':
    #     filename = 'WTG'+filename
    # elif args.abtype=='NTG':
    #     filename = 'NTG'+filename
    # directory = directory + args.data

    # filepath = os.path.join(directory,filename)
    # torch.save(train_models,filepath)


