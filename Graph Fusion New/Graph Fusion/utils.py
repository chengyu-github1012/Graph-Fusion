import numpy as np
import scipy.sparse as sp
import torch
import os
import csv
import scipy.io as sio
import sys
from sklearn.model_selection import train_test_split
import json
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeClassifier
from torch import nn


def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot
line = 'cpac'
root_folder = 'E:\GAT-\pyGAT-master (1)\pyGAT-master\data1'
data_folder = os.path.join(root_folder, 'ABIDE_pcp/cpac/filt_noglobal')
phenotype = os.path.join(root_folder, 'ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv')
def get_ids(num_subjects=None):
    """

    return:
        subject_IDs    : list of all subject IDs
    """

    subject_IDs = np.genfromtxt(os.path.join(data_folder, 'subject_IDs.txt'), dtype=str)
    # os.path.join(data_folder, 'subject_IDs.txt') 输出 ABIDE_pcp/cpac/filt_noglobal\subject_IDs.txt

    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]

    return subject_IDs

# 获取受试者列表的表型值
def get_subject_score(subject_list, score):
    scores_dict = {}

    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['SUB_ID'] in subject_list:
                scores_dict[row['SUB_ID']] = row[score]
    # print('scores_dict:', scores_dict)
    return scores_dict

def create_affinity_graph_from_scores(scores, subject_list):
    """
        scores       : list of phenotypic information to be used to construct the affinity graph
        subject_list : list of subject IDs

    return:
        graph        : adjacency matrix of the population graph (num_subjects x num_subjects)
    """

    num_nodes = len(subject_list)
    graph = np.zeros((num_nodes, num_nodes))

    for l in scores:
        label_dict = get_subject_score(subject_list, l)

        # quantitative phenotypic scores
        if l in ['AGE_AT_SCAN', 'FIQ']:
            for k in range(num_nodes):
                for j in range(k, num_nodes):
                    try:
                        val = abs(float(label_dict[subject_list[k]]) - float(label_dict[subject_list[j]]))
                        if val < 2:
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:  # missing label
                        pass

        else:
            for k in range(num_nodes):
                for j in range(k, num_nodes):
                    if label_dict[subject_list[k]] == label_dict[subject_list[j]]:
                        graph[k, j] += 1
                        graph[j, k] += 1

    return graph

# 加载预先计算的fMRI连接网络
def get_networks(subject_list, kind, atlas_name="aal", variable='connectivity'):
    """
        subject_list : list of subject IDs
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation

        atlas_name   : name of the parcellation atlas used

        variable     : variable name in the .mat file that has been used to save the precomputed networks



    return:
        matrix      : feature matrix of connectivity networks (num_subjects x network_size)

    """

    all_networks = []
    for subject in subject_list:
        fl = os.path.join(data_folder, subject,
                          subject + "_" + atlas_name + "_" + kind + ".mat")
        # os.path.join() 是连接两个或更多的路径名组件
        matrix = sio.loadmat(fl)[variable]
        all_networks.append(matrix)
    # all_networks=np.array(all_networks)

    idx = np.triu_indices_from(all_networks[0], 1)
    norm_networks = [np.arctanh(mat) for mat in all_networks]
    vec_networks = [mat[idx] for mat in norm_networks]
    matrix = np.vstack(vec_networks)
    return matrix

def read_data(data_file,label,start,test_size,random_state):

   dataset = pd.read_csv(data_file, header=None, names=["labels"])
   # print('dataset:', dataset)
   train_data= dataset.iloc[: , (start-1):].values

   train_data = train_data.tolist()  # [[0], [0], [0], [0], [0], [0],
   with open('E:\GAT-\pyGAT-master (1)\pyGAT-master\data1\ABIDE_pcp\cpac/filt_noglobal\label1.txt', 'w') as f:
       f.write(json.dumps(train_data)) # 需要用到到json 模块  json.dumps 函数
       f.close()
   train_target = dataset.iloc[:, (label-1):label]
   # print(train_target.shape)  # (871, 1)
   train_x, test_x, train_y, test_y = train_test_split(train_data, train_target, test_size=test_size,
                                                      random_state=random_state, stratify=train_target)
   # print('train_X1:', train_x)
   return train_x, train_y, test_x, test_y, dataset

def site_percentage(train_ind, perc, subject_list):
    """
        train_ind    : indices of the training samples
        perc         : percentage of training set used (使用训练集的百分比)
        subject_list : list of subject IDs

    return:
        labeled_indices      : indices of the subset of training samples(训练样本子集的指标)
    """

    train_list = subject_list[train_ind]
    sites = get_subject_score(train_list, score='SITE_ID') #寻找在所有站点中的同时也在训练集中站点
    # print(f'sites:{sites}') 输出：sites:{'50145': 'OHSU', '50146': 'OHSU', '50147': 'OHSU',。。}
    unique = np.unique(list(sites.values())).tolist()#np.unique函数排除重复元素之后，升序排列 toist是将转换为列表形式
    site = np.array([unique.index(sites[train_list[x]]) for x in range(len(train_list))])
    # index可以查询到unique中某个字符（sites[train_list[x]]）的位置下标，np.array函数是将括号里的内容转换为数组
    labeled_indices = []

    for i in np.unique(site):
        id_in_site = np.argwhere(site == i).flatten()#np.argwhere函数返回site==i的数组下标 flatten()函数可以执行展平操作，返回一个一维数组

        num_nodes = len(id_in_site)
        labeled_num = int(round(perc * num_nodes))#round函数返回四舍五入的值 例如round（56.4567,2）返回56.45保留小数点后两位
        labeled_indices.extend(train_ind[id_in_site[:labeled_num]])
    # print(f'labeled_indices:{labeled_indices}')
    print(len(labeled_indices))
    return labeled_indices

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def get_train_test_masks(labels, idx_train,  idx_test, idx_val):
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return y_train,  y_test, y_val, train_mask, val_mask, test_mask

# 降维
def feature_selection(matrix, labels, train_ind, fnum):
    """
        matrix       : feature matrix (num_subjects x num_features)
        labels       : ground truth labels (num_subjects x 1)
        train_ind    : indices of the training samples
        fnum         : size of the feature vector after feature selection 特征选择后的特征向量的大小

    return:
        x_data      : feature matrix of lower dimension (num_subjects x fnum) 降维后的矩阵
    """

    estimator = RidgeClassifier()  # 岭回归
    selector = RFE(estimator, fnum, step=100, verbose=1)


    featureX = matrix[train_ind, :]

    featureY = labels[train_ind]
    # print(f'featureY:{featureY}')
    # print(featureY.shape)
    print('X:', featureX.shape)
    print('Y:', featureY.shape)
    selector = selector.fit(featureX, featureY.ravel())
    print(f'selector:{selector}')
    x_data = selector.transform(matrix)
    print('x_data:', x_data)

    print("Number of labeled samples %d" % len(train_ind))
    print("Number of features selected %d" % x_data.shape[1])

    return x_data

def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def l2_regularization(model, l2_alpha):
    for module in model.modules():
        if type(module) is nn.Conv2d:
            module.weight.grad.data.add_(l2_alpha * module.weight.data)