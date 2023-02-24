from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch_geometric.data import Data
from utils import load_data, accuracy, get_ids, create_affinity_graph_from_scores, get_networks, get_subject_score, read_data, site_percentage, get_train_test_masks,feature_selection, l2_regularization
from models import GAT, SpGAT
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from RE import Regularization
from scipy.spatial import distance
import igraph as ig

# Training settings
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')  # 0.0001
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 loss on parameters).')  # 1e-4
    parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
    parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha for the leaky_relu.') # 0.2
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    parser.add_argument('--connectivity', default='correlation', help='Type of connectivity used for network '
                                                                      'construction 用于网络构建的连接类型(default: correlation, '
                                                                      'options: correlation, partial correlation, '
                                                                      'tangent)')
    parser.add_argument('--atlas', default='ho',
                        help='atlas for network construction (node definition) 网络构建图集(节点定义) (default: ho, '
                             'see preprocessed-connectomes-project.org/abide/Pipelines.html '
                             'for more options )')
    parser.add_argument('--num_features', default=2100, type=int, help='Number of features to keep for '
                                                                           'the feature selection step  为特征选择步骤保留的特征数量(default: 2000)')  # 2100
    parser.add_argument('--num_training', default=0.9, type=float, help='Percentage of training set used for ' 
                                                                        'training (default: 1.0)')  # 0.9
    parser.add_argument('--folds', default=0, type=int, help='For cross validation, specifies which fold will be '
                                                             'used. All folds are used if set to 11 (default: 11)')

    args = parser.parse_args()
    params = dict()
    params['num_features'] = args.num_features
    params['num_training'] = args.num_training
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    atlas = args.atlas
    connectivity = args.connectivity


    lr = 1e-4  # 1e-4
    hid_c = 12
    n_epoch = 2800 # 2500 2800
    seed = 20
    q = 0.19  # 0.2 0.5 0.19
    #################################
    n_class = 2

    # start_time = time.time()

    np.random.seed(seed)  # for numpy
    random.seed(seed)
    torch.manual_seed(seed)  # for cpu/GPU
    torch.cuda.manual_seed(seed)  # for current GPU
    torch.cuda.manual_seed_all(seed)  # for all GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    subject_IDs = get_ids()
    labels = get_subject_score(subject_IDs, score='DX_GROUP')

    sites = get_subject_score(subject_IDs, score='SITE_ID')
    unique = np.unique(list(sites.values())).tolist()

    num_classes = n_class
    num_nodes = len(subject_IDs)
    y_data = np.zeros([num_nodes, num_classes])
    y = np.zeros([num_nodes, 1])
    site = np.zeros([num_nodes, 1], dtype=np.int)

    for i in range(num_nodes):
        y_data[i, int(labels[subject_IDs[i]]) - 1] = 1
        y[i] = int(labels[subject_IDs[i]])
        site[i] = unique.index(sites[subject_IDs[i]])

    features = get_networks(subject_IDs, kind=connectivity, atlas_name=atlas)
    print('features11:', features)

    # train/test
    skf = StratifiedKFold(n_splits=10)
    cv_splits = list(skf.split(features, np.squeeze(y)))

    train_index = cv_splits[args.folds][0]
    test_index = cv_splits[args.folds][1]
    eval_index = cv_splits[args.folds][1]

    def sample_mask(idx, l):
        """Create mask."""
        mask = np.zeros(l)
        mask[idx] = 1
        return np.array(mask, dtype=np.bool)


    labeled_ind = site_percentage(train_index, params['num_training'], subject_IDs)

    features = feature_selection(features, y, labeled_ind, params['num_features'])

    distv = distance.pdist(features, metric='correlation')
    dist = distance.squareform(distv)
    sigma = np.mean(dist)
    sparse_graph = np.exp(- dist ** 2 / (2 * sigma ** 2))
    adj = create_affinity_graph_from_scores(['SEX', 'SITE_ID'], subject_IDs)
    adj = torch.tensor(adj)
    # adjj = adj * sparse_graph
    adj1 = create_affinity_graph_from_scores(['SEX'], subject_IDs)
    adj1 = torch.tensor(adj1)
    # adjj1 = adj1 * sparse_graph
    adj2 = create_affinity_graph_from_scores(['SITE_ID'], subject_IDs)
    adj2 = torch.tensor(adj2)
    # adjj2 = adj2 * sparse_graph

    a = []
    b = []
    c = []
    for i in range(len(adj)):
        for j in range(len(adj)):
            if adj[i][j] != 0:
                a.append(i)
                b.append(j)
                c.append(adj[i][j])
    edge_index = torch.tensor([a, b], dtype=torch.long)
    edge_attr = torch.tensor(c, dtype=torch.float)
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(y - 1, dtype=int)
    print(f'y: {y}')
    y = np.squeeze(y)
    print('edge_index:', edge_index)






    data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)

    print(f'edge_attr: {data.edge_attr}')
    print(f'x: {data.x}')
    edge = []
    edge_attr = data.edge_attr.tolist()
    for i in range(data.num_edges):
        e = []
        e.append(edge_attr[i])
        edge.append(e)
    edge = torch.tensor(edge, dtype=torch.float)
    print(f'edge: {edge}')  # 边权

    data.num_classes = num_classes
    data.train_mask = sample_mask(train_index, data.num_nodes)
    data.test_mask = sample_mask(test_index, data.num_nodes)
    data.eval_mask = sample_mask(eval_index, data.num_nodes)
    train_mask = sample_mask(train_index, data.num_nodes)
    test_mask = sample_mask(test_index, data.num_nodes)
    eval_mask = sample_mask(eval_index, data.num_nodes)
    data.features = torch.tensor(features, dtype=torch.float64)
    data.adj = torch.tensor(adj, dtype=torch.float64)


    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'y: {data.y}')  # y
    print(f'edge_index: {data.edge_index}')  # edge_index
    print(f'edge_index[0]: {data.edge_index[0]}')
    print(f'labels: {labels}')  #
    print(f'data.num_classes: {data.num_classes}')
    print(f'Number of node features: {data.num_node_features}')
    print(f'Number of edge features: {data.num_edge_features}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'if edge indices are ordered and do not contain duplicate entries.: {data.is_coalesced()}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(f'Number of testing nodes: {data.test_mask.sum()}')  # test
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
    print(f'Contains self-loops: {data.contains_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

    # my_net = GAT(in_c=data.num_node_features, hid_c=hid_c, out_c=n_class)

    #################################################################

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # my_net = my_net.to(device)
    # data = data.to(device)
    #
    # optimizer = torch.optim.Adam(my_net.parameters(), lr=lr)
    if args.sparse:  # 稀疏算法
        my_net = SpGAT(nfeat=args.num_features,
                      nhid=args.hidden,
                      nclass=int(labels.max()) + 1,  # 7
                      dropout=args.dropout,
                      nheads=args.nb_heads,
                      alpha=args.alpha)
    else:
        my_net = GAT(nfeat=args.num_features,
                    nhid=args.hidden,
                    nclass=n_class,
                    # nclass=int(labels.max()) + 1,
                    dropout=args.dropout,
                    nheads=args.nb_heads,
                    alpha=args.alpha)
    print('model', list(my_net.named_parameters()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    my_net = my_net.to(device)
    data = data.to(device)
    if args.weight_decay > 0:
        reg_loss = Regularization(my_net, args.weight_decay, p=2).to(device)
    else:
        print("no regularization")
    optimizer = optim.Adam(my_net.parameters(),
                           lr=lr, weight_decay=args.weight_decay
                           )

    # tmp = filter(lambda x: x.requires_grad, list(my_net.parameters()))
    # num = sum(map(lambda x: np.prod(x.shape), tmp))
    # print(my_net)
    # print('Total trainable tensors:', num)
    my_net.train()
    for epoch in range(n_epoch):
        optimizer.zero_grad()

        # output = my_net(x, adj)
        output = my_net(x, adj, adj1, adj2)
        loss = F.nll_loss(output[train_mask], y[train_mask])
        flood = (loss - q).abs() + q
        # if args.weight_decay > 0:
        #     loss = loss + reg_loss(my_net)
        acc_train = accuracy(output[train_mask], y[train_mask])
        flood.backward()
        # l2_regularization(my_net, 0.01)
        optimizer.step()
        # if not args.fastmode:
        # Evaluate validation set performance separately, 单独评估验证集的性能
        # deactivates dropout during validation run. 在验证运行期间停用dropout
        if not args.fastmode:
            my_net.eval()
            # output = model(features, adj)
            output = my_net(x,adj, adj1, adj2)

        loss1 = F.nll_loss(output[eval_mask], y[eval_mask])
        # if args.weight_decay > 0:
        #     loss1 = loss1 + reg_loss(my_net)
        acc_val = accuracy(output[eval_mask], y[eval_mask])
        #
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss.data.item()),
              'acc_train: {:.4f}'.format(acc_train.data.item()),
              'loss_val: {:.4f}'.format(loss1.data.item()),
              'acc_val: {:.4f}'.format(acc_val.data.item())
              )
        # print("Epoch", epoch + 1, "Loss", loss.item(),
        #       'acc_train: {:.4f}'.format(acc_train.data.item())
        #       )

    my_net.eval()
    _, prediction = my_net(x,adj, adj1, adj2).max(dim=1)

    targ = y
    print(f'prediction[data.test_mask]p: {prediction[test_mask]}')
    print(f'targ[data.test_mask]: {targ[test_mask]}')

    test_correct = prediction[test_mask].eq(targ[test_mask]).sum().item()
    test_num = test_mask.sum().item()

    print("Accuracy of Test Samples: ", test_correct / test_num)

    y_test = targ[test_mask].data.cpu().numpy()
    y_pred = prediction[test_mask].data.cpu().numpy()
    auc_score = roc_auc_score(y_test, y_pred)
    print("AUC of Test Samples: ", auc_score)


if __name__ == '__main__':
    main()