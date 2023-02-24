import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.b = nn.Parameter(torch.empty(size=(2, 1, 1)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.conv_ = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=32, stride=2, dilation=2)
        self.relu_ = nn.ReLU(True)
        self.pool_ = nn.MaxPool2d(kernel_size=16, stride=16)
        self.softmax_ = nn.Softmax(dim=0)
        self.f_ = nn.Flatten(0)
        self.f1_ = nn.Flatten(0)
        self.f2_ = nn.Flatten(0)
        self.L_ = nn.Linear(in_features=2916, out_features=1)
        # self.L_1 = nn.Linear(in_features=871, out_features=420)

    def forward(self, h, adj, adj1, adj2):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        # print('W:', self.W)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = attention[np.newaxis]
        attention1 = torch.where(adj1 > 0, e, zero_vec)
        attention1 = F.softmax(attention1, dim=1)
        attention1 = attention1[np.newaxis]
        attention2 = torch.where(adj2 > 0, e, zero_vec)
        attention2 = F.softmax(attention2, dim=1)
        attention2 = attention2[np.newaxis]
        attention_fusion = self.fusion(attention1, attention2)
        # attention_fusion = self.fusion(attention_fusion1, attention2)
        attention_fusion = attention_fusion.squeeze(0)
        # print('att:', attention_fusion.shape)

        attention_fusion = F.dropout(attention_fusion, self.dropout, training=self.training)
        h_prime = torch.matmul(attention_fusion, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
    def fusion(self, att, att1):
        data = torch.cat((att, att1), 0)
        x = self.pool_(data)
        x1 = self.f_(x[0]).detach().numpy()  # (676,) 拉直
        x2 = self.f1_(x[1]).detach().numpy()
        x1 = torch.tensor(np.expand_dims(x1, axis=0))  # [1, 676]
        x2 = torch.tensor(np.expand_dims(x2, axis=0))
        data = torch.cat((x1, x2), 0)  # [2, 676]
        result = self.L_(data)  # [3,1]
        result = self.leakyrelu(result)
        result = self.softmax_(result)
        result1 = result[0]
        result2 = result[1]
        mul2 = torch.mul(att, result1)
        mul3 = torch.mul(att1, result2)
        attention_fusion1 = mul2 + mul3
        return attention_fusion1

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # print('a:', self.a)
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
