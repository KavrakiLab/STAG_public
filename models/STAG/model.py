import warnings
warnings.filterwarnings('ignore')

from typing import Callable, Tuple, Union
import math
import torch
from torch import Tensor
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import softmax

class EVTConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, nn: Callable, heads: int = 1, concat: bool = False,
                  beta: bool = False, dropout: float = 0.,
                  edge_dim: int = None, aggr: str = 'add',
                 root_weight: bool = True, bias: bool = True, **kwargs):
        super().__init__(aggr=aggr, node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None
        self.nn = nn
        self.root_weight = root_weight
        self.beta = beta and root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
#         self.lin_value = Linear(in_channels[0], heads * out_channels)

        self.in_channels_l = in_channels[0]

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.nn)
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
#         self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:

        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
#         value = self.lin_value(x[0]).view(-1, H, C)
        value = x[0]

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, query=query, key=key, value=value, edge_attr=edge_attr, size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r

        return out


    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: Tensor, index: Tensor, ptr: OptTensor) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr1 = self.lin_edge(edge_attr)

            key_j = key_j + edge_attr1.view(-1, self.heads,self.out_channels)

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = value_j

        weight = self.nn(edge_attr)
        weight = weight.view(-1, self.in_channels_l, self.out_channels)
        out = torch.matmul(out.unsqueeze(1), weight).squeeze(1)
        out = out.unsqueeze(1) * alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr}, nn={self.nn},  heads={self.heads})')

class STAG(nn.Module):

    def __init__(self):
        self.node_feats_in = 19
        self.edge_feats_in = 19
        self.global_feats_in = None
        super(STAG,self).__init__()

        # node MLP1
        self.node_mlp1_lin1 = nn.Linear(self.node_feats_in,32)
        self.node_norm = gnn.norm.LayerNorm(32)
        self.node_mlp1_lin2 = nn.Linear(32,32)
        # edge MLP1
        self.edge_mlp1_lin1 = nn.Linear((32*2)+self.edge_feats_in,16)
        self.edge_norm = nn.LayerNorm(16)
        self.edge_mlp1_lin2 = nn.Linear(16,16)
        # GCN Conv block 1
        self.block1_conv1 = EVTConv(32,32,nn = nn.Sequential(nn.Linear(16,16),nn.LayerNorm(16),nn.LeakyReLU(),nn.Dropout(0.125),nn.Linear(16,32*32)),heads=1,dropout=0.125)
        self.conv1_norm = gnn.norm.LayerNorm(32)
        self.block1_conv2 = EVTConv(32,32,nn = nn.Sequential(nn.Linear(16,16),nn.LayerNorm(16),nn.LeakyReLU(),nn.Dropout(0.125),nn.Linear(16,32*32)),heads=1,dropout=0.125)
        self.conv2_norm = gnn.norm.LayerNorm(32)
        self.block1_conv3 = EVTConv(32,32,nn = nn.Sequential(nn.Linear(16,16),nn.LayerNorm(16),nn.LeakyReLU(),nn.Dropout(0.125),nn.Linear(16,32*32)),heads=1,dropout=0.125)
        self.conv3_norm = gnn.norm.LayerNorm(32)
        # global MLP 2
        self.global_mlp2_lin1 = nn.Linear(32,32)
        self.global_norm = nn.LayerNorm(32)
        self.global_mlp2_lin2 = nn.Linear(32,1)

        self.explaining = False

    def forward(self,x,edge_index,edge_attr,batch):
        row, col = edge_index
        # node MLP1
        x = self.node_mlp1_lin1(x)
        x = self.node_norm(x,batch)
        x = F.leaky_relu(x)
        x = F.dropout(x,p=0.125,training=self.training)
        x = self.node_mlp1_lin2(x)
        # edge MLP1
        edge_attr = torch.cat([x[row], x[col], edge_attr], dim=-1)
        edge_attr = self.edge_mlp1_lin1(edge_attr)
        edge_attr = self.edge_norm(edge_attr)
        edge_attr = F.leaky_relu(edge_attr)
        edge_attr = F.dropout(edge_attr,p=0.125,training=self.training)
        edge_attr = self.edge_mlp1_lin2(edge_attr)
        # GCN Conv block 1
        x = self.block1_conv1(x,edge_index,edge_attr)
        x = self.conv1_norm(x,batch)
        x = F.leaky_relu(x)
        x = self.block1_conv2(x,edge_index,edge_attr)
        x = self.conv2_norm(x,batch)
        x = F.leaky_relu(x)
        x = self.block1_conv3(x,edge_index,edge_attr)
        x = self.conv3_norm(x,batch)
        #global MLP 2
        global_attr = gnn.global_mean_pool(x,batch)
        global_attr = self.global_mlp2_lin1(global_attr)
        global_attr = self.global_norm(global_attr)
        global_attr = F.leaky_relu(global_attr)
        global_attr = F.dropout(global_attr,p=0.125,training=self.training)
        global_attr = self.global_mlp2_lin2(global_attr)

        if self.explaining:
            return global_attr

        return global_attr.squeeze()
