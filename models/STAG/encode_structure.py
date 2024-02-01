import torch
import torch_geometric.nn as gnn

from Bio.PDB import PDBParser

import numpy as np

from torch_geometric.utils import remove_isolated_nodes
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from .utils import *

class Node():
    def __init__(self,coord,chain,indx,res,data):
        self.coord = coord
        self.chain = chain
        self.indx = indx
        self.res = res
        self.data = data

class Graph():
    def __init__(self,nodes,label,fold,edge_attr=None):
        self.nodes = nodes
        self.edge_index = None
        self.label = label
        self.fold = fold
        self.edge_attr=edge_attr
        self.global_attr = None
        self.index = None
    def add_node(self,node):
        self.nodes.append(node)
    def set_edge_index(self,edge_index):
        self.edge_index = edge_index
    def set_edge_attr(self,edge_attr):
        self.edge_attr = edge_attr
    def set_global(self,global_attr):
        self.global_attr = global_attr
    def get_node_data(self):
        return torch.stack([node.data for node in self.nodes])
    def get_coords(self):
        return torch.Tensor(np.array([node.coord for node in self.nodes]))
    def get_P_coords(self):
        return torch.Tensor(np.array([node.coord for node in self.nodes if node.chain == 'P']))
    def get_M_coords(self):
        return torch.Tensor(np.array([node.coord for node in self.nodes if node.chain == 'M']))
    def get_A_coords(self):
        return torch.Tensor(np.array([node.coord for node in self.nodes if node.chain == 'A']))
    def get_B_coords(self):
        return torch.Tensor(np.array([node.coord for node in self.nodes if node.chain == 'B']))

def visualize_nodes3d(graph):
    pos = graph.get_coords()
    edge_index = graph.edge_index
    fig = plt.figure(figsize=(4, 4))
    ax = plt.axes(projection='3d')
    P_pos = graph.get_P_coords()
    M_pos = graph.get_M_coords()
    A_pos = graph.get_A_coords()
    B_pos = graph.get_B_coords()
    ax.scatter3D(P_pos[:, 0], P_pos[:, 1], P_pos[:,2],  s=20, zorder=1000, color='red')
    ax.scatter3D(M_pos[:, 0], M_pos[:, 1], M_pos[:,2],  s=20, zorder=1000, color='darkviolet')
    ax.scatter3D(A_pos[:, 0], A_pos[:, 1], A_pos[:,2],  s=20, zorder=1000, color='turquoise')
    ax.scatter3D(B_pos[:, 0], B_pos[:, 1], B_pos[:,2],  s=20, zorder=1000, color='lime')
    for (src, dst) in edge_index.t().tolist():
        src = pos[src].tolist()
        dst = pos[dst].tolist()
        ax.plot3D([src[0], dst[0]], [src[1], dst[1]], [src[2], dst[2]], linewidth=1, color='black',alpha=0.5)
    plt.axis('off')
    plt.show()

def get_graph(pdb,label,fold,roi_radius = 14, edge_radius = 8):
    parser = PDBParser()
    structure = parser.get_structure('roi',pdb)
    res_graph = Graph([],label,fold)
    for chain in structure[0]:
        if True:
            for res in chain:
                CA = []
                for atom in res.get_atoms():
                    if atom.get_name() in ['CA']:
                        CA = np.array(atom.get_coord())
                res_graph.add_node(Node(CA,chain.id,res.id[1],d[res.get_resname()],torch.tensor(np.concatenate((chain_1_hot[cn[chain.id]],np.array(kidera.loc[d[res.get_resname()]],dtype=np.float32),np.array(atchley.loc[d[res.get_resname()]],dtype=np.float32))),dtype=torch.float32)))

#     res_graph.set_edge_index(gnn.radius_graph(res_graph.get_coords(), r=edge_radius, loop=False))
    edge_index = []
    for c1,node1 in enumerate(res_graph.nodes):
        for c2,node2 in enumerate(res_graph.nodes):
            if cn[node1.chain] in [0,1] and cn[node2.chain] in [2,3]:
                if np.linalg.norm(node1.coord-node2.coord) < roi_radius:
                    edge_index.append([c1,c2])
                    edge_index.append([c2,c1])
    edge_index = torch.from_numpy(np.array(edge_index).T.astype(int))
    res_graph.set_edge_index(edge_index)

    edge_indicies, _, mask = remove_isolated_nodes(res_graph.edge_index)
    res_graph.edge_index = edge_indicies
    res_graph.nodes = [res_graph.nodes[i] for i in range(len(mask)) if mask[i]]
    res_graph.set_edge_index(gnn.radius_graph(res_graph.get_coords(), r=edge_radius, loop=False))
    return res_graph

def add_edge_data(graph):
    edge_attrs = []
    for (src, dst) in graph.edge_index.t().tolist():
        d = np.linalg.norm(graph.nodes[src].coord - graph.nodes[dst].coord)
        length_scale_list = [1.5 ** x for x in range(15)]
        rbf_dists_ij = np.exp(-np.array([(d**2)/ls for ls in length_scale_list]))
        connect_type = (chain_1_hot[cn[graph.nodes[src].chain]]+chain_1_hot[cn[graph.nodes[dst].chain]])/2
        edge_attrs.append(np.concatenate((rbf_dists_ij,connect_type)))
    graph.set_edge_attr(torch.tensor(np.stack(edge_attrs),dtype=torch.float32))
