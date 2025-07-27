
import dgl 
import numpy as np 
from rdkit import Chem
import torch 

from copy import deepcopy 

class BaseMolAugmentor:
    def __init__(self, mask_atom_rate=0.25, mask_bond_rate=0.25):
        self.mask_atom_rate = mask_atom_rate
        self.mask_bond_rate = mask_bond_rate
    
    def augment(self, graph:dgl.DGLGraph) -> dgl.DGLGraph:
        # graph: base graph, not complete graph
        # atom_feature: graph.ndata['h']
        # bond_feature: graph.edata['e']
        # 可以有其他字段
        # 注意要返回深复制的graph
        
        # graph = deepcopy(graph) # 比下边的复制慢
        new_graph = dgl.graph(graph.edges(), num_nodes=graph.number_of_nodes())
        # new_graph.ndata['h'] = graph.ndata['h'].clone()
        # new_graph.edata['e'] = graph.edata['e'].clone()
        # Clone all node data
        for key in graph.ndata:
            new_graph.ndata[key] = graph.ndata[key].clone()

        # Clone all edge data
        for key in graph.edata:
            new_graph.edata[key] = graph.edata[key].clone()
            
        graph = new_graph
        
        n_nodes = graph.number_of_nodes()
        # 因为是有向图，要求每条无向边被两条连续的有向边表示; 之前没调用过add_self_loop
        # self-loop设置需要和dataset一致
        n_edges_double = graph.number_of_edges()
        
        # if n_edges_double == 1:
        #     # 由分子的预处理代码问题导致
        #     # 没有键的分子, 一般是单原子或离子化合物, e.g., NaCl
        #     n_edges_double = 0
        assert n_edges_double % 2 == 0, f"The graph should be undirected, but got {n_edges_double} directed edges"
        n_edges = n_edges_double // 2
        mask_edges = np.random.choice(n_edges, int(n_edges * self.mask_bond_rate), replace=False)
        mask_edges_id = [2*i for i in mask_edges] + [2*i+1 for i in mask_edges]
        
        mask_nodes_id = np.random.choice(n_nodes, int(n_nodes * self.mask_atom_rate), replace=False)
        # np.random.choice(0, 0, replace=False) = array([], dtype=int64)
        
        for key in graph.ndata:
            graph.ndata[key][mask_nodes_id] = 0
        # graph.ndata['h'][mask_nodes_id] = 0
        graph = dgl.remove_edges(graph, mask_edges_id)
        
        return graph 
    
    def __call__(self, *args, **kwargs):
        return self.augment(*args, **kwargs)

if __name__ == "__main__":
    torch.random.manual_seed(1)
    augmentor = BaseMolAugmentor(0.5, 0.5)
    # edges = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 0), (0, 3)]
    # edges = [(0, 0)]
    # src, tgt = list(zip(*edges))
    # g = dgl.graph((src, tgt))
    # g.ndata['h'] = torch.randn(1, 2)
    # g.edata['e'] = torch.randn(1, 3)
    # print(g)
    # aug_g = aug(g)
    # print(aug_g)
    # print(aug_g.ndata['h'])
    # print(aug_g.edata['e'])
    
    from vemol.chem_utils.smiles2graph.base_graph import smiles2base_graph
    from tqdm import tqdm 
    smiles = 'N1CCOCC1'
    graph = smiles2base_graph(smiles)
    aug_graph = augmentor(graph)
    print(graph, graph.ndata['h'].sum(dim=-1)) 
    print(aug_graph, aug_graph.ndata['h'].sum(dim=-1))
    # for _ in tqdm(range(5000)):
    #     aug_graph = augmentor(graph)
    
    
        
        
        
        
        
        
        
        
    
