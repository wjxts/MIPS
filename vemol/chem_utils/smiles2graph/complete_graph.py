from typing import Optional, List, Tuple

import dgl 
import networkx as nx
from rdkit import Chem
import torch

from vemol.chem_utils.featurizer.standard_featurizer import StandardFeaturizer
from vemol.chem_utils.graph_ops import extract_edges_from_mol 
from vemol.chem_utils.featurizer.standard_featurizer import INF


def generate_complete_graph(edges: List[Tuple[int, int]], max_length):
    nx_graph = nx.Graph(edges)
    paths_dict = dict(nx.algorithms.all_pairs_shortest_path(nx_graph, cutoff=max_length-1))
    src, dst, distance, paths = [], [], [], []
    for i in paths_dict:
        for j in paths_dict[i]:
           
            src.append(i)
            dst.append(j)
            distance.append([len(paths_dict[i][j])-1])
            paths.append(paths_dict[i][j]+[-INF]*(max_length-len(paths_dict[i][j])))
    
    return src, dst, distance, paths

def smiles2complete_graph(smiles: str, max_length: int) -> Optional[dgl.DGLGraph]:
    # max_length: max_length of the path
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        # raise ValueError(f"Invalid SMILES: {smiles}")
        return None 
    featurizer = StandardFeaturizer()
    
    atom_features = featurizer.featurize_atoms(mol)
    # bond_features = featurizer.featurize_bonds(mol)
    
    edges = extract_edges_from_mol(mol, add_self_loop=True)
    
    src, dst, distance, paths = generate_complete_graph(edges, max_length)
    
    distance = torch.LongTensor(distance) # E*1
    paths = torch.LongTensor(paths) # E*(max_length+1)
    graph = dgl.graph((src, dst), num_nodes=mol.GetNumAtoms())
    graph.ndata['h'] = torch.FloatTensor(atom_features)
    graph.edata['distance'] = distance
    graph.edata['paths'] = paths
    return graph


def base_graph2complete_graph(base_graph: dgl.DGLGraph, max_length: int) -> dgl.DGLGraph:
    # add self loop to avoid zero edges and imply the number of nodes
    base_graph = dgl.add_self_loop(base_graph)
    
    src, tgt = base_graph.edges()
    src, tgt = src.tolist(), tgt.tolist()
    edges = list(zip(src, tgt))
    src, dst, distance, paths = generate_complete_graph(edges, max_length=max_length)
    distance = torch.LongTensor(distance) # E*1
    paths = torch.LongTensor(paths) # E*(max_length)
    graph = dgl.graph((src, dst), num_nodes=base_graph.num_nodes())
    for key in base_graph.ndata:
        graph.ndata[key] = base_graph.ndata[key].clone()

    graph.edata['distance'] = distance
    graph.edata['paths'] = paths
    assert 'distance' in graph.edata, f"distance not in edata: {graph}"
    return graph

if __name__ == "__main__":

    
    from tqdm import tqdm 
    smiles = 'CCC(NC(=O)c1scnc1C1CC1)C(=O)N1CCOCC1'
    
    from vemol.chem_utils.smiles2graph.base_graph import smiles2base_graph
    
    base_graph = smiles2base_graph(smiles)
    graph = base_graph2complete_graph(base_graph, max_length=3)
    print(graph)
   