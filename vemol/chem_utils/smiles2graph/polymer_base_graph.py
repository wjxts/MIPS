from typing import Optional, Literal, List

import dgl 
import networkx as nx
from rdkit import Chem
import torch 

from vemol.chem_utils.smiles2graph.base_graph import smiles2base_graph, mol2base_graph
from vemol.chem_utils.polymer_utils import main_chain_length_poly_graph, repeat_poly_graph

MAIN_CHAIN_INDEX = 40

def add_main_chain_embedding(graph: dgl.DGLGraph, indices: List[int]) -> dgl.DGLGraph:
    graph.ndata['h'][indices, MAIN_CHAIN_INDEX] = 1 # inplace operation
    return graph 

def base_graph2polymer_base_graph(graph: dgl.DGLGraph,
                                  op: Literal['star_keep', 'star_remove', 'star_sub', 'star_link'],
                                  main_chain_embed: bool=False) -> Optional[dgl.DGLGraph]:
    # inplace改变
    
    # mol = Chem.MolFromSmiles(smiles)
    # wildcard_atom_idxs = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == '*']
    # connected_atom_idxs = [
    # list(mol.GetAtomWithIdx(wildcard_id).GetNeighbors())[0].GetIdx()
    # for wildcard_id in wildcard_atom_idxs
    # ]
    # path = list(Chem.rdmolops.GetShortestPath(mol, wildcard_atom_idxs[0], wildcard_atom_idxs[1]))
    # assert len(connected_atom_idxs) == 2, f"{smiles}: the number of connected atoms is not 2"
    # print(path)
    
        
    wildcard_atom_idxs = graph.ndata['h'][:, 100].nonzero().squeeze().tolist()
    connected_atom_idxs = graph.in_edges(wildcard_atom_idxs, form='uv')[0].tolist()
    nx_g = dgl.to_networkx(graph.cpu())
    src_node = wildcard_atom_idxs[0]
    dst_node = wildcard_atom_idxs[1]
    path = nx.shortest_path(nx_g, source=src_node, target=dst_node)
    # print(path);exit()
    if len(path)<=5:
        graph = repeat_poly_graph(graph) # 和GT模型有关, 主链长度需要>=2*max_length-1, 取max_length=3
        
    # 先变换feature, 后边随着remove变换就行了
    if main_chain_embed:
        graph = add_main_chain_embedding(graph, path)
    if op == 'star_keep':
        return graph
    elif op == 'star_remove':
        graph.remove_nodes(wildcard_atom_idxs)
        return graph
    elif op == 'star_sub':
        for key in graph.ndata.keys():
            graph.ndata[key][wildcard_atom_idxs[0]] = graph.ndata[key][connected_atom_idxs[1]]
            graph.ndata[key][wildcard_atom_idxs[1]] = graph.ndata[key][connected_atom_idxs[0]]
        return graph 
    elif op == 'star_link':
        graph.add_edges(connected_atom_idxs[0], connected_atom_idxs[1])
        graph.add_edges(connected_atom_idxs[1], connected_atom_idxs[0])
        graph.remove_nodes(wildcard_atom_idxs)
        return graph 
    else:
        raise ValueError(f"Invalid op: {op}")

def psmiles2polymer_base_graph(smiles: str, 
                       op: Literal['star_keep', 'star_remove', 'star_sub', 'star_link'],
                       main_chain_embed: bool=False) -> Optional[dgl.DGLGraph]:
    # 需要是aug_psmiles
    # mol = Chem.MolFromSmiles(smiles)
    # base_graph = mol2base_graph(mol)
    base_graph = smiles2base_graph(smiles)
    graph = base_graph2polymer_base_graph(base_graph, op, main_chain_embed)
    return graph
    


if __name__ == "__main__":
    # smiles_list = ['*C(*)C', '*CO*', '*CCO*', '*CC(O*)C(F)(F)F']
    # op_list = ['star_keep', 'star_remove', 'star_sub', 'star_link']
    # for smiles in smiles_list:
    #     for op in op_list:
    #         print(smiles, op)
    #         g = psmiles2base_graph(smiles, op)
    #         assert g is not None, "None graph"
    #         print(f"num nodes: {g.number_of_nodes()}, num edges: {g.number_of_edges()//2}")    
            
    smiles = "*CCOCCOC(=O)O*"
    # op = 'star_link'
    op = 'star_sub'
    g = psmiles2polymer_base_graph(smiles, op)
    print(torch.allclose(g.ndata['h'][0], g.ndata['h'][3]))
    print(torch.allclose(g.ndata['h'][1], g.ndata['h'][4]))


