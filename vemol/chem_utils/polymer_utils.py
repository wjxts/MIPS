from copy import deepcopy

import dgl
import networkx as nx
import numpy as np
import pandas as pd 
import rdkit 
from rdkit import Chem
import torch 

ALLOWED_ATOM_SET = {'P', 'N', 'F', 'O', 'Br', 'H', 'I', 'Cl', 'C', 'Si', 'S', '*'}

def augment_polymer(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    mol_copy = deepcopy(mol)
    star_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == '*']
    connected_atoms = [
        list(mol.GetAtomWithIdx(wildcard).GetNeighbors())[0].GetIdx()
        for wildcard in star_indices
    ]
    # print(f"star indices: {star_indices}")
    # print(f"connected atoms: {connected_atoms}")
    assert connected_atoms[0] > star_indices[0] and connected_atoms[1] < star_indices[1], "wrong order"
    editable_monomer1 = Chem.RWMol(mol)
    editable_monomer2 = Chem.RWMol(mol_copy)
    editable_monomer1.RemoveAtom(star_indices[1])  # 移除 monomer1 的 '*'
    editable_monomer2.RemoveAtom(star_indices[0])  # 移除 monomer2 的 '*'
    
    # 连接两个分子
    combined = Chem.CombineMols(editable_monomer1, editable_monomer2)
    editable_combined = Chem.RWMol(combined)
    minus = 1 if connected_atoms[1] > star_indices[0] else 0
    # print(f"minus: {minus}")
    editable_combined.AddBond(connected_atoms[1], connected_atoms[0]+editable_monomer1.GetNumAtoms()-minus, Chem.BondType.SINGLE)
    final_mol = editable_combined.GetMol()
    
    # final_smiles = Chem.MolToSmiles(final_mol, canonical=False)
    final_smiles = Chem.MolToSmiles(final_mol)
    return final_smiles


def main_chain_length(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    star_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == '*']
    path = Chem.rdmolops.GetShortestPath(mol, star_indices[0], star_indices[1])
    return len(path) - 2  # 路径上包含的原子数, 减去两个 '*' 的原子


def recursive_augment_polymer(smiles: str) -> str:
    while main_chain_length(smiles) <= 2:
        smiles = augment_polymer(smiles)
    return smiles

def psmiles_star_op(smiles: str, op: str) -> Chem.Mol:
    if op=='star_keep':
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    star_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == '*']
    connected_atoms = [
        list(mol.GetAtomWithIdx(wildcard).GetNeighbors())[0].GetIdx()
        for wildcard in star_indices
    ]
    # print(f"star indices: {star_indices}")
    # print(f"connected atoms: {connected_atoms}")
    # assert connected_atoms[0] > star_indices[0] and connected_atoms[1] < star_indices[1], "wrong order"
    emol = Chem.RWMol(mol)
    if op=='star_sub':
        atom = emol.GetAtomWithIdx(star_indices[0])
        sub_index = emol.GetAtomWithIdx(connected_atoms[1]).GetAtomicNum()
        atom.SetAtomicNum(sub_index)

        atom = emol.GetAtomWithIdx(star_indices[1])
        sub_index = emol.GetAtomWithIdx(connected_atoms[0]).GetAtomicNum()
        atom.SetAtomicNum(sub_index)
        
    elif op=='star_link':
        emol.AddBond(connected_atoms[0], connected_atoms[1], Chem.BondType.SINGLE)
        emol.RemoveAtom(star_indices[1])
        emol.RemoveAtom(star_indices[0])
        
    elif op=='star_remove':
        emol.RemoveAtom(star_indices[1])
        emol.RemoveAtom(star_indices[0])
        
    else:
        raise ValueError(f"Invalid op: {op}")
    new_mol = emol.GetMol()
    
    return new_mol
    # Chem.SanitizeMol(new_mol)
    # new_smiles = Chem.MolToSmiles(new_mol)
    # return new_smiles

def psmiles_star_sub(smiles :str) ->str:
    mol = Chem.MolFromSmiles(smiles)
    star_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == '*']
    connected_atoms = [
        list(mol.GetAtomWithIdx(wildcard).GetNeighbors())[0].GetIdx()
        for wildcard in star_indices
    ]
    # print(f"star indices: {star_indices}")
    # print(f"connected atoms: {connected_atoms}")
    # assert connected_atoms[0] > star_indices[0] and connected_atoms[1] < star_indices[1], "wrong order"
    emol = Chem.RWMol(mol)
    atom = emol.GetAtomWithIdx(star_indices[0])
    sub_index = emol.GetAtomWithIdx(connected_atoms[1]).GetAtomicNum()
    atom.SetAtomicNum(sub_index)

    atom = emol.GetAtomWithIdx(star_indices[1])
    sub_index = emol.GetAtomWithIdx(connected_atoms[0]).GetAtomicNum()
    atom.SetAtomicNum(sub_index)
    new_mol = emol.GetMol()
    new_smiles = Chem.MolToSmiles(new_mol)
    return new_smiles

def psmiles_star_link(smiles :str) ->str:
    mol = Chem.MolFromSmiles(smiles)
    star_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == '*']
    connected_atoms = [
        list(mol.GetAtomWithIdx(wildcard).GetNeighbors())[0].GetIdx()
        for wildcard in star_indices
    ]
    # print(f"star indices: {star_indices}")
    # print(f"connected atoms: {connected_atoms}")
    # assert connected_atoms[0] > star_indices[0] and connected_atoms[1] < star_indices[1], "wrong order"
    emol = Chem.RWMol(mol)
    emol.AddBond(connected_atoms[0], connected_atoms[1], Chem.BondType.SINGLE)
    emol.RemoveAtom(star_indices[1])
    emol.RemoveAtom(star_indices[0])
    
    new_mol = emol.GetMol()
    Chem.SanitizeMol(new_mol)
    new_smiles = Chem.MolToSmiles(new_mol)
    return new_smiles


def repeat_poly_graph(g: dgl.DGLGraph) -> dgl.DGLGraph:
    # 不改变原图
    star_nodes_id = g.ndata['h'][:, 100].nonzero().squeeze().tolist()
    star_neighbors = g.in_edges(star_nodes_id, form='uv')[0].tolist()
    repeat_g = deepcopy(dgl.batch([g, g])) # 拓扑和特征浅复制, batch_num_nodes有两个元素
    # print(repeat_g.batch_num_nodes())
    g_num_nodes = g.num_nodes()
    tem_nodes = star_neighbors[1], star_neighbors[0]+g_num_nodes
    # print(tem_nodes)
    repeat_g.add_edges(tem_nodes, tem_nodes[::-1])
    remove_star_nodes_id = [star_nodes_id[1], star_nodes_id[0]+g_num_nodes]
    # print(remove_star_nodes_id)
    repeat_g.remove_nodes(remove_star_nodes_id)
    # print(repeat_g.batch_num_nodes()) # batch_num_nodes只有一个元素
    return repeat_g


def minus(idx, star_nodes_id):
    minus_idx = 0
    for i in star_nodes_id:
        if i < idx:
            minus_idx += 1
    return minus_idx


def main_chain_length_poly_graph(g: dgl.DGLGraph) -> int:
    # 主链原子个数
    star_nodes_id = g.ndata['h'][:, 100].nonzero().squeeze().tolist()
    star_neighbors = g.in_edges(star_nodes_id, form='uv')[0].tolist()
    nx_g = dgl.to_networkx(g.cpu())
    src_node = star_neighbors[0]
    dst_node = star_neighbors[1]
    shortest_path = nx.shortest_path(nx_g, source=src_node, target=dst_node)
    return len(shortest_path)

def translation_augument(g: dgl.DGLGraph) -> dgl.DGLGraph:
    star_nodes_id = g.ndata['h'][:, 100].nonzero().squeeze().tolist()
    star_neighbors = g.in_edges(star_nodes_id, form='uv')[0].tolist()
    
    nx_g = dgl.to_networkx(g.cpu())
    src_node = star_neighbors[0]
    dst_node = star_neighbors[1]
    shortest_path = nx.shortest_path(nx_g, source=src_node, target=dst_node)

    star_features = deepcopy(g.ndata['h'][star_nodes_id[0]])
    trans_g = deepcopy(g)
    g_num_nodes = trans_g.num_nodes()
    trans_g.add_edges(star_neighbors, star_neighbors[::-1])
    trans_g.remove_nodes(star_nodes_id)
    
    loop_shortest_path = [i-minus(i, star_nodes_id) for i in shortest_path]
    bond_cands = [(loop_shortest_path[i], loop_shortest_path[i+1]) for i in range(len(loop_shortest_path)-1)]

    break_bond_id = np.random.choice(range(len(bond_cands)), size=(1,))[0]

    break_bond = bond_cands[break_bond_id]

    edge_ids = trans_g.edge_ids(break_bond, break_bond[::-1], return_uv=False)
    trans_g.remove_edges(edge_ids)
    trans_g.add_nodes(2, data={'h': torch.stack([star_features, star_features], dim=0)})
    new_edge1 = (break_bond[0], g_num_nodes-2)
    new_edge2 = (break_bond[1], g_num_nodes-1)
    new_edges = [new_edge1, new_edge1[::-1], new_edge2, new_edge2[::-1]]
    src, dst = list(zip(*new_edges))
    trans_g.add_edges(src, dst)

    return trans_g

import matplotlib.pyplot as plt

def visualize_graph(g: dgl.DGLGraph):
    node_labels = (torch.argmax(g.ndata['h'][:101], dim=1)+1).tolist()
    node_labels = {i: str(label) for i, label in enumerate(node_labels)}
    nx_graph = dgl.to_networkx(g)
    plt.figure(figsize=(6, 4))
    pos = nx.spring_layout(nx_graph)
    # 绘制图结构
    nx.draw(nx_graph, pos, with_labels=False, node_color='lightblue', node_size=500, font_size=10)
    # 添加节点数字标签
    nx.draw_networkx_labels(nx_graph, pos, labels=node_labels, font_size=12, font_color='red')
    # 显示图像
    plt.title("DGL Graph Visualized with NetworkX")




if __name__ == "__main__":
    from vemol.chem_utils.smiles2graph.polymer_base_graph import psmiles2polymer_base_graph
    smiles = '*CC(F)CC(*)F'
    base_graph = psmiles2polymer_base_graph(smiles, op='star_keep')
    repeat_poly_graph(base_graph)