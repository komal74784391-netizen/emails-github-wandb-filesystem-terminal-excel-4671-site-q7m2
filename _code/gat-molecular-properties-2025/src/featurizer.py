"""Molecular graph featurisation utilities."""

import numpy as np

ATOM_FEATURES = 32
BOND_FEATURES = 8


def one_hot(value, size):
    vec = np.zeros(size, dtype=np.float32)
    vec[int(value) % size] = 1.0
    return vec


def featurize_atom(atom):
    return one_hot(atom.GetAtomicNum(), ATOM_FEATURES)


def featurize_bond(bond):
    return one_hot(bond.GetBondTypeAsDouble(), BOND_FEATURES)


def featurize_molecule(mol):
    nodes = [featurize_atom(a) for a in mol.GetAtoms()]
    edges = [featurize_bond(b) for b in mol.GetBonds()]
    adjacency = np.eye(len(nodes), dtype=np.float32)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[i, j] = adjacency[j, i] = 1.0
    return np.stack(nodes), np.stack(edges), adjacency
