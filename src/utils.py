from copy import deepcopy
from functools import reduce

import numpy as np
from IPython.core.display import display
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import re


def smiles_to_svg(smiles):
    mol = Chem.MolFromSmiles(
        smiles,
        replacements={
            '[Del]': '',
            '()': '',
        },
        sanitize=False
    )
    mc = Chem.Mol(mol.ToBinary())
    for atom in mc.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    drawer = rdMolDraw2D.MolDraw2DSVG(500, 300)
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg.replace('svg:', '')


def display_numbered(mol):
    mol = deepcopy(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    display(mol)


def get_largest_fragment(mol):
    frags = list(map(lambda s: Chem.MolFromSmiles(s, sanitize=False), Chem.MolToSmiles(mol).split('.')))
    i = np.argmax(list(map(lambda f: f.GetNumAtoms(), frags)))
    return frags[i]


def get_ligands2(molecule, root_pattern_smiles, ligand_root_indices):
    root_pattern = Chem.MolFromSmiles(root_pattern_smiles)
    chains = Chem.ReplaceCore(molecule, root_pattern, labelByIndex=True)
    if chains is None:
        return None
    pieces = Chem.GetMolFrags(chains, asMols=True)
    ligands = [Chem.MolToSmiles(x, True) for x in pieces]
    R_idxs = []
    Rs = []
    for index in ligand_root_indices:
        if index == 0:
            R_idxs.append('*')
        else:
            R_idxs.append('[{}*]'.format(index))

    def get_ligand(index):
        for ligand in ligands:
            if index == "*":
                if "*]" in ligand:
                    continue
            if index in ligand:
                return re.sub(r"\[\d\*\]", "*", ligand)
        if index not in ligands:
            return ''

    Rs.append(list(map(get_ligand, R_idxs)))
    return Rs


def remove_titanium(molecule):
    molecule = Chem.DeleteSubstructs(molecule, Chem.MolFromSmiles('[Ti]', sanitize=False))
    smiles = Chem.MolToSmiles(molecule)
    return Chem.MolFromSmiles(list(filter(lambda s: len(s) > 18, smiles.split('.')))[0])


def remove_bridge(molecule, root_pattern_smiles, removal_indices):
    root_pattern = Chem.MolFromSmiles(root_pattern_smiles)
    matches = molecule.GetSubstructMatches(root_pattern)
    if len(matches) == 0:
        return None
    match = matches[0]

    e_mol = Chem.EditableMol(molecule)
    indexes_to_delete = list(map(lambda i: match[i], removal_indices))
    indexes_to_delete.sort(reverse=True)
    for i in indexes_to_delete:
        e_mol.RemoveAtom(i)
    molecule = e_mol.GetMol()
    return get_largest_fragment(molecule)


def VdW_volume(mol):
    non_aromatic_idx = 0
    Ra = 0
    Rna = 0
    H_count = 0
    structure = Chem.MolFromSmiles(mol)
    atoms_to_count = ['C', 'N', 'O', 'F', 'Cl', 'Br']
    H_structure = Chem.AddHs(structure)
    for atom in (structure.GetAtoms()):
        H_count += int(atom.GetTotalNumHs())
    count = {}
    for atoms in atoms_to_count:
        count["{}-count".format(atoms)] = len(structure.GetSubstructMatches(Chem.MolFromSmiles(atoms)))
    bonds = H_structure.GetBonds()
    info = structure.GetRingInfo()
    rings = info.AtomRings()
    for ring in rings:
        for id in ring:
            if not structure.GetAtomWithIdx(id).GetIsAromatic():
                non_aromatic_idx += 1
        if non_aromatic_idx == 0:
            Ra += 1
        else:
            Rna += 1
        non_aromatic_idx = 0

    atom_count = list(count.values())
    atom_contributions = [20.58, 15.60, 14.71, 13.31, 22.45, 26.52]
    products = [a * b for a, b in zip(atom_count, atom_contributions)]

    Total_Volume = 0.801 * ((sum(products) + (int(H_count) * 7.24)) - (len(bonds) * 5.92) - (int(Ra) * 14.7) - (
            int(Rna) * 3.8)) + 0.18
    return Total_Volume


def replace_rounds(haystack, replacements):
    return reduce(lambda haystack, sub: re.sub(sub[0], sub[1], haystack), replacements, haystack)
