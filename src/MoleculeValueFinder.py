import csv
from copy import deepcopy
from functools import reduce
from math import exp
from typing import Tuple, List, Set, Iterable

import numpy as np
from IPython.core.display import display
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem.rdchem import Mol

from src.utils import replace_rounds, display_numbered

halogen_score = 4

atom_scores = {
    'O': 5,
    'N': 10,
    'F': halogen_score,
    'Cl': halogen_score,
    'Br': halogen_score,
    'S': 20
}


class MoleculeValueFinder:

    def __init__(self, data_file_path: str):
        self.ligand_values = {}
        self.molecules = {}
        with open(data_file_path) as file:
            _, *rows = list(csv.reader(file, delimiter='\t', quotechar='"'))

        for row in rows:
            smiles = row[0].replace('[R]', '*')
            self.ligand_values[smiles] = (float(row[1]), float(row[2]))
            self.molecules[smiles] = Chem.MolFromSmiles(smiles)

    def get_values(self, query_smiles: str) -> Tuple[float, float, float]:
        if query_smiles == '' or query_smiles == '*':
            return 0, 0, 1.32
        query_smiles = replace_rounds(query_smiles, [
            (r'\[C+@+H*\]', 'C'),
            (r'\[N\+\]\(\=O\)\[O\-\]', 'N(=O)=O'),
            (r'\[O\+\]', 'O')
        ])
        if query_smiles.count("*") > 1:
            chain = Chem.MolFromSmiles('N')
            products = Chem.ReplaceSubstructs(Chem.MolFromSmiles(query_smiles), Chem.MolFromSmarts('[#0]'), chain)
            query_smiles = Chem.MolToSmiles(products[0])

        exact_match = self.ligand_values.get(query_smiles)
        if exact_match is not None:
            return exact_match[0], exact_match[1], self.vd_w_volume(query_smiles)

        query_mol = Chem.MolFromSmiles(query_smiles)
        root_matches = list(
            filter(
                lambda ligand: ligand[1] == query_smiles[1],
                self.ligand_values
            )
        )

        best_match = None
        best_match_score = float("-inf")

        for match in root_matches:
            score = self.get_match_score(query_mol, self.molecules[match])
            if score >= best_match_score:
                best_match = match
                best_match_score = score
        display('ORIGINAL')
        display_numbered(query_mol)
        display('MATCH')
        display_numbered(self.molecules[best_match])
        display('SCORE')
        display(best_match_score)
        mcs = rdFMCS.FindMCS([query_mol, self.molecules[best_match]])
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
        display('MCS')
        display_numbered(mcs_mol)
        match_values = self.ligand_values[best_match]
        return match_values[0], match_values[1], self.vd_w_volume(query_smiles)

    @staticmethod
    def get_match_score(query: Mol, match: Mol) -> float:
        score = 0
        mcs = rdFMCS.FindMCS([query, match], ringMatchesRingOnly=True, completeRingsOnly=True)
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)

        if '#0' not in mcs.smartsString:
            return float("-inf")

        def get_root_distance_weight(mol: Mol, atom_idx: int) -> float:
            if atom_idx == 0:
                return 0
            return exp(-len(Chem.GetShortestPath(mol, 0, atom_idx)) / 7)

        for atom in mcs_mol.GetAtoms():
            if atom.GetIdx() == 0:
                continue
            w = get_root_distance_weight(mcs_mol, atom.GetIdx())
            score += (atom_scores.get(atom.GetSymbol()) or 1) * w

        q_copy = deepcopy(query)
        for atom in q_copy.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
        query_remainder = Chem.DeleteSubstructs(q_copy, mcs_mol)
        for atom in query_remainder.GetAtoms():
            w = get_root_distance_weight(query, atom.GetAtomMapNum())
            score -= (atom_scores.get(atom.GetSymbol()) or 1) * w

        m_copy = deepcopy(match)
        for atom in m_copy.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
        match_remainder = Chem.DeleteSubstructs(m_copy, mcs_mol)
        for atom in match_remainder.GetAtoms():
            w = get_root_distance_weight(match, atom.GetAtomMapNum())
            score -= (atom_scores.get(atom.GetSymbol()) or 1) * w

        return score

    def get_smallest_root_match(self, mol: Mol) -> Mol:
        search_space: Set[Mol] = set(self.molecules.values())

        all_idxs = set(range(0, mol.GetNumAtoms()))
        included_idxs = {0}

        bonds = {}

        def register_bond(from_idx: int, to_idx: int):
            entry = bonds.get(from_idx)
            if entry is None:
                entry = []
                bonds[from_idx] = entry
            entry.append(to_idx)

        for bond in mol.GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            register_bond(begin, end)
            register_bond(end, begin)

        while len(included_idxs) < mol.GetNumAtoms():
            frontier_permutations = reduce(
                lambda perms, from_idx: perms | set(
                    map(
                        lambda to_idx: frozenset([*included_idxs, to_idx]),
                        filter(lambda idx: idx not in included_idxs, bonds[from_idx])
                    )
                ),
                included_idxs,
                set()
            )

            new_search_space = set()
            for perm in frontier_permutations:
                e_mol = Chem.EditableMol(mol)
                perm_idxs = list(all_idxs - perm)
                perm_idxs.sort(reverse=True)
                for idx in perm_idxs:
                    e_mol.RemoveAtom(idx)
                display('mul')
                display_numbered(e_mol.GetMol())
                new_search_space |= set(self.find_superstructures(e_mol.GetMol(), search_space))
                included_idxs |= perm

            if len(new_search_space) == 0:
                return self.get_smallest_mol(list(search_space))
            search_space = new_search_space
            if len(search_space) < 100:
                display("from mul")
                for s in search_space:
                    display_numbered(s)

            if len(frontier_permutations) > 1:
                e_mol = Chem.EditableMol(mol)
                perm_idxs = list(all_idxs - included_idxs)
                perm_idxs.sort(reverse=True)
                for idx in perm_idxs:
                    e_mol.RemoveAtom(idx)
                display('single')
                display_numbered(e_mol.GetMol())
                new_search_space = set(self.find_superstructures(e_mol.GetMol(), search_space))

            if len(new_search_space) == 0:
                return self.get_smallest_mol(list(search_space))
            search_space = new_search_space
            if len(search_space) < 100:
                display("from mul")
                for s in search_space:
                    display_numbered(s)

    @staticmethod
    def find_superstructures(substructure: Mol, search_space: Iterable[Mol]) -> List[Mol]:
        return list(
            filter(
                lambda mol: mol.HasSubstructMatch(substructure),
                search_space
            )
        )

    @staticmethod
    def get_smallest_mol(mols: List[Mol]) -> Mol:
        mols.sort(key=Mol.GetNumAtoms)
        return mols[0]

    @staticmethod
    def vd_w_volume(smiles: str) -> float:
        if smiles == '':
            return 1.32
        non_aromatic_idx = 0
        ra = 0
        rna = 0
        h_count = 0
        structure = Chem.MolFromSmiles(smiles)
        atoms_to_count = ['C', 'N', 'O', 'F', 'Cl', 'Br']
        # display(structure)
        h_structure = Chem.AddHs(structure)
        for atom in (structure.GetAtoms()):
            h_count += int(atom.GetTotalNumHs())
        count = {}
        for atom in atoms_to_count:
            count[f'{atom}-count'] = len(structure.GetSubstructMatches(Chem.MolFromSmiles(atom)))
        bonds = h_structure.GetBonds()
        info = structure.GetRingInfo()
        rings = info.AtomRings()
        for ring in rings:
            for idx in ring:
                if not structure.GetAtomWithIdx(idx).GetIsAromatic():
                    non_aromatic_idx += 1
            if non_aromatic_idx == 0:
                ra += 1
            else:
                rna += 1
            non_aromatic_idx = 0

        atom_count = list(count.values())
        atom_contributions = [20.58, 15.60, 14.71, 13.31, 22.45, 26.52]
        products = [a * b for a, b in zip(atom_count, atom_contributions)]

        return 0.801 * ((sum(products) + (int(h_count) * 7.24)) - (len(bonds) * 5.92) - (int(ra) * 14.7) - (
                int(rna) * 3.8)) + 0.18
