import csv
from copy import deepcopy
from functools import reduce
from math import exp
from typing import Tuple, List, Set, Iterable, Dict

import numpy as np
from IPython.core.display import display
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem.rdchem import Mol

from src.utils import replace_rounds, display_numbered

halogen_score = 5

atom_scores = {
    'O': 6,
    'N': 7,
    'c': 2,
    'n': 14,
    'F': halogen_score,
    'Cl': halogen_score,
    'Br': halogen_score,
    'S': 20
}


class MoleculeValueFinder:
    ligand_values: Dict[str, Tuple[float, float]]
    molecules: Dict[str, Mol]

    root_distance_cache: Dict[Tuple[Mol, int], float]

    def __init__(self, data_file_path: str):
        self.ligand_values = {}
        self.molecules = {}
        self.root_distance_cache = {}
        with open(data_file_path) as file:
            _, *rows = list(csv.reader(file, delimiter='\t', quotechar='"'))

        for row in rows:
            smiles = row[0].replace('[R]', '*')
            self.ligand_values[smiles] = (float(row[1]), float(row[2]))
            self.molecules[smiles] = Chem.MolFromSmiles(smiles)

    def get_values(self, query_smiles: str) -> Tuple[float, float, float]:
        if query_smiles == '' or query_smiles == '*':
            return 0, 0, 1.32
        query_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(query_smiles))
        query_smiles = replace_rounds(query_smiles, [
            (r'\[C+@+H*\]', 'C'),
            (r'N+@+H*', 'N'),
            (r'\[N\+\]\(\=O\)\[O\-\]', 'N(=O)=O'),
            (r'\[O\+\]', 'O')
        ])
        volume = self.vd_w_volume(query_smiles)
        if query_smiles.count("*") > 1:
            volume = int(self.vd_w_volume(query_smiles)) * 0.5
            chain = Chem.MolFromSmiles('N')
            products = Chem.ReplaceSubstructs(Chem.MolFromSmiles(query_smiles), Chem.MolFromSmarts('[#0]'), chain)
            query_smiles = Chem.MolToSmiles(products[0])

        exact_match = self.ligand_values.get(query_smiles)
        if exact_match is not None:
            return exact_match[0], exact_match[1], volume

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

        mcs = rdFMCS.FindMCS([query_mol, self.molecules[best_match]])
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)

        match_values = self.ligand_values[best_match]
        return match_values[0], match_values[1], volume

    def get_root_distance_weight(self, mol: Mol, atom_idx: int) -> float:
        if atom_idx == 0:
            return 0
        cache_key = (mol, atom_idx)
        cache_result = self.root_distance_cache.get(cache_key)
        if cache_result is not None:
            return cache_result
        result = exp(-len(Chem.GetShortestPath(mol, 0, atom_idx)) / 7)
        self.root_distance_cache[cache_key] = result
        return result

    def get_match_score(self, query: Mol, match: Mol) -> float:
        score = 0
        mcs = rdFMCS.FindMCS([query, match], completeRingsOnly=True)
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
        smiles = Chem.MolToSmiles(mcs_mol)
        mcs_mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if '#0' not in mcs.smartsString:
            return float("-inf")

        def get_atom_symbol(atom):
            if atom.GetIsAromatic():
                return atom.GetSymbol().lower()
            return atom.GetSymbol()

        matching_atoms = ""
        for atom in mcs_mol.GetAtoms():
            matching_atoms += str(get_atom_symbol(atom))
            if atom.GetIdx() == 0:
                continue
            w = self.get_root_distance_weight(mcs_mol, atom.GetIdx())
            score += (atom_scores.get(get_atom_symbol(atom)) or 1) * w

        m_copy = deepcopy(match)
        for atom in m_copy.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())

        match_remainder = Chem.DeleteSubstructs(m_copy, mcs_mol)
        remaining_match_atoms = ""
        for atom in match_remainder.GetAtoms():
            remaining_match_atoms += str(get_atom_symbol(atom))
            w = self.get_root_distance_weight(match, atom.GetAtomMapNum())
            score -= (atom_scores.get(get_atom_symbol(atom)) or 1) * w

        q_copy = deepcopy(query)
        for atom in q_copy.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
        query_remainder = Chem.DeleteSubstructs(q_copy, mcs_mol)
        remaining_query_atoms = ""
        for atom in query_remainder.GetAtoms():
            remaining_query_atoms += str(get_atom_symbol(atom))
            w = self.get_root_distance_weight(query, atom.GetAtomMapNum())
            score -= (atom_scores.get(get_atom_symbol(atom)) or 1) * w

        match_count = q_copy.GetSubstructMatches(mcs_mol)
        if len(match_count) > 1:
            indices = [x for x in match_count[0] if x not in match_count[1]]
            for index in indices:
                atom = q_copy.GetAtomWithIdx(index)
                score -= (atom_scores.get(get_atom_symbol(atom)) or 1) * w
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
