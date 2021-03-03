import csv
from typing import Tuple, Dict

import numpy as np
from IPython.core.display import display
from rdkit import Chem
from rdkit.Chem import rdFMCS
import re

from src.utils import replace_rounds


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
            return 0, 0, 0
        query_smiles = replace_rounds(query_smiles, [
            (r'\[C+@+H*\]', 'C'),
            (r'\[N\+\]\(\=O\)\[O\-\]', 'N(=O)=O'),
            (r'\[O\+\]', 'O')
        ])  # .split('C(*)')[0]
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
                lambda ligand: ligand[1] == query_smiles[1] and query_mol.HasSubstructMatch(self.molecules[ligand]),
                self.ligand_values
            )
        )

        largest_match_idx = np.argmax(list(map(lambda match: self.molecules[match].GetNumAtoms(), root_matches)))
        match = self.ligand_values[root_matches[largest_match_idx]]
        return match[0], match[1], self.vd_w_volume(query_smiles)

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
