import csv

import numpy as np
from IPython.core.display import display
from rdkit import Chem
from rdkit.Chem import rdFMCS
import re
from src.utils import VdW_volume

from src.utils import replace_rounds


class LigandValueFinder:
    def __init__(self, data_file_path):
        self.ligand_values = {}
        self.molecules = {}
        with open(data_file_path) as file:
            _, *rows = list(csv.reader(file, delimiter='\t', quotechar='"'))

        for row in rows:
            smiles = row[0].replace('[R]', '*')
            self.ligand_values[smiles] = (float(row[1]), float(row[2]))
            self.molecules[smiles] = Chem.MolFromSmiles(smiles)


    def get_values(self, query_smiles):
        if query_smiles == '':
            return 0, 0, ''
        query_smiles = replace_rounds(query_smiles, [
            (r'\[C+@+H*\]', 'C'),
            (r'\[N\+\]\(\=O\)\[O\-\]', 'N(=O)=O'),
            (r'\[O\+\]', 'O')
        ])#.split('C(*)')[0]
        if query_smiles.count("*") > 1:
            chain = Chem.MolFromSmiles('N')
            products = Chem.ReplaceSubstructs(Chem.MolFromSmiles(query_smiles), Chem.MolFromSmarts('[#0]'), chain)
            query_smiles = Chem.MolToSmiles(products[0])

        exact_match = self.ligand_values.get(query_smiles)
        if exact_match is not None:
            return *exact_match, query_smiles

        query_mol = Chem.MolFromSmiles(query_smiles)

        root_matches = list(
            filter(
                lambda ligand: ligand[1] == query_smiles[1] and query_mol.HasSubstructMatch(self.molecules[ligand]),
                self.ligand_values
            )
        )


        largest_match_idx = np.argmax(list(map(lambda match: self.molecules[match].GetNumAtoms(), root_matches)))
        return *self.ligand_values[root_matches[largest_match_idx]], root_matches[largest_match_idx]

