from rdkit import Chem

from src.ComplexProcessor import ComplexProcessor
from src.MoleculeValueFinder import MoleculeValueFinder
from src.utils import display_numbered
from src.MoleculeValueFinder import MoleculeValueFinder

if __name__ == '__main__':
    smiles = '*C(c1ccc2c(c1)OCO2)N(C)C'
    mvf = MoleculeValueFinder('./res/substituent-parameters.tsv')
    display_numbered(Chem.MolFromSmiles(smiles))
    mvf.get_values(smiles)