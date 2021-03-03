from enum import Enum
from typing import Optional, List

from src.MoleculeValueFinder import MoleculeValueFinder
from src.utils import smiles_to_svg


class Molecule:
    smiles: str
    hansch_pi: Optional[float]
    hammett: Optional[float]
    vd_w_volume: Optional[float]

    def __init__(self, smiles: str):
        self.smiles = smiles

    def set_values(self, value_finder: MoleculeValueFinder):
        (
            self.hansch_pi,
            self.hammett,
            self.smiles
        ) = value_finder.get_values(self.smiles)

    def get_svg(self) -> str:
        return smiles_to_svg(self.smiles)


class Complex(Molecule):
    class_name: str
    ligands: List[Optional[Molecule]]
    bridge: Optional[Molecule]

    def __init__(self, smiles: str, class_name: str):
        super().__init__(smiles)
        self.class_name = class_name

    def get_row_elements(self):
        return [
            f'{self.get_svg()}<br>{self.smiles}',
            self.class_name,
            *list(
                map(
                    lambda ligand:
                    "No ligand"
                    if ligand is None else
                    f'{ligand.get_svg()}<br>{ligand.smiles}: values',
                    self.ligands
                )
            )
        ]
