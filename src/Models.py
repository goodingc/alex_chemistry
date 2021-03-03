from enum import Enum
from typing import Optional, List

from src.MoleculeValueFinder import MoleculeValueFinder
from src.utils import smiles_to_svg, smiles_html


class Molecule:
    smiles: str
    hansch_pi: Optional[float]
    hammett: Optional[float]
    vd_w_volume: Optional[float]

    def __init__(self, smiles: str):
        self.smiles = smiles
        self.hansch_pi = None
        self.hammett = None
        self.vd_w_volume = None

    def set_values(self, value_finder: MoleculeValueFinder):
        (
            self.hansch_pi,
            self.hammett,
            self.vd_w_volume
        ) = value_finder.get_values(self.smiles)

    def get_svg(self) -> str:
        return smiles_to_svg(self.smiles)

    def get_html(self):
        elements = [
            self.get_svg(),
            f'SMILES: {smiles_html(self.smiles)}'
        ]
        if self.hansch_pi is not None:
            elements.append(f'Hansch π: {self.hansch_pi}')
        if self.hammett is not None:
            elements.append(f'Hammett: {self.hammett}')
        if self.vd_w_volume is not None:
            elements.append(f'VdW Volume: {self.vd_w_volume}')
        return '<br>'.join(elements)

    def __str__(self):
        return self.smiles


class Complex(Molecule):
    class_name: str
    ligands: List[Optional[Molecule]]
    bridge: Optional[Molecule]

    def __init__(self, smiles: str, class_name: str):
        super().__init__(smiles)
        self.class_name = class_name
        self.ligands = []
        self.bridge = None

    def get_row_elements(self) -> List[str]:
        return [
            self.get_html(),
            self.class_name,
            *list(
                map(
                    lambda ligand: "No ligand" if ligand is None else ligand.get_html(),
                    self.ligands
                )
            ),
            "No bridge" if self.bridge is None else self.bridge.get_html()
        ]
