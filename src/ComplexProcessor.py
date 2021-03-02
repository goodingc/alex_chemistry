import re
from functools import reduce

import openpyxl
from IPython.display import HTML, display
from openpyxl import Workbook
from rdkit import Chem

from src.LigandValueFinder import LigandValueFinder
from src.utils import remove_titanium, remove_bridge, get_ligands2, smiles_to_svg, VdW_volume, get_bridge_idty, replace_rounds


def metallocene(molecule):
    x = reduce(
        lambda mol, smiles:
        Chem.DeleteSubstructs(
            mol,
            Chem.MolFromSmiles(smiles, sanitize=False)
        ),
        [
            'Cl[Ti]Cl',
            '[Ti]',
            '[Cl-]',
        ],
        molecule
    )
    root_pattern = Chem.MolFromSmiles("c1cccc1.c1cccc1", sanitize=False)
    chains = Chem.ReplaceCore(x, root_pattern, labelByIndex=True)
    if chains is None:
        return None
    pieces = Chem.GetMolFrags(chains, asMols=True)
    ligands = [Chem.MolToSmiles(x, True) for x in pieces]
    Rs = []
    for ligand in ligands:
        ligand = replace_rounds(ligand, [(r'\[C+@+H*\]', 'C'), (r"\[\d\*\]", "*")])
        if ligand.count("*") > 1:
            Rs.append(ligand.split('C(*)')[0])
            Rs.append("*C"+ligand.split("C(*)")[1])
            return [*Rs, *([None] * (7 - len(Rs)))]
        if len(ligand) < 5:
            continue
        else:
            Rs.append(re.sub(r"\[\d\*\]", "*", ligand))
    ligands = [*Rs, *([""] * (2 - len(Rs)))]
    return [*ligands, *([None] * (7 - len(ligands)))]


def onno(molecule):
    molecule = remove_titanium(molecule)
    molecule_br = remove_bridge(molecule, "NCCN", [1, 2])
    return [*get_ligands2(molecule_br, "NCC1=CC=CC=C1O", [0, 3, 4, 5, 6, 1])[0], get_bridge_idty(molecule, "N.N")]
    # return [*get_ligands(molecule, "NCC1=CC=CC=C1O", [0, 3, 4, 5, 6])[0], None]


def ono(molecule):
    molecule = remove_titanium(molecule)
    return [*get_ligands2(molecule, "NCC1=CC=CC=C1O", [0, 3, 4, 5, 6, 1])[0], None]
    # return [*get_ligands(molecule, "NCC1=CC=CC=C1O", [0, 3, 4, 5, 6]), None]


def onnoen(molecule):
    molecule = remove_titanium(molecule)
    molecule_br = remove_bridge(molecule, "NC(C=CC=C1)=C1N", [1, 6])
    # return [None, *get_ligands(molecule, "OC1=CC=CC=C1C=N", [5, 4, 3 ,2])[0], None]
    return [None, *get_ligands2(molecule_br, "OC1=CC=CC=C1C=N", [5, 4, 3, 2, 0])[0], get_bridge_idty(molecule, "N.N")]


def onnoalen2(molecule):
    # display(molecule)
    molecule = remove_titanium(molecule)
    # display(molecule)
    try_1 = remove_bridge(molecule, "NCCN", [1, 2])
    if try_1 is None:
        molecule_br = remove_bridge(molecule, "NC(C=CC=C1)=C1N", [1, 6])
    else:
        molecule_br = try_1
    try_1 = get_ligands2(molecule_br, "NCC1=CC=CC=C1O", [0, 3, 4, 5, 6, 1])
    if try_1 is None:
        return [None, *get_ligands2(molecule_br, "OC1=CC=CC=C1C=N", [5, 4, 3, 2, 7])[0], get_bridge_idty(molecule, "N.N")]
    return [*try_1[0], get_bridge_idty(molecule, "N.N")]


def onoen(molecule):
    molecule = remove_titanium(molecule)
    return [*get_ligands2(molecule, "OC1=CC=CC=C1C=N", [8, 5, 4, 3, 2, 7])[0], None]


extraction_dictionary = {
    'M': metallocene,
    'ONNO': onno,
    'ONO': ono,
    'ONNOen': onnoen,
    'ONNOalen': onnoalen2,
    'ONOen': onoen
}


class ComplexProcessor:

    def __init__(self, input_workbook_path):
        workbook = openpyxl.load_workbook(input_workbook_path, data_only=True)
        input_sheet = workbook['ChemOffice1']

        self.complexes = {}

        row = 2
        name = input_sheet.cell(row=row, column=1).value
        while name is not None:
            name = input_sheet.cell(row=row, column=1).value
            class_name = input_sheet.cell(row=row, column=3).value
            if class_name not in extraction_dictionary:
                row += 1
                continue
            smiles = input_sheet.cell(row=row, column=2).value
            self.complexes[row] = (smiles, class_name)
            row += 1

    def extract_ligands(self):
        self.ligands = {}
        for row in self.complexes:
            self.ligands[row] = extraction_dictionary[self.complexes[row][1]](Chem.MolFromSmiles(
                self.complexes[row][0],
                replacements={
                    '[Del]': '',
                    '()': '',
                },
                sanitize=False
            ))

    def find_ligand_values(self, data_file_path):
        self.ligand_values = {}
        value_finder = LigandValueFinder(data_file_path)
        for row in self.ligands:
            self.ligand_values[row] = list(
                map(lambda ligand: None if ligand is None else value_finder.get_values(ligand), self.ligands[row])
            )
        pass

    def display(self):
        rows = ""

        for row in self.ligands:
            rows += f"""
            <tr>
                <td>{row}</td>
                <td>{smiles_to_svg(self.complexes[row][0])}<br>{self.complexes[row][0]}</td>
                <td>{self.complexes[row][1]}</td>
                {''.join(
                list(
                    map(
                        lambda l_v: 
                        "<td>No ligand</td>" 
                        if l_v[0] is None else 
                        f'<td>{smiles_to_svg(l_v[0])}<br>{l_v[0]}: {l_v[1][:2]}<br>{smiles_to_svg(l_v[1][2])}</td>',
                        zip(self.ligands[row], self.ligand_values[row])
                    )
                )
            )}
            </tr>
            """

        html = f"""
        <table style="font-size: 8pt">
            <hr>
                <th>Row Index</th>
                <th>Molecule</th>
                <th>Class</th>
                <th>R<sup>1</sup></th>
                <th>R<sup>2</sup></th>
                <th>R<sup>3</sup></th>
                <th>R<sup>4</sup></th>
                <th>R<sup>5</sup></th>
                <th>R<sup>6</sup></th>
                <th>R<sup>7</sup></th>
            {rows}
        </table>
        """
        # f = open("out.html", "w")
        # f.write(html)
        # f.close()
        display(HTML(html))

    def write_to_sheet(self, output_workbook_path):
        workbook = Workbook()
        sheet = workbook.active

        titles = [
            'Complex',
            'Class',
            'R1',
            'HanschPi',
            'Hammett',
            'VdW',
            'R2',
            'HanschPi',
            'Hammett',
            'VdW',
            'R3',
            'HanschPi',
            'Hammett',
            'VdW',
            'R4',
            'HanschPi',
            'Hammett',
            'VdW',
            'R5',
            'HanschPi',
            'Hammett',
            'VdW',
            'R6',
            'HanschPi',
            'Hammett',
            'VdW',
            'R7',
            'HanschPi',
            'Hammett',
            'VdW',
        ]
        for (column, value) in zip(range(len(titles)), titles):
            sheet.cell(1, column + 1, value)

        for row in self.ligands:
            sheet.cell(row, 1, self.complexes[row][0])
            sheet.cell(row, 2, self.complexes[row][1])
            ligands = self.ligands[row][2]
            for i in range(7):
                sheet.cell(row, (4*(i+1))-1, 'No ligand' if self.ligands[row][i] is None else self.ligands[row][i])
                sheet.cell(row, (4*(i+1))+2, 'N/A' if self.ligands[row][i] is None else VdW_volume(self.ligands[row][i]))
                sheet.cell(row, (4 * (i + 1)), 'N/A' if self.ligands[row][i] is None else self.ligand_values[row][i][0])
                sheet.cell(row, (4 * (i + 1))+1, 'N/A' if self.ligands[row][i] is None else self.ligand_values[row][i][1])
        workbook.save(output_workbook_path)

if __name__ == '__main__':
    x = ComplexProcessor('../res/Titanium spreadsheet MkII - Copy.xlsx')
    x.extract_ligands()
    x.find_ligand_values('../res/substituent-parameters.tsv')