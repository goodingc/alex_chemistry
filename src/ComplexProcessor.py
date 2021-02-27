import re
from functools import reduce

import openpyxl
from IPython.display import HTML, display
from openpyxl import Workbook
from rdkit import Chem

from src.LigandValueFinder import LigandValueFinder
from src.utils import remove_titanium, remove_bridge, get_ligands2, smiles_to_svg


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
        if len(ligand) < 5:
            continue
        else:
            Rs.append(re.sub(r"\[\d\*\]", "*", ligand))
    ligands = [*Rs, *([""] * (2 - len(Rs)))]
    return [*ligands, *([None] * (6 - len(ligands)))]


def onno(molecule):
    molecule = remove_titanium(molecule)
    molecule = remove_bridge(molecule, "NCCN", [1, 2])
    return [*get_ligands2(molecule, "NCC1=CC=CC=C1O", [0, 3, 4, 5, 6, 1])[0]]
    # return [*get_ligands(molecule, "NCC1=CC=CC=C1O", [0, 3, 4, 5, 6])[0], None]


def ono(molecule):
    molecule = remove_titanium(molecule)
    return [*get_ligands2(molecule, "NCC1=CC=CC=C1O", [0, 3, 4, 5, 6, 1])[0]]
    # return [*get_ligands(molecule, "NCC1=CC=CC=C1O", [0, 3, 4, 5, 6]), None]


def onnoen(molecule):
    molecule = remove_titanium(molecule)
    molecule = remove_bridge(molecule, "NC(C=CC=C1)=C1N", [1, 6])
    # return [None, *get_ligands(molecule, "OC1=CC=CC=C1C=N", [5, 4, 3 ,2])[0], None]
    return [None, *get_ligands2(molecule, "OC1=CC=CC=C1C=N", [5, 4, 3, 2, 0])[0]]


def onnoalen2(molecule):
    molecule = remove_titanium(molecule)
    try_1 = remove_bridge(molecule, "NCCN", [1, 2])
    if try_1 is None:
        molecule = remove_bridge(molecule, "NC(C=CC=C1)=C1N", [1, 6])
    else:
        molecule = try_1
    try_1 = get_ligands2(molecule, "NCC1=CC=CC=C1O", [0, 3, 4, 5, 6, 1])
    if try_1 is None:
        return [None, *get_ligands2(molecule, "OC1=CC=CC=C1C=N", [5, 4, 3, 2, 7])[0]]
    return [*try_1[0], None]


def onoen(molecule):
    molecule = remove_titanium(molecule)
    return get_ligands2(molecule, "OC1=CC=CC=C1C=N", [8, 5, 4, 3, 2, 7])[0]


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
        <table>
            <hr>
                <th>Row Index</th>
                <th>Molecule</th>
                <th>Class</th>
                <th>Ligand 1</th>
                <th>Ligand 2</th>
                <th>Ligand 3</th>
                <th>Ligand 4</th>
                <th>Ligand 5</th>
                <th>Ligand 6</th>
            </hr>
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
            'R2',
            'R3',
            'R4',
            'R5',
            'R6',
        ]
        for (column, value) in zip(range(len(titles)), titles):
            sheet.cell(1, column + 1, value)

        for row in self.ligands:
            sheet.cell(row, 1, self.ligands[row][0])
            sheet.cell(row, 2, self.ligands[row][1])
            ligands = self.ligands[row][2]
            for i in range(6):
                sheet.cell(row, i + 3, 'No ligand' if ligands[i] is None else ligands[i])

        workbook.save(output_workbook_path)

if __name__ == '__main__':
    x = ComplexProcessor('../res/Titanium spreadsheet MkII - Copy.xlsx')
    x.extract_ligands()
    x.find_ligand_values('../res/substituent-parameters.tsv')