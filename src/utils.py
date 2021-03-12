from copy import deepcopy
from functools import reduce
from typing import List, Tuple, Optional

import numpy as np
from IPython.core.display import display, HTML
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import re

from rdkit.Chem.rdchem import Mol


def replace_rounds(haystack: str, replacements: List[Tuple[str, str]]) -> str:
    return reduce(lambda haystack, sub: re.sub(sub[0], sub[1], haystack), replacements, haystack)


def is_transition_metal(at):
    n = at.GetAtomicNum()
    return (n>=22 and n<=29) or (n>=40 and n<=47) or (n>=72 and n<=79)


def set_dative_bonds(mol, fromAtoms=(7,8)):
    pt = Chem.GetPeriodicTable()
    rwmol = Chem.RWMol(mol)
    rwmol.UpdatePropertyCache(strict=False)
    metals = [at for at in rwmol.GetAtoms() if is_transition_metal(at)]
    for metal in metals:
        for nbr in metal.GetNeighbors():
            if nbr.GetAtomicNum() in fromAtoms and \
               nbr.GetExplicitValence()>pt.GetDefaultValence(nbr.GetAtomicNum()) and \
               rwmol.GetBondBetweenAtoms(nbr.GetIdx(),metal.GetIdx()).GetBondType() == Chem.BondType.SINGLE:
                rwmol.RemoveBond(nbr.GetIdx(),metal.GetIdx())
                rwmol.AddBond(nbr.GetIdx(),metal.GetIdx(),Chem.BondType.DATIVE)
    return rwmol


def smiles_to_svg(smiles: str) -> str:
    if smiles.count("Ti") > 0:
        smiles = replace_rounds(smiles, [
            (r'c\dcccc\d', "C1=CC=CC1"),
            (r'.c\d', '.C1'),
            (r'cccc\d', '=CC=CC1')
        ])
    mol = Chem.MolFromSmiles(
        smiles,
        replacements={
            '[Del]': '',
            '()': '',
        },
        sanitize=False
    )
    mol = set_dative_bonds(mol)
    mc = Chem.Mol(mol.ToBinary())
    for atom in mc.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    drawer = rdMolDraw2D.MolDraw2DSVG(250, 150)
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return f"<span title='{smiles}'>{svg.replace('svg:', '')}</span>"


def display_numbered(mol: Mol):
    mol = deepcopy(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    display(mol)


def get_largest_fragment(mol: Mol) -> Mol:
    frags = list(map(lambda s: Chem.MolFromSmiles(s, sanitize=False), Chem.MolToSmiles(mol).split('.')))
    i = np.argmax(list(map(lambda f: f.GetNumAtoms(), frags)))
    return frags[i]


def get_ligands(molecule: Mol, root_pattern_smiles: str, ligand_root_indices: List[int]) -> Optional[List[str]]:
    root_pattern = Chem.MolFromSmiles(root_pattern_smiles)
    chains = Chem.ReplaceCore(molecule, root_pattern, labelByIndex=True)
    if chains is None:
        return None
    pieces = Chem.GetMolFrags(chains, asMols=True)
    ligands = [Chem.MolToSmiles(x, True) for x in pieces]
    root_indicies = []
    for index in ligand_root_indices:
        if index == 0:
            root_indicies.append('*')
        else:
            root_indicies.append('[{}*]'.format(index))

    def get_ligand(index: str) -> str:
        for ligand in ligands:
            if index == "*":
                if "*]" in ligand:
                    continue
            if index in ligand:
                return re.sub(r"\[\d\*\]", "*", ligand)
        if index not in ligands:
            return ''

    return list(map(get_ligand, root_indicies))


def remove_titanium(molecule: Mol) -> Mol:
    molecule = Chem.DeleteSubstructs(molecule, Chem.MolFromSmiles('[Ti]', sanitize=False))
    smiles = Chem.MolToSmiles(molecule)
    return Chem.MolFromSmiles(list(filter(lambda s: len(s) > 18, smiles.split('.')))[0])


def get_bridge_idty(ligand: Mol, class_pattern: str) -> Optional[list[str]]:
    ligand = Chem.DeleteSubstructs(ligand, Chem.MolFromSmiles("[N+](=O)[O-]", sanitize=False))
    root_pattern = Chem.MolFromSmiles(class_pattern, sanitize=False)
    chains = Chem.ReplaceCore(ligand, root_pattern)
    # display(chains)
    if chains is None:
        return None
    pieces = Chem.GetMolFrags(chains, asMols=True)
    ligands = sorted([Chem.MolToSmiles(x, True) for x in pieces], key=len)
    bridge = []
    for ligand in ligands:
        if (Chem.MolFromSmiles(ligand)).GetNumAtoms() < 20 and ligand.count("*") > 1:
            bridge.append(re.sub(r"\[\d\*\]", "*", ligand))
    return bridge



def remove_bridge(molecule: Mol, root_pattern_smiles: str, removal_indices: List[int]) -> Optional[Mol]:
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


def display_table(headers: List[str], data: List[List[str]]):
    html = f"""
    <table>
        <hr>
            {''.join(map(lambda header: f'<th>{header}</th>', headers))}
        </hr>
        {''.join(map(lambda row: f"<tr>{''.join(map(lambda cell: f'<td>{cell}</td>', row))}</tr>", data))}
    </table>
    """
    display(HTML(html))


def smiles_html(smiles: str) -> str:
    if len(smiles) > 10:
        return f"""
        <span title="{smiles}">{smiles[:min(len(smiles), 10)]}...</span>
        """
    return smiles
