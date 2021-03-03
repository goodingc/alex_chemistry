from src.ComplexProcessor import ComplexProcessor


if __name__ == '__main__':
    x = ComplexProcessor('./res/Titanium spreadsheet MkII - Copy.xlsx')
    x.extract_ligands()
    x.find_ligand_values('./res/substituent-parameters.tsv')
    x.display()