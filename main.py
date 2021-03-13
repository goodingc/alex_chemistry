from src.ComplexProcessor import ComplexProcessor


if __name__ == '__main__':
    x = ComplexProcessor('./res/Titanium spreadsheet MkII - Copy.xlsx', limit=50)
    x.extract_ligands()
    x.find_ligand_values_thread_pool('./res/substituent-parameters.tsv')
    print(0)