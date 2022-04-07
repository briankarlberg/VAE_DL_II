import pandas as pd
import numpy as np
import time # Optional in script

array_arg = 0

config = np.load('ctrp_unique_507.npy', allow_pickle=True)

start = time.time()
cell_lines_by_genes = pd.read_csv(
    '/Users/karlberb/work/drug_response/drug_response/CCLE_gene_expression_by_cell_line.tsv',
    sep = '\t')
end = time.time() - start
# print(str(round(end, 1)))

CTRPv2 = pd.read_csv(
    '/Users/karlberb/work/drug_response/drug_response/drug_response_CTRPv2.tsv',
    sep = '\t')
CTRPv2_unique_compounds = CTRPv2.Drug.unique()

drug_frame = CTRPv2[CTRPv2.Drug == CTRPv2_unique_compounds[array_arg]]

drug_cell_lines = drug_frame.Cell_line

clg = cell_lines_by_genes.Cell_line
subset_cell_lines_direct = pd.Series(list(set(clg) & set(drug_cell_lines)))

subset_of_drug_frame_gexp_hits = drug_frame[
    drug_frame.Cell_line.isin(subset_cell_lines_direct)].copy()

expression_cell_line_subset = cell_lines_by_genes[
    cell_lines_by_genes['Cell_line'].isin(drug_cell_lines)].copy()

subset_of_drug_frame_gexp_hits.set_index('Cell_line',inplace = True)
expression_cell_line_subset.set_index('Cell_line',inplace = True)

expression_cell_line_subset.drop(columns = ['Project','Label'],axis = 1,inplace = True)
subset_of_drug_frame_gexp_hits.drop(columns = ['Drug'],axis = 1,inplace = True)

vae_in = pd.concat([subset_of_drug_frame_gexp_hits,
                    expression_cell_line_subset],
                   axis = 1)