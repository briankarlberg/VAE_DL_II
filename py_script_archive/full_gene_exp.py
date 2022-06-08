import pandas as pd
import argparse
from pathlib import Path

results_path = Path("../results")

parser = argparse.ArgumentParser()
parser.add_argument("--drug_index", "-di", action="store",
                    type=int, required=True, help="Index number of drug in config")
parser.add_argument("--cell_lines", "-cl", action="store", type=str, required=True,
                    help="The file containing all cell lines")

parser.add_argument("--drug_response", "-dr", action="store", type=str, required=True,
                    help="The file containing all drug responses")

args = parser.parse_args()
drug_index = args.drug_index
# File location for cell lines data
cell_line_by_genes_file = args.cell_lines
# File location for drug response data
drug_response_file = args.drug_response

# config = np.load('./data/ctrp_unique_507.npy', allow_pickle=True)

# Load cell line by genes dataset
cell_lines_by_genes = pd.read_csv(cell_line_by_genes_file, sep='\t')

# Load drug response dataset
CTRPv2 = pd.read_csv(drug_response_file, sep='\t')
CTRPv2_unique_compounds = CTRPv2.Drug.unique()

drug_frame = CTRPv2[CTRPv2.Drug == CTRPv2_unique_compounds[drug_index]]

drug_cell_lines = drug_frame.Cell_line

clg = cell_lines_by_genes.Cell_line
subset_cell_lines_direct = pd.Series(list(set(clg) & set(drug_cell_lines)))

subset_of_drug_frame_gexp_hits = drug_frame[
    drug_frame.Cell_line.isin(subset_cell_lines_direct)].copy()

expression_cell_line_subset = cell_lines_by_genes[
    cell_lines_by_genes['Cell_line'].isin(drug_cell_lines)].copy()

subset_of_drug_frame_gexp_hits.set_index('Cell_line', inplace=True)
expression_cell_line_subset.set_index('Cell_line', inplace=True)

expression_cell_line_subset.drop(columns=['Project', 'Label'], axis=1, inplace=True)
subset_of_drug_frame_gexp_hits.drop(columns=['Drug'], axis=1, inplace=True)

vae_in = pd.concat([subset_of_drug_frame_gexp_hits,
                    expression_cell_line_subset],
                   axis=1)

# Create results path if it does not exist
if not results_path.exists():
    results_path.mkdir(parents=True, exist_ok=True)

# Write data frame to results folder
vae_in.to_csv(Path(results_path, '../full_expression_test.tsv'), sep='\t', index=False)
