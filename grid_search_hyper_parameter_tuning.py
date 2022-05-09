import pandas as pd
from bayes_opt import BayesianOptimization
from library.three_encoder_vae.three_encoder_architecture import ThreeEncoderArchitecture
from functools import partial
from library.preprocessing.splits import SplitHandler
import argparse
import os
from library.data.data_loader import DataLoader

learning_rates = [0.0001, 0.0002, 0.0003, 0.0004]
amount_of_layers = [5, 8, 12]
latent_space = [200, 500, 1000]
loss_function = ['adam', 'sme']


def get_args():
    """
       Load all provided cli args
       """
    parser = argparse.ArgumentParser()

    parser.add_argument("-cg", "--coding_genes", action="store", required=True, help="The file to use for coding genes")
    parser.add_argument("-ncg", "--encoding_genes", action="store", required=True,
                        help="The file to use for non coding gene")
    parser.add_argument("-mf", "--molecular_fingerprint", action="store", required=True,
                        help="The file to use for the molecular fingerprint")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    coding_gene_data: pd.DataFrame = DataLoader.load_data(args.coding_genes)
    non_coding_gene_data: pd.DataFrame = DataLoader.load_data(args.non_coding_genes)
    molecular_fingerprint_data: pd.DataFrame = DataLoader.load_data(args.molecular_fingerprint)

    coding_gene_train_data, coding_gene_validation_data = SplitHandler.create_splits(input_data=coding_gene_data,
                                                                                     without_val=True)
    non_coding_gene_train_data, non_coding_gene_validation_data = SplitHandler.create_splits(
        input_data=non_coding_gene_data, without_val=True)
    molecular_fingerprint_train_data, molecular_fingerprint_validation_data = SplitHandler.create_splits(
        input_data=molecular_fingerprint_data, without_val=True)

    # define constants during run time
    fit_with_partial = partial(ThreeEncoderArchitecture.build_three_variational_auto_encoder, coding_gene_train_data,
                               coding_gene_validation_data, non_coding_gene_train_data,
                               non_coding_gene_validation_data, molecular_fingerprint_train_data,
                               molecular_fingerprint_validation_data)

    lower_layer_boundaries: dict = {
        "coding_genes": [1000, 500, 200],
        "non_coding_genes": [1000, 500, 200],
        "molecular_fingerprints": [1000, 500, 200],
    }

    upper_layer_boundaries: dict = {
        "coding_genes": [2000, 1000, 500],
        "non_coding_genes": [2000, 1000, 500],
        "molecular_fingerprints": [2000, 1000, 500],
    }

    # Bounded region of parameter space
    boundaries = {'embedding_dimension': (500, 1000),
                  'amount_of_layers': (lower_layer_boundaries, upper_layer_boundaries),
                  'learning_rate': (0.0001, 0.1)}

    optimizer = BayesianOptimization(
        f=fit_with_partial,
        pbounds=boundaries,
        random_state=1,
    )

    optimizer.maximize(init_points=2, n_iter=3)

    print(optimizer.max)

