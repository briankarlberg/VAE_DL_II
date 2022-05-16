import pandas as pd
import argparse
from library.data.data_loader import DataLoader
from library.preprocessing.splits import SplitHandler
from library.preprocessing.preprocessing import Preprocessing
from library.three_encoder_vae.three_encoder_architecture import ThreeEncoderArchitecture
from library.plotting.plots import Plotting

base_path = "latent_space_generation"


def get_args():
    """
       Load all provided cli args
       """
    parser = argparse.ArgumentParser()

    parser.add_argument("-cg", "--coding_genes", action="store", required=True, help="The file to use for coding genes")
    parser.add_argument("-ncg", "--non_coding_genes", action="store", required=True,
                        help="The file to use for non coding gene")
    parser.add_argument("-mf", "--molecular_fingerprint", action="store", required=True,
                        help="The file to use for the molecular fingerprint")
    parser.add_argument("-s", "--scaling", action="store", required=False,
                        help="Which type of scaling should be used", choices=["min", "s"], default="s")
    return parser.parse_args()


if __name__ == '__main__':
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

    # Normalize
    coding_gene_train_data = Preprocessing.normalize(data=coding_gene_train_data,
                                                     features=coding_gene_data.columns.tolist(), method=args.scaling)
    coding_gene_validation_data = Preprocessing.normalize(data=coding_gene_validation_data,
                                                          features=coding_gene_data.columns.tolist(),
                                                          method=args.scaling)

    non_coding_gene_train_data = Preprocessing.normalize(data=non_coding_gene_train_data,
                                                         features=non_coding_gene_data.columns.tolist(),
                                                         method=args.scaling)
    non_coding_gene_validation_data = Preprocessing.normalize(data=non_coding_gene_validation_data,
                                                              features=non_coding_gene_data.columns.tolist(),
                                                              method=args.scaling)

    molecular_fingerprint_train_data = Preprocessing.normalize(data=molecular_fingerprint_train_data,
                                                               features=molecular_fingerprint_data.columns.tolist(),
                                                               method=args.scaling)
    molecular_fingerprint_validation_data = Preprocessing.normalize(data=molecular_fingerprint_validation_data,
                                                                    features=molecular_fingerprint_data.columns.tolist(),
                                                                    method=args.scaling)

    amount_of_layers: dict = {
        "coding_genes": [15000, 10000, 5000, 2500, 1000, 500, 200],
        "non_coding_genes": [30000, 25000, 20000, 15000, 10000, 5000, 2500, 1000, 500, 200],
        "molecular_fingerprint": [1000, 500, 200],
    }

    vae: ThreeEncoderArchitecture = ThreeEncoderArchitecture(base_path=base_path, csv_logger=False, plot_model=True)
    input()
    model, encoder, decoder, history = vae.build_three_variational_auto_encoder(
        training_data=(coding_gene_train_data, non_coding_gene_train_data, molecular_fingerprint_train_data),
        validation_data=(
            coding_gene_validation_data, non_coding_gene_validation_data, molecular_fingerprint_validation_data),
        amount_of_layers=amount_of_layers,
        embedding_dimension=1000
    )

    plotter: Plotting = Plotting(base_path=base_path)
    plotter.plot_model_performance(history=history, file_name="Model History")
