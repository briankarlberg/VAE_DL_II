import pandas as pd
import argparse
from library.data.data_loader import DataLoader
from library.preprocessing.splits import SplitHandler
from library.preprocessing.preprocessing import Preprocessing
from library.three_encoder_vae.three_encoder_architecture import ThreeEncoderArchitecture
from library.plotting.plots import Plotting
from library.multi_three_encoder_vae.multi_three_encoder_architecture import MultiThreeEncoderArchitecture
from pathlib import Path
from library.data.folder_management import FolderManagement
from library.regression_vae.regression_vae import RegressionVAE

base_path = Path("latent_space_generation")


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
    parser.add_argument("-r", "--response_data", action="store", required=False,
                        help="The file containing the response data ")
    parser.add_argument("-lt", "--latent_space", action="store", required=True,
                        help="Defines the latent space dimensions")
    parser.add_argument("-s", "--scaling", action="store", required=False,
                        help="Which type of scaling should be used", choices=["min", "s"], default="s")
    parser.add_argument("-m", "--model", action="store", required=False, choices=["o", "n", "r"], default="o")
    return parser.parse_args()

test_variable = 0
loss_inspection = 'Here is line 37 in latent_space_exploration.py'
inspectDF = pd.DataFrame()
inspectDF['Variable value at point in script'] = [test_variable]
inspectDF['Message from script'] = [loss_inspection]
inspectDF.to_csv('inspectDF_'+inspection_version+'_.tsv',
                sep = '\t')

if __name__ == '__main__':
    args = get_args()

    if not base_path.exists():
        FolderManagement.create_directory(base_path)

    coding_gene_data: pd.DataFrame = DataLoader.load_data(args.coding_genes)
    non_coding_gene_data: pd.DataFrame = DataLoader.load_data(args.non_coding_genes)
    molecular_fingerprint_data: pd.DataFrame = DataLoader.load_data(args.molecular_fingerprint)

    # Create split
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

    latent_space: int = args.latent_space
    amount_of_layers: dict = {
        "coding_genes": [coding_gene_data.shape[1], coding_gene_data.shape[1] / 2, coding_gene_data.shape[1] / 3,
                         coding_gene_data.shape[1] / 4, latent_space],
        "non_coding_genes": [non_coding_gene_data.shape[1], coding_gene_data.shape[1] / 2,
                             coding_gene_data.shape[1] / 3, coding_gene_data.shape[1] / 4, latent_space],
        "molecular_fingerprint": [molecular_fingerprint_train_data.shape[1], coding_gene_data.shape[1] / 2,
                                  coding_gene_data.shape[1] / 3, coding_gene_data.shape[1] / 4, latent_space]
    }

    if args.model == 'o':
        vae: ThreeEncoderArchitecture = ThreeEncoderArchitecture()
        model, encoder, decoder, history = vae.build_three_variational_auto_encoder(
            training_data=(coding_gene_train_data, non_coding_gene_train_data, molecular_fingerprint_train_data),
            validation_data=(
                coding_gene_validation_data, non_coding_gene_validation_data, molecular_fingerprint_validation_data),
            amount_of_layers=amount_of_layers,
            embedding_dimension=latent_space, folder=str(base_path)
        )

    elif args.model == 'n':
        vae: MultiThreeEncoderArchitecture = MultiThreeEncoderArchitecture()
        vae.build_three_variational_auto_encoder(
            training_data=(coding_gene_train_data, non_coding_gene_train_data, molecular_fingerprint_train_data),
            validation_data=(
                coding_gene_validation_data, non_coding_gene_validation_data, molecular_fingerprint_validation_data),
            amount_of_layers=amount_of_layers,
            embedding_dimension=latent_space, folder=str(base_path)
        )

        history = vae.history

    elif args.model == 'r':

        if args.response_data is None:
            raise ValueError("Please provide response data, when using the regression vae")

        y: pd.DataFrame = DataLoader.load_data(file_name=args.response_data)

        y = Preprocessing.normalize(data=y, features=y.columns.tolist(), method=args.scaling)

        amount_of_layers["decoder"] = [latent_space, latent_space]

        vae: RegressionVAE = RegressionVAE()
        vae.build_regression_vae(
            training_data=(coding_gene_train_data, non_coding_gene_train_data, molecular_fingerprint_train_data),
            validation_data=(
                coding_gene_validation_data, non_coding_gene_validation_data, molecular_fingerprint_validation_data),
            amount_of_layers=amount_of_layers,
            embedding_dimension=latent_space, folder=str(base_path),
            target_value=y)

        history = vae.history

    else:
        raise ValueError("Please specify a model to run")

    plotter: Plotting = Plotting(base_path=str(base_path))
    plotter.plot_model_performance(history=history, file_name="model_history")