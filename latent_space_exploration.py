import sys

sys.path.append('/home/groups/EllrottLab/drug_resp/VAE/VAE_DL_II/venv/lib/python3.7/site-packages')

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
from library.coding_gene_vae.coding_gene_vae import CodingGeneVae
from library.vae.vae import CodingGeneModel
from typing import Tuple

base_path = Path("latent_space_generation")


def get_args():
    """
       Load all provided cli args
       """
    parser = argparse.ArgumentParser()

    parser.add_argument("-cg", "--coding_genes", action="store", required=False,
                        help="The file to use for coding genes")
    parser.add_argument("-ncg", "--non_coding_genes", action="store", required=False,
                        help="The file to use for non coding gene")
    parser.add_argument("-mf", "--molecular_fingerprint", action="store", required=False,
                        help="The file to use for the molecular fingerprint")
    parser.add_argument("-r", "--response_data", action="store", required=False,
                        help="The file containing the response data ")
    parser.add_argument("-lt", "--latent_space", action="store", required=True,
                        help="Defines the latent space dimensions")
    parser.add_argument("-s", "--scaling", action="store", required=False,
                        help="Which type of scaling should be used", choices=["min", "s"], default="s")
    parser.add_argument("-m", "--model", action="store", required=False, choices=["o", "n", "r", "cg"], default="o")
    return parser.parse_args()


inspection_version = '2022-06-13_run0'
test_variable = 9
# loss_inspection = 'Here is line 42 in latent_space_exploration.py'
template = 'Coding gene VAE test'
inspectDF = pd.DataFrame()
# inspectDF['Variable value at point in script'] = [test_variable]
inspectDF['Variable value at point in script'] = [test_variable]
inspectDF['Message from script'] = [template]
inspectDF.to_csv('inspectDF_' + inspection_version + '.tsv',
                 sep='\t')


def load_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data: pd.DataFrame = DataLoader.load_data(file_name=file_path)

    # Create split
    train_data, validation_data = SplitHandler.create_splits(input_data=data,
                                                             without_val=True)

    # Normalize
    train_data = Preprocessing.normalize(data=train_data,
                                         features=train_data.columns.tolist(),
                                         method=args.scaling)
    validation_data = Preprocessing.normalize(data=validation_data,
                                              features=validation_data.columns.tolist(),
                                              method=args.scaling)

    return train_data, validation_data


if __name__ == '__main__':
    args = get_args()

    if not base_path.exists():
        FolderManagement.create_directory(base_path)

    latent_space: int = int(args.latent_space)

    if args.model == 'o' or args.model == 'n':

        coding_gene_train_data, coding_gene_validation_data = load_data(args.coding_genes)
        non_coding_gene_train_data, non_coding_gene_validation_data = load_data(args.non_coding_genes)
        mf_train_data, mf_validation_data = load_data(args.molecular_fingerprint)

        amount_of_layers: dict = {
            "coding_genes": [coding_gene_train_data.shape[1], coding_gene_train_data.shape[1] / 2,
                             coding_gene_train_data.shape[1] / 3,
                             coding_gene_train_data.shape[1] / 4, latent_space],
            "non_coding_genes": [non_coding_gene_train_data.shape[1], non_coding_gene_train_data.shape[1] / 2,
                                 non_coding_gene_train_data.shape[1] / 3, non_coding_gene_train_data.shape[1] / 4,
                                 latent_space],
            "molecular_fingerprint": [mf_train_data.shape[1], mf_train_data.shape[1] / 2,
                                      mf_train_data.shape[1] / 3, mf_train_data.shape[1] / 4,
                                      latent_space]
        }

        if args.model == 'o':
            vae: ThreeEncoderArchitecture = ThreeEncoderArchitecture()
            model, encoder, decoder, history = vae.build_three_variational_auto_encoder(
                training_data=(coding_gene_train_data, non_coding_gene_train_data, mf_train_data),
                validation_data=(
                    coding_gene_validation_data, non_coding_gene_validation_data,
                    mf_validation_data),
                amount_of_layers=amount_of_layers,
                embedding_dimension=latent_space, folder=str(base_path)
            )

        elif args.model == 'n':
            vae: MultiThreeEncoderArchitecture = MultiThreeEncoderArchitecture()
            vae.build_three_variational_auto_encoder(
                training_data=(coding_gene_train_data, non_coding_gene_train_data, mf_train_data),
                validation_data=(
                    coding_gene_validation_data,
                    non_coding_gene_validation_data,
                    mf_validation_data),
                amount_of_layers=amount_of_layers,
                embedding_dimension=latent_space, folder=str(base_path)
            )

            history = vae.history

    elif args.model == 'cg':

        if args.coding_genes is None:
            raise ValueError(
                "Coding Gene VAE needs data. Please specify the data by using --coding_genes as an cli argument")

        coding_gene_train_data, coding_gene_validation_data = load_data(args.coding_genes)
        vae: CodingGeneModel = CodingGeneModel(input_dimensions=coding_gene_train_data.shape[1],
                                               save_path=str(base_path), embedding_dimension=latent_space)
        vae.compile_model()
        vae.train_model(train_data=coding_gene_train_data, validation_data=coding_gene_validation_data)

        history = vae.history

    else:
        raise ValueError("Please specify a model to run")

    plotter: Plotting = Plotting(base_path=str(base_path))
    plotter.plot_model_performance(history=history, file_name="model_history")
