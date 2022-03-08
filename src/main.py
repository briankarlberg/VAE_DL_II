from args import ArgumentParser
import pandas as pd
from preprocessing.preprocessing import Preprocessing
from preprocessing.splits import create_splits
from preprocessing.feature_selection import feature_selection
from vae.vae import DRT_VAE
from plotting.plots import plot_model_performance

if __name__ == "__main__":
    args = ArgumentParser.get_args()
    data = pd.read_csv(args.file, sep='\t')
    data = feature_selection(data)
    print(data)
    input()
    normalized_data: pd.DataFrame = Preprocessing.normalize(data, args.normalization, args.feature_range)
    print(normalized_data)
    X_train, X_val, X_test = create_splits(normalized_data)
    print(X_train, X_val, X_test)
    model, enc, dec, hist = DRT_VAE.build_model(X_train, X_val, X_train.shape[1], 10)
    plot_model_performance(hist, None, 'plot_test_v1')
