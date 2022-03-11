from args import ArgumentParser
import pandas as pd
from preprocessing.preprocessing import Preprocessing
from preprocessing.splits import create_splits
from preprocessing.feature_selection import feature_selection
from vae.vae import DRT_VAE
from pathlib import Path
from evaluation.evaluation import Evaluation

if __name__ == "__main__":
    args = ArgumentParser.get_args()
    data = pd.read_csv(args.file, sep='\t', index_col=0)
    # data = feature_selection(data)
    Preprocessing.normalize(data, args.normalization, args.feature_range)
    X_train, X_val, X_test = create_splits(data)

    X_train = Preprocessing.normalize(data=X_train, method=args.normalization, feature_range=args.feature_range)
    X_val = Preprocessing.normalize(data=X_val, method=args.normalization, feature_range=args.feature_range)
    X_test = Preprocessing.normalize(data=X_test, method=args.normalization, feature_range=args.feature_range)

    print(X_train, X_val, X_test)
    input()
    model, enc, dec, hist = DRT_VAE.build_model(X_train, X_val, X_train.shape[1],
                                                embedding_dimension=20)
    mean, var, latent_space = enc.predict(X_test)
    prediction = dec.predict(latent_space)
    r2_scores = Evaluation.calculate_r2_score(X_test, prediction, list(data.columns))

    file_name = Path(args.file).stem
    r2_scores.to_csv(Path("results", f"{file_name}_r2_score.csv"), index=False)
