from args import ArgumentParser
import pandas as pd
from preprocessing.preprocessing import Preprocessing

if __name__ == "__main__":
    args = ArgumentParser.get_args()
    data = pd.read_csv(args.file, sep='\t')
    print(data)
    input()
    normalized_data: pd.DataFrame = Preprocessing.normalize(data, args.normalization, args.feature_range)
    print(normalized_data)