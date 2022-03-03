from args import ArgumentParser
import pandas as pd

if __name__ == "__main__":
    args = ArgumentParser.get_args()
    data = pd.read_csv(args.file, sep = '\t')
    print(data)

