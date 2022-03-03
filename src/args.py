import argparse


class ArgumentParser:

    @staticmethod
    def get_args():
        """
        Load all provided cli args
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--file", action="store", required=True, help="The file used for training the model")

        return parser.parse_args()
