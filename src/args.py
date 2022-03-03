import argparse


class ArgumentParser:

    @staticmethod
    def get_args():
        """
        Load all provided cli args
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--file", "-f", action="store", required=True, help="The file used for training the model")
        parser.add_argument("--normalization", "-n", action="store", required=False,
                            help="The normalization method to use", default="s")
        parser.add_argument("--feature_range", "-fr", action="store", required=False,
                            help="The feature range to use for normalization. "
                                 "If normalization method selected does not use a feature range, this is ignored")

        return parser.parse_args()
