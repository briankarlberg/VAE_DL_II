from library.three_encoder_vae.three_encoder_architecture import ThreeEncoderArchitecture
import numpy as np
import pandas as pd

if __name__ == '__main__':
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))
    df2 = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))
    df3 = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))

    layers: dict = {
        "coding_genes": [5, 4, 3, 2],
        "non_coding_genes": [5, 4, 3, 2],
        "molecular_fingerprint": [5, 4, 3, 2],
    }

    ThreeEncoderArchitecture.build_three_variational_auto_encoder(training_data=(df, df2, df3),
                                                                  validation_data=(df, df2, df3),
                                                                  output_dimensions=df3.shape[1],
                                                                  embedding_dimension=5,
                                                                  amount_of_layers=layers)
