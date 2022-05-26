from keras import layers, regularizers
import pandas as pd
import keras
from library.multi_three_encoder_vae.sampling import Sampling
from tensorflow.keras.layers import Dense, Dropout, Input, Multiply
from tensorflow.keras.models import Model
from typing import Tuple
from keras.callbacks import TerminateOnNaN, CSVLogger, EarlyStopping
import os
from keras import backend as K
from tensorflow.keras.losses import MeanSquaredError


# https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf


class RegressionVAE:

    def __init__(self):
        self._coding_gene_encoder: Model = None
        self._non_coding_gene_encoder: Model = None
        self._molecular_fingerprints_encoder: Model = None
        self._decoder: Model = None

        self._vae: Model = None
        self._history = None

    @property
    def coding_gene_encoder(self) -> Model:
        return self._coding_gene_encoder

    @property
    def non_coding_gene_encoder(self) -> Model:
        return self._non_coding_gene_encoder

    @property
    def molecular_fingerprints_encoder(self) -> Model:
        return self._molecular_fingerprints_encoder

    @property
    def decoder(self) -> Model:
        return self._decoder

    @property
    def vae(self):
        return self._vae

    @property
    def history(self):
        return self._history

    def build_regression_vae(self, training_data: Tuple,
                             validation_data: Tuple,
                             target_value: pd.DataFrame,
                             embedding_dimension: int,
                             amount_of_layers: dict,
                             folder: str,
                             activation='relu',
                             learning_rate: float = 1e-3,
                             optimizer: str = "adam"):
        """
        Compiles the vae based on the given input parameters
        """

        if len(training_data) != 3:
            raise ValueError("Training and validation data must contain two datasets!")

        embedding_dimension = int(embedding_dimension)

        coding_gene_training_data: pd.DataFrame = training_data[0]
        non_coding_gene_training_data: pd.DataFrame = training_data[1]
        molecular_fingerprints_training_data: pd.DataFrame = training_data[2]

        coding_gene_validation_data: pd.DataFrame = validation_data[0]
        non_coding_gene_validation_data: pd.DataFrame = validation_data[1]
        molecular_fingerprints_validation_data: pd.DataFrame = validation_data[2]

        coding_gene_layers: list = amount_of_layers.get("coding_genes")
        non_coding_gene_layers: list = amount_of_layers.get("non_coding_genes")
        molecular_fingerprint_layers: list = amount_of_layers.get("molecular_fingerprint")
        decoder_layers: list = amount_of_layers.get("decoder")

        r = regularizers.l1_l2(10e-5)

        # Switch network when layers are redefined

        self._coding_gene_encoder, coding_z_mean, coding_z_log_var = self.__create_encoder_model(
            input_dimensions=coding_gene_training_data.shape[1], activation=activation,
            layer_dimensions=coding_gene_layers, r=r, model_name="coding_genes_encoder",
            embedding_dimensions=embedding_dimension)

        self._coding_gene_encoder.summary()

        self._non_coding_gene_encoder, non_coding_z_mean, non_coding_z_log_var = self.__create_encoder_model(
            input_dimensions=non_coding_gene_training_data.shape[1], activation=activation,
            layer_dimensions=non_coding_gene_layers, r=r, model_name="non_coding_genes_encoder",
            embedding_dimensions=embedding_dimension)

        self._non_coding_gene_encoder.summary()

        self._molecular_fingerprints_encoder, mf_z_mean, mf_z_log_var = self.__create_encoder_model(
            input_dimensions=molecular_fingerprints_training_data.shape[1], activation=activation,
            layer_dimensions=molecular_fingerprint_layers, r=r, model_name="molecular_fingerprint_encoder",
            embedding_dimensions=embedding_dimension)

        self._molecular_fingerprints_encoder.summary()

        self._decoder: Model = self.__create_decoder_model(
            input_dimensions=embedding_dimension, activation=activation,
            layer_dimensions=decoder_layers, r=r, model_name="decoder")

        self._decoder.summary()

        #   VAE loss terms w/ KL divergence
        def CodingLoss(true, pred):
            reconstruction_loss_fn = MeanSquaredError()
            recon_loss = reconstruction_loss_fn(true, pred)
            kl_loss = 1 + coding_z_log_var * 2 - K.square(coding_z_mean) - K.exp(coding_z_log_var * 2)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5

            vae_loss = K.mean(recon_loss + kl_loss)
            return vae_loss / 2

        def NonCodingLoss(true, pred):
            reconstruction_loss_fn = MeanSquaredError()
            recon_loss = reconstruction_loss_fn(true, pred)
            kl_loss = 1 + non_coding_z_log_var * 2 - K.square(non_coding_z_mean) - K.exp(non_coding_z_log_var * 2)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5

            vae_loss = K.mean(recon_loss + kl_loss)
            return vae_loss / 2

        def MFLoss(true, pred):
            reconstruction_loss_fn = MeanSquaredError()
            recon_loss = reconstruction_loss_fn(true, pred)
            kl_loss = 1 + mf_z_log_var * 2 - K.square(mf_z_mean) - K.exp(mf_z_log_var * 2)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5

            vae_loss = K.mean(recon_loss + kl_loss)
            return vae_loss / 2

        def CombLoss(true, pred):
            l1 = CodingLoss(true, pred)
            l2 = NonCodingLoss(true, pred)
            l3 = MFLoss(true, pred)
            return l1 + l2 + l3

        output = self._decoder(Multiply()([self._coding_gene_encoder.output[2], self._non_coding_gene_encoder.output[2],
                                           self._molecular_fingerprints_encoder.output[2]]))
        self._vae = Model(inputs=[self._coding_gene_encoder.input, self._non_coding_gene_encoder.input,
                                  self._molecular_fingerprints_encoder.input], outputs=output, name="vae")

        losses = {"decoder": CombLoss}
        loss_weights = {"decoder": 1.0}

        self._vae.compile(loss=losses, loss_weights=loss_weights, optimizer=optimizer)
        self._vae.summary()

        callbacks = []

        early_stopping = EarlyStopping(monitor="reconstruction_loss",
                                       mode="min", patience=5,
                                       restore_best_weights=True)
        callbacks.append(early_stopping)

        term_nan = TerminateOnNaN()
        callbacks.append(term_nan)

        csv_logger = CSVLogger(os.path.join(folder, 'training.log'), separator='\t')
        callbacks.append(csv_logger)

        self._history = self._vae.fit(
            x=[coding_gene_training_data, non_coding_gene_training_data, molecular_fingerprints_training_data],
            y=target_value,
            epochs=100,
            callbacks=callbacks,
            batch_size=128)

    @staticmethod
    def __create_encoder_model(input_dimensions: int, activation: str, r: int, layer_dimensions: list,
                               embedding_dimensions: int, model_name: str):
        """
        Create the model for the markers
        @param input_dimensions:
        @param layer_dimensions: A list of layers dimensions
        The value is the amount of dimensions. e.g. [200,100,50] will create 3 layers.
        One with 200 dimensions, another with 100 and a third with 50 dimensions
        @param activation:
        @param model_name:
        @param r:
        @return:
        """

        inputs = Input(shape=(input_dimensions,))
        x = inputs

        for i, layer in enumerate(layer_dimensions):
            x = Dense(layer, activation=activation, activity_regularizer=r,
                      name=f"{model_name}_layer_{i}")(x)
            x = Dropout(0.3, name=f"{model_name}_layer_{i}_dropout")(x)

        z_mean = Dense(embedding_dimensions, name=f'{model_name}_z_mean')(x)
        z_log_var = Dense(embedding_dimensions, name=f'{model_name}_z_log_var')(x)

        z = Sampling()([z_mean, z_log_var])

        return Model(inputs=inputs, outputs=[z_mean, z_log_var, z], name=model_name), z_mean, z_log_var

    @staticmethod
    def __create_decoder_model(input_dimensions: int, activation: str, r: int, layer_dimensions: list,
                               model_name: str):
        """
        Create the model for the markers
        @param input_dimensions:
        @param layer_dimensions: A list of layers dimensions
        The value is the amount of dimensions. e.g. [200,100,50] will create 3 layers.
        One with 200 dimensions, another with 100 and a third with 50 dimensions
        @param activation:
        @param model_name:
        @param r:
        @return:
        """

        inputs = keras.Input(shape=(input_dimensions,))
        x = inputs

        for i, layer in enumerate(layer_dimensions):
            x = layers.Dense(layer, activation=activation, activity_regularizer=r,
                             name=f"{model_name}_layer_{i}")(x)

        x = Dense(1, activation=activation, name=f"{model_name}_layer_output")(x)

        return Model(inputs, x, name=model_name)
