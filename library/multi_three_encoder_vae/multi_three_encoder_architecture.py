from keras import layers, regularizers
import pandas as pd
import keras
from library.multi_three_encoder_vae.multi_three_encoder_model import MultiThreeEncoderVAE
from library.multi_three_encoder_vae.sampling import Sampling
from keras.layers import Dense, Dropout, Input
import tensorflow as tf
from tensorflow.keras.models import Model
from typing import Tuple
from keras.callbacks import TerminateOnNaN, CSVLogger, ModelCheckpoint, EarlyStopping
import os


# https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf


class MultiThreeEncoderArchitecture:

    def __init__(self):
        self._coding_gene_encoder: Model = None
        self._non_coding_gene_encoder: Model = None
        self._molecular_fingerprints_encoder: Model = None

        self._coding_gene_decoder: Model = None
        self._non_coding_gene_decoder: Model = None
        self._molecular_fingerprints_decoder: Model = None

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
    def coding_gene_decoder(self) -> Model:
        return self._coding_gene_decoder

    @property
    def non_coding_gene_decoder(self) -> Model:
        return self._non_coding_gene_decoder

    @property
    def molecular_fingerprints_decoder(self) -> Model:
        return self._molecular_fingerprints_decoder

    @property
    def vae(self):
        return self._vae

    @property
    def history(self):
        return self._history

    def build_three_variational_auto_encoder(self, training_data: Tuple,
                                             validation_data: Tuple,
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

        r = regularizers.l1_l2(10e-5)

        # Switch network when layers are redefined

        self._coding_gene_encoder: Model = self.__create_encoder_model(
            input_dimensions=coding_gene_training_data.shape[1], activation=activation,
            layer_dimensions=coding_gene_layers, r=r, model_name="coding_genes_encoder",
            embedding_dimensions=embedding_dimension)

        self._non_coding_gene_encoder = self.__create_encoder_model(
            input_dimensions=non_coding_gene_training_data.shape[1], activation=activation,
            layer_dimensions=non_coding_gene_layers, r=r, model_name="non_coding_genes_encoder",
            embedding_dimensions=embedding_dimension)

        self._molecular_fingerprints_encoder = self.__create_encoder_model(
            input_dimensions=molecular_fingerprints_training_data.shape[1], activation=activation,
            layer_dimensions=molecular_fingerprint_layers, r=r, model_name="molecular_fingerprint_encoder",
            embedding_dimensions=embedding_dimension)

        # Reverse layer lists
        coding_gene_layers.reverse()
        non_coding_gene_layers.reverse()
        molecular_fingerprint_layers.reverse()

        self._coding_gene_decoder: Model = self.__create_decoder_model(
            input_dimensions=embedding_dimension, activation=activation,
            layer_dimensions=coding_gene_layers, r=r, model_name="coding_genes_decoder")

        self._coding_gene_decoder.summary()

        self._non_coding_gene_decoder: Model = self.__create_decoder_model(
            input_dimensions=embedding_dimension, activation=activation,
            layer_dimensions=non_coding_gene_layers, r=r, model_name="non_coding_genes_decoder")

        self._non_coding_gene_decoder.summary()

        self._molecular_fingerprints_decoder: Model = self.__create_decoder_model(
            input_dimensions=embedding_dimension, activation=activation,
            layer_dimensions=molecular_fingerprint_layers, r=r, model_name="molecular_fingerprint_decoder")

        self._molecular_fingerprints_decoder.summary()

        callbacks = []

        early_stopping = EarlyStopping(monitor="reconstruction_loss",
                                       mode="min", patience=5,
                                       restore_best_weights=True)
        callbacks.append(early_stopping)

        term_nan = TerminateOnNaN()
        callbacks.append(term_nan)

        csv_logger = CSVLogger(os.path.join(folder, 'training.log'), separator='\t')
        callbacks.append(csv_logger)

        self._vae = MultiThreeEncoderVAE(self._coding_gene_encoder, self._non_coding_gene_encoder,
                                         self._molecular_fingerprints_encoder,
                                         self._coding_gene_decoder, self._non_coding_gene_decoder,
                                         self._molecular_fingerprints_decoder)
        self._vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), run_eagerly=True)
        # self._vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

        self._history = self._vae.fit(
            x=[coding_gene_training_data, non_coding_gene_training_data, molecular_fingerprints_training_data],
            epochs=500,
            callbacks=callbacks,
            batch_size=256,
            shuffle=True,
            verbose=1)

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

        # Model(inputs=encoder_input, outputs=[z_mean, z_log_var, z], name="marker_encoder")
        return Model(inputs=inputs, outputs=[z_mean, z_log_var, z], name=model_name)

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

        return Model(inputs=inputs, outputs=x, name=model_name)
