from keras import layers, regularizers
import pandas as pd
import keras
from library.three_encoder_vae.sampling import Sampling
from keras.layers import concatenate
from library.three_encoder_vae.three_encoder_model import ThreeEncoderVAE
import tensorflow as tf
from library.three_encoder_vae.custom_callbacks import WeightsForBatch
from tensorflow.keras.models import Model
from typing import Tuple
from keras.callbacks import TerminateOnNaN, CSVLogger, ModelCheckpoint, EarlyStopping
from pathlib import Path


# https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf


class ThreeEncoderArchitecture:

    @staticmethod
    def build_three_variational_auto_encoder(training_data: Tuple,
                                             validation_data: Tuple,
                                             embedding_dimension: int,
                                             amount_of_layers: dict,
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

        coding_gene_encoder: Model = ThreeEncoderArchitecture.__create_sub_model(
            input_dimensions=coding_gene_training_data.shape[1], activation=activation,
            layer_dimensions=coding_gene_layers, r=r, model_name="coding_genes_encoder")

        non_coding_gene_encoder = ThreeEncoderArchitecture.__create_sub_model(
            input_dimensions=non_coding_gene_training_data.shape[1], activation=activation,
            layer_dimensions=non_coding_gene_layers, r=r, model_name="non_coding_genes_encoder")

        molecular_fingerprints_encoder = ThreeEncoderArchitecture.__create_sub_model(
            input_dimensions=molecular_fingerprints_training_data.shape[1], activation=activation,
            layer_dimensions=molecular_fingerprint_layers, r=r, model_name="molecular_fingerprint_encoder")

        combined_input = concatenate(
            [coding_gene_encoder.output, non_coding_gene_encoder.output, molecular_fingerprints_encoder.output])

        # Latent space
        z_mean = layers.Dense(embedding_dimension, name="z_mean")(combined_input)
        z_log_var = layers.Dense(embedding_dimension, name="z_log_var")(combined_input)
        z = Sampling()([z_mean, z_log_var])

        encoder = keras.Model(
            inputs=[coding_gene_encoder.input, non_coding_gene_encoder.input, molecular_fingerprints_encoder.input],
            outputs=[z_mean, z_log_var, z], name="encoder")
        encoder.summary()

        # Reverse layer lists
        coding_gene_layers.reverse()
        non_coding_gene_layers.reverse()
        molecular_fingerprint_layers.reverse()

        coding_gene_decoder: Model = ThreeEncoderArchitecture.__create_sub_model(
            input_dimensions=embedding_dimension, activation=activation,
            layer_dimensions=coding_gene_layers, r=r, model_name="coding_genes_decoder")

        coding_gene_decoder.summary()

        non_coding_gene_decoder: Model = ThreeEncoderArchitecture.__create_sub_model(
            input_dimensions=embedding_dimension, activation=activation,
            layer_dimensions=non_coding_gene_layers, r=r, model_name="non_coding_genes_decoder")

        non_coding_gene_decoder.summary()

        molecular_fingerprints_decoder: Model = ThreeEncoderArchitecture.__create_sub_model(
            input_dimensions=embedding_dimension, activation=activation,
            layer_dimensions=non_coding_gene_layers, r=r, model_name="molecular_fingerprint_decoder")

        molecular_fingerprints_decoder.summary()

        decoder = keras.Model(
            inputs=[coding_gene_decoder.input, non_coding_gene_decoder.input, molecular_fingerprints_decoder.input],
            outputs=[coding_gene_decoder.output, non_coding_gene_decoder.output,
                     molecular_fingerprints_decoder.output],
            name="decoder")
        decoder.summary()

        callbacks = []

        early_stopping = EarlyStopping(monitor="reconstruction_loss",
                                       mode="min", patience=5,
                                       restore_best_weights=True)
        callbacks.append(early_stopping)

        term_nan = TerminateOnNaN()
        callbacks.append(term_nan)

        csv_logger = CSVLogger(Path.joinpath("results", 'training.log'),
                               separator='\t')
        callbacks.append(csv_logger)

        vae = ThreeEncoderVAE(encoder, decoder)
        vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        # plot_model(encoder, "encoder.png", show_shapes=True)
        # plot_model(decoder, "decoder.png", show_shapes=True)
        # input()

        history = vae.fit(
            [coding_gene_training_data, non_coding_gene_training_data, molecular_fingerprints_training_data],
            validation_data=(
                [coding_gene_validation_data, non_coding_gene_validation_data, molecular_fingerprints_validation_data],
                [coding_gene_validation_data, non_coding_gene_validation_data, molecular_fingerprints_validation_data]),
            epochs=500,
            callbacks=callbacks,
            batch_size=256,
            shuffle=True,
            verbose=1)

        return vae, encoder, decoder, history

    @staticmethod
    def __create_sub_model(input_dimensions: int, activation: str, r: int, layer_dimensions: list,
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
        x = None
        # Iteration count
        i: int = 0
        for layer in layer_dimensions:
            if i == 0:
                x = layers.Dense(layer, activation=activation, activity_regularizer=r,
                                 name=f"{model_name}_layer_{i}")(inputs)
            else:
                x = layers.Dense(layer, activation=activation, activity_regularizer=r,
                                 name=f"{model_name}_layer_{i}")(x)

            # Increase by 1
            i += 1

        return Model(inputs, x, name=model_name)

    @staticmethod
    def __create_coding_gene_encoder(input_dimensions: int, activation: str, r: int, layer_dimensions: list):
        """
        Create the model for the markers
        @param input_dimensions:
        @param activation:
        @param r:
        @return:
        """
        inputs = keras.Input(shape=(input_dimensions,))
        x = None
        # Keep track of iteration
        i: int = 0
        for layer_dimension in layer_dimensions:
            if i == 0:
                x = layers.Dense(input_dimensions, activation=activation, activity_regularizer=r,
                                 name=f"coding_gene_encoder_layer_{i}")(inputs)
            else:
                x = layers.Dense(layer_dimension, activation=activation, activity_regularizer=r,
                                 name=f"coding_gene_encoder_layer_{i}")(x)

            # Increase counter by 1
            i += 1
        model = Model(inputs, x)

        return model

    @staticmethod
    def __create_coding_gene_decoder(embedding_dimensions: int, activation: str, r: int, output_layers: list):
        """
        Create the model for the markers
        @param embedding_dimensions:
        @param output_layers:
        @param activation:
        @param r:
        @return:
        """

        decoder_inputs = keras.Input(shape=(embedding_dimensions,))
        x = None
        # Iteration count
        i: int = 0
        for layer in output_layers:
            if i == 0:
                x = layers.Dense(layer, activation=activation, activity_regularizer=r,
                                 name=f"coding_gene_decoder_layer_{i}")(decoder_inputs)
            else:
                x = layers.Dense(layer, activation=activation, activity_regularizer=r,
                                 name=f"coding_gene_decoder_layer_{i}")(x)

            # Increase by 1
            i += 1

        return Model(decoder_inputs, x)

    @staticmethod
    def __create_non_coding_gene_encoder(input_dimensions: int, activation: str, r: int, amount_of_layers: int):
        """
        Create the model for the markers
        @param input_dimensions:
        @param activation:
        @param r:
        @return:
        """
        inputs = keras.Input(shape=(input_dimensions,))
        x = None
        for layer in range(amount_of_layers):
            if layer == 0:
                x = layers.Dense(input_dimensions, activation=activation, activity_regularizer=r,
                                 name=f"coding_gene_{layer}")(inputs)
            else:
                x = layers.Dense(input_dimensions, activation=activation, activity_regularizer=r,
                                 name=f"coding_gene_{layer}")(x)

        model = Model(inputs, x)

        return model

    @staticmethod
    def __create_non_coding_gene_decoder(embedding_dimensions: int, activation: str, r: int, output_layers: list):
        """
        Create the model for the markers
        @param embedding_dimensions:
        @param output_layers:
        @param activation:
        @param r:
        @return:
        """

        decoder_inputs = keras.Input(shape=(embedding_dimensions,))
        x = None
        # Iteration count
        i: int = 0
        for layer in output_layers:
            if i == 0:
                x = layers.Dense(layer, activation=activation, activity_regularizer=r,
                                 name=f"coding_gene_decoder_layer_{i}")(decoder_inputs)
            else:
                x = layers.Dense(layer, activation=activation, activity_regularizer=r,
                                 name=f"coding_gene_decoder_layer_{i}")(x)

            # Increase by 1
            i += 1

        return Model(decoder_inputs, x)

    @staticmethod
    def __create_molecular_fingerprint_encoder(input_dimensions: int, activation: str, r: int, amount_of_layers: int):
        """
        Create the model for the markers
        @param input_dimensions:
        @param activation:
        @param r:
        @return:
        """
        inputs = keras.Input(shape=(input_dimensions,))
        x = None
        for layer in range(amount_of_layers):
            if layer == 0:
                x = layers.Dense(input_dimensions, activation=activation, activity_regularizer=r,
                                 name=f"coding_gene_{layer}")(inputs)
            else:
                x = layers.Dense(input_dimensions, activation=activation, activity_regularizer=r,
                                 name=f"coding_gene_{layer}")(x)

        model = Model(inputs, x)

        return model
