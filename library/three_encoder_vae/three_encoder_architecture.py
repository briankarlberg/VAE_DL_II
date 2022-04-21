from keras import layers, regularizers
import pandas as pd
import keras
from library.three_encoder_vae.sampling import Sampling
from keras.layers import concatenate
from library.three_encoder_vae.three_encoder_model import ThreeEncoderVAE
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from library.three_encoder_vae.custom_callbacks import CustomCallback, WeightsForBatch
from tensorflow.keras.models import Model
from typing import Tuple


# https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf


class ThreeEncoderArchitecture:

    @staticmethod
    def build_three_variational_auto_encoder(training_data: Tuple,
                                             validation_data: Tuple,
                                             output_dimensions: int,
                                             embedding_dimension: int, activation='relu',
                                             learning_rate: float = 1e-3,
                                             amount_of_layers: Tuple = (5, 5, 5),
                                             optimizer: str = "adam",
                                             use_ml_flow: bool = True):
        """
        Compiles the vae based on the given input parameters
        """

        if len(training_data) != 2:
            raise ValueError("Training and validation data must contain two datasets!")

        coding_gene_training_data: pd.DataFrame = training_data[0]
        non_coding_gene_training_data: pd.DataFrame = training_data[1]
        molecular_fingerprints_training_data: pd.DataFrame = training_data[2]

        coding_gene_validation_data: pd.DataFrame = validation_data[0]
        non_coding_gene_validation_data: pd.DataFrame = validation_data[1]
        molecular_fingerprints_validation_data: pd.DataFrame = validation_data[2]

        coding_gene_layers: int = amount_of_layers[0]
        non_coding_gene_layers: int = amount_of_layers[0]
        molecular_fingerprint_layers: int = amount_of_layers[0]

        r = regularizers.l1_l2(10e-5)

        # Switch network when layers are redefined

        coding_gene_encoder = ThreeEncoderArchitecture.__create_coding_gene_encoder(
            input_dimensions=coding_gene_training_data.shape[1], activation='relu',
            amount_of_layers=coding_gene_layers, r=r)

        non_coding_gene_encoder = ThreeEncoderArchitecture.__create_non_coding_gene_encoder(
            input_dimensions=non_coding_gene_training_data.shape[1], activation=activation,
            amount_of_layers=non_coding_gene_layers, r=r)

        molecular_fingerprints_encoder = ThreeEncoderArchitecture.__create_molecular_fingerprint_encoder(
            input_dimensions=molecular_fingerprints_training_data.shape[1], activation=activation, r=r,
            amount_of_layers=molecular_fingerprint_layers)

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

        # Build the decoder
        decoder_inputs = keras.Input(shape=(embedding_dimension,))
        h1 = layers.Dense(output_dimensions / 3, activation=activation, name="decoding_h1")(decoder_inputs)
        h2 = layers.Dense(output_dimensions / 2.5, activation=activation, name="decoding_h2")(h1)
        h3 = layers.Dense(output_dimensions / 2, activation=activation, name="decoding_h3")(h2)
        h4 = layers.Dense(output_dimensions / 1.5, activation=activation, name="decoding_h4")(h3)

        decoder_outputs = layers.Dense(output_dimensions, name="decoder_output")(h4)
        decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")
        decoder.summary()

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="reconstruction_loss",
                                                          mode="min", patience=5,
                                                          restore_best_weights=True)
        vae = ThreeEncoderVAE(encoder, decoder)
        vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

        history = vae.fit(
            [coding_gene_training_data, non_coding_gene_training_data, molecular_fingerprints_training_data],
            validation_data=(
                [coding_gene_validation_data, non_coding_gene_validation_data, molecular_fingerprints_validation_data],
                [coding_gene_validation_data, non_coding_gene_validation_data, molecular_fingerprints_validation_data]),
            epochs=500,
            callbacks=[early_stopping, WeightsForBatch()],
            batch_size=256,
            shuffle=True,
            verbose=1)

        return vae, encoder, decoder, history

    @staticmethod
    def __create_coding_gene_encoder(input_dimensions: int, activation: str, r: int, amount_of_layers: int):
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
