import tensorflow as tf
from keras import layers, regularizers
import pandas as pd
import keras
from library.vae.sampling import Sampling
from library.vae.vae_model import CodingGeneVAE
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger


class CodingGeneModel:

    def __init__(self, input_dimensions: int, embedding_dimension: int, save_path: str):
        self._input_dimensions = input_dimensions
        self._vae = None
        self._decoder = None
        self._encoder = None
        self._save_path = save_path
        self._embedding_dimension = embedding_dimension
        self._history = None

    @property
    def history(self):
        return self._history

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    @property
    def vae(self):
        return self._vae

    def compile_model(self, activation='relu'):
        """
        Model is being used for files which do not contain all possible features
        """
        r = regularizers.l1_l2(10e-5)

        encoder_inputs = keras.Input(shape=(self._input_dimensions,))
        h1 = layers.Dense(self._input_dimensions, activation=activation, activity_regularizer=r)(encoder_inputs)
        h2 = layers.Dense(self._input_dimensions / 2, activation=activation, activity_regularizer=r)(h1)
        h3 = layers.Dense(self._input_dimensions / 3, activation=activation, activity_regularizer=r)(h2)
        h4 = layers.Dense(self._input_dimensions / 4, activation=activation, activity_regularizer=r)(h3)

        z_mean = layers.Dense(self._embedding_dimension, name="z_mean")(h4)
        z_log_var = layers.Dense(self._embedding_dimension, name="z_log_var")(h4)
        z = Sampling()([z_mean, z_log_var])
        self._encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        # self._encoder.summary()

        # Build the decoder
        decoder_inputs = keras.Input(shape=(self._embedding_dimension,))
        h1 = layers.Dense(self._input_dimensions / 4, activation=activation)(decoder_inputs)
        h2 = layers.Dense(self._input_dimensions / 3, activation=activation)(h1)
        h3 = layers.Dense(self._input_dimensions / 2, activation=activation)(h2)

        decoder_outputs = layers.Dense(self._input_dimensions)(h3)
        self._decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")
        # self._decoder.summary()

        # Train the VAE
        # Create the VAR, compile, and run.

        self._vae = CodingGeneVAE(self._encoder, self._decoder)
        self._vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

        plot_model(self._encoder, to_file=os.path.join(self._save_path, 'coding_gene_encoder_model.png'),
                   show_shapes=True)
        plot_model(self._decoder, to_file=os.path.join(self._save_path, 'coding_gene_decoder_model.png'),
                   show_shapes=True)
        # plot_model(self._vae, to_file=os.path.join(self._save_path, 'coding_gene_model.png'), show_shapes=True)

    def train_model(self, train_data: np.ndarray, validation_data: np.ndarray):
        callbacks = []

        early_stop = EarlyStopping(monitor="reconstruction_loss",
                                   mode="min", patience=5,
                                   restore_best_weights=True)
        callbacks.append(early_stop)

        csv_logger = CSVLogger(os.path.join(self._save_path, 'training.log'),
                               separator='\t')
        callbacks.append(csv_logger)

        self._history = self._vae.fit(train_data,
                                      validation_data=(validation_data, validation_data),
                                      epochs=500,
                                      callbacks=callbacks,
                                      batch_size=32,
                                      shuffle=True,
                                      verbose=1)
