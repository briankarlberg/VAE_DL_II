import tensorflow as tf
from keras import layers, regularizers
import pandas as pd
import keras
from vae.sampling import Sampling
from vae.vae_model import VAE
import tensorflow as tf
import mlflow
import mlflow.tensorflow
import numpy as np


class DRT_VAE:
    @staticmethod
    def build_model(train_data: np.ndarray, validation_data: np.ndarray, input_dimension: int,
                    embedding_dimension: int, activation='relu'):
        r = regularizers.l1_l2(10e-5)

        encoder_inputs = keras.Input(shape=(input_dimension,))
        h1 = layers.Dense(input_dimension, activation=activation, activity_regularizer=r)(encoder_inputs)
        h2 = layers.Dense(input_dimension / 2, activation=activation, activity_regularizer=r)(h1)
        h3 = layers.Dense(input_dimension / 3, activation=activation, activity_regularizer=r)(h2)
        h4 = layers.Dense(input_dimension / 4, activation=activation, activity_regularizer=r)(h3)

        z_mean = layers.Dense(embedding_dimension, name="z_mean")(h4)
        z_log_var = layers.Dense(embedding_dimension, name="z_log_var")(h4)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()

        # Build the decoder
        decoder_inputs = keras.Input(shape=(embedding_dimension,))
        h1 = layers.Dense(input_dimension / 4, activation=activation)(decoder_inputs)
        h2 = layers.Dense(input_dimension / 3, activation=activation)(h1)
        h3 = layers.Dense(input_dimension / 2, activation=activation)(h2)

        decoder_outputs = layers.Dense(input_dimension)(h3)
        decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")
        decoder.summary()

        # Train the VAE
        # Create the VAR, compile, and run.

        callback = tf.keras.callbacks.EarlyStopping(monitor="reconstruction_loss",
                                                    mode="min", patience=5,
                                                    restore_best_weights=True)
        vae = VAE(encoder, decoder)
        vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

        history = vae.fit(train_data,
                          validation_data=(validation_data, validation_data),
                          epochs=100,
                          callbacks=callback,
                          batch_size=32,
                          shuffle=True,
                          verbose=1)

        return vae, encoder, decoder, history
