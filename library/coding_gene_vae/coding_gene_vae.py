import pandas as pd
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from library.coding_gene_vae.sampling import Sampling
import tensorflow as tf
from typing import List
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, CSVLogger
import os


class CodingGeneVae:

    def __init__(self, input_dimension: int, embedding_dimension: int, layer_count: int):
        self._layer_count = layer_count
        self._embedding_dimension = embedding_dimension
        self._encoder = self.__build_encoder(layers=layer_count, input_dimensions=input_dimension)
        self._decoder = self.__build_decoder(layers=layer_count, input_dimensions=input_dimension)
        self._vae = None
        self._history = None

    @property
    def history(self):
        return self._history

    def __build_encoder(self, layers: int, input_dimensions: int) -> Model:
        input_layer = Input(shape=(input_dimensions,), name="encoder_input")

        x = input_layer
        for layer in range(layers):
            layer += 2
            x = Dense(units=input_dimensions / layer, activation='relu')(x)
            x = Dropout(0.2)(x)

        self._z_mean = Dense(self._embedding_dimension, name="z_mean")(x)
        self._z_log_var = Dense(self._embedding_dimension, name="z_log_var")(x)
        z = Sampling()([self._z_mean, self._z_log_var])

        return Model(inputs=input_layer, outputs=[self._z_mean, self._z_log_var, z], name="encoder")

    def __build_decoder(self, layers: int, input_dimensions: int):

        input_layer = Input(shape=(self._embedding_dimension,))

        x = input_layer
        for layer in reversed(range(layers)):
            layer += 2
            x = Dense(units=input_dimensions / layer, activation='relu')(x)

        return Model(inputs=input_layer, outputs=x, name="decoder")

    def build_model(self):
        output = self._decoder(self._encoder.outputs[2])
        self._vae = Model(inputs=self._encoder.input, outputs=output, name="coding_gene_vae")

        #   VAE loss terms w/ KL divergence
        def Loss(true, pred):
            reconstruction_loss_fn = tf.keras.losses.MeanSquaredError()
            reconstruction_loss = reconstruction_loss_fn(true, pred)
            kl_loss = -0.5 * (1 + self._z_log_var - tf.square(self._z_mean) - tf.exp(self._z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + 0.0001 * kl_loss

            return total_loss

        losses = {"decoder": Loss}
        loss_weight = {"decoder": 1.0}

        self._vae.compile(loss=losses, loss_weights=loss_weight, optimizer="adam")

        self._vae.summary()

    def train(self, training_data: pd.DataFrame, validation_data: pd.DataFrame, save_path: str):

        callbacks: List = []

        early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=8)
        callbacks.append(early_stop)

        term_nan = TerminateOnNaN()
        callbacks.append(term_nan)

        csv_logger = CSVLogger(os.path.join(save_path, 'training.log'),
                               separator='\t')
        callbacks.append(csv_logger)

        self._history = self._vae.fit(x={"encoder_input": training_data},
                                      validation_data=(validation_data, validation_data),
                                      epochs=500,
                                      callbacks=callbacks,
                                      shuffle=True,
                                      batch_size=96,
                                      verbose=1)
