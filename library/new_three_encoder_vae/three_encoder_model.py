import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from typing import Tuple


class NewThreeEncoderVAE(keras.Model):
    def __init__(self, coding_encoder, non_coding_encoder, mf_encoder, coding_decoder, non_coding_decoder, mf_decoder,
                 **kwargs):
        # TODO: add params for shapes of the data, to make it more generic in the call step
        super(NewThreeEncoderVAE, self).__init__(**kwargs)
        self.coding_encoder = coding_encoder
        self.non_coding_encoder = non_coding_encoder
        self.mf_encoder = mf_encoder

        self.coding_decoder = coding_decoder
        self.non_coding_decoder = non_coding_decoder
        self.mf_decoder = mf_decoder

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            coding_recon_loss, coding_kl_loss = self.calculate_loss(self.coding_encoder,
                                                                    self.coding_decoder, data[0][0])

            non_coding_recon_loss, non_coding_kl_loss = self.calculate_loss(self.non_coding_encoder,
                                                                            self.non_coding_decoder, data[0][1])

            mf_recon_loss, mf_kl_loss = self.calculate_loss(self.mf_encoder, self.mf_decoder, data[0][2])

            total_reconstruction_loss = coding_recon_loss + non_coding_recon_loss + mf_recon_loss
            total_kl_loss = coding_kl_loss + non_coding_kl_loss + mf_kl_loss
            total_loss = total_reconstruction_loss + total_kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(total_reconstruction_loss)
        self.kl_loss_tracker.update_state(total_kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, inputs):

        coding_indices = range(0, 19186)
        non_coding_indices = range(19186, 57820)
        molecular_fingerprint_indices = range(57820, 59868)

        if type(inputs) is tuple:
            coding_genes = inputs[0]
            non_coding_genes = inputs[1]
            molecular_fingerprints = inputs[2]
        else:
            coding_genes = tf.gather(inputs, coding_indices, axis=1)
            non_coding_genes = tf.gather(inputs, non_coding_indices, axis=1)
            molecular_fingerprints = tf.gather(inputs, molecular_fingerprint_indices, axis=1)

        z_mean, z_log_var, z = self.encoder([coding_genes, non_coding_genes, molecular_fingerprints])
        return self.decoder([z, z, z])

    def calculate_loss(self, encoder: Model, decoder: Model, data) -> Tuple:
        z_mean, z_log_var, z = encoder(data)
        reconstruction = decoder(z)
        reconstruction_loss_fn = keras.losses.MeanSquaredError()

        # Calculate los
        recon_loss = reconstruction_loss_fn(data, reconstruction)

        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

        return recon_loss, kl_loss