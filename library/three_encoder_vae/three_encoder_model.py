import tensorflow as tf
from tensorflow import keras
from keras.layers import concatenate


class ThreeEncoderVAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(ThreeEncoderVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
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
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder([z, z, z])
            reconstruction_loss_fn = keras.losses.MeanSquaredError()
            # Concatenate encoders
            data = concatenate([data[0][0], data[0][1]], data[0][2])
            reconstruction_loss = reconstruction_loss_fn(data, reconstruction)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
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
