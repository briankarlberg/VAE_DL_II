import tensorflow as tf
from tensorflow.layers


class DRT_VAE:

    def __init__(self):


    def build_model(self):
        inputs_dim = data.inputs_dim

        r = regularizers.l1_l2(10e-5)
        # mlflow.log_param("regularizer", r)

        encoder_inputs = keras.Input(shape=(inputs_dim,))
        h1 = layers.Dense(inputs_dim, activation=activation, activity_regularizer=r)(encoder_inputs)
        h2 = layers.Dense(inputs_dim / 2, activation=activation, activity_regularizer=r)(h1)
        h3 = layers.Dense(inputs_dim / 3, activation=activation, activity_regularizer=r)(h2)

        z_mean = layers.Dense(latent_space_dimensions, name="z_mean")(h3)
        z_log_var = layers.Dense(latent_space_dimensions, name="z_log_var")(h3)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()

        # Build the decoder
        decoder_inputs = keras.Input(shape=(latent_space_dimensions,))
        h1 = layers.Dense(inputs_dim / 3, activation=activation)(decoder_inputs)
        h2 = layers.Dense(inputs_dim / 2, activation=activation)(h1)

        decoder_outputs = layers.Dense(inputs_dim)(h2)
        decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")
        decoder.summary()
        # mlflow.log_param("decoder_summary", decoder.summary())

        # Visualize the model.
        # tf.keras.utils.plot_model(model, to_file="model.png")

        # Train the VAE
        # Create the VAR, compile, and run.

        callback = tf.keras.callbacks.EarlyStopping(monitor="reconstruction_loss",
                                                    mode="min", patience=5,
                                                    restore_best_weights=True)
        vae = VAE(encoder, decoder)
        vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

        history = vae.fit(data.X_train,
                          validation_data=(data.X_val, data.X_val),
                          epochs=100,
                          callbacks=callback,
                          batch_size=256,
                          shuffle=True,
                          verbose=1)