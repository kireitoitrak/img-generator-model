import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator

# vae architecture
latent_dim = 32

# encoder
encoder_inputs = keras.Input(shape=(128, 128, 3))
x = layers.Conv2D(32, 4, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 4, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

# reparameterization
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])

# decoder
decoder_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(32 * 32 * 64, activation="relu")(decoder_inputs)
x = layers.Reshape((32, 32, 64))(x)
x = layers.Conv2DTranspose(64, 4, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 4, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(3, 4, activation="sigmoid", padding="same")(x)

# vae model
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")
vae_inputs = encoder_inputs
vae_outputs = decoder(encoder(vae_inputs)[2])
vae = keras.Model(vae_inputs, vae_outputs, name="vae")

# custom loss function for vae
def vae_loss(y_true, y_pred, z_mean, z_log_var):
    reconstruction_loss = keras.losses.mean_squared_error(y_true, y_pred)
    reconstruction_loss *= 128 * 128 * 3  # Image dimensions
    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    return tf.reduce_mean(reconstruction_loss + kl_loss)

# custom loss function model compile
vae.compile(optimizer=keras.optimizers.Adam(), loss=lambda y_true, y_pred: vae_loss(y_true, y_pred, z_mean, z_log_var))

# dataset preprocessing
data_generator = ImageDataGenerator(rescale=1.0/255.0)  # Normalize the pixel values

# dataset loading
batch_size = 32
image_size = (128, 128)  # image size matches the input size of the model
your_dataset = data_generator.flow_from_directory(
    'C:/catalogue/',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='input',  # 'input' for unlabeled data
    shuffle=True,  # shuffle data during training
)

# Train the VAE using the loaded dataset
epochs = 3
# Train the VAE
vae.fit(your_dataset, epochs=epochs)