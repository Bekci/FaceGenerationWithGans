from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.initializers import TruncatedNormal
import os
import time
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from keras.preprocessing import image

def get_random_samples(n, b):
    return np.random.randint(0,n, b)

def read_images_from_list(path_list, im_size):
    num_img = []
    for p in path_list:
        img = image.load_img(p, target_size=im_size)
        num_img.append(image.img_to_array(img))
    return np.array(num_img)

def get_batch_of_images(path, batch_size, target_size):
    image_names = np.array(os.listdir(path))
    n = len(image_names)
    idx = get_random_samples(n, batch_size)
    batch_paths = [os.path.join(path, im_name) for im_name in image_names[idx]]
    num_imgs = read_images_from_list(batch_paths, target_size)

    return np.array([normalize_image(x) for x in num_imgs])


def get_generator(z_dim):
    model = Sequential()

    model.add(Dense(8*8*1024, input_dim=z_dim))
    model.add(Reshape((8,8,1024)))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(512, kernel_size=[5,5], strides=[2,2], padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(256, kernel_size=[5,5], strides=[2,2], padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, kernel_size=[5,5], strides=[2,2], padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(64, kernel_size=[5,5], strides=[2,2], padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(3, kernel_size=[5,5], strides=[1,1], padding='same'))

    model.add(Activation('tanh'))

    print("Generator Model")
    model.summary()

    z = Input(shape=(z_dim, ))

    img = model(z)

    return Model(inputs=z, outputs=img)

def get_discriminator(img_shape):

    layer_inits = TruncatedNormal(mean=0.0, stddev=0.02, seed=None)
    model = Sequential()

    model.add(Conv2D(64, kernel_size=[5, 5], strides=[2, 2], input_shape=img_shape, padding='same', kernel_initializer=layer_inits))
    model.add(BatchNormalization(epsilon= 0.00005))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(128, kernel_size=[5, 5], strides=[2, 2], input_shape=img_shape, padding='same', kernel_initializer=layer_inits))
    model.add(BatchNormalization(epsilon= 0.00005))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(256, kernel_size=[5, 5], strides=[2, 2], input_shape=img_shape, padding='same', kernel_initializer=layer_inits))
    model.add(BatchNormalization(epsilon= 0.00005))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(512, kernel_size=[5, 5], strides=[1, 1], input_shape=img_shape, padding='same', kernel_initializer=layer_inits))
    model.add(BatchNormalization(epsilon= 0.00005))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(1024, kernel_size=[5, 5], strides=[2, 2], input_shape=img_shape, padding='same', kernel_initializer=layer_inits))
    model.add(BatchNormalization(epsilon= 0.00005))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    img = Input(shape=img_shape)
    prediction = model(img)

    #print('Discriminator Summary')
    #model.summary()

    return Model(img, prediction)

def sample_images(iteration, z_dim ,image_grid_rows=4, image_grid_columns=4):

    # Sample random noise
    z = np.random.normal(0, 1,
              (image_grid_rows * image_grid_columns, z_dim))

    # Generate images from random noise
    gen_imgs = generator.predict(z)

    gen_imgs = (127.5 * gen_imgs + 1).astype(np.uint8)
    # Set image grid
    fig, axs = plt.subplots(image_grid_rows, image_grid_columns,
                                    figsize=(4,4), sharey=True, sharex=True)
    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            # Output image grid
            axs[i,j].imshow(gen_imgs[cnt, :,:,:])
            axs[i,j].axis('off')
            cnt += 1
    plt.savefig('sample_' + str(iteration) + '.png')

# Apply normalization to each image
def normalize_image(x):
    return x / 127.5 - 1.


def train_model(iterations, batch_size, sample_interval, discriminator, generator, im_path, combined, target_size):

    accs = []
    losses = []

    for iteration in range(iterations):
        start_time = time.time()

        real_images = get_batch_of_images(im_path, batch_size, target_size) + tf.random_normal(shape=tf.shape((batch_size, target_size[0],target_size[1])),
                                                     mean=0.0,
                                                     stddev=random.uniform(0.0, 0.1),
                                                     dtype=tf.float32)

        fake = np.ones((batch_size, 1)) * random.uniform(0.0, 0.1)
        real = np.ones((batch_size, 1))  * random.uniform(0.9, 1.0)

        # Generate random samples
        noise_data = np.random.normal(-1, 1, (batch_size, NOISE_VEC_DIM))
        generated_images = generator.predict(noise_data)

        d_loss_fake = discriminator.train_on_batch(generated_images, fake)

        d_loss_real = discriminator.train_on_batch(real_images, real)

        d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

        g_loss = combined.train_on_batch(noise_data, real)

        end_time = time.time()

        if iteration % sample_interval == 0:

            print('Iteration:{} discriminator loss:{}, generator loss:{}, elapsed time: {}'
                  .format(iteration, d_loss, g_loss, end_time - start_time ))
            losses.append( (d_loss, g_loss) )

            sample_images(iteration, NOISE_VEC_DIM)

            generator.save('generator.h5')

if __name__ == '__main__':

    real_images_dir = os.path.join('resized_celeba')
    losses_file = open('losses.txt', 'w')

    img_rows = 128
    img_cols = 128
    channels = 3
    IMAGE_SIZE = (img_rows, img_cols)
    batch_size = 32
    NOISE_VEC_DIM = 100
    img_shape = (img_rows, img_cols, channels)

    discriminator = get_discriminator(img_shape)
    d_optimizer = Adam(0.00004, 0.5)

    discriminator.compile(loss='binary_crossentropy',
                          optimizer=d_optimizer)

    discriminator.trainable = False

    generator = get_generator(NOISE_VEC_DIM)

    z = Input(shape=(NOISE_VEC_DIM,))
    img = generator(z)

    prediction = discriminator(img)

    g_optimizer = Adam(0.00008, 0.5)
    combined = Model(z, prediction)
    combined.compile(loss='binary_crossentropy', optimizer=g_optimizer)

    iterations = 1000
    sample_interval = 100

    train_model(iterations, batch_size, sample_interval, discriminator, generator, real_images_dir, combined, IMAGE_SIZE)
    losses_file.close()
