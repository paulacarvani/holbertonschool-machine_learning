#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


def shear_image(image, intensity):
    shear = tf.keras.preprocessing.image.random_shear
    return shear(image, intensity, row_axis=0, col_axis=1, channel_axis=2)


if __name__ == "__main__":
    import tensorflow_datasets as tfds
    from tensorflow.keras.utils import save_img

    tf.compat.v1.enable_eager_execution()
    tf.compat.v1.set_random_seed(3)

    doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
    for image, _ in doggies.shuffle(10).take(1):
        save_img("./images/before_shear.jpg", image)
        save_img("./images/shear.jpg", shear_image(image, 50))
