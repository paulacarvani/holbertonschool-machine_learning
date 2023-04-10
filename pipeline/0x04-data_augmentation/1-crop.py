#!/usr/bin/env python3
import tensorflow as tf


def crop_image(image, size):
    return tf.image.random_crop(image, size)


if __name__ == "__main__":
    import tensorflow_datasets as tfds
    from tensorflow.keras.utils import save_img
    from tensorflow.python.ops.numpy_ops import np_config
    np_config.enable_numpy_behavior()

    tf.compat.v1.enable_eager_execution()
    tf.compat.v1.set_random_seed(1)

    doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
    for image, _ in doggies.shuffle(10).take(1):
        save_img("./images/before_random_crop.jpg", image)
        cropped = crop_image(image, (200, 200, 3))
        cropped = tf.image.resize(cropped, (image.shape[0], image.shape[1]))
        save_img("./images/random_crop.jpg", cropped)
