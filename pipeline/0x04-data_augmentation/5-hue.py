#!/usr/bin/env python3
import tensorflow as tf


def change_hue(image, delta):
    return tf.image.adjust_hue(image, delta)


if __name__ == "__main__":
    import tensorflow_datasets as tfds
    from tensorflow.keras.utils import save_img

    tf.compat.v1.enable_eager_execution()
    tf.compat.v1.set_random_seed(5)

    doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
    for image, _ in doggies.shuffle(10).take(1):
        save_img("./images/before_hue.jpg", image)
        save_img("./images/hue.jpg", change_hue(image, -0.5))
