#!/usr/bin/env python3
import tensorflow as tf

def flip_image(image):
    return tf.image.flip_left_right(image)


if __name__ == "__main__":
    import tensorflow_datasets as tfds
    from tensorflow.keras.utils import save_img

    tf.compat.v1.enable_eager_execution()
    tf.compat.v1.set_random_seed(0)

    doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
    for image, _ in doggies.shuffle(10).take(1):
        save_img("./images/before_flip.jpg", image)
        save_img("./images/flip.jpg", flip_image(image))